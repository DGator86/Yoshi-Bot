"""AdaptiveLearner - ML optimization after each hour.

Analyzes prediction outcomes and adapts engine parameters to improve
future predictions. Implements online learning with exploration.
"""
from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearnerConfig:
    """Configuration for AdaptiveLearner."""
    # Learning rate
    learning_rate: float = 0.1

    # Exploration
    exploration_rate: float = 0.1  # Probability of random exploration
    exploration_decay: float = 0.995  # Decay per hour

    # Constraints
    min_samples_to_adapt: int = 10
    max_parameter_change: float = 0.3  # Max 30% change per adaptation

    # Parameter bounds
    parameter_bounds: Dict[str, tuple] = field(default_factory=lambda: {
        "volatility_scale": (0.5, 2.0),
        "drift_scale": (0.3, 2.0),
        "jump_intensity": (0.01, 0.10),
        "weight_monte_carlo": (0.2, 0.6),
        "weight_gradient_boost": (0.1, 0.5),
        "weight_random_forest": (0.1, 0.4),
        "confidence_base": (0.5, 0.85),
        "confidence_vol_penalty": (0.1, 0.5),
    })

    # Optimization targets
    target_direction_accuracy: float = 0.60
    target_ci_coverage: float = 0.90

    # Storage
    storage_path: Optional[str] = None


class AdaptiveLearner:
    """Online learning system for prediction parameters.

    After each hour:
    1. Analyzes prediction outcomes
    2. Computes gradients for each parameter
    3. Updates parameters to improve direction accuracy and CI coverage
    4. Occasionally explores new parameter regions

    Implements a simple policy gradient approach with exploration.
    """

    def __init__(
        self,
        engine,
        config: Optional[LearnerConfig] = None,
    ):
        """Initialize AdaptiveLearner.

        Args:
            engine: HourlyPredictionEngine to adapt
            config: Configuration object
        """
        self.engine = engine
        self.config = config or LearnerConfig()

        if self.config.storage_path is None:
            self.config.storage_path = os.path.expanduser("~/.yoshi/learner_state.json")

        self.storage_path = Path(self.config.storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # State
        self._parameter_history: List[Dict] = []
        self._performance_history: List[Dict] = []
        self._exploration_rate = self.config.exploration_rate

        # Load state
        self._load_state()

    def _load_state(self):
        """Load learner state from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    state = json.load(f)
                self._parameter_history = state.get("parameter_history", [])
                self._performance_history = state.get("performance_history", [])
                self._exploration_rate = state.get("exploration_rate", self.config.exploration_rate)
                logger.info(f"Loaded learner state with {len(self._performance_history)} history points")
            except Exception as e:
                logger.error(f"Failed to load learner state: {e}")

    def _save_state(self):
        """Save learner state to storage."""
        try:
            state = {
                "parameter_history": self._parameter_history[-100:],  # Keep last 100
                "performance_history": self._performance_history[-100:],
                "exploration_rate": self._exploration_rate,
            }
            with open(self.storage_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learner state: {e}")

    def _compute_reward(self, results: Dict[str, dict]) -> float:
        """Compute reward signal from hourly results.

        Args:
            results: Dict of symbol -> evaluation results

        Returns:
            Reward value (higher is better)
        """
        if not results:
            return 0.0

        # Aggregate metrics across symbols
        direction_accuracies = []
        ci_coverages = []
        errors = []

        for symbol, metrics in results.items():
            if metrics:
                direction_accuracies.append(1.0 if metrics.get("direction_correct", False) else 0.0)
                ci_coverages.append(1.0 if metrics.get("in_ci_range", False) else 0.0)
                if metrics.get("prediction_error_pct") is not None:
                    errors.append(metrics["prediction_error_pct"])

        if not direction_accuracies:
            return 0.0

        avg_direction = np.mean(direction_accuracies)
        avg_ci_coverage = np.mean(ci_coverages) if ci_coverages else 0.0
        avg_error = np.mean(errors) if errors else 10.0  # Penalize missing errors

        # Reward = weighted combination
        # Direction accuracy is most important for Kalshi
        reward = (
            0.6 * avg_direction +
            0.3 * avg_ci_coverage +
            0.1 * max(0, 1 - avg_error / 5)  # Error penalty (5% error = 0 contribution)
        )

        return reward

    def _compute_gradients(
        self,
        current_params: Dict[str, float],
        results: Dict[str, dict],
    ) -> Dict[str, float]:
        """Compute parameter gradients based on results.

        Uses finite difference approximation based on parameter history.

        Args:
            current_params: Current parameter values
            results: Evaluation results

        Returns:
            Dict of parameter -> gradient estimate
        """
        gradients = {param: 0.0 for param in current_params}

        if len(self._performance_history) < 3:
            return gradients

        # Use recent history to estimate gradients
        recent = self._performance_history[-10:]

        for param in current_params:
            # Find variations in this parameter
            param_values = [h["params"].get(param, current_params[param]) for h in recent]
            rewards = [h["reward"] for h in recent]

            if len(set(param_values)) < 2:
                continue  # No variation, can't estimate gradient

            # Simple correlation-based gradient
            param_array = np.array(param_values)
            reward_array = np.array(rewards)

            # Normalize
            param_norm = (param_array - param_array.mean()) / (param_array.std() + 1e-9)
            reward_norm = (reward_array - reward_array.mean()) / (reward_array.std() + 1e-9)

            # Correlation as gradient estimate
            gradient = np.mean(param_norm * reward_norm)
            gradients[param] = gradient

        return gradients

    def _apply_gradients(
        self,
        current_params: Dict[str, float],
        gradients: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply gradients to update parameters.

        Args:
            current_params: Current parameter values
            gradients: Parameter gradients

        Returns:
            Updated parameters
        """
        new_params = current_params.copy()

        for param, gradient in gradients.items():
            if param not in self.config.parameter_bounds:
                continue

            bounds = self.config.parameter_bounds[param]
            current_value = current_params[param]

            # Compute update
            delta = self.config.learning_rate * gradient * current_value

            # Clamp delta to max change
            max_delta = self.config.max_parameter_change * current_value
            delta = np.clip(delta, -max_delta, max_delta)

            # Apply update
            new_value = current_value + delta

            # Clamp to bounds
            new_value = np.clip(new_value, bounds[0], bounds[1])

            new_params[param] = float(new_value)

        return new_params

    def _explore(self, current_params: Dict[str, float]) -> Dict[str, float]:
        """Randomly explore parameter space.

        Args:
            current_params: Current parameters

        Returns:
            Explored parameters
        """
        new_params = current_params.copy()

        # Randomly perturb one or two parameters
        params_to_explore = np.random.choice(
            list(current_params.keys()),
            size=min(2, len(current_params)),
            replace=False,
        )

        for param in params_to_explore:
            if param not in self.config.parameter_bounds:
                continue

            bounds = self.config.parameter_bounds[param]

            # Random value within bounds, biased toward current
            current = current_params[param]
            range_width = bounds[1] - bounds[0]

            # Gaussian perturbation around current
            perturbation = np.random.normal(0, range_width * 0.1)
            new_value = current + perturbation

            # Clamp to bounds
            new_value = np.clip(new_value, bounds[0], bounds[1])

            new_params[param] = float(new_value)
            logger.info(f"Exploring {param}: {current:.4f} -> {new_value:.4f}")

        return new_params

    def adapt(self, results: Dict[str, dict]) -> Dict[str, float]:
        """Adapt parameters based on hourly results.

        Args:
            results: Dict of symbol -> evaluation results

        Returns:
            New parameter values
        """
        # Get current parameters
        current_params = self.engine.get_parameters()

        # Compute reward
        reward = self._compute_reward(results)
        logger.info(f"Hourly reward: {reward:.4f}")

        # Record performance
        self._performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "params": current_params.copy(),
            "reward": reward,
            "results": {k: {key: v for key, v in val.items() if not isinstance(v, (dict, list))}
                       for k, val in results.items() if val},
        })

        # Check if we should explore
        if np.random.random() < self._exploration_rate:
            logger.info("Exploration mode - randomly perturbing parameters")
            new_params = self._explore(current_params)
        else:
            # Compute and apply gradients
            gradients = self._compute_gradients(current_params, results)
            new_params = self._apply_gradients(current_params, gradients)

            # Log significant changes
            for param in new_params:
                old_val = current_params[param]
                new_val = new_params[param]
                if abs(new_val - old_val) / (old_val + 1e-9) > 0.01:
                    logger.info(f"Adapting {param}: {old_val:.4f} -> {new_val:.4f}")

        # Decay exploration rate
        self._exploration_rate *= self.config.exploration_decay
        self._exploration_rate = max(self._exploration_rate, 0.01)  # Minimum 1%

        # Update engine
        self.engine.update_parameters(new_params)

        # Record new parameters
        self._parameter_history.append({
            "timestamp": datetime.now().isoformat(),
            "params": new_params,
        })

        # Save state
        self._save_state()

        return new_params

    def get_performance_summary(self) -> Dict:
        """Get summary of learning performance.

        Returns:
            Dict with learning metrics
        """
        if not self._performance_history:
            return {"n_adaptations": 0, "avg_reward": 0}

        recent = self._performance_history[-24:]  # Last 24 hours
        rewards = [h["reward"] for h in recent]

        return {
            "n_adaptations": len(self._performance_history),
            "avg_reward": np.mean(rewards) if rewards else 0,
            "reward_trend": np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 3 else 0,
            "exploration_rate": self._exploration_rate,
            "current_params": self.engine.get_parameters(),
        }

    def reset(self):
        """Reset learner state (use for debugging)."""
        self._parameter_history = []
        self._performance_history = []
        self._exploration_rate = self.config.exploration_rate
        self._save_state()
        logger.info("Learner state reset")

    def force_explore(self):
        """Force exploration on next adaptation."""
        self._exploration_rate = 1.0
        logger.info("Forced exploration on next adaptation")
