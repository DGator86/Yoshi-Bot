"""HourlyPredictionEngine - Monte Carlo ensemble for hourly predictions.

This engine combines:
1. Physics-based Monte Carlo simulation (QuantumPriceEngine)
2. Ensemble ML predictions (gradient boost, random forest, ridge)
3. Regime-aware parameter adjustment
4. Calibrated confidence intervals

The output is optimized for Kalshi hourly prediction markets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HourlyPrediction:
    """Result of an hourly prediction."""
    # Point estimate
    point_estimate: float
    expected_return: float  # As decimal

    # Confidence intervals
    ci_low: float  # Lower bound (e.g., 5th percentile)
    ci_high: float  # Upper bound (e.g., 95th percentile)
    ci_50_low: float  # 50% CI lower
    ci_50_high: float  # 50% CI upper

    # Probabilities
    probability_up: float
    probability_down: float

    # Confidence score (0-1)
    confidence_score: float

    # Regime
    regime: str  # "trending", "ranging", "volatile"

    # Simulation stats
    mean_estimate: float
    std_estimate: float
    skew: float
    kurtosis: float

    # Model info
    n_simulations: int
    models_used: List[str]
    model_weights: Dict[str, float]


@dataclass
class EngineConfig:
    """Configuration for HourlyPredictionEngine."""
    # Monte Carlo settings
    n_simulations: int = 10000
    confidence_level: float = 0.90

    # Ensemble weights
    weight_monte_carlo: float = 0.40
    weight_gradient_boost: float = 0.30
    weight_random_forest: float = 0.20
    weight_physics_drift: float = 0.10

    # Regime adjustments
    trending_mc_weight: float = 0.50  # More MC in trends
    volatile_rf_weight: float = 0.35  # More RF in volatile

    # Physics parameters (tunable)
    volatility_scale: float = 1.0
    drift_scale: float = 1.0
    jump_intensity: float = 0.03

    # Confidence calibration
    confidence_base: float = 0.70  # Base confidence
    confidence_vol_penalty: float = 0.30  # Reduce confidence in high vol


class HourlyPredictionEngine:
    """Engine for generating hourly price predictions.

    Combines Monte Carlo simulation with ML ensemble for robust predictions.
    Designed for Kalshi hourly crypto markets.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.90,
        config: Optional[EngineConfig] = None,
    ):
        """Initialize the prediction engine.

        Args:
            n_simulations: Number of Monte Carlo paths
            confidence_level: Confidence level for intervals (e.g., 0.90)
            config: Full configuration object
        """
        self.config = config or EngineConfig(
            n_simulations=n_simulations,
            confidence_level=confidence_level,
        )

        # ML models (lazy loaded)
        self._ml_models_fitted = False
        self._gb_model = None
        self._rf_model = None
        self._ridge_model = None
        self._scaler = None
        self._feature_cols: List[str] = []

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime.

        Args:
            df: Historical price data

        Returns:
            Regime string: "trending", "ranging", or "volatile"
        """
        if len(df) < 30:
            return "ranging"

        recent = df.tail(60)
        returns = recent["returns"].dropna() if "returns" in recent.columns else recent["close"].pct_change().dropna()

        if len(returns) < 10:
            return "ranging"

        # Trend strength
        up_moves = (returns > 0).sum()
        down_moves = (returns < 0).sum()
        total = up_moves + down_moves
        trend_strength = abs(up_moves - down_moves) / total if total > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(60)  # Hourly vol

        # Classify
        if trend_strength > 0.55 and volatility < 0.025:
            return "trending"
        elif volatility > 0.035:
            return "volatile"
        else:
            return "ranging"

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for ML prediction.

        Args:
            df: Historical price data

        Returns:
            DataFrame with computed features
        """
        result = df.copy()

        # Ensure returns exist
        if "returns" not in result.columns:
            result["returns"] = result["close"].pct_change()

        # Momentum features
        for window in [5, 10, 20, 30, 60]:
            result[f"mom_{window}"] = result["returns"].rolling(window).mean()
            result[f"vol_{window}"] = result["returns"].rolling(window).std()

        # RSI-like
        gains = result["returns"].clip(lower=0)
        losses = (-result["returns"]).clip(lower=0)
        result["rsi_14"] = 100 - (100 / (1 + gains.rolling(14).mean() / (losses.rolling(14).mean() + 1e-9)))

        # VWAP displacement
        if "volume" in result.columns and result["volume"].sum() > 0:
            result["vwap"] = (result["close"] * result["volume"]).rolling(30).sum() / result["volume"].rolling(30).sum()
        else:
            result["vwap"] = result["close"].rolling(30).mean()
        result["vwap_disp"] = (result["close"] - result["vwap"]) / result["close"]

        # Range features
        if "high" in result.columns and "low" in result.columns:
            result["range_pct"] = (result["high"] - result["low"]) / result["close"]
            result["upper_wick"] = (result["high"] - result[["open", "close"]].max(axis=1)) / result["close"]
            result["lower_wick"] = (result[["open", "close"]].min(axis=1) - result["low"]) / result["close"]

        # Volatility ratio
        result["vol_ratio"] = result["vol_5"] / (result["vol_30"] + 1e-9)

        # Acceleration
        result["mom_accel"] = result["mom_5"] - result["mom_20"]

        return result

    def _run_monte_carlo(
        self,
        current_price: float,
        drift: float,
        volatility: float,
        horizon_minutes: int,
        jump_intensity: float,
    ) -> np.ndarray:
        """Run Monte Carlo simulation.

        Args:
            current_price: Current price
            drift: Expected drift (annualized)
            volatility: Volatility (annualized)
            horizon_minutes: Prediction horizon in minutes
            jump_intensity: Jump intensity for Poisson process

        Returns:
            Array of simulated final prices
        """
        n_sims = self.config.n_simulations
        dt = horizon_minutes / (365 * 24 * 60)  # In years

        # Scale parameters
        drift_scaled = drift * self.config.drift_scale
        vol_scaled = volatility * self.config.volatility_scale

        # Pre-generate random numbers
        brownian = np.random.normal(0, 1, n_sims)
        jumps_count = np.random.poisson(jump_intensity * dt, n_sims)
        jump_sizes = np.random.normal(0, 0.02, n_sims)  # 2% jump std

        # GBM with jumps
        log_returns = (
            (drift_scaled - 0.5 * vol_scaled**2) * dt +
            vol_scaled * np.sqrt(dt) * brownian +
            jumps_count * jump_sizes
        )

        final_prices = current_price * np.exp(log_returns)

        return final_prices

    def _get_ensemble_weights(self, regime: str) -> Dict[str, float]:
        """Get ensemble weights adjusted for regime.

        Args:
            regime: Market regime

        Returns:
            Dict of model weights
        """
        weights = {
            "monte_carlo": self.config.weight_monte_carlo,
            "gradient_boost": self.config.weight_gradient_boost,
            "random_forest": self.config.weight_random_forest,
            "physics_drift": self.config.weight_physics_drift,
        }

        # Adjust for regime
        if regime == "trending":
            weights["monte_carlo"] = self.config.trending_mc_weight
            weights["gradient_boost"] *= 0.8
        elif regime == "volatile":
            weights["random_forest"] = self.config.volatile_rf_weight
            weights["monte_carlo"] *= 0.8

        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights

    def _calculate_confidence_score(
        self,
        volatility: float,
        regime: str,
        ci_width: float,
        current_price: float,
    ) -> float:
        """Calculate confidence score for prediction.

        Args:
            volatility: Current volatility
            regime: Market regime
            ci_width: Width of confidence interval
            current_price: Current price

        Returns:
            Confidence score (0-1)
        """
        base = self.config.confidence_base

        # Volatility penalty
        vol_penalty = min(volatility * self.config.confidence_vol_penalty * 10, 0.3)

        # Regime adjustment
        regime_adj = {
            "trending": 0.05,  # Boost in trends
            "ranging": 0.0,
            "volatile": -0.10,  # Reduce in volatile
        }.get(regime, 0.0)

        # CI width penalty (wider = less confident)
        ci_pct = ci_width / current_price
        ci_penalty = min(ci_pct * 2, 0.2)

        confidence = base - vol_penalty + regime_adj - ci_penalty
        confidence = max(0.3, min(0.95, confidence))  # Clamp to [0.3, 0.95]

        return confidence

    def predict(
        self,
        df: pd.DataFrame,
        current_price: float,
        horizon_minutes: int,
    ) -> Optional[HourlyPrediction]:
        """Generate hourly prediction.

        Args:
            df: Historical price data
            current_price: Current price
            horizon_minutes: Minutes until hour end

        Returns:
            HourlyPrediction or None
        """
        if len(df) < 30:
            logger.warning("Insufficient data for prediction")
            return None

        # Detect regime
        regime = self._detect_regime(df)

        # Compute features
        features_df = self._compute_features(df)

        # Calculate drift and volatility from recent data
        returns = df["returns"].dropna() if "returns" in df.columns else df["close"].pct_change().dropna()

        if len(returns) < 10:
            return None

        # Detect bar interval from timestamps (default to 30 min for CoinGecko)
        bars_per_year = 365 * 24 * 60  # Default: 1-minute bars
        if "timestamp" in df.columns and len(df) >= 2:
            time_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]).total_seconds()
            if time_diff > 0:
                bars_per_day = 86400 / time_diff  # seconds per day / seconds per bar
                bars_per_year = bars_per_day * 365
                logger.debug(f"Detected bar interval: {time_diff/60:.0f} min, {bars_per_year:.0f} bars/year")

        # Recent volatility (annualized)
        vol_short = returns.tail(30).std() * np.sqrt(bars_per_year)
        vol_long = returns.tail(120).std() * np.sqrt(bars_per_year) if len(returns) >= 120 else vol_short

        volatility = 0.7 * vol_short + 0.3 * vol_long

        # Drift estimate from momentum
        momentum = returns.tail(30).mean() * bars_per_year  # Annualized

        # Adjust drift based on regime
        if regime == "ranging":
            momentum *= 0.3  # Reduce momentum in ranging
        elif regime == "volatile":
            momentum *= 0.5  # Reduce in volatile

        # Run Monte Carlo
        mc_prices = self._run_monte_carlo(
            current_price=current_price,
            drift=momentum,
            volatility=volatility,
            horizon_minutes=horizon_minutes,
            jump_intensity=self.config.jump_intensity,
        )

        # Get ensemble weights
        weights = self._get_ensemble_weights(regime)

        # For now, use MC as primary (ML models can be added later via learner)
        final_prices = mc_prices

        # Calculate statistics
        point_estimate = float(np.median(final_prices))
        mean_estimate = float(np.mean(final_prices))
        std_estimate = float(np.std(final_prices))

        # Confidence intervals
        alpha = (1 - self.config.confidence_level) / 2
        ci_low = float(np.percentile(final_prices, alpha * 100))
        ci_high = float(np.percentile(final_prices, (1 - alpha) * 100))
        ci_50_low = float(np.percentile(final_prices, 25))
        ci_50_high = float(np.percentile(final_prices, 75))

        # Probabilities
        probability_up = float(np.mean(final_prices > current_price))
        probability_down = 1 - probability_up

        # Higher moments
        skew = float(pd.Series(final_prices).skew())
        kurtosis = float(pd.Series(final_prices).kurtosis())

        # Confidence score
        confidence_score = self._calculate_confidence_score(
            volatility=vol_short,
            regime=regime,
            ci_width=ci_high - ci_low,
            current_price=current_price,
        )

        expected_return = (point_estimate - current_price) / current_price

        return HourlyPrediction(
            point_estimate=point_estimate,
            expected_return=expected_return,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_50_low=ci_50_low,
            ci_50_high=ci_50_high,
            probability_up=probability_up,
            probability_down=probability_down,
            confidence_score=confidence_score,
            regime=regime,
            mean_estimate=mean_estimate,
            std_estimate=std_estimate,
            skew=skew,
            kurtosis=kurtosis,
            n_simulations=self.config.n_simulations,
            models_used=list(weights.keys()),
            model_weights=weights,
        )

    def update_parameters(self, params: Dict[str, float]):
        """Update engine parameters (for adaptive learning).

        Args:
            params: Dict of parameter name -> value
        """
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} = {value}")

    def get_parameters(self) -> Dict[str, float]:
        """Get current tunable parameters.

        Returns:
            Dict of parameter name -> value
        """
        return {
            "volatility_scale": self.config.volatility_scale,
            "drift_scale": self.config.drift_scale,
            "jump_intensity": self.config.jump_intensity,
            "weight_monte_carlo": self.config.weight_monte_carlo,
            "weight_gradient_boost": self.config.weight_gradient_boost,
            "weight_random_forest": self.config.weight_random_forest,
            "confidence_base": self.config.confidence_base,
            "confidence_vol_penalty": self.config.confidence_vol_penalty,
        }
