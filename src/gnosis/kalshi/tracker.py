"""PredictionTracker - Store and evaluate predictions.

Tracks predictions, compares them to actuals, and provides
performance metrics for ML adaptation.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """A single prediction record with outcome."""
    # Identification
    symbol: str
    prediction_time: datetime
    hour_end: datetime

    # Prediction
    predicted_price: float
    predicted_return_pct: float
    ci_low: float
    ci_high: float
    confidence_pct: float
    probability_direction: float
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    regime: str

    # Benchmark (at prediction time)
    benchmark_price: float

    # Actual outcome (filled after hour ends)
    actual_price: Optional[float] = None
    actual_return_pct: Optional[float] = None
    actual_direction: Optional[str] = None

    # Evaluation (filled after hour ends)
    direction_correct: Optional[bool] = None
    in_ci_range: Optional[bool] = None
    prediction_error_pct: Optional[float] = None
    ci_width_pct: Optional[float] = None

    # Model info
    model_version: str = "v1"
    parameters: Dict = field(default_factory=dict)

    def evaluate(self, actual_price: float):
        """Evaluate prediction against actual outcome.

        Args:
            actual_price: Actual price at hour end
        """
        self.actual_price = actual_price
        self.actual_return_pct = (actual_price - self.benchmark_price) / self.benchmark_price * 100

        # Direction
        if self.actual_return_pct > 0.05:
            self.actual_direction = "UP"
        elif self.actual_return_pct < -0.05:
            self.actual_direction = "DOWN"
        else:
            self.actual_direction = "NEUTRAL"

        # Direction correctness
        self.direction_correct = self.predicted_direction == self.actual_direction

        # CI range check
        self.in_ci_range = self.ci_low <= actual_price <= self.ci_high

        # Prediction error
        self.prediction_error_pct = abs(self.predicted_return_pct - self.actual_return_pct)

        # CI width
        self.ci_width_pct = (self.ci_high - self.ci_low) / self.benchmark_price * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert datetime to ISO format
        d["prediction_time"] = self.prediction_time.isoformat()
        d["hour_end"] = self.hour_end.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        """Create from dictionary."""
        d["prediction_time"] = datetime.fromisoformat(d["prediction_time"])
        d["hour_end"] = datetime.fromisoformat(d["hour_end"])
        return cls(**d)


class PredictionTracker:
    """Track predictions and evaluate performance.

    Stores predictions, matches them with actuals, and provides
    metrics for the adaptive learner.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize PredictionTracker.

        Args:
            storage_path: Path to store prediction history (default: ~/.yoshi/predictions.json)
        """
        if storage_path is None:
            storage_path = os.path.expanduser("~/.yoshi/predictions.json")
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._pending: Dict[str, List[PredictionRecord]] = {}  # Awaiting evaluation
        self._evaluated: List[PredictionRecord] = []  # Evaluated predictions
        self._last_hour_results: Dict[str, dict] = {}

        # Load existing data
        self._load()

    def _load(self):
        """Load predictions from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self._evaluated = [PredictionRecord.from_dict(d) for d in data.get("evaluated", [])]
                logger.info(f"Loaded {len(self._evaluated)} historical predictions")
            except Exception as e:
                logger.error(f"Failed to load predictions: {e}")
                self._evaluated = []

    def _save(self):
        """Save predictions to storage."""
        try:
            data = {
                "evaluated": [r.to_dict() for r in self._evaluated[-1000:]],  # Keep last 1000
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")

    def record_prediction(self, signal) -> PredictionRecord:
        """Record a new prediction.

        Args:
            signal: PredictionSignal object

        Returns:
            PredictionRecord
        """
        record = PredictionRecord(
            symbol=signal.symbol,
            prediction_time=signal.timestamp,
            hour_end=signal.hour_end,
            predicted_price=signal.predicted_price,
            predicted_return_pct=signal.predicted_return_pct,
            ci_low=signal.range_low,
            ci_high=signal.range_high,
            confidence_pct=signal.prediction_confidence_pct,
            probability_direction=signal.probability_direction,
            predicted_direction=signal.direction.value,
            regime=signal.regime.value,
            benchmark_price=signal.current_price,
            model_version=signal.model_version,
        )

        # Add to pending
        if signal.symbol not in self._pending:
            self._pending[signal.symbol] = []
        self._pending[signal.symbol].append(record)

        logger.debug(f"Recorded prediction for {signal.symbol}: {signal.predicted_price:.2f}")
        return record

    def evaluate_last_hour(self) -> Dict[str, dict]:
        """Evaluate all pending predictions from the last hour.

        Call this at the start of each hour to evaluate predictions
        from the previous hour.

        Returns:
            Dict of symbol -> evaluation results
        """
        results = {}
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        for symbol, records in self._pending.items():
            # Find predictions for the hour that just ended
            to_evaluate = [
                r for r in records
                if r.hour_end <= now and r.hour_end > hour_ago
            ]

            if not to_evaluate:
                continue

            # Get actual price (use the most recent prediction's hour_end)
            # In production, you'd fetch the actual price from the exchange
            actual_price = self._get_actual_price(symbol, to_evaluate[-1].hour_end)

            if actual_price is None:
                logger.warning(f"Could not get actual price for {symbol}")
                continue

            # Evaluate each prediction
            for record in to_evaluate:
                record.evaluate(actual_price)
                self._evaluated.append(record)

            # Remove from pending
            self._pending[symbol] = [r for r in records if r not in to_evaluate]

            # Compute aggregate results
            direction_correct = sum(1 for r in to_evaluate if r.direction_correct) / len(to_evaluate)
            in_ci_range = sum(1 for r in to_evaluate if r.in_ci_range) / len(to_evaluate)
            avg_error = sum(r.prediction_error_pct or 0 for r in to_evaluate) / len(to_evaluate)

            # Use last prediction for representative values
            last = to_evaluate[-1]
            results[symbol] = {
                "direction_correct": last.direction_correct,
                "in_ci_range": last.in_ci_range,
                "predicted_return_pct": last.predicted_return_pct,
                "actual_return_pct": last.actual_return_pct,
                "prediction_error_pct": last.prediction_error_pct,
                "direction_accuracy": direction_correct,
                "ci_coverage": in_ci_range,
                "avg_error": avg_error,
                "n_predictions": len(to_evaluate),
            }

        self._last_hour_results = results
        self._save()

        return results

    def _get_actual_price(self, symbol: str, hour_end: datetime) -> Optional[float]:
        """Get actual price at hour end.

        In production, this would fetch from exchange API.
        For now, uses the monitor if available.

        Args:
            symbol: Symbol
            hour_end: Hour end time

        Returns:
            Actual price or None
        """
        # Try to import monitor to get price
        try:
            from .monitor import PriceMonitor
            import asyncio

            monitor = PriceMonitor(symbols=[symbol])

            # Run async initialization
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, can't use run_until_complete
                # Return None and let caller handle it
                return None
            else:
                loop.run_until_complete(monitor.update_all())

            snapshot = monitor.get_latest(symbol)
            if snapshot:
                return snapshot.price
        except Exception as e:
            logger.debug(f"Could not fetch actual price: {e}")

        return None

    def get_sample_count(self) -> int:
        """Get total number of evaluated predictions."""
        return len(self._evaluated)

    def get_summary(self) -> Dict:
        """Get performance summary.

        Returns:
            Dict with performance metrics
        """
        if not self._evaluated:
            return {
                "total_predictions": 0,
                "direction_accuracy": 0,
                "ci_coverage": 0,
                "avg_error_pct": 0,
            }

        # Overall metrics
        direction_correct = sum(1 for r in self._evaluated if r.direction_correct)
        in_ci = sum(1 for r in self._evaluated if r.in_ci_range)
        errors = [r.prediction_error_pct for r in self._evaluated if r.prediction_error_pct is not None]

        # By symbol
        by_symbol = {}
        for symbol in set(r.symbol for r in self._evaluated):
            symbol_records = [r for r in self._evaluated if r.symbol == symbol]
            by_symbol[symbol] = {
                "count": len(symbol_records),
                "direction_accuracy": sum(1 for r in symbol_records if r.direction_correct) / len(symbol_records),
                "ci_coverage": sum(1 for r in symbol_records if r.in_ci_range) / len(symbol_records),
            }

        # By regime
        by_regime = {}
        for regime in set(r.regime for r in self._evaluated):
            regime_records = [r for r in self._evaluated if r.regime == regime]
            by_regime[regime] = {
                "count": len(regime_records),
                "direction_accuracy": sum(1 for r in regime_records if r.direction_correct) / len(regime_records),
            }

        return {
            "total_predictions": len(self._evaluated),
            "direction_accuracy": direction_correct / len(self._evaluated) if self._evaluated else 0,
            "ci_coverage": in_ci / len(self._evaluated) if self._evaluated else 0,
            "avg_error_pct": sum(errors) / len(errors) if errors else 0,
            "by_symbol": by_symbol,
            "by_regime": by_regime,
            "last_hour": self._last_hour_results,
        }

    def get_learning_data(self, lookback_hours: int = 24) -> List[PredictionRecord]:
        """Get recent evaluated predictions for learning.

        Args:
            lookback_hours: How many hours back to look

        Returns:
            List of evaluated PredictionRecords
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        return [
            r for r in self._evaluated
            if r.hour_end > cutoff
        ]

    def clear_old_data(self, keep_hours: int = 168):
        """Clear data older than specified hours.

        Args:
            keep_hours: Number of hours to keep (default: 1 week)
        """
        cutoff = datetime.now() - timedelta(hours=keep_hours)
        old_count = len(self._evaluated)
        self._evaluated = [r for r in self._evaluated if r.hour_end > cutoff]
        new_count = len(self._evaluated)
        logger.info(f"Cleared {old_count - new_count} old predictions")
        self._save()
