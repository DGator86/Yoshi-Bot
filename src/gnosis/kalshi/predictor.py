"""KalshiPredictor - Main orchestrator for hourly crypto predictions.

This is the central coordinator that:
1. Monitors prices continuously
2. Generates predictions for end-of-hour
3. Sends Telegram alerts when confidence thresholds are met
4. Tracks predictions vs actuals
5. Triggers ML adaptation after each hour
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Predicted direction."""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Regime(Enum):
    """Market regime."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


@dataclass
class PredictionSignal:
    """A prediction signal ready for Kalshi trading.

    This is what gets sent to Telegram and used for trading decisions.
    """
    # Identification
    symbol: str  # BTC, ETH, SOL
    timestamp: datetime
    hour_end: datetime  # When this hour segment ends
    minutes_until_close: int

    # Current state
    current_price: float

    # Prediction
    predicted_price: float
    predicted_return_pct: float
    direction: Direction

    # Confidence intervals
    range_low: float
    range_high: float
    range_confidence_pct: float  # e.g., 90 for 90% CI

    # Prediction confidence
    prediction_confidence_pct: float  # How confident in point estimate
    probability_direction: float  # P(direction is correct)

    # Context
    regime: Regime

    # Model info
    model_version: str = "v1"
    n_simulations: int = 10000

    def to_telegram_message(self) -> str:
        """Format as Telegram message."""
        direction_emoji = "🟢" if self.direction == Direction.UP else "🔴" if self.direction == Direction.DOWN else "⚪"
        regime_emoji = "📈" if self.regime == Regime.TRENDING else "📊" if self.regime == Regime.RANGING else "⚡"

        lines = [
            f"{direction_emoji} **{self.symbol}** Prediction",
            f"",
            f"⏰ **Hour Close:** {self.hour_end.strftime('%H:%M')} ({self.minutes_until_close} min)",
            f"",
            f"💰 **Current:** ${self.current_price:,.2f}",
            f"🎯 **Predicted:** ${self.predicted_price:,.2f} ({self.predicted_return_pct:+.2f}%)",
            f"",
            f"📊 **{self.range_confidence_pct:.0f}% Range:** ${self.range_low:,.2f} - ${self.range_high:,.2f}",
            f"",
            f"✅ **Confidence:** {self.prediction_confidence_pct:.0f}%",
            f"📈 **P({self.direction.value}):** {self.probability_direction:.0%}",
            f"",
            f"{regime_emoji} **Regime:** {self.regime.value}",
        ]

        return "\n".join(lines)

    def meets_threshold(self, min_confidence: float = 70.0, min_prob_direction: float = 0.60) -> bool:
        """Check if signal meets trading thresholds."""
        return (
            self.prediction_confidence_pct >= min_confidence and
            self.probability_direction >= min_prob_direction and
            self.direction != Direction.NEUTRAL
        )


@dataclass
class KalshiConfig:
    """Configuration for KalshiPredictor."""
    # Symbols to track
    symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Prediction settings
    prediction_interval_seconds: int = 60  # How often to generate predictions
    min_minutes_before_close: int = 5  # Don't predict if < 5 min left

    # Alert thresholds
    alert_confidence_threshold: float = 70.0  # Min confidence to alert
    alert_probability_threshold: float = 0.60  # Min P(direction) to alert

    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Learning
    learning_enabled: bool = True
    min_samples_for_learning: int = 10  # Need at least N samples before adapting

    # Model
    n_simulations: int = 10000
    confidence_level: float = 0.90  # For confidence intervals


class KalshiPredictor:
    """Main orchestrator for Kalshi hourly predictions.

    Usage:
        predictor = KalshiPredictor(config)
        await predictor.start()  # Runs continuously

    Or for single predictions:
        signal = predictor.predict_now("BTC")
    """

    def __init__(self, config: Optional[KalshiConfig] = None):
        """Initialize KalshiPredictor.

        Args:
            config: Configuration object
        """
        self.config = config or KalshiConfig()

        # Import components (lazy to avoid circular imports)
        from .monitor import PriceMonitor
        from .engine import HourlyPredictionEngine
        from .telegram import TelegramNotifier
        from .tracker import PredictionTracker
        from .learner import AdaptiveLearner

        # Initialize components
        self.monitor = PriceMonitor(symbols=self.config.symbols)
        self.engine = HourlyPredictionEngine(
            n_simulations=self.config.n_simulations,
            confidence_level=self.config.confidence_level,
        )
        self.notifier = TelegramNotifier(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
        )
        self.tracker = PredictionTracker()
        self.learner = AdaptiveLearner(self.engine)

        # State
        self._running = False
        self._last_predictions: Dict[str, PredictionSignal] = {}
        self._callbacks: List[Callable[[PredictionSignal], None]] = []

    def add_callback(self, callback: Callable[[PredictionSignal], None]):
        """Add callback for new prediction signals."""
        self._callbacks.append(callback)

    def _get_hour_end(self) -> datetime:
        """Get the end of the current hour."""
        now = datetime.now()
        return now.replace(minute=59, second=59, microsecond=0)

    def _get_minutes_until_close(self) -> int:
        """Get minutes until the current hour closes."""
        now = datetime.now()
        return 59 - now.minute

    def predict_now(self, symbol: str) -> Optional[PredictionSignal]:
        """Generate a prediction for the given symbol right now.

        Args:
            symbol: Symbol to predict (BTC, ETH, SOL)

        Returns:
            PredictionSignal or None if insufficient data
        """
        # Get current price data
        snapshot = self.monitor.get_latest(symbol)
        if snapshot is None:
            logger.warning(f"No price data for {symbol}")
            return None

        # Get historical data for prediction
        history = self.monitor.get_history(symbol, bars=200)
        if len(history) < 60:
            logger.warning(f"Insufficient history for {symbol}: {len(history)} bars")
            return None

        # Calculate time until hour end
        hour_end = self._get_hour_end()
        minutes_until_close = self._get_minutes_until_close()

        if minutes_until_close < self.config.min_minutes_before_close:
            logger.info(f"Too close to hour end ({minutes_until_close} min), skipping")
            return None

        # Generate prediction
        prediction = self.engine.predict(
            df=history,
            current_price=snapshot.price,
            horizon_minutes=minutes_until_close,
        )

        if prediction is None:
            return None

        # Determine direction
        expected_return = (prediction.point_estimate - snapshot.price) / snapshot.price
        if abs(expected_return) < 0.001:  # < 0.1% is neutral
            direction = Direction.NEUTRAL
        elif expected_return > 0:
            direction = Direction.UP
        else:
            direction = Direction.DOWN

        # Map regime
        regime_map = {
            "trending": Regime.TRENDING,
            "ranging": Regime.RANGING,
            "volatile": Regime.VOLATILE,
        }
        regime = regime_map.get(prediction.regime, Regime.RANGING)

        # Create signal
        signal = PredictionSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            hour_end=hour_end,
            minutes_until_close=minutes_until_close,
            current_price=snapshot.price,
            predicted_price=prediction.point_estimate,
            predicted_return_pct=expected_return * 100,
            direction=direction,
            range_low=prediction.ci_low,
            range_high=prediction.ci_high,
            range_confidence_pct=self.config.confidence_level * 100,
            prediction_confidence_pct=prediction.confidence_score * 100,
            probability_direction=prediction.probability_up if direction == Direction.UP else (1 - prediction.probability_up),
            regime=regime,
            n_simulations=self.config.n_simulations,
        )

        return signal

    async def _prediction_loop(self):
        """Main prediction loop - runs continuously."""
        while self._running:
            try:
                for symbol in self.config.symbols:
                    signal = self.predict_now(symbol)

                    if signal is not None:
                        self._last_predictions[symbol] = signal

                        # Track prediction
                        self.tracker.record_prediction(signal)

                        # Check if meets threshold
                        if signal.meets_threshold(
                            min_confidence=self.config.alert_confidence_threshold,
                            min_prob_direction=self.config.alert_probability_threshold,
                        ):
                            # Send alert
                            await self.notifier.send_signal(signal)

                        # Notify callbacks
                        for callback in self._callbacks:
                            try:
                                callback(signal)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Prediction loop error: {e}")

            # Wait for next interval
            await asyncio.sleep(self.config.prediction_interval_seconds)

    async def _hourly_learning_loop(self):
        """Learning loop - runs at the end of each hour."""
        while self._running:
            # Wait until next hour boundary
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()

            await asyncio.sleep(wait_seconds)

            if not self._running:
                break

            try:
                # Evaluate predictions from the last hour
                results = self.tracker.evaluate_last_hour()

                if results and self.config.learning_enabled:
                    # Check if we have enough samples
                    total_samples = self.tracker.get_sample_count()

                    if total_samples >= self.config.min_samples_for_learning:
                        # Run adaptation
                        logger.info("Running hourly ML adaptation...")
                        self.learner.adapt(results)
                        logger.info("Adaptation complete")
                    else:
                        logger.info(f"Not enough samples for learning: {total_samples}/{self.config.min_samples_for_learning}")

            except Exception as e:
                logger.error(f"Learning loop error: {e}")

    async def _price_update_loop(self):
        """Price monitoring loop."""
        while self._running:
            try:
                await self.monitor.update_all()
            except Exception as e:
                logger.error(f"Price update error: {e}")
            await asyncio.sleep(5)  # Update every 5 seconds

    async def start(self):
        """Start the predictor (runs continuously)."""
        logger.info("Starting KalshiPredictor...")
        self._running = True

        # Initialize monitor
        await self.monitor.initialize()

        # Start all loops
        await asyncio.gather(
            self._price_update_loop(),
            self._prediction_loop(),
            self._hourly_learning_loop(),
        )

    def stop(self):
        """Stop the predictor."""
        logger.info("Stopping KalshiPredictor...")
        self._running = False

    def get_latest_signal(self, symbol: str) -> Optional[PredictionSignal]:
        """Get the most recent prediction signal for a symbol."""
        return self._last_predictions.get(symbol)

    def get_all_signals(self) -> Dict[str, PredictionSignal]:
        """Get all latest prediction signals."""
        return self._last_predictions.copy()

    def get_performance_summary(self) -> Dict:
        """Get performance summary from tracker."""
        return self.tracker.get_summary()
