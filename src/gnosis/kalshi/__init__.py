"""Kalshi Hourly Prediction System.

A self-improving prediction system for hourly crypto price markets.

Components:
- KalshiPredictor: Main orchestrator
- PriceMonitor: Continuous price feeds for BTC/ETH/SOL
- HourlyPredictionEngine: Monte Carlo ensemble predictions
- TelegramNotifier: Alert delivery
- PredictionTracker: Store and compare predictions vs actuals
- AdaptiveLearner: ML optimization after each hour

Usage:
    # Start the predictor
    python -m gnosis.kalshi.run

    # With Telegram notifications
    python -m gnosis.kalshi.run --telegram-token YOUR_TOKEN --telegram-chat CHAT_ID

    # Single prediction
    python -m gnosis.kalshi.run --once --symbol BTC

Programmatic usage:
    from gnosis.kalshi import KalshiPredictor, KalshiConfig

    config = KalshiConfig(symbols=["BTC", "ETH"])
    predictor = KalshiPredictor(config)
    await predictor.start()
"""
from .predictor import KalshiPredictor, KalshiConfig, PredictionSignal, Direction, Regime
from .monitor import PriceMonitor, PriceSnapshot
from .engine import HourlyPredictionEngine, HourlyPrediction, EngineConfig
from .telegram import TelegramNotifier
from .tracker import PredictionTracker, PredictionRecord
from .learner import AdaptiveLearner, LearnerConfig

__all__ = [
    # Main predictor
    "KalshiPredictor",
    "KalshiConfig",
    "PredictionSignal",
    "Direction",
    "Regime",
    # Price monitoring
    "PriceMonitor",
    "PriceSnapshot",
    # Prediction engine
    "HourlyPredictionEngine",
    "HourlyPrediction",
    "EngineConfig",
    # Notifications
    "TelegramNotifier",
    # Tracking
    "PredictionTracker",
    "PredictionRecord",
    # Learning
    "AdaptiveLearner",
    "LearnerConfig",
]
