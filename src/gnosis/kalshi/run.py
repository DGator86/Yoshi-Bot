#!/usr/bin/env python3
"""YoshiBot Kalshi Predictor - Main Entry Point.

Run this script to start the Kalshi hourly prediction system.

Usage:
    # Start with default settings
    python -m gnosis.kalshi.run

    # With Telegram notifications
    python -m gnosis.kalshi.run --telegram-token YOUR_TOKEN --telegram-chat YOUR_CHAT_ID

    # Single prediction (no continuous monitoring)
    python -m gnosis.kalshi.run --once --symbol BTC

    # With custom confidence threshold
    python -m gnosis.kalshi.run --confidence 75

Example:
    python -m gnosis.kalshi.run --telegram-token "123456:ABC..." --telegram-chat "-100123..." --confidence 70
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("yoshibot")


def print_banner():
    """Print YoshiBot banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🦖  YOSHIBOT KALSHI PREDICTOR  🦖                       ║
    ║                                                           ║
    ║   Hourly Crypto Price Predictions for Kalshi Markets      ║
    ║   BTC • ETH • SOL                                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YoshiBot Kalshi Predictor - Hourly crypto price predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single prediction and exit (no continuous monitoring)",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        choices=["BTC", "ETH", "SOL", "ALL"],
        help="Symbol to predict (default: BTC, use ALL for all symbols)",
    )

    parser.add_argument(
        "--telegram-token",
        type=str,
        default=os.environ.get("TELEGRAM_BOT_TOKEN"),
        help="Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)",
    )

    parser.add_argument(
        "--telegram-chat",
        type=str,
        default=os.environ.get("TELEGRAM_CHAT_ID"),
        help="Telegram chat ID (or set TELEGRAM_CHAT_ID env var)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=70.0,
        help="Minimum confidence threshold for alerts (default: 70)",
    )

    parser.add_argument(
        "--probability",
        type=float,
        default=0.60,
        help="Minimum probability threshold for direction (default: 0.60)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Prediction interval in seconds (default: 60)",
    )

    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable adaptive learning",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_single_prediction(args):
    """Run a single prediction and exit."""
    from .predictor import KalshiPredictor, KalshiConfig

    symbols = ["BTC", "ETH", "SOL"] if args.symbol == "ALL" else [args.symbol]

    config = KalshiConfig(
        symbols=symbols,
        telegram_bot_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat,
        n_simulations=args.simulations,
    )

    predictor = KalshiPredictor(config)

    # Initialize monitor synchronously
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(predictor.monitor.initialize())

    print(f"\n📊 Generating prediction(s) at {datetime.now().strftime('%H:%M:%S')}...\n")

    for symbol in symbols:
        signal = predictor.predict_now(symbol)

        if signal:
            print(signal.to_telegram_message())
            print("\n" + "="*60 + "\n")

            # Check if meets threshold
            if signal.meets_threshold(args.confidence, args.probability):
                print(f"✅ Signal meets threshold - would trigger alert!")
            else:
                print(f"⚠️  Signal below threshold (confidence: {signal.prediction_confidence_pct:.0f}% < {args.confidence}%)")
        else:
            print(f"❌ Could not generate prediction for {symbol}")


async def run_continuous(args):
    """Run continuous prediction loop."""
    from .predictor import KalshiPredictor, KalshiConfig

    symbols = ["BTC", "ETH", "SOL"] if args.symbol == "ALL" else [args.symbol]

    config = KalshiConfig(
        symbols=symbols,
        prediction_interval_seconds=args.interval,
        alert_confidence_threshold=args.confidence,
        alert_probability_threshold=args.probability,
        telegram_bot_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat,
        learning_enabled=not args.no_learning,
        n_simulations=args.simulations,
    )

    predictor = KalshiPredictor(config)

    # Add console callback
    def console_callback(signal):
        now = datetime.now().strftime("%H:%M:%S")
        direction = "↑" if signal.direction.value == "UP" else "↓" if signal.direction.value == "DOWN" else "→"
        print(
            f"[{now}] {signal.symbol} ${signal.current_price:,.0f} "
            f"{direction} ${signal.predicted_price:,.0f} "
            f"({signal.predicted_return_pct:+.2f}%) "
            f"Conf: {signal.prediction_confidence_pct:.0f}% "
            f"P({signal.direction.value}): {signal.probability_direction:.0%} "
            f"[{signal.regime.value}]"
        )

    predictor.add_callback(console_callback)

    # Handle shutdown
    def shutdown_handler(sig, frame):
        print("\n\n🛑 Shutting down YoshiBot...")
        predictor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Print status
    print(f"🎯 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Prediction interval: {args.interval}s")
    print(f"📊 Confidence threshold: {args.confidence}%")
    print(f"📈 Probability threshold: {args.probability:.0%}")
    print(f"🎰 Monte Carlo simulations: {args.simulations:,}")
    print(f"🧠 Adaptive learning: {'ON' if not args.no_learning else 'OFF'}")

    if config.telegram_bot_token:
        print(f"📱 Telegram: Enabled")
    else:
        print(f"📱 Telegram: Disabled (set --telegram-token to enable)")

    print("\n" + "="*60)
    print("Starting predictions... (Ctrl+C to stop)")
    print("="*60 + "\n")

    # Start predictor
    await predictor.start()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    if args.once:
        run_single_prediction(args)
    else:
        asyncio.run(run_continuous(args))


if __name__ == "__main__":
    main()
