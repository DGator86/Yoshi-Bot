"""PriceMonitor - Continuous price feeds for BTC/ETH/SOL.

Fetches real-time prices from multiple sources and maintains
a rolling history for prediction.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    """A single price snapshot."""
    symbol: str
    price: float
    timestamp: datetime
    volume_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h_pct: Optional[float] = None
    source: str = "binance"


@dataclass
class OHLCVBar:
    """OHLCV bar for historical data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceMonitor:
    """Monitors prices for multiple crypto assets.

    Maintains real-time prices and rolling history for each symbol.
    Uses Binance public API by default.
    """

    # Symbol mappings for different exchanges
    SYMBOL_MAP = {
        "BTC": {"binance": "BTCUSDT", "coingecko": "bitcoin"},
        "ETH": {"binance": "ETHUSDT", "coingecko": "ethereum"},
        "SOL": {"binance": "SOLUSDT", "coingecko": "solana"},
    }

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        history_size: int = 500,  # Keep last 500 bars (~8 hours at 1-min)
    ):
        """Initialize PriceMonitor.

        Args:
            symbols: List of symbols to monitor (default: BTC, ETH, SOL)
            history_size: Number of bars to keep in history
        """
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        self.history_size = history_size

        # State
        self._latest: Dict[str, PriceSnapshot] = {}
        self._history: Dict[str, Deque[OHLCVBar]] = {
            s: deque(maxlen=history_size) for s in self.symbols
        }
        self._initialized = False

    async def initialize(self):
        """Initialize by fetching historical data."""
        logger.info("Initializing PriceMonitor...")

        for symbol in self.symbols:
            try:
                await self._fetch_history(symbol)
                logger.info(f"Loaded {len(self._history[symbol])} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load history for {symbol}: {e}")

        self._initialized = True
        logger.info("PriceMonitor initialized")

    async def _fetch_history(self, symbol: str, bars: int = 200):
        """Fetch historical OHLCV data from Binance.

        Args:
            symbol: Symbol to fetch
            bars: Number of bars to fetch
        """
        try:
            import aiohttp
        except ImportError:
            # Fallback to sync requests
            return await self._fetch_history_sync(symbol, bars)

        binance_symbol = self.SYMBOL_MAP.get(symbol, {}).get("binance", f"{symbol}USDT")
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": "1m",
            "limit": bars,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for candle in data:
                        bar = OHLCVBar(
                            timestamp=datetime.fromtimestamp(candle[0] / 1000),
                            open=float(candle[1]),
                            high=float(candle[2]),
                            low=float(candle[3]),
                            close=float(candle[4]),
                            volume=float(candle[5]),
                        )
                        self._history[symbol].append(bar)
                else:
                    logger.error(f"Binance API error: {response.status}")

    async def _fetch_history_sync(self, symbol: str, bars: int = 200):
        """Sync fallback for fetching history."""
        import requests

        binance_symbol = self.SYMBOL_MAP.get(symbol, {}).get("binance", f"{symbol}USDT")
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": "1m",
            "limit": bars,
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for candle in data:
                bar = OHLCVBar(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                )
                self._history[symbol].append(bar)

    async def _fetch_current_price(self, symbol: str) -> Optional[PriceSnapshot]:
        """Fetch current price from Binance.

        Args:
            symbol: Symbol to fetch

        Returns:
            PriceSnapshot or None
        """
        try:
            import aiohttp
            use_async = True
        except ImportError:
            use_async = False

        binance_symbol = self.SYMBOL_MAP.get(symbol, {}).get("binance", f"{symbol}USDT")

        if use_async:
            async with aiohttp.ClientSession() as session:
                # Get ticker
                url = f"https://api.binance.com/api/v3/ticker/24hr"
                params = {"symbol": binance_symbol}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return PriceSnapshot(
                            symbol=symbol,
                            price=float(data["lastPrice"]),
                            timestamp=datetime.now(),
                            volume_24h=float(data["volume"]),
                            high_24h=float(data["highPrice"]),
                            low_24h=float(data["lowPrice"]),
                            change_24h_pct=float(data["priceChangePercent"]),
                            source="binance",
                        )
        else:
            import requests
            url = f"https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": binance_symbol}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return PriceSnapshot(
                    symbol=symbol,
                    price=float(data["lastPrice"]),
                    timestamp=datetime.now(),
                    volume_24h=float(data["volume"]),
                    high_24h=float(data["highPrice"]),
                    low_24h=float(data["lowPrice"]),
                    change_24h_pct=float(data["priceChangePercent"]),
                    source="binance",
                )

        return None

    async def update_all(self):
        """Update prices for all symbols."""
        for symbol in self.symbols:
            try:
                snapshot = await self._fetch_current_price(symbol)
                if snapshot:
                    self._latest[symbol] = snapshot

                    # Also update history with latest bar
                    if self._history[symbol]:
                        last_bar = self._history[symbol][-1]
                        now = datetime.now()

                        # If we're in a new minute, add a new bar
                        if now.minute != last_bar.timestamp.minute or now.hour != last_bar.timestamp.hour:
                            new_bar = OHLCVBar(
                                timestamp=now.replace(second=0, microsecond=0),
                                open=snapshot.price,
                                high=snapshot.price,
                                low=snapshot.price,
                                close=snapshot.price,
                                volume=0,
                            )
                            self._history[symbol].append(new_bar)
                        else:
                            # Update current bar
                            last_bar.high = max(last_bar.high, snapshot.price)
                            last_bar.low = min(last_bar.low, snapshot.price)
                            last_bar.close = snapshot.price

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")

    def get_latest(self, symbol: str) -> Optional[PriceSnapshot]:
        """Get latest price snapshot for a symbol.

        Args:
            symbol: Symbol to get

        Returns:
            PriceSnapshot or None
        """
        return self._latest.get(symbol)

    def get_history(self, symbol: str, bars: Optional[int] = None) -> pd.DataFrame:
        """Get historical data as DataFrame.

        Args:
            symbol: Symbol to get
            bars: Number of bars to return (default: all)

        Returns:
            DataFrame with OHLCV columns
        """
        history = self._history.get(symbol)
        if not history:
            return pd.DataFrame()

        if bars:
            history = list(history)[-bars:]
        else:
            history = list(history)

        df = pd.DataFrame([
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in history
        ])

        if len(df) > 0:
            df["returns"] = df["close"].pct_change()
            df["symbol"] = symbol
            df["bar_idx"] = range(len(df))

        return df

    def get_all_latest(self) -> Dict[str, PriceSnapshot]:
        """Get latest prices for all symbols."""
        return self._latest.copy()
