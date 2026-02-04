"""PriceMonitor - Continuous price feeds for BTC/ETH/SOL.

Fetches real-time prices from multiple sources with automatic fallback:
1. Binance (primary)
2. CoinGecko (fallback - works globally)
3. Kraken (fallback)
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    source: str = "unknown"


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
    Uses multiple data sources with automatic fallback.
    """

    # Symbol mappings for different exchanges
    SYMBOL_MAP = {
        "BTC": {
            "binance": "BTCUSDT",
            "coingecko": "bitcoin",
            "kraken": "XXBTZUSD",
            "coinbase": "BTC-USD",
        },
        "ETH": {
            "binance": "ETHUSDT",
            "coingecko": "ethereum",
            "kraken": "XETHZUSD",
            "coinbase": "ETH-USD",
        },
        "SOL": {
            "binance": "SOLUSDT",
            "coingecko": "solana",
            "kraken": "SOLUSD",
            "coinbase": "SOL-USD",
        },
    }

    # Data sources in priority order
    SOURCES = ["binance", "coingecko", "kraken", "coinbase"]

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        history_size: int = 500,
        preferred_source: Optional[str] = None,
    ):
        """Initialize PriceMonitor.

        Args:
            symbols: List of symbols to monitor (default: BTC, ETH, SOL)
            history_size: Number of bars to keep in history
            preferred_source: Preferred data source (default: auto)
        """
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        self.history_size = history_size
        self.preferred_source = preferred_source

        # State
        self._latest: Dict[str, PriceSnapshot] = {}
        self._history: Dict[str, Deque[OHLCVBar]] = {
            s: deque(maxlen=history_size) for s in self.symbols
        }
        self._working_source: Dict[str, str] = {}  # Track which source works for each symbol
        self._initialized = False

    async def initialize(self):
        """Initialize by fetching historical data."""
        logger.info("Initializing PriceMonitor...")

        for symbol in self.symbols:
            success = False
            for source in self.SOURCES:
                if self.preferred_source and source != self.preferred_source:
                    continue
                try:
                    await self._fetch_history(symbol, source=source)
                    if len(self._history[symbol]) > 0:
                        self._working_source[symbol] = source
                        logger.info(f"Loaded {len(self._history[symbol])} bars for {symbol} from {source}")
                        success = True
                        break
                except Exception as e:
                    logger.debug(f"Failed to load from {source}: {e}")
                    continue

            if not success:
                logger.warning(f"Could not load history for {symbol} from any source")

        # Also fetch current prices
        await self.update_all()

        self._initialized = True
        logger.info("PriceMonitor initialized")

    async def _fetch_history(self, symbol: str, bars: int = 200, source: str = "binance"):
        """Fetch historical OHLCV data.

        Args:
            symbol: Symbol to fetch
            bars: Number of bars to fetch
            source: Data source to use
        """
        if source == "binance":
            await self._fetch_history_binance(symbol, bars)
        elif source == "coingecko":
            await self._fetch_history_coingecko(symbol, bars)
        elif source == "kraken":
            await self._fetch_history_kraken(symbol, bars)
        elif source == "coinbase":
            await self._fetch_history_coinbase(symbol, bars)

    async def _fetch_history_binance(self, symbol: str, bars: int = 200):
        """Fetch from Binance."""
        import requests

        binance_symbol = self.SYMBOL_MAP.get(symbol, {}).get("binance", f"{symbol}USDT")
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_symbol, "interval": "1m", "limit": bars}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Binance API error: {response.status_code}")

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

    async def _fetch_history_coingecko(self, symbol: str, bars: int = 200):
        """Fetch from CoinGecko (free, no API key needed, global)."""
        import requests

        cg_id = self.SYMBOL_MAP.get(symbol, {}).get("coingecko", symbol.lower())

        # CoinGecko market_chart gives us hourly data for last 24h or minutely for last hour
        # For 200 1-minute bars, we'll use the OHLC endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
        params = {"vs_currency": "usd", "days": "1"}  # Last 24 hours

        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            raise Exception(f"CoinGecko API error: {response.status_code}")

        data = response.json()
        # CoinGecko OHLC returns [timestamp, open, high, low, close]
        for candle in data[-bars:]:
            bar = OHLCVBar(
                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=0,  # CoinGecko OHLC doesn't include volume
            )
            self._history[symbol].append(bar)

    async def _fetch_history_kraken(self, symbol: str, bars: int = 200):
        """Fetch from Kraken."""
        import requests

        kraken_symbol = self.SYMBOL_MAP.get(symbol, {}).get("kraken", f"{symbol}USD")
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": kraken_symbol, "interval": 1}  # 1 minute

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Kraken API error: {response.status_code}")

        data = response.json()
        if data.get("error"):
            raise Exception(f"Kraken error: {data['error']}")

        # Kraken returns data in result with pair name as key
        result = data.get("result", {})
        pair_data = None
        for key in result:
            if key != "last":
                pair_data = result[key]
                break

        if not pair_data:
            raise Exception("No data from Kraken")

        for candle in pair_data[-bars:]:
            bar = OHLCVBar(
                timestamp=datetime.fromtimestamp(candle[0]),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[6]),
            )
            self._history[symbol].append(bar)

    async def _fetch_history_coinbase(self, symbol: str, bars: int = 200):
        """Fetch from Coinbase."""
        import requests

        cb_symbol = self.SYMBOL_MAP.get(symbol, {}).get("coinbase", f"{symbol}-USD")
        url = f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles"
        params = {"granularity": 60}  # 1 minute

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Coinbase API error: {response.status_code}")

        data = response.json()
        # Coinbase returns [timestamp, low, high, open, close, volume] in reverse order
        for candle in reversed(data[-bars:]):
            bar = OHLCVBar(
                timestamp=datetime.fromtimestamp(candle[0]),
                open=float(candle[3]),
                high=float(candle[2]),
                low=float(candle[1]),
                close=float(candle[4]),
                volume=float(candle[5]),
            )
            self._history[symbol].append(bar)

    async def _fetch_current_price(self, symbol: str) -> Optional[PriceSnapshot]:
        """Fetch current price with fallback sources.

        Args:
            symbol: Symbol to fetch

        Returns:
            PriceSnapshot or None
        """
        # Try working source first
        sources = self.SOURCES.copy()
        if symbol in self._working_source:
            working = self._working_source[symbol]
            sources.remove(working)
            sources.insert(0, working)

        for source in sources:
            try:
                snapshot = await self._fetch_price_from_source(symbol, source)
                if snapshot:
                    self._working_source[symbol] = source
                    return snapshot
            except Exception as e:
                logger.debug(f"Failed to fetch {symbol} from {source}: {e}")
                continue

        return None

    async def _fetch_price_from_source(self, symbol: str, source: str) -> Optional[PriceSnapshot]:
        """Fetch price from a specific source."""
        import requests

        if source == "binance":
            binance_symbol = self.SYMBOL_MAP.get(symbol, {}).get("binance", f"{symbol}USDT")
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}")
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

        elif source == "coingecko":
            cg_id = self.SYMBOL_MAP.get(symbol, {}).get("coingecko", symbol.lower())
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": cg_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            }
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}")
            data = response.json()
            coin_data = data.get(cg_id, {})
            return PriceSnapshot(
                symbol=symbol,
                price=float(coin_data["usd"]),
                timestamp=datetime.now(),
                volume_24h=coin_data.get("usd_24h_vol"),
                change_24h_pct=coin_data.get("usd_24h_change"),
                source="coingecko",
            )

        elif source == "kraken":
            kraken_symbol = self.SYMBOL_MAP.get(symbol, {}).get("kraken", f"{symbol}USD")
            url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}")
            data = response.json()
            if data.get("error"):
                raise Exception(str(data["error"]))
            result = data.get("result", {})
            ticker = list(result.values())[0] if result else {}
            return PriceSnapshot(
                symbol=symbol,
                price=float(ticker["c"][0]),  # Last trade price
                timestamp=datetime.now(),
                volume_24h=float(ticker["v"][1]),  # 24h volume
                high_24h=float(ticker["h"][1]),
                low_24h=float(ticker["l"][1]),
                source="kraken",
            )

        elif source == "coinbase":
            cb_symbol = self.SYMBOL_MAP.get(symbol, {}).get("coinbase", f"{symbol}-USD")
            url = f"https://api.exchange.coinbase.com/products/{cb_symbol}/ticker"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}")
            data = response.json()
            return PriceSnapshot(
                symbol=symbol,
                price=float(data["price"]),
                timestamp=datetime.now(),
                volume_24h=float(data.get("volume", 0)),
                source="coinbase",
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
        """Get latest price snapshot for a symbol."""
        return self._latest.get(symbol)

    def get_history(self, symbol: str, bars: Optional[int] = None) -> pd.DataFrame:
        """Get historical data as DataFrame."""
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

    def get_working_sources(self) -> Dict[str, str]:
        """Get which data source is working for each symbol."""
        return self._working_source.copy()
