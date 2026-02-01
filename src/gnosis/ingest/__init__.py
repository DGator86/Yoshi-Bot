"""Ingest package exports."""
from __future__ import annotations

from .loader import load_or_create_prints, create_data_manifest
from .loader import generate_stub_prints as _generate_stub_prints

# CCXT-based live data fetching (optional - requires ccxt package)
try:
    from .ccxt_loader import (
        CCXTLoader,
        fetch_live_prints,
        fetch_live_ohlcv,
    )
    _HAS_CCXT = True
except ImportError:
    _HAS_CCXT = False
    CCXTLoader = None
    fetch_live_prints = None
    fetch_live_ohlcv = None

__all__ = [
    "load_or_create_prints",
    "create_data_manifest",
    "generate_stub_prints",
    # CCXT exports (available if ccxt is installed)
    "CCXTLoader",
    "fetch_live_prints",
    "fetch_live_ohlcv",
]


def generate_stub_prints(
    symbols: list[str],
    n_days: int = 365,
    trades_per_day: int = 50000,
    start_date: str = "2023-01-01",
    seed: int = 1337,
):
    """
    Stable public API for tests:
      generate_stub_prints(symbols, n_days=..., trades_per_day=..., seed=...)
    """
    return _generate_stub_prints(
        symbols=symbols,
        start_date=start_date,
        n_days=n_days,
        trades_per_day=trades_per_day,
        seed=seed,
    )
