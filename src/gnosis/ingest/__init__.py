"""Ingest package exports."""
from __future__ import annotations

from .loader import load_or_create_prints, create_data_manifest
from .loader import generate_stub_prints as _generate_stub_prints

# Optional CoinGecko imports - gracefully handle if dependencies missing
try:
    from .coingecko import fetch_coingecko_prints, CoinGeckoClient
    _COINGECKO_AVAILABLE = True
except ImportError:
    _COINGECKO_AVAILABLE = False
    fetch_coingecko_prints = None
    CoinGeckoClient = None

__all__ = [
    "load_or_create_prints",
    "create_data_manifest",
    "generate_stub_prints",
]

# Only export CoinGecko classes if available
if _COINGECKO_AVAILABLE:
    __all__.extend(["fetch_coingecko_prints", "CoinGeckoClient"])

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
