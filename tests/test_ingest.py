"""Tests for data ingestion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from gnosis.ingest import generate_stub_prints


def test_generate_stub_prints():
    """Test stub print generation."""
    df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=100, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "side" in df.columns
    assert df["symbol"].iloc[0] == "BTCUSDT"


def test_generate_stub_prints_deterministic():
    """Test that stub generation is deterministic."""
    df1 = generate_stub_prints(["BTCUSDT"], n_days=1, trades_per_day=50, seed=123)
    df2 = generate_stub_prints(["BTCUSDT"], n_days=1, trades_per_day=50, seed=123)

    pd.testing.assert_frame_equal(df1, df2)
