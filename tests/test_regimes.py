"""Tests for regime classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.ingest import generate_stub_prints
from gnosis.domains import DomainAggregator, compute_features
from gnosis.regimes import KPCOFGSClassifier


def test_kpcofgs_classifier():
    """Test KPCOFGS classification."""
    prints_df = generate_stub_prints(["BTCUSDT"], n_days=2, trades_per_day=500, seed=42)

    config = {"domains": {"D0": {"n_trades": 100}}}
    agg = DomainAggregator(config)
    bars = agg.aggregate(prints_df, "D0")
    features = compute_features(bars)

    classifier = KPCOFGSClassifier({})
    result = classifier.classify(features)

    assert "K" in result.columns
    assert "P" in result.columns
    assert "C" in result.columns
    assert "O" in result.columns
    assert "F" in result.columns
    assert "G" in result.columns
    assert "S" in result.columns
