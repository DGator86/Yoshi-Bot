"""Tests for validation harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from gnosis.harness import WalkForwardHarness, pinball_loss, coverage, sharpness


def test_walkforward_harness():
    """Test walk-forward fold generation."""
    df = pd.DataFrame({"x": range(1000)})

    config = {
        "outer_folds": 3,
        "train_days": 180,
        "val_days": 30,
        "test_days": 30,
    }
    harness = WalkForwardHarness(config)
    folds = list(harness.generate_folds(df))

    assert len(folds) > 0
    for fold in folds:
        assert fold.train_end <= fold.val_start
        assert fold.val_end <= fold.test_start


def test_pinball_loss():
    """Test pinball loss computation."""
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.15, 0.15, 0.35])

    loss = pinball_loss(y_true, y_pred, 0.5)
    assert isinstance(loss, float)
    assert loss >= 0


def test_coverage():
    """Test coverage computation."""
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q_low = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    q_high = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    cov = coverage(y_true, q_low, q_high)
    assert cov == 1.0  # All within intervals


def test_sharpness():
    """Test sharpness computation."""
    q_low = np.array([0.0, 0.1])
    q_high = np.array([0.2, 0.3])

    sharp = sharpness(q_low, q_high)
    assert sharp == 0.2  # Average width
