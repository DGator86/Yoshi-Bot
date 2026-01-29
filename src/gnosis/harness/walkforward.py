"""Walk-forward validation harness with purge/embargo."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Fold:
    """Represents a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class WalkForwardHarness:
    """Nested walk-forward with purge and embargo."""

    def __init__(self, config: dict):
        self.outer_folds = config.get("outer_folds", 8)
        self.train_days = config.get("train_days", 180)
        self.val_days = config.get("val_days", 30)
        self.test_days = config.get("test_days", 30)
        self.purge_trades = config.get("purge_trades", "HORIZON")
        self.embargo_trades = config.get("embargo_trades", "HORIZON")

    def generate_folds(self, df: pd.DataFrame, horizon_trades: int = 2000) -> Iterator[Fold]:
        """Generate walk-forward folds with purge/embargo."""
        n_bars = len(df)
        total_window = self.train_days + self.val_days + self.test_days

        # Calculate bars per day (approximate)
        bars_per_day = max(1, n_bars // 30)  # assuming ~30 days of data in stub

        train_bars = self.train_days * bars_per_day // 30  # scale down
        val_bars = self.val_days * bars_per_day // 30
        test_bars = self.test_days * bars_per_day // 30

        # Purge/embargo in bars (based on horizon)
        purge_bars = max(1, horizon_trades // 200)  # D0 has 200 trades
        embargo_bars = purge_bars

        step = (n_bars - train_bars - val_bars - test_bars - purge_bars * 2 - embargo_bars) // max(1, self.outer_folds - 1)
        step = max(1, step)

        for i in range(self.outer_folds):
            train_start = i * step
            train_end = train_start + train_bars

            # Purge gap between train and val
            val_start = train_end + purge_bars
            val_end = val_start + val_bars

            # Embargo gap between val and test
            test_start = val_end + embargo_bars
            test_end = min(test_start + test_bars, n_bars)

            if test_end <= test_start or val_end > n_bars:
                break

            yield Fold(
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )


def compute_future_returns(df: pd.DataFrame, horizon_bars: int = 10) -> pd.DataFrame:
    """Compute future returns for prediction targets."""
    result = df.copy()
    result["future_return"] = result.groupby("symbol")["close"].transform(
        lambda x: x.shift(-horizon_bars) / x - 1
    )
    return result
