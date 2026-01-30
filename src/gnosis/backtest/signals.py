"""Signal generation from forecasts."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    mode: str = "x_hat_threshold"  # "x_hat_threshold", "quantile_skew"
    long_threshold: float = 0.0  # Go long when x_hat > threshold
    use_abstain: bool = True  # Respect abstain flag from predictions
    min_confidence: float = 0.0  # Min S_pmax to take position


class SignalGenerator:
    """Convert forecasts into trading signals (LONG=1, FLAT=0)."""

    def __init__(self, config: SignalConfig):
        self.config = config

    def generate_single(self, row: pd.Series) -> int:
        """Generate signal for a single prediction row.

        Args:
            row: Series with x_hat, abstain, and optionally S_pmax

        Returns:
            1 for LONG, 0 for FLAT
        """
        # Respect abstain flag
        if self.config.use_abstain and row.get("abstain", False):
            return 0

        # Check confidence threshold
        if "S_pmax" in row and row["S_pmax"] < self.config.min_confidence:
            return 0

        if self.config.mode == "x_hat_threshold":
            x_hat = row.get("x_hat", 0.0)
            if pd.isna(x_hat):
                return 0
            return 1 if x_hat > self.config.long_threshold else 0

        elif self.config.mode == "quantile_skew":
            q05 = row.get("q05", 0.0)
            q50 = row.get("q50", 0.0)
            q95 = row.get("q95", 0.0)
            if pd.isna(q05) or pd.isna(q50) or pd.isna(q95):
                return 0
            upside = q95 - q50
            downside = q50 - q05
            # Go long if more upside than downside
            if downside > 0 and upside > downside:
                return 1
            return 0

        return 0

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Add 'signal' column to predictions DataFrame.

        Args:
            predictions_df: DataFrame with forecast columns

        Returns:
            Copy of DataFrame with 'signal' column added (1=LONG, 0=FLAT)
        """
        df = predictions_df.copy()
        df["signal"] = df.apply(self.generate_single, axis=1)
        return df
