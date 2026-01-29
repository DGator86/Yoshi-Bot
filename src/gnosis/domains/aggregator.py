"""Domain aggregation based on trade counts."""
import numpy as np
import pandas as pd


class DomainAggregator:
    """Aggregates print data into domains (D0-D3) based on trade counts."""

    def __init__(self, domain_config: dict):
        self.domain_config = domain_config

    def aggregate(self, prints_df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Aggregate prints into domain bars based on trade count."""
        n_trades = self.domain_config["domains"][domain]["n_trades"]

        bars = []
        for symbol in prints_df["symbol"].unique():
            sym_prints = prints_df[prints_df["symbol"] == symbol].copy()
            sym_prints = sym_prints.sort_values("timestamp")

            n_bars = len(sym_prints) // n_trades
            for i in range(n_bars):
                start_idx = i * n_trades
                end_idx = (i + 1) * n_trades
                chunk = sym_prints.iloc[start_idx:end_idx]

                bars.append({
                    "symbol": symbol,
                    "bar_idx": i,
                    "timestamp_start": chunk["timestamp"].iloc[0],
                    "timestamp_end": chunk["timestamp"].iloc[-1],
                    "open": chunk["price"].iloc[0],
                    "high": chunk["price"].max(),
                    "low": chunk["price"].min(),
                    "close": chunk["price"].iloc[-1],
                    "volume": chunk["quantity"].sum(),
                    "n_trades": len(chunk),
                    "buy_volume": chunk[chunk["side"] == "BUY"]["quantity"].sum(),
                    "sell_volume": chunk[chunk["side"] == "SELL"]["quantity"].sum(),
                })

        return pd.DataFrame(bars)


def compute_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features from domain bars."""
    df = bars_df.copy()

    # Returns
    df["returns"] = df.groupby("symbol")["close"].pct_change()

    # Realized volatility (trailing 20 bars)
    df["realized_vol"] = df.groupby("symbol")["returns"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )

    # Order flow imbalance
    df["ofi"] = (df["buy_volume"] - df["sell_volume"]) / (df["buy_volume"] + df["sell_volume"] + 1e-9)

    # Price range
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    return df
