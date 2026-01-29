"""Scoring and calibration metrics."""
import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute pinball (quantile) loss."""
    errors = y_true - y_pred
    return np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))


def coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Compute interval coverage."""
    in_interval = (y_true >= q_low) & (y_true <= q_high)
    return np.mean(in_interval)


def sharpness(q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Compute sharpness (average interval width)."""
    return np.mean(q_high - q_low)


def crps_empirical(y_true: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    """Approximate CRPS using quantile predictions."""
    # Simple approximation using available quantiles
    total = 0.0
    sorted_qs = sorted(quantiles.keys())

    for i, q in enumerate(sorted_qs):
        pred = quantiles[q]
        total += pinball_loss(y_true, pred, q)

    return total / len(sorted_qs)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    target_col: str = "future_return",
) -> dict:
    """Evaluate prediction quality."""
    # Merge predictions with actuals
    merged = predictions_df.merge(
        actuals_df[["symbol", "bar_idx", target_col]],
        on=["symbol", "bar_idx"],
        how="inner"
    )

    y_true = merged[target_col].values
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]

    if len(y_true) == 0:
        return {
            "pinball_05": np.nan,
            "pinball_50": np.nan,
            "pinball_95": np.nan,
            "coverage_90": np.nan,
            "sharpness": np.nan,
            "crps_approx": np.nan,
            "mae": np.nan,
            "n_samples": 0,
        }

    q05 = merged["q05"].values[valid_mask]
    q50 = merged["q50"].values[valid_mask]
    q95 = merged["q95"].values[valid_mask]

    return {
        "pinball_05": pinball_loss(y_true, q05, 0.05),
        "pinball_50": pinball_loss(y_true, q50, 0.50),
        "pinball_95": pinball_loss(y_true, q95, 0.95),
        "coverage_90": coverage(y_true, q05, q95),
        "sharpness": sharpness(q05, q95),
        "crps_approx": crps_empirical(y_true, {0.05: q05, 0.50: q50, 0.95: q95}),
        "mae": np.mean(np.abs(y_true - q50)),
        "n_samples": len(y_true),
    }
