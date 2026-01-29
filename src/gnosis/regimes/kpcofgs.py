"""KPCOFGS regime classification."""
import numpy as np
import pandas as pd


class KPCOFGSClassifier:
    """Simple rule-based KPCOFGS regime classifier."""

    def __init__(self, regimes_config: dict):
        self.config = regimes_config

    def classify(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar into KPCOFGS regimes."""
        df = features_df.copy()

        # K - Kinetics (trend vs mean-revert)
        df["K"] = np.where(
            df["returns"].abs() > df["realized_vol"] * 1.5,
            "K_TRENDING",
            np.where(
                df["returns"].abs() < df["realized_vol"] * 0.5,
                "K_MEAN_REVERTING",
                "K_BALANCED"
            )
        )

        # P - Pressure (volatility regime)
        vol_expanding = df["realized_vol"] > df["realized_vol"].shift(1) * 1.2
        vol_contracting = df["realized_vol"] < df["realized_vol"].shift(1) * 0.8
        df["P"] = np.where(vol_expanding, "P_VOL_EXPANDING",
                          np.where(vol_contracting, "P_VOL_CONTRACTING", "P_VOL_STABLE"))

        # C - Current (order flow)
        df["C"] = np.where(
            df["ofi"] > 0.3, "C_BUY_FLOW_DOMINANT",
            np.where(df["ofi"] < -0.3, "C_SELL_FLOW_DOMINANT", "C_FLOW_NEUTRAL")
        )

        # O - Oscillation (price structure)
        df["O"] = np.where(
            (df["returns"] > 0) & (df["range_pct"] > df["range_pct"].rolling(20).mean()),
            "O_BREAKOUT",
            np.where(
                (df["returns"] < 0) & (df["range_pct"] > df["range_pct"].rolling(20).mean()),
                "O_BREAKDOWN",
                np.where(df["ofi"].abs() > 0.5, "O_SWEEP_REVERT", "O_RANGE")
            )
        )

        # F - Flow (momentum state)
        mom = df["returns"].rolling(5).mean()
        mom_change = mom - mom.shift(5)
        df["F"] = np.where(
            mom_change > df["realized_vol"] * 0.5, "F_ACCEL",
            np.where(mom_change < -df["realized_vol"] * 0.5, "F_DECEL",
                    np.where(mom.abs() < df["realized_vol"] * 0.2, "F_STALL", "F_REVERSAL"))
        )

        # G - Gear (tactical state)
        df["G"] = np.where(
            (df["K"] == "K_TRENDING") & (df["F"] == "F_ACCEL"), "G_TREND_CONT",
            np.where(
                (df["K"] == "K_TRENDING") & (df["F"] == "F_DECEL"), "G_TREND_EXH",
                np.where(
                    (df["K"] == "K_MEAN_REVERTING") & (df["returns"] > 0), "G_MR_BOUNCE",
                    np.where(
                        (df["K"] == "K_MEAN_REVERTING") & (df["returns"] < 0), "G_MR_FADE",
                        np.where(df["O"] == "O_BREAKOUT", "G_BO_HOLD", "G_BO_FAIL")
                    )
                )
            )
        )

        # S - Species (specific setup)
        df["S"] = "S_UNCERTAIN"  # Default
        df.loc[(df["G"] == "G_TREND_CONT") & (df["F"] == "F_DECEL"), "S"] = "S_TC_PULLBACK_RESUME"
        df.loc[(df["G"] == "G_TREND_CONT") & (df["F"] == "F_ACCEL"), "S"] = "S_TC_ACCEL_BREAK"
        df.loc[(df["G"] == "G_TREND_EXH"), "S"] = "S_TX_TOPPING_ROLL"
        df.loc[(df["G"] == "G_MR_BOUNCE") & (df["ofi"] > 0.3), "S"] = "S_MR_OVERSHOOT_SNAPBACK"
        df.loc[(df["G"] == "G_MR_FADE"), "S"] = "S_MR_GRIND_BACK"
        df.loc[(df["G"] == "G_BO_HOLD"), "S"] = "S_BO_LEVEL_BREAK_HOLD"
        df.loc[(df["G"] == "G_BO_FAIL"), "S"] = "S_BO_LEVEL_BREAK_FAIL"
        df.loc[df["O"] == "O_RANGE", "S"] = "S_RANGE_EDGE_FADE"
        df.loc[(df["O"] == "O_SWEEP_REVERT") & (df["returns"] > 0), "S"] = "S_SWEEP_UP_REVERT"
        df.loc[(df["O"] == "O_SWEEP_REVERT") & (df["returns"] < 0), "S"] = "S_SWEEP_DOWN_REVERT"

        return df
