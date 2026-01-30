"""Performance statistics calculation."""
from typing import Dict, Optional

import numpy as np
import pandas as pd


class StatsCalculator:
    """Calculate performance statistics from backtest results."""

    @staticmethod
    def compute(
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365 * 24,  # Assume hourly-ish bars
    ) -> Dict:
        """Compute comprehensive performance statistics.

        Args:
            equity_curve: DataFrame with 'timestamp', 'equity' columns
            trades_df: DataFrame with 'pnl', 'fee' columns
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Number of periods per year for annualization

        Returns:
            Dict with performance metrics
        """
        stats = {}

        # Basic equity stats
        if equity_curve.empty:
            stats["total_return"] = 0.0
            stats["total_return_pct"] = 0.0
            stats["final_equity"] = initial_capital
        else:
            final_equity = equity_curve["equity"].iloc[-1]
            stats["final_equity"] = final_equity
            stats["total_return"] = (final_equity - initial_capital) / initial_capital
            stats["total_return_pct"] = stats["total_return"] * 100

        stats["initial_capital"] = initial_capital

        # Time range
        if not equity_curve.empty:
            stats["backtest_start"] = str(equity_curve["timestamp"].iloc[0])
            stats["backtest_end"] = str(equity_curve["timestamp"].iloc[-1])
        else:
            stats["backtest_start"] = None
            stats["backtest_end"] = None

        # Returns series for Sharpe/Sortino
        if len(equity_curve) > 1:
            returns = equity_curve["equity"].pct_change().dropna()
            if len(returns) > 0:
                mean_ret = returns.mean()
                std_ret = returns.std()

                # Sharpe ratio (annualized)
                if std_ret > 0:
                    sharpe = (mean_ret - risk_free_rate / periods_per_year) / std_ret
                    stats["sharpe_ratio"] = sharpe * np.sqrt(periods_per_year)
                else:
                    stats["sharpe_ratio"] = 0.0

                # Sortino ratio (annualized)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        sortino = (
                            mean_ret - risk_free_rate / periods_per_year
                        ) / downside_std
                        stats["sortino_ratio"] = sortino * np.sqrt(periods_per_year)
                    else:
                        stats["sortino_ratio"] = 0.0
                else:
                    stats["sortino_ratio"] = float("inf") if mean_ret > 0 else 0.0
            else:
                stats["sharpe_ratio"] = 0.0
                stats["sortino_ratio"] = 0.0
        else:
            stats["sharpe_ratio"] = 0.0
            stats["sortino_ratio"] = 0.0

        # Max drawdown
        if len(equity_curve) > 0:
            equity = equity_curve["equity"].values
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            stats["max_drawdown"] = float(drawdown.max())
            stats["max_drawdown_pct"] = stats["max_drawdown"] * 100
        else:
            stats["max_drawdown"] = 0.0
            stats["max_drawdown_pct"] = 0.0

        # Calmar ratio (annual return / max drawdown)
        if stats["max_drawdown"] > 0:
            # Approximate annualized return
            n_periods = len(equity_curve)
            if n_periods > 0:
                annual_return = (
                    (1 + stats["total_return"]) ** (periods_per_year / n_periods) - 1
                )
                stats["calmar_ratio"] = annual_return / stats["max_drawdown"]
            else:
                stats["calmar_ratio"] = 0.0
        else:
            stats["calmar_ratio"] = float("inf") if stats["total_return"] > 0 else 0.0

        # Trade statistics
        if not trades_df.empty:
            stats["n_trades"] = len(trades_df)
            stats["total_fees"] = float(trades_df["fee"].sum())

            # PnL stats (only for closing trades with realized PnL)
            pnls = trades_df["pnl"]
            nonzero_pnls = pnls[pnls != 0]

            if len(nonzero_pnls) > 0:
                stats["avg_trade_pnl"] = float(nonzero_pnls.mean())
                winners = nonzero_pnls[nonzero_pnls > 0]
                losers = nonzero_pnls[nonzero_pnls < 0]

                stats["win_rate"] = len(winners) / len(nonzero_pnls)

                # Profit factor
                gross_profit = winners.sum() if len(winners) > 0 else 0
                gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
                if gross_loss > 0:
                    stats["profit_factor"] = gross_profit / gross_loss
                else:
                    stats["profit_factor"] = float("inf") if gross_profit > 0 else 0.0
            else:
                stats["avg_trade_pnl"] = 0.0
                stats["win_rate"] = 0.0
                stats["profit_factor"] = 0.0
        else:
            stats["n_trades"] = 0
            stats["total_fees"] = 0.0
            stats["avg_trade_pnl"] = 0.0
            stats["win_rate"] = 0.0
            stats["profit_factor"] = 0.0

        return stats
