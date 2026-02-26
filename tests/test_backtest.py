"""Unit tests for backtest metrics computation."""

import numpy as np
import pandas as pd
import pytest


class TestMetrics:
    def _make_equity_curve(self, returns: list[float]) -> pd.DataFrame:
        """Build a simple equity curve from a list of daily returns."""
        dates = pd.bdate_range("2024-01-01", periods=len(returns))
        equity = [100_000.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity = equity[1:]  # drop initial
        df = pd.DataFrame({"equity": equity, "cash": equity, "num_positions": 0}, index=dates)
        df["daily_return"] = pd.Series(returns, index=dates)
        return df

    def test_compute_metrics(self):
        from backtest.metrics import compute_metrics

        returns = [0.01, -0.005, 0.008, 0.002, -0.003] * 50
        eq_df = self._make_equity_curve(returns)
        trades_df = pd.DataFrame()

        metrics = compute_metrics(eq_df, trades_df, 100_000.0)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "annual_return" in metrics

    def test_sharpe_positive_for_positive_returns(self):
        from backtest.metrics import compute_metrics

        returns = [0.01] * 100
        eq_df = self._make_equity_curve(returns)

        metrics = compute_metrics(eq_df, pd.DataFrame(), 100_000.0)
        # Consistently positive returns should give a positive Sharpe
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown_calculation(self):
        from backtest.metrics import compute_metrics

        # Create a clear drawdown: up, up, down, down, up
        returns = [0.05, 0.05, -0.10, -0.10, 0.05]
        eq_df = self._make_equity_curve(returns)

        metrics = compute_metrics(eq_df, pd.DataFrame(), 100_000.0)
        # Max drawdown should be negative
        assert metrics["max_drawdown"] < 0

    def test_trade_metrics(self):
        from backtest.metrics import compute_metrics

        returns = [0.01, -0.005] * 50
        eq_df = self._make_equity_curve(returns)

        trades = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "JPM"],
                "entry_date": ["2024-01-01", "2024-01-05", "2024-01-10"],
                "exit_date": ["2024-01-03", "2024-01-08", "2024-01-15"],
                "entry_price": [100.0, 200.0, 150.0],
                "exit_price": [105.0, 195.0, 160.0],
                "shares": [10, 5, 8],
                "pnl": [50.0, -25.0, 80.0],
                "pnl_pct": [0.05, -0.025, 0.067],
                "reason": ["take_profit", "stop_loss", "take_profit"],
            }
        )

        metrics = compute_metrics(eq_df, trades, 100_000.0)
        assert "win_rate" in metrics
        assert "total_trades" in metrics
        assert metrics["total_trades"] == 3
        assert abs(metrics["win_rate"] - 2 / 3) < 0.01
