"""Unit tests for execution optimization modules."""

import numpy as np
import pytest

from strategy.execution import (
    AdaptiveSlippageModel,
    VWAPExecutor,
    TurnoverOptimizer,
)


class TestAdaptiveSlippage:
    def test_estimate_slippage(self):
        model = AdaptiveSlippageModel()
        slippage = model.estimate_slippage(
            price=150.0, shares=100, avg_daily_volume=5_000_000.0,
            market_cap=2e12, vix_level=18.0,
        )
        # Should be in reasonable range (0-100 bps)
        assert 0 < slippage < 100

    def test_higher_vix_higher_slippage(self):
        model = AdaptiveSlippageModel()
        low_vix = model.estimate_slippage(
            price=150.0, shares=100, avg_daily_volume=5_000_000.0,
            market_cap=2e12, vix_level=12.0,
        )
        high_vix = model.estimate_slippage(
            price=150.0, shares=100, avg_daily_volume=5_000_000.0,
            market_cap=2e12, vix_level=35.0,
        )
        assert high_vix > low_vix

    def test_small_cap_higher_slippage(self):
        model = AdaptiveSlippageModel()
        large = model.estimate_slippage(
            price=150.0, shares=100, avg_daily_volume=5_000_000.0,
            market_cap=500e9, vix_level=18.0,
        )
        small = model.estimate_slippage(
            price=150.0, shares=100, avg_daily_volume=500_000.0,
            market_cap=2e9, vix_level=18.0,
        )
        assert small > large


class TestVWAPExecutor:
    def test_simulate(self):
        executor = VWAPExecutor()
        result = executor.simulate_vwap_execution(
            total_shares=1000,
            open_price=150.0,
            high=155.0,
            low=148.0,
            close=152.0,
            daily_volume=5_000_000,
        )
        assert "avg_fill_price" in result
        assert result["avg_fill_price"] > 0
        assert "slippage_vs_vwap_bps" in result
        assert "execution_pct" in result


class TestTurnoverOptimizer:
    def test_filter_trades(self):
        optimizer = TurnoverOptimizer(min_alpha_to_trade=0.003, cost_per_trade_bps=10.0)

        trades = [
            {"ticker": "AAPL", "action": "OPEN_LONG", "expected_return": 0.05},
            {"ticker": "MSFT", "action": "OPEN_LONG", "expected_return": 0.001},
            {"ticker": "JPM", "action": "CLOSE", "expected_return": 0.03, "reason": "stop_loss"},
        ]

        filtered = optimizer.filter_trades(trades, current_positions={})
        tickers = [t["ticker"] for t in filtered]
        # AAPL (high return) and JPM (stop exit) should survive
        assert "AAPL" in tickers
        assert "JPM" in tickers
        # MSFT (low alpha) should be filtered out
        assert "MSFT" not in tickers

    def test_rebalance_frequency(self):
        optimizer = TurnoverOptimizer()
        freq = optimizer.compute_optimal_rebalance_frequency(
            signal_autocorrelation=0.8, avg_trade_cost_bps=10.0, signal_ic=0.05,
        )
        assert freq in ("daily", "weekly", "biweekly", "monthly")
