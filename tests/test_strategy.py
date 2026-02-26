"""Unit tests for strategy modules (portfolio, entry/exit, risk)."""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import make_config


class TestPortfolioState:
    def test_equity_calculation(self):
        from strategy.portfolio import PortfolioState, Position

        portfolio = PortfolioState(cash=50_000.0, initial_capital=100_000.0)
        portfolio.positions["AAPL"] = Position(
            ticker="AAPL",
            entry_date="2024-01-01",
            entry_price=150.0,
            shares=100,
            direction="long",
            stop_loss=145.0,
            take_profit=160.0,
            trailing_stop_trigger=153.0,
            sector="Technology",
        )

        prices = {"AAPL": 155.0}
        equity = portfolio.total_equity(prices)
        assert equity == 50_000.0 + 100 * 155.0

    def test_sector_exposure(self):
        from strategy.portfolio import PortfolioState, Position

        portfolio = PortfolioState(cash=80_000.0, initial_capital=100_000.0)
        portfolio.positions["AAPL"] = Position(
            ticker="AAPL",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=100,
            direction="long",
            stop_loss=95.0,
            take_profit=110.0,
            trailing_stop_trigger=103.0,
            sector="Technology",
        )

        prices = {"AAPL": 100.0}
        exposure = portfolio.sector_exposure(prices)
        total_eq = portfolio.total_equity(prices)
        assert "Technology" in exposure
        assert abs(exposure["Technology"] - 100 * 100 / total_eq) < 0.01


class TestPositionSizing:
    def test_size_respects_max(self):
        from strategy.portfolio import PortfolioConstructor, PortfolioState
        from models.predict import Prediction

        config = make_config()
        pc = PortfolioConstructor(config)
        portfolio = PortfolioState(cash=100_000.0, initial_capital=100_000.0)

        pred = Prediction(
            date="2024-01-01",
            ticker="AAPL",
            action="BUY",
            confidence=0.80,
            predicted_class=1,
            probabilities={"BUY": 0.80, "HOLD": 0.15, "SELL": 0.05},
        )

        shares = pc.size_position(pred, portfolio, 150.0, 5.0, "Technology", {"AAPL": 150.0})
        assert shares > 0
        # Position value should not exceed max_position_size_pct of equity
        max_val = 100_000.0 * config["strategy"]["max_position_size_pct"] / 100
        assert shares * 150.0 <= max_val + 1  # +1 for rounding


class TestEntryExit:
    def test_compute_stop_levels(self):
        from strategy.entry_exit import EntryExitEngine

        config = make_config()
        engine = EntryExitEngine(config)

        stop, target, trail = engine.compute_stop_levels(100.0, atr=5.0)
        assert stop < 100.0
        assert target > 100.0
        assert trail > 100.0
        assert trail < target

    def test_fallback_stop_levels(self):
        from strategy.entry_exit import EntryExitEngine

        config = make_config()
        engine = EntryExitEngine(config)

        # Without ATR, should use percentage-based fallback
        stop, target, trail = engine.compute_stop_levels(100.0)
        assert stop < 100.0
        assert target > 100.0


class TestRiskManager:
    def test_check_can_trade(self):
        from strategy.risk_manager import RiskManager
        from strategy.portfolio import PortfolioState

        config = make_config()
        rm = RiskManager(config)
        portfolio = PortfolioState(cash=100_000.0, initial_capital=100_000.0)

        can_trade, reason = rm.check_can_trade(portfolio, {})
        assert can_trade is True

    def test_drawdown_blocks_trading(self):
        from strategy.risk_manager import RiskManager
        from strategy.portfolio import PortfolioState

        config = make_config()
        config["strategy"]["max_portfolio_drawdown_pct"] = 5.0
        rm = RiskManager(config)

        # First call to set _peak_equity high
        portfolio_peak = PortfolioState(cash=100_000.0, initial_capital=100_000.0)
        rm.update_day_start(portfolio_peak, {})

        # Now simulate drawdown > 5%: equity dropped to 90k from peak of 100k
        portfolio_down = PortfolioState(cash=90_000.0, initial_capital=100_000.0)
        can_trade, reason = rm.check_can_trade(portfolio_down, {})
        assert can_trade is False
