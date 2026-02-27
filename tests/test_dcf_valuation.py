"""Tests for signals.dcf_valuation module."""

import numpy as np
import pandas as pd
import pytest

from signals.dcf_valuation import (
    compute_dcf_valuation,
    compute_implied_growth_rate,
    compute_roic,
    compute_terminal_value,
    compute_wacc,
    project_fcf,
)


# ── Synthetic Data Builders ──────────────────────────────────────────


def _make_quarterly_income(n_quarters: int = 4) -> pd.DataFrame:
    """Build a quarterly income statement matching yfinance format.

    Rows = line items, columns = dates (descending, most recent first).
    """
    dates = pd.date_range("2024-01-01", periods=n_quarters, freq="QS")[::-1]
    data = {
        d: {
            "Total Revenue": 1e9,
            "Gross Profit": 4e8,
            "Operating Income": 2e8,
            "Net Income": 1.5e8,
            "EBITDA": 3e8,
            "Interest Expense": 1e7,
            "Tax Provision": 3e7,
            "Basic EPS": 3.0,
        }
        for d in dates
    }
    return pd.DataFrame(data)


def _make_annual_income(n_years: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2021-12-31", periods=n_years, freq="YS")[::-1]
    revenues = [4e9, 3.6e9, 3.2e9, 2.8e9][:n_years]
    data = {
        d: {
            "Total Revenue": rev,
            "Gross Profit": rev * 0.4,
            "Operating Income": rev * 0.2,
            "Net Income": rev * 0.15,
            "EBITDA": rev * 0.3,
            "Interest Expense": 4e7,
            "Tax Provision": rev * 0.15 * 0.25,
        }
        for d, rev in zip(dates, revenues)
    }
    return pd.DataFrame(data)


def _make_quarterly_cashflow(n_quarters: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_quarters, freq="QS")[::-1]
    data = {
        d: {
            "Operating Cash Flow": 2.5e8,
            "Capital Expenditure": -5e7,
            "Free Cash Flow": 2e8,
            "Depreciation And Amortization": 3e7,
        }
        for d in dates
    }
    return pd.DataFrame(data)


def _make_quarterly_balance_sheet(n_quarters: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_quarters, freq="QS")[::-1]
    data = {
        d: {
            "Total Assets": 5e9,
            "Current Assets": 2e9,
            "Current Liabilities": 1e9,
            "Long Term Debt": 8e8,
            "Cash And Cash Equivalents": 5e8,
            "Stockholders Equity": 3e9,
            "Retained Earnings": 2e9,
            "Total Liabilities Net Minority Interest": 2e9,
        }
        for d in dates
    }
    return pd.DataFrame(data)


def _make_annual_balance_sheet(n_years: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2023-12-31", periods=n_years, freq="YS")[::-1]
    data = {
        d: {
            "Total Assets": 5e9,
            "Long Term Debt": 8e8,
            "Cash And Cash Equivalents": 5e8,
            "Stockholders Equity": 3e9,
        }
        for d in dates
    }
    return pd.DataFrame(data)


def _make_statements() -> dict:
    return {
        "quarterly_income": _make_quarterly_income(),
        "annual_income": _make_annual_income(),
        "quarterly_cashflow": _make_quarterly_cashflow(),
        "quarterly_balance_sheet": _make_quarterly_balance_sheet(),
        "annual_balance_sheet": _make_annual_balance_sheet(),
    }


def _make_info(market_cap: float = 10e9) -> dict:
    return {
        "marketCap": market_cap,
        "enterpriseValue": market_cap + 3e8,
        "currentPrice": 150.0,
        "sharesOutstanding": market_cap / 150.0,
        "beta": 1.1,
    }


# ── Tests: compute_wacc ──────────────────────────────────────────────


class TestComputeWACC:
    def test_returns_reasonable_range(self):
        info = _make_info()
        statements = _make_statements()
        wacc = compute_wacc(info, statements, risk_free_rate=0.04)
        assert 0.05 <= wacc <= 0.25

    def test_small_cap_premium(self):
        # Use no-debt scenario so only equity weight matters and the 1.5% premium is visible
        info_large = _make_info(market_cap=5e9)
        info_small = _make_info(market_cap=5e8)
        # Remove debt from balance sheet so capital structure effects don't mask the premium
        statements = _make_statements()
        no_debt_bs = statements["quarterly_balance_sheet"].copy()
        no_debt_bs.loc["Long Term Debt"] = 0.0
        statements_no_debt = {**statements, "quarterly_balance_sheet": no_debt_bs}
        no_debt_abs = statements["annual_balance_sheet"].copy()
        no_debt_abs.loc["Long Term Debt"] = 0.0
        statements_no_debt["annual_balance_sheet"] = no_debt_abs

        wacc_large = compute_wacc(info_large, statements_no_debt, risk_free_rate=0.04)
        wacc_small = compute_wacc(info_small, statements_no_debt, risk_free_rate=0.04)

        assert wacc_small > wacc_large, "Small-cap WACC should include illiquidity premium"

    def test_nan_when_no_capital(self):
        info = {"marketCap": 0, "beta": 1.0}
        statements = {
            "annual_income": pd.DataFrame(),
            "quarterly_balance_sheet": pd.DataFrame(),
            "annual_balance_sheet": pd.DataFrame(),
        }
        wacc = compute_wacc(info, statements, risk_free_rate=0.04)
        assert np.isnan(wacc)

    def test_clamps_to_minimum(self):
        info = _make_info(market_cap=100e9)
        info["beta"] = 0.0
        statements = _make_statements()
        wacc = compute_wacc(info, statements, risk_free_rate=0.01, equity_risk_premium=0.01)
        assert wacc >= 0.05


# ── Tests: project_fcf ───────────────────────────────────────────────


class TestProjectFCF:
    def test_caps_at_2x_base(self):
        projected = project_fcf(1e8, revenue_growth_rate=0.50, terminal_growth=0.02, years=5)
        cap = 2.0 * 1e8
        assert all(p <= cap for p in projected), "Projections should be capped at 2x base"

    def test_decays_growth(self):
        projected = project_fcf(1e8, revenue_growth_rate=0.20, terminal_growth=0.02, years=5)
        assert len(projected) == 5
        # Year-over-year growth should decrease
        growths = [projected[i] / projected[i - 1] - 1 for i in range(1, len(projected))]
        for i in range(1, len(growths)):
            assert growths[i] <= growths[i - 1] + 1e-9, "Growth rate should decay"

    def test_empty_for_zero_years(self):
        assert project_fcf(1e8, 0.10, years=0) == []

    def test_single_year_uses_terminal_growth(self):
        projected = project_fcf(1e8, revenue_growth_rate=0.20, terminal_growth=0.03, years=1)
        assert len(projected) == 1
        expected = 1e8 * 1.03
        assert abs(projected[0] - expected) < 1.0


# ── Tests: compute_terminal_value ────────────────────────────────────


class TestComputeTerminalValue:
    def test_gordon_growth(self):
        tv = compute_terminal_value(final_year_fcf=1e8, terminal_growth=0.02, wacc=0.10)
        expected = 1e8 * 1.02 / (0.10 - 0.02)
        assert abs(tv - expected) < 1.0

    def test_zero_when_wacc_below_growth(self):
        tv = compute_terminal_value(1e8, terminal_growth=0.10, wacc=0.05)
        assert tv == 0.0


# ── Tests: compute_dcf_valuation ─────────────────────────────────────


class TestComputeDCFValuation:
    def test_returns_all_expected_keys(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)

        expected_keys = [
            "intrinsic_value_per_share",
            "margin_of_safety",
            "dcf_upside_pct",
            "wacc",
            "roic",
            "roic_vs_wacc_spread",
            "fcf_yield",
            "implied_growth_rate",
            "ev_to_fcf",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_reasonable_intrinsic_value(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)

        iv = result["intrinsic_value_per_share"]
        assert not np.isnan(iv), "Intrinsic value should be computed"
        assert iv > 0, "Intrinsic value should be positive"

    def test_handles_missing_data_gracefully(self):
        statements = {
            "quarterly_income": pd.DataFrame(),
            "annual_income": pd.DataFrame(),
            "quarterly_cashflow": pd.DataFrame(),
            "quarterly_balance_sheet": pd.DataFrame(),
            "annual_balance_sheet": pd.DataFrame(),
        }
        info = {}
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)

        assert np.isnan(result["intrinsic_value_per_share"])
        assert np.isnan(result["margin_of_safety"])

    def test_wacc_populated(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)
        assert 0.05 <= result["wacc"] <= 0.25


# ── Tests: compute_implied_growth_rate ───────────────────────────────


class TestComputeImpliedGrowthRate:
    def test_returns_rate_in_range(self):
        rate = compute_implied_growth_rate(
            current_fcf=8e8, enterprise_value=20e9, wacc=0.10, projection_years=5
        )
        assert -0.10 <= rate <= 0.30

    def test_nan_for_negative_fcf(self):
        rate = compute_implied_growth_rate(
            current_fcf=-1e8, enterprise_value=20e9, wacc=0.10
        )
        assert np.isnan(rate)

    def test_nan_for_zero_ev(self):
        rate = compute_implied_growth_rate(
            current_fcf=8e8, enterprise_value=0, wacc=0.10
        )
        assert np.isnan(rate)


# ── Tests: compute_roic ──────────────────────────────────────────────


class TestComputeROIC:
    def test_basic_calculation(self):
        statements = _make_statements()
        roic = compute_roic(statements)
        assert not np.isnan(roic)
        # NOPAT = operating_income * (1 - tax_rate)
        # invested_capital = equity + debt - cash = 3e9 + 8e8 - 5e8 = 3.3e9
        # operating_income (annual, latest) = 4e9 * 0.2 = 8e8
        # Should be a positive number below 1
        assert 0 < roic < 1.0

    def test_nan_for_empty_statements(self):
        statements = {"annual_income": pd.DataFrame(), "quarterly_balance_sheet": pd.DataFrame(),
                       "annual_balance_sheet": pd.DataFrame()}
        roic = compute_roic(statements)
        assert np.isnan(roic)

    def test_nan_when_no_equity(self):
        dates = pd.date_range("2024-12-31", periods=1, freq="YS")[::-1]
        ai = pd.DataFrame({dates[0]: {"Operating Income": 2e8, "Net Income": 1.5e8, "Tax Provision": 3e7}})
        bs = pd.DataFrame({dates[0]: {"Stockholders Equity": -1e9, "Long Term Debt": 5e8,
                                       "Cash And Cash Equivalents": 2e9}})
        statements = {"annual_income": ai, "quarterly_balance_sheet": bs,
                       "annual_balance_sheet": bs}
        roic = compute_roic(statements)
        # invested_capital = -1e9 + 5e8 - 2e9 = -2.5e9 (non-positive)
        assert np.isnan(roic)
