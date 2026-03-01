"""Tests for signals.dcf_valuation module."""

import numpy as np
import pandas as pd
import pytest

from signals.dcf_valuation import (
    _compute_fcff,
    _distress_probability,
    _get_total_debt,
    _trailing_4q_sum,
    compute_dcf_valuation,
    compute_implied_growth_rate,
    compute_roic,
    compute_terminal_value,
    compute_wacc,
    project_fcf,
)
from signals.model_verifier import verify_dcf


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
            "Stock Based Compensation": 2e7,
            "Change In Working Capital": -1e7,
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
        "sector": "Technology",
        "country": "United States",
    }


# ── Tests: _trailing_4q_sum (Bug Fix 1A) ────────────────────────────


class TestTrailing4QSum:
    def test_returns_sum_with_4_quarters(self):
        df = _make_quarterly_cashflow(4)
        result = _trailing_4q_sum(df, "Operating Cash Flow")
        assert result == pytest.approx(4 * 2.5e8)

    def test_returns_nan_with_3_quarters(self):
        df = _make_quarterly_cashflow(3)
        result = _trailing_4q_sum(df, "Operating Cash Flow")
        assert np.isnan(result), "Should return NaN when fewer than 4 quarters"

    def test_returns_nan_with_1_quarter(self):
        df = _make_quarterly_cashflow(1)
        result = _trailing_4q_sum(df, "Operating Cash Flow")
        assert np.isnan(result)

    def test_returns_nan_for_missing_line_item(self):
        df = _make_quarterly_cashflow(4)
        result = _trailing_4q_sum(df, "Nonexistent Item")
        assert np.isnan(result)


# ── Tests: _get_total_debt (Bug Fix 1B) ─────────────────────────────


class TestGetTotalDebt:
    def test_uses_current_debt_when_available(self):
        statements = _make_statements()
        bs = statements["quarterly_balance_sheet"].copy()
        bs.loc["Current Debt"] = 1e8
        statements["quarterly_balance_sheet"] = bs
        total = _get_total_debt(statements)
        # Long Term Debt (8e8) + Current Debt (1e8) = 9e8
        assert total == pytest.approx(9e8)

    def test_fallback_to_combined_item(self):
        statements = _make_statements()
        bs = statements["quarterly_balance_sheet"].copy()
        bs.loc["Current Debt And Capital Lease Obligation"] = 1.5e8
        statements["quarterly_balance_sheet"] = bs
        total = _get_total_debt(statements)
        # Long Term Debt (8e8) + combined fallback (1.5e8) = 9.5e8
        assert total == pytest.approx(9.5e8)

    def test_no_double_count_when_both_present(self):
        statements = _make_statements()
        bs = statements["quarterly_balance_sheet"].copy()
        bs.loc["Current Debt"] = 1e8
        bs.loc["Current Debt And Capital Lease Obligation"] = 2e8
        statements["quarterly_balance_sheet"] = bs
        total = _get_total_debt(statements)
        # Should use Current Debt (1e8), NOT the combined (2e8)
        assert total == pytest.approx(9e8)


# ── Tests: _compute_fcff (delta_wc flag) ────────────────────────────


class TestComputeFCFF:
    def test_returns_tuple(self):
        statements = _make_statements()
        result = _compute_fcff(statements)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_delta_wc_estimated_false_when_present(self):
        statements = _make_statements()
        _, delta_wc_estimated = _compute_fcff(statements)
        assert delta_wc_estimated is False

    def test_delta_wc_estimated_true_when_missing(self):
        statements = _make_statements()
        cf = statements["quarterly_cashflow"].copy()
        cf = cf.drop("Change In Working Capital", errors="ignore")
        statements["quarterly_cashflow"] = cf
        _, delta_wc_estimated = _compute_fcff(statements)
        assert delta_wc_estimated is True


# ── Tests: compute_wacc ──────────────────────────────────────────────


class TestComputeWACC:
    def test_returns_reasonable_range(self):
        info = _make_info()
        statements = _make_statements()
        wacc = compute_wacc(info, statements, risk_free_rate=0.04)
        assert 0.04 <= wacc <= 0.25

    def test_small_cap_premium(self):
        info_large = _make_info(market_cap=5e9)
        info_small = _make_info(market_cap=5e8)
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
        assert wacc >= 0.04

    def test_country_risk_premium_for_us(self):
        info = _make_info()
        info["country"] = "United States"
        statements = _make_statements()
        wacc_us = compute_wacc(info, statements, risk_free_rate=0.04)

        info_br = _make_info()
        info_br["country"] = "Brazil"
        wacc_br = compute_wacc(info_br, statements, risk_free_rate=0.04)

        assert wacc_br > wacc_us, "Brazilian company should have higher WACC due to CRP"

    def test_country_risk_default_for_unknown(self):
        info = _make_info()
        info["country"] = "Narnia"
        statements = _make_statements()
        wacc = compute_wacc(info, statements, risk_free_rate=0.04)
        # Should not crash; uses 1% default CRP
        assert 0.04 <= wacc <= 0.30


# ── Tests: project_fcf ───────────────────────────────────────────────


class TestProjectFCF:
    def test_caps_at_2x_base(self):
        projected = project_fcf(1e8, revenue_growth_rate=0.50, terminal_growth=0.02, years=5)
        cap = 2.0 * 1e8
        assert all(p <= cap for p in projected), "Projections should be capped at 2x base"

    def test_decays_growth(self):
        projected = project_fcf(1e8, revenue_growth_rate=0.20, terminal_growth=0.02, years=5)
        assert len(projected) == 5
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

    def test_convex_fade_faster_than_linear_in_early_years(self):
        """Convex decay should produce lower growth in year 2 than linear would."""
        projected = project_fcf(1e8, revenue_growth_rate=0.20, terminal_growth=0.02, years=5)
        year2_growth = projected[1] / projected[0] - 1
        # Linear year 2 growth would be: 0.20 + (1/4)*(0.02-0.20) = 0.155
        # Convex year 2 growth should be lower than 0.155
        assert year2_growth < 0.155, "Convex fade should be faster than linear in early years"

    def test_convex_fade_converges_to_terminal(self):
        projected = project_fcf(1e8, revenue_growth_rate=0.20, terminal_growth=0.02, years=5)
        final_growth = projected[-1] / projected[-2] - 1
        assert abs(final_growth - 0.02) < 0.01, "Final year should approach terminal growth"


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
            "delta_wc_estimated",
            "implied_reinvestment_rate",
            "bear_iv",
            "base_iv",
            "bull_iv",
            "scenario_range_pct",
            "tv_gordon",
            "tv_exit_multiple",
            "tv_divergence_pct",
            "distress_probability",
            "distress_adjusted_iv",
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
        assert 0.04 <= result["wacc"] <= 0.25

    def test_delta_wc_estimated_flag(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)
        assert result["delta_wc_estimated"] is False

        # Remove ΔWC from cashflow
        cf = statements["quarterly_cashflow"].copy()
        cf = cf.drop("Change In Working Capital", errors="ignore")
        statements_no_wc = {**statements, "quarterly_cashflow": cf}
        result2 = compute_dcf_valuation(statements_no_wc, info, risk_free_rate=0.04)
        assert result2["delta_wc_estimated"] is True

    def test_scenarios_bear_lt_bull(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)

        bear = result["bear_iv"]
        bull = result["bull_iv"]

        if not np.isnan(bear) and not np.isnan(bull):
            assert bear < bull, "Bear IV should be less than bull"

    def test_scenario_range_pct_positive(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)
        sr = result["scenario_range_pct"]
        if not np.isnan(sr):
            assert sr > 0, "Scenario range should be positive"

    def test_tv_gordon_populated(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)
        assert not np.isnan(result["tv_gordon"]), "Gordon TV should be computed"
        assert result["tv_gordon"] > 0

    def test_exit_multiple_tv_populated(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)
        # With Technology sector and EBITDA data, should be computed
        tv_exit = result["tv_exit_multiple"]
        assert not np.isnan(tv_exit), "Exit multiple TV should be computed for Technology"
        assert tv_exit > 0

    def test_distress_probability_with_altman_z(self):
        statements = _make_statements()
        info = _make_info()
        # Safe company — Z > 3.0
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04, altman_z=4.5)
        assert result["distress_probability"] == 0.0
        assert not np.isnan(result["distress_adjusted_iv"])

    def test_distress_discount_for_low_z(self):
        statements = _make_statements()
        info = _make_info()
        result_safe = compute_dcf_valuation(statements, info, risk_free_rate=0.04, altman_z=5.0)
        result_grey = compute_dcf_valuation(statements, info, risk_free_rate=0.04, altman_z=2.5)

        iv_safe = result_safe["distress_adjusted_iv"]
        iv_grey = result_grey["distress_adjusted_iv"]
        if not np.isnan(iv_safe) and not np.isnan(iv_grey):
            assert iv_grey < iv_safe, "Grey-zone company should get lower adjusted IV"

    def test_reinvestment_rate_in_result(self):
        statements = _make_statements()
        info = _make_info()
        result = compute_dcf_valuation(statements, info, risk_free_rate=0.04)
        # With positive ROIC and revenue growth, should be computed
        reinv = result["implied_reinvestment_rate"]
        if not np.isnan(reinv):
            assert reinv > 0


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
        assert np.isnan(roic)


# ── Tests: _distress_probability ─────────────────────────────────────


class TestDistressProbability:
    def test_zero_for_safe_z(self):
        assert _distress_probability(4.5) == 0.0
        assert _distress_probability(3.1) == 0.0

    def test_small_for_near_boundary(self):
        assert _distress_probability(2.8) == 0.02

    def test_moderate_for_grey_zone(self):
        assert _distress_probability(2.0) == 0.10

    def test_high_for_distressed(self):
        assert _distress_probability(0.5) == 0.50

    def test_zero_for_nan(self):
        assert _distress_probability(np.nan) == 0.0


# ── Tests: Model Verifier (Bug Fixes 1D, 1F) ────────────────────────


class TestVerifierImpliedGrowth:
    def test_error_when_implied_growth_exceeds_wacc(self):
        dcf_result = {
            "wacc": 0.10,
            "implied_growth_rate": 0.12,
            "intrinsic_value_per_share": 100.0,
            "dcf_upside_pct": 0.20,
            "fcf_yield": 0.05,
            "roic_vs_wacc_spread": 0.05,
            "margin_of_safety": 0.20,
        }
        info = {"currentPrice": 80.0}
        result = verify_dcf(dcf_result, info)
        assert not result.passed, "Should fail when implied growth >= WACC"
        assert any("Gordon Growth" in e for e in result.errors)

    def test_tightened_iv_bounds(self):
        dcf_result = {
            "wacc": 0.10,
            "intrinsic_value_per_share": 900.0,  # 6x price → error
            "dcf_upside_pct": 5.0,
            "fcf_yield": 0.05,
            "implied_growth_rate": 0.10,
            "roic_vs_wacc_spread": 0.05,
            "margin_of_safety": 0.80,
        }
        info = {"currentPrice": 150.0}
        result = verify_dcf(dcf_result, info)
        assert not result.passed, "IV at 6x price should be an error with tightened bounds"
