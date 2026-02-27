"""Tests for signals.fundamental_deep module."""

import numpy as np
import pandas as pd
import pytest

from signals.fundamental_deep import (
    compute_balance_sheet_health,
    compute_cash_flow_quality,
    compute_deep_fundamentals,
    compute_earnings_quality,
    compute_industry_relative_metrics,
    compute_institutional_blindspot,
    compute_piotroski_f_score,
    compute_profitability_trajectory,
)


# ── Synthetic Data Builders ──────────────────────────────────────────


def _make_quarterly_income(n_quarters: int = 8) -> pd.DataFrame:
    """Quarterly income statement: line items as index, dates as columns (descending)."""
    dates = pd.date_range("2022-01-01", periods=n_quarters, freq="QS")[::-1]
    base_rev = 1e9
    data = {}
    for i, d in enumerate(dates):
        rev = base_rev * (1 + 0.02 * i)  # slight growth across quarters
        data[d] = {
            "Total Revenue": rev,
            "Gross Profit": rev * 0.40,
            "Operating Income": rev * 0.20,
            "Net Income": rev * 0.15,
            "EBITDA": rev * 0.30,
            "Interest Expense": 1e7,
            "Tax Provision": rev * 0.15 * 0.25,
            "Basic EPS": 3.0 + 0.05 * i,
            "Cost Of Revenue": rev * 0.60,
            "Selling General And Administration": rev * 0.10,
        }
    return pd.DataFrame(data)


def _make_quarterly_balance_sheet(n_quarters: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n_quarters, freq="QS")[::-1]
    data = {}
    for i, d in enumerate(dates):
        data[d] = {
            "Total Assets": 5e9,
            "Current Assets": 2e9,
            "Current Liabilities": 1e9,
            "Long Term Debt": 8e8 - 1e7 * i,  # decreasing debt
            "Cash And Cash Equivalents": 5e8,
            "Stockholders Equity": 3e9,
            "Retained Earnings": 2e9 + 5e7 * i,
            "Total Liabilities Net Minority Interest": 2e9,
            "Accounts Receivable": 2e8,
            "Inventory": 1e8,
            "Accounts Payable": 1.5e8,
            "Ordinary Shares Number": 1e8,
        }
    return pd.DataFrame(data)


def _make_quarterly_cashflow(n_quarters: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n_quarters, freq="QS")[::-1]
    data = {}
    for i, d in enumerate(dates):
        data[d] = {
            "Operating Cash Flow": 2.5e8,
            "Capital Expenditure": -5e7,
            "Free Cash Flow": 2e8,
            "Depreciation And Amortization": 3e7,
            "Repurchase Of Capital Stock": -2e7,
            "Cash Dividends Paid": -1e7,
            "Net Income": 1.5e8,
        }
    return pd.DataFrame(data)


def _make_annual_income(n_years: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2023-12-31", periods=n_years, freq="YS")[::-1]
    data = {}
    revenues = [4e9, 3.6e9][:n_years]
    for d, rev in zip(dates, revenues):
        data[d] = {
            "Total Revenue": rev,
            "Gross Profit": rev * 0.40,
            "Operating Income": rev * 0.20,
            "Net Income": rev * 0.15,
            "Tax Provision": rev * 0.15 * 0.25,
        }
    return pd.DataFrame(data)


def _make_annual_balance_sheet(n_years: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2023-12-31", periods=n_years, freq="YS")[::-1]
    data = {}
    for i, d in enumerate(dates):
        data[d] = {
            "Total Assets": 5e9,
            "Current Assets": 2e9,
            "Current Liabilities": 1e9 + 5e7 * i,  # worsening for year 1
            "Long Term Debt": 8e8 + 2e7 * i,
            "Cash And Cash Equivalents": 5e8,
            "Stockholders Equity": 3e9,
            "Ordinary Shares Number": 1e8,
        }
    return pd.DataFrame(data)


def _make_info() -> dict:
    return {
        "marketCap": 10e9,
        "currentPrice": 150.0,
        "fullTimeEmployees": 5000,
        "dividendYield": 0.015,
        "numberOfAnalystOpinions": 3,
        "heldPercentInstitutions": 0.25,
        "targetMeanPrice": 180.0,
        "shortRatio": 2.5,
        "regularMarketPrice": 150.0,
    }


def _make_statements() -> dict:
    return {
        "quarterly_income": _make_quarterly_income(),
        "quarterly_balance_sheet": _make_quarterly_balance_sheet(),
        "quarterly_cashflow": _make_quarterly_cashflow(),
        "annual_income": _make_annual_income(),
        "annual_balance_sheet": _make_annual_balance_sheet(),
    }


# ── Tests: compute_profitability_trajectory ──────────────────────────


class TestProfitabilityTrajectory:
    def test_returns_all_7_keys(self):
        qi = _make_quarterly_income()
        qbs = _make_quarterly_balance_sheet()
        info = _make_info()
        result = compute_profitability_trajectory(qi, qbs, info)

        expected_keys = [
            "roe_4q_trend",
            "roic_latest",
            "roic_4q_trend",
            "gross_margin_4q_trend",
            "operating_margin_4q_trend",
            "net_margin_4q_trend",
            "revenue_per_employee",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        assert len(result) == 7

    def test_revenue_per_employee_computed(self):
        qi = _make_quarterly_income()
        qbs = _make_quarterly_balance_sheet()
        info = _make_info()
        result = compute_profitability_trajectory(qi, qbs, info)
        assert not np.isnan(result["revenue_per_employee"])
        assert result["revenue_per_employee"] > 0

    def test_handles_empty_dataframes(self):
        result = compute_profitability_trajectory(
            pd.DataFrame(), pd.DataFrame(), {}
        )
        for val in result.values():
            assert np.isnan(val)


# ── Tests: compute_balance_sheet_health ──────────────────────────────


class TestBalanceSheetHealth:
    def test_altman_z_score_computed(self):
        info = _make_info()
        qbs = _make_quarterly_balance_sheet()
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        result = compute_balance_sheet_health(info, qbs, qi, qcf)

        z = result["altman_z_score"]
        assert not np.isnan(z)
        # For a healthy company, Z should be positive
        assert z > 0

    def test_returns_6_keys(self):
        info = _make_info()
        qbs = _make_quarterly_balance_sheet()
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        result = compute_balance_sheet_health(info, qbs, qi, qcf)

        expected_keys = [
            "altman_z_score",
            "interest_coverage",
            "current_ratio",
            "debt_to_fcf",
            "net_debt_to_ebitda",
            "cash_as_pct_of_assets",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_current_ratio_value(self):
        info = _make_info()
        qbs = _make_quarterly_balance_sheet()
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        result = compute_balance_sheet_health(info, qbs, qi, qcf)
        # Current Assets=2e9, Current Liabilities=1e9 => ratio ~2.0
        assert abs(result["current_ratio"] - 2.0) < 0.1


# ── Tests: compute_piotroski_f_score ─────────────────────────────────


class TestPiotrosiFScore:
    def test_score_range_0_to_9(self):
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        qbs = _make_quarterly_balance_sheet()
        ai = _make_annual_income()
        ab = _make_annual_balance_sheet()
        result = compute_piotroski_f_score(qi, qcf, qbs, ai, ab)

        score = result["piotroski_f_score"]
        assert 0 <= score <= 9

    def test_returns_9_flags(self):
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        qbs = _make_quarterly_balance_sheet()
        ai = _make_annual_income()
        ab = _make_annual_balance_sheet()
        result = compute_piotroski_f_score(qi, qcf, qbs, ai, ab)

        for i in range(1, 10):
            key = f"piotroski_flag_{i}"
            assert key in result, f"Missing key: {key}"
            assert result[key] in (0.0, 1.0), f"Flag {i} should be 0.0 or 1.0"

    def test_score_equals_sum_of_flags(self):
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        qbs = _make_quarterly_balance_sheet()
        ai = _make_annual_income()
        ab = _make_annual_balance_sheet()
        result = compute_piotroski_f_score(qi, qcf, qbs, ai, ab)

        flag_sum = sum(result[f"piotroski_flag_{i}"] for i in range(1, 10))
        assert result["piotroski_f_score"] == flag_sum


# ── Tests: compute_cash_flow_quality ─────────────────────────────────


class TestCashFlowQuality:
    def test_negative_accruals_when_ocf_exceeds_ni(self):
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        qbs = _make_quarterly_balance_sheet()
        info = _make_info()
        result = compute_cash_flow_quality(qi, qcf, qbs, info)

        # OCF TTM = 4 * 2.5e8 = 1e9, NI TTM = 4 * 1.5e8 = 6e8
        # accruals = NI - OCF = 6e8 - 1e9 = -4e8 (negative => good)
        assert result["accruals_ratio"] < 0

    def test_returns_7_keys(self):
        qi = _make_quarterly_income()
        qcf = _make_quarterly_cashflow()
        qbs = _make_quarterly_balance_sheet()
        info = _make_info()
        result = compute_cash_flow_quality(qi, qcf, qbs, info)

        expected_keys = [
            "accruals_ratio",
            "fcf_to_net_income",
            "ocf_margin",
            "ocf_margin_4q_trend",
            "capex_intensity",
            "cash_conversion_cycle",
            "fcf_growth_3yr_cagr",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_handles_empty_data(self):
        result = compute_cash_flow_quality(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
        )
        for val in result.values():
            assert np.isnan(val)


# ── Tests: compute_earnings_quality ──────────────────────────────────


class TestEarningsQuality:
    def test_persistence_high_for_stable_earnings(self):
        qi = _make_quarterly_income(n_quarters=8)
        qcf = _make_quarterly_cashflow(n_quarters=8)
        result = compute_earnings_quality(qi, qcf)

        # Stable EPS (3.0 + small trend) should have high autocorrelation
        persistence = result["earnings_persistence"]
        if not np.isnan(persistence):
            # With a monotonic trend, autocorrelation should be positive
            assert persistence > 0

    def test_returns_5_keys(self):
        qi = _make_quarterly_income(n_quarters=8)
        qcf = _make_quarterly_cashflow(n_quarters=8)
        result = compute_earnings_quality(qi, qcf)

        expected_keys = [
            "earnings_persistence",
            "earnings_volatility",
            "revenue_growth_consistency",
            "sga_as_pct_revenue",
            "sga_trend",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# ── Tests: compute_institutional_blindspot ───────────────────────────


class TestInstitutionalBlindspot:
    def test_returns_all_6_keys(self):
        info = _make_info()
        holders = pd.DataFrame()
        insider = pd.DataFrame()
        result = compute_institutional_blindspot(info, holders, insider)

        expected_keys = [
            "analyst_count",
            "inst_ownership_pct",
            "insider_cluster_buy",
            "price_vs_analyst_target",
            "short_interest_days",
            "blindspot_score",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        assert len(result) == 6

    def test_blindspot_score_computed(self):
        info = _make_info()
        result = compute_institutional_blindspot(info, pd.DataFrame(), pd.DataFrame())
        # Low analyst count (3) + low inst ownership (25%) => high blindspot
        assert not np.isnan(result["blindspot_score"])
        assert 0 <= result["blindspot_score"] <= 1.0

    def test_insider_cluster_detection(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="3D")
        insider_df = pd.DataFrame({
            "Transaction": ["Purchase", "Purchase", "Purchase", "Sale", "Purchase"],
            "Start Date": dates,
            "Shares": [1000, 2000, 500, 1000, 300],
        })
        info = _make_info()
        result = compute_institutional_blindspot(info, pd.DataFrame(), insider_df)
        # 3 purchases within 12 days (<30 days) => cluster buy = 1.0
        assert result["insider_cluster_buy"] == 1.0


# ── Tests: compute_industry_relative_metrics ─────────────────────────


class TestIndustryRelativeMetrics:
    def test_percentiles_between_0_and_1(self):
        ticker_fund = {"pe_ratio": 20.0, "pb_ratio": 5.0, "altman_z_score": 3.5}
        all_fund = {
            "AAPL": {"pe_ratio": 25.0, "pb_ratio": 8.0, "altman_z_score": 4.0},
            "MSFT": {"pe_ratio": 30.0, "pb_ratio": 10.0, "altman_z_score": 5.0},
            "GOOG": {"pe_ratio": 15.0, "pb_ratio": 3.0, "altman_z_score": 2.0},
            "TEST": {"pe_ratio": 20.0, "pb_ratio": 5.0, "altman_z_score": 3.5},
        }
        industry_map = {
            "AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech", "TEST": "Tech",
        }
        result = compute_industry_relative_metrics("TEST", ticker_fund, all_fund, industry_map)

        for key, val in result.items():
            if not np.isnan(val):
                assert 0.0 <= val <= 1.0, f"{key} percentile out of range: {val}"

    def test_returns_composite_rank(self):
        ticker_fund = {"pe_ratio": 20.0}
        all_fund = {
            "A": {"pe_ratio": 10.0}, "B": {"pe_ratio": 30.0}, "TEST": {"pe_ratio": 20.0}
        }
        industry_map = {"A": "Tech", "B": "Tech", "TEST": "Tech"}
        result = compute_industry_relative_metrics("TEST", ticker_fund, all_fund, industry_map)
        assert "industry_fundamental_rank" in result


# ── Tests: compute_deep_fundamentals (master) ────────────────────────


class TestComputeDeepFundamentals:
    def test_returns_all_feature_categories(self):
        statements = _make_statements()
        info = _make_info()
        features = compute_deep_fundamentals(statements, info)

        # Should have features from categories 1-6
        assert "roe_4q_trend" in features  # cat 1
        assert "altman_z_score" in features  # cat 2
        assert "accruals_ratio" in features  # cat 3
        assert "earnings_persistence" in features  # cat 4
        assert "piotroski_f_score" in features  # cat 6
        assert len(features) > 20

    def test_handles_all_empty_statements(self):
        statements = {
            "quarterly_income": pd.DataFrame(),
            "quarterly_balance_sheet": pd.DataFrame(),
            "quarterly_cashflow": pd.DataFrame(),
            "annual_income": pd.DataFrame(),
            "annual_balance_sheet": pd.DataFrame(),
        }
        features = compute_deep_fundamentals(statements, {})
        # Should not raise; returned dict may have nans but should be populated
        assert isinstance(features, dict)
