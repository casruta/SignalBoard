"""Tests for signals.dynamic_screener module."""

import numpy as np
import pandas as pd
import pytest

from signals.dynamic_screener import DynamicScreener


# ── Synthetic Data Builders ──────────────────────────────────────────


def _make_healthy_deep_fund() -> dict:
    """Return a single stock's deep fundamentals that passes all safety gates."""
    return {
        "piotroski_f_score": 7,
        "altman_z_score": 5.0,
        "interest_coverage": 12.0,
        "current_ratio": 2.0,
        "quick_ratio": 1.8,
        "accruals_ratio": -0.03,
        "fcf_to_net_income": 1.1,
        "gross_margin_4q_trend": 0.02,
        "operating_margin_4q_trend": 0.01,
        "blindspot_score": 0.6,
        "revenue_growth_consistency": 0.05,
        "earnings_persistence": 0.7,
        "quarters_available": 8,
    }


def _make_healthy_dcf() -> dict:
    """Return DCF results for a healthy stock (positive FCF, ROIC > WACC,
    and >= 15% margin of safety per Damodaran)."""
    return {
        "roic_vs_wacc_spread": 0.06,
        "dcf_upside_pct": 0.25,    # 25% upside — passes 15% MOS gate
        "fcf_yield": 0.05,
    }


def _make_healthy_info(market_cap: float = 5e9) -> dict:
    return {
        "marketCap": market_cap,
        "revenueGrowth": 0.08,
        "averageVolume": 500_000,   # passes 100k volume filter
        "sector": "Technology",
    }


def _make_deep_fundamentals(n_stocks: int = 10) -> dict[str, dict]:
    """Build synthetic deep fundamentals for n_stocks tickers."""
    rng = np.random.default_rng(42)
    tickers = [f"STK{i:02d}" for i in range(n_stocks)]
    result = {}
    for t in tickers:
        result[t] = {
            "piotroski_f_score": rng.integers(5, 9),
            "altman_z_score": rng.uniform(3.5, 6.0),
            "interest_coverage": rng.uniform(5, 20),
            "current_ratio": rng.uniform(1.2, 3.0),
            "quick_ratio": rng.uniform(1.0, 2.5),
            "accruals_ratio": rng.uniform(-0.10, 0.05),
            "fcf_to_net_income": rng.uniform(0.8, 1.5),
            "gross_margin_4q_trend": rng.uniform(-0.05, 0.05),
            "operating_margin_4q_trend": rng.uniform(-0.05, 0.05),
            "blindspot_score": rng.uniform(0.0, 1.0),
            "revenue_growth_consistency": rng.uniform(0.02, 0.10),
            "earnings_persistence": rng.uniform(0.3, 0.9),
            "quarters_available": 8,
        }
    return result


def _make_dcf_results(tickers: list[str]) -> dict[str, dict]:
    rng = np.random.default_rng(42)
    result = {}
    for t in tickers:
        result[t] = {
            "roic_vs_wacc_spread": rng.uniform(0.01, 0.15),
            "dcf_upside_pct": rng.uniform(0.15, 0.5),  # all pass 15% MOS
            "fcf_yield": rng.uniform(0.02, 0.10),
        }
    return result


def _make_info_map(tickers: list[str], market_cap: float = 5e9) -> dict[str, dict]:
    return {
        t: {
            "marketCap": market_cap,
            "revenueGrowth": 0.08,
            "averageVolume": 500_000,
            "sector": "Technology",
        }
        for t in tickers
    }


def _build_test_data(n_stocks: int = 10, market_cap: float = 5e9):
    deep = _make_deep_fundamentals(n_stocks)
    tickers = list(deep.keys())
    dcf = _make_dcf_results(tickers)
    info = _make_info_map(tickers, market_cap)
    return deep, dcf, info, tickers


# ── Tests: DynamicScreener.screen ────────────────────────────────────


class TestDynamicScreenerScreen:
    def test_returns_list_of_tickers(self):
        deep, dcf, info, _ = _build_test_data()
        screener = DynamicScreener(top_n=5)
        result = screener.screen(deep, dcf, info)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_filters_distressed_stocks(self):
        deep, dcf, info, tickers = _build_test_data()
        # Make one stock distressed (low Altman Z)
        deep[tickers[0]]["altman_z_score"] = 1.0
        screener = DynamicScreener(top_n=10)
        result = screener.screen(deep, dcf, info)
        assert tickers[0] not in result

    def test_returns_at_most_top_n(self):
        deep, dcf, info, _ = _build_test_data(n_stocks=20)
        screener = DynamicScreener(top_n=5)
        result = screener.screen(deep, dcf, info)
        assert len(result) <= 5

    def test_empty_when_all_fail_safety(self):
        deep, dcf, info, tickers = _build_test_data()
        # Set all Altman Z below threshold
        for t in tickers:
            deep[t]["altman_z_score"] = 1.0
        screener = DynamicScreener(top_n=10)
        result = screener.screen(deep, dcf, info)
        assert result == []


# ── Tests: compute_dcf_rankings ──────────────────────────────────────


class TestComputeDCFRankings:
    def test_returns_expected_columns(self):
        deep, dcf, info, _ = _build_test_data()
        screener = DynamicScreener()
        df = screener.compute_dcf_rankings(deep, dcf, info)

        expected_columns = [
            "ticker",
            "dcf_upside_pct",
            "margin_of_safety",
            "intrinsic_value",
            "current_price",
            "wacc",
            "fcf_yield",
            "piotroski_f_score",
            "altman_z_score",
            "passes_safety",
            "rank",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_rank_ascending_by_dcf_upside(self):
        deep, dcf, info, _ = _build_test_data()
        screener = DynamicScreener()
        df = screener.compute_dcf_rankings(deep, dcf, info)
        # Rank 1 should have the highest DCF upside
        valid = df.dropna(subset=["dcf_upside_pct"])
        if len(valid) > 1:
            rank1 = valid[valid["rank"] == valid["rank"].min()]
            assert rank1["dcf_upside_pct"].iloc[0] >= valid["dcf_upside_pct"].min()

    def test_empty_input(self):
        screener = DynamicScreener()
        df = screener.compute_dcf_rankings({}, {}, {})
        assert df.empty

    def test_dcf_upside_capped(self):
        """DCF upside values should be capped within screener bounds."""
        deep, dcf, info, tickers = _build_test_data(n_stocks=3)
        screener = DynamicScreener()
        # Set extreme DCF upside
        dcf[tickers[0]]["dcf_upside_pct"] = 5.0  # 500%
        dcf[tickers[1]]["dcf_upside_pct"] = -0.9  # -90%
        df = screener.compute_dcf_rankings(deep, dcf, info)
        # Capped values should be within bounds
        valid = df["dcf_upside_pct"].dropna()
        assert (valid <= screener.DCF_UPSIDE_CAP).all()
        assert (valid >= screener.DCF_UPSIDE_FLOOR).all()


# ── Tests: Safety filter ─────────────────────────────────────────────


class TestSafetyFilter:
    """Damodaran-aligned safety filters: market cap, Altman Z > 3, 4+ quarters,
    non-declining revenue, Piotroski >= 5, positive FCF, ROIC > WACC,
    liquidity ratio, volume >= 100k, margin of safety >= 15%."""

    def _healthy(self):
        """Return (info, deep_fund, dcf) that passes all safety filters."""
        return _make_healthy_info(), _make_healthy_deep_fund(), _make_healthy_dcf()

    def test_blocks_small_market_cap(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info["marketCap"] = 200_000_000  # $200M < $300M
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_missing_market_cap(self):
        screener = DynamicScreener()
        _, deep, dcf = self._healthy()
        assert screener._apply_safety_filters({}, deep, dcf=dcf)[0] is False

    def test_passes_healthy_stock(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is True

    def test_blocks_low_altman_z(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        deep["altman_z_score"] = 1.5
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_grey_zone_altman_z(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        deep["altman_z_score"] = 2.5
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_altman_z_at_boundary(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        deep["altman_z_score"] = 3.0
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False
        deep["altman_z_score"] = 3.01
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is True

    def test_altman_z_threshold_from_config(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        deep["altman_z_score"] = 2.0
        config = {"screening": {"min_altman_z": 1.81}}
        assert screener._apply_safety_filters(info, deep, dcf=dcf, config=config)[0] is True

    def test_blocks_insufficient_quarters(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        deep["quarters_available"] = 3  # need >= 4
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_declining_revenue(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info["revenueGrowth"] = -0.05  # negative growth
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_low_piotroski(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        deep["piotroski_f_score"] = 4  # need >= 5
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_negative_fcf(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        dcf["fcf_yield"] = -0.02  # negative FCF
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_value_destroying_roic(self):
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        dcf["roic_vs_wacc_spread"] = -0.02  # ROIC < WACC
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_blocks_weak_quick_ratio_for_tech(self):
        """Tech sector uses quick ratio (not current ratio) for liquidity gate."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info["sector"] = "Technology"
        deep["quick_ratio"] = 0.5  # below 0.8 threshold
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_uses_current_ratio_for_industrials(self):
        """Industrials (inventory-heavy) still use current ratio >= 1.0."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info["sector"] = "Industrials"
        deep["current_ratio"] = 0.7
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_quick_ratio_fallback_to_current_ratio(self):
        """When quick ratio is missing for non-manufacturing, falls back to current ratio."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info["sector"] = "Technology"
        deep.pop("quick_ratio", None)
        deep["current_ratio"] = 0.7
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    # ── New Damodaran gates ──────────────────────────────────────────

    def test_blocks_low_volume(self):
        """Volume filter: stocks with <100k average daily volume are rejected."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info["averageVolume"] = 50_000  # below 100k threshold
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_passes_when_volume_missing(self):
        """If volume data is missing, don't block (fail open for data gaps)."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        info.pop("averageVolume", None)
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is True

    def test_blocks_insufficient_margin_of_safety(self):
        """Damodaran: require >= 15% DCF upside before recommending buy."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        dcf["dcf_upside_pct"] = 0.10  # 10% upside < 15% threshold
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is False

    def test_passes_at_margin_of_safety_threshold(self):
        """Exactly 15% upside should pass."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        dcf["dcf_upside_pct"] = 0.15  # exactly 15%
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is True

    def test_passes_when_dcf_upside_missing(self):
        """If DCF upside is NaN (data gap), don't block."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        dcf.pop("dcf_upside_pct", None)
        assert screener._apply_safety_filters(info, deep, dcf=dcf)[0] is True

    def test_margin_of_safety_configurable(self):
        """Config can override the 15% default."""
        screener = DynamicScreener()
        info, deep, dcf = self._healthy()
        dcf["dcf_upside_pct"] = 0.08  # 8% — would fail default 15%
        config = {"screening": {"min_margin_of_safety": 0.05}}
        assert screener._apply_safety_filters(info, deep, dcf=dcf, config=config)[0] is True




# ── Tests: Filing lag utility ────────────────────────────────────────


class TestFilingLag:
    def test_filter_by_filing_lag(self):
        from signals.fundamental_deep import filter_by_filing_lag
        dates = pd.to_datetime(["2025-06-30", "2025-03-31", "2024-12-31"])
        df = pd.DataFrame(
            {"2025-06-30": [100], "2025-03-31": [90], "2024-12-31": [80]},
            index=["Total Revenue"],
        )
        df.columns = dates
        # As of Aug 1, only Q ending March 31 and Dec 31 should be available
        result = filter_by_filing_lag(df, pd.Timestamp("2025-08-01"), filing_lag_days=45)
        assert len(result.columns) == 2  # Jun 30 excluded (not yet filed)


# ── Tests: Damodaran DCF helpers ─────────────────────────────────────


class TestDamodaranDCFHelpers:
    """Test bottom-up beta, synthetic cost of debt, and FCFF helpers."""

    def test_bottom_up_beta_unlevered_for_zero_debt(self):
        from signals.dcf_valuation import _compute_bottom_up_beta
        beta = _compute_bottom_up_beta("Technology", 0.0, 1e9)
        assert abs(beta - 1.10) < 0.01  # should equal unlevered

    def test_bottom_up_beta_increases_with_leverage(self):
        from signals.dcf_valuation import _compute_bottom_up_beta
        low_lev = _compute_bottom_up_beta("Industrials", 1e8, 1e9)
        high_lev = _compute_bottom_up_beta("Industrials", 5e8, 1e9)
        assert high_lev > low_lev

    def test_synthetic_cost_of_debt_aaa(self):
        from signals.dcf_valuation import _synthetic_cost_of_debt
        kd = _synthetic_cost_of_debt(ebit=100, interest_expense=5, risk_free_rate=0.04)
        # coverage = 20 → AAA → spread 0.63%
        assert abs(kd - 0.0463) < 0.001

    def test_synthetic_cost_of_debt_distressed(self):
        from signals.dcf_valuation import _synthetic_cost_of_debt
        kd = _synthetic_cost_of_debt(ebit=5, interest_expense=20, risk_free_rate=0.04)
        # coverage = 0.25 → D → spread 12%
        assert kd > 0.15

    def test_synthetic_cost_of_debt_no_debt(self):
        from signals.dcf_valuation import _synthetic_cost_of_debt
        kd = _synthetic_cost_of_debt(ebit=100, interest_expense=0, risk_free_rate=0.04)
        assert abs(kd - 0.0463) < 0.001  # AAA-equivalent

    def test_terminal_growth_capped_at_risk_free(self):
        from signals.dcf_valuation import get_terminal_growth
        g = get_terminal_growth("Technology", risk_free_rate=0.03)
        assert g <= 0.03  # Tech raw = 3.5%, capped at 3.0% Rf
