"""Tests for signals.dynamic_screener module."""

import numpy as np
import pandas as pd
import pytest

from signals.dynamic_screener import DynamicScreener


# ── Synthetic Data Builders ──────────────────────────────────────────


def _make_deep_fundamentals(n_stocks: int = 10) -> dict[str, dict]:
    """Build synthetic deep fundamentals for n_stocks tickers."""
    rng = np.random.default_rng(42)
    tickers = [f"STK{i:02d}" for i in range(n_stocks)]
    result = {}
    for t in tickers:
        result[t] = {
            "piotroski_f_score": rng.integers(3, 9),
            "altman_z_score": rng.uniform(2.0, 6.0),
            "interest_coverage": rng.uniform(5, 20),
            "accruals_ratio": rng.uniform(-0.10, 0.05),
            "fcf_to_net_income": rng.uniform(0.8, 1.5),
            "gross_margin_4q_trend": rng.uniform(-0.05, 0.05),
            "operating_margin_4q_trend": rng.uniform(-0.05, 0.05),
            "blindspot_score": rng.uniform(0.0, 1.0),
            "quarters_available": 4,
        }
    return result


def _make_dcf_results(tickers: list[str]) -> dict[str, dict]:
    rng = np.random.default_rng(42)
    result = {}
    for t in tickers:
        result[t] = {
            "roic_vs_wacc_spread": rng.uniform(-0.05, 0.15),
            "dcf_upside_pct": rng.uniform(-0.3, 0.5),
        }
    return result


def _make_info_map(tickers: list[str], market_cap: float = 5e9) -> dict[str, dict]:
    return {t: {"marketCap": market_cap} for t in tickers}


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


# ── Tests: compute_composite_scores ──────────────────────────────────


class TestComputeCompositeScores:
    def test_returns_expected_columns(self):
        deep, dcf, info, _ = _build_test_data()
        screener = DynamicScreener()
        df = screener.compute_composite_scores(deep, dcf, info)

        expected_columns = [
            "ticker",
            "composite_score",
            "rank",
            "piotroski_score",
            "roic_spread_score",
            "cash_flow_score",
            "balance_sheet_score",
            "dcf_score",
            "blindspot_score",
            "margin_score",
            "passes_safety",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_scores_between_0_and_1(self):
        deep, dcf, info, _ = _build_test_data()
        screener = DynamicScreener()
        df = screener.compute_composite_scores(deep, dcf, info)

        score_cols = [
            "piotroski_score", "roic_spread_score", "cash_flow_score",
            "balance_sheet_score", "dcf_score", "blindspot_score", "margin_score",
        ]
        for col in score_cols:
            assert (df[col] >= 0).all() and (df[col] <= 1).all(), f"{col} out of [0,1]"

    def test_empty_input(self):
        screener = DynamicScreener()
        df = screener.compute_composite_scores({}, {}, {})
        assert df.empty


# ── Tests: Regime weight adjustment ──────────────────────────────────


class TestRegimeAdjustment:
    def test_risk_off_increases_balance_sheet_weight(self):
        screener = DynamicScreener()
        neutral = screener._get_regime_weights("neutral")
        risk_off = screener._get_regime_weights("risk_off")
        # In risk_off, balance_sheet weight factor is 1.5 so its relative weight
        # should be higher than in neutral
        assert risk_off["balance_sheet"] > neutral["balance_sheet"]

    def test_risk_on_increases_dcf_weight(self):
        screener = DynamicScreener()
        neutral = screener._get_regime_weights("neutral")
        risk_on = screener._get_regime_weights("risk_on")
        assert risk_on["dcf_upside"] > neutral["dcf_upside"]

    def test_weights_sum_to_one(self):
        screener = DynamicScreener()
        for regime in ["neutral", "risk_off", "risk_on"]:
            weights = screener._get_regime_weights(regime)
            assert abs(sum(weights.values()) - 1.0) < 1e-9


# ── Tests: Safety filter ─────────────────────────────────────────────


class TestSafetyFilter:
    def test_blocks_small_market_cap(self):
        screener = DynamicScreener()
        info = {"marketCap": 200_000_000}  # $200M < $300M
        deep_fund = {"altman_z_score": 4.0, "quarters_available": 4}
        assert screener._apply_safety_filters(info, deep_fund) is False

    def test_blocks_missing_market_cap(self):
        screener = DynamicScreener()
        info = {}
        deep_fund = {"altman_z_score": 4.0, "quarters_available": 4}
        assert screener._apply_safety_filters(info, deep_fund) is False

    def test_passes_healthy_stock(self):
        screener = DynamicScreener()
        info = {"marketCap": 5e9}
        deep_fund = {"altman_z_score": 4.0, "quarters_available": 4}
        assert screener._apply_safety_filters(info, deep_fund) is True

    def test_blocks_low_altman_z(self):
        screener = DynamicScreener()
        info = {"marketCap": 5e9}
        deep_fund = {"altman_z_score": 1.5, "quarters_available": 4}
        assert screener._apply_safety_filters(info, deep_fund) is False

    def test_blocks_insufficient_quarters(self):
        screener = DynamicScreener()
        info = {"marketCap": 5e9}
        deep_fund = {"altman_z_score": 4.0, "quarters_available": 1}
        assert screener._apply_safety_filters(info, deep_fund) is False
