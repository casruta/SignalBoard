"""Dynamic screening: adaptive composite ranking to find under-the-radar quality stocks.

Replaces rigid threshold filters (ROE>12%, D/E<2, P/E<35) with an adaptive
composite ranking system that thinks like a professional analyst.  Instead of
AND-filters that exclude stocks for failing one criterion, stocks are scored
holistically so strength in one area can compensate for weakness in another.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DynamicScreener:
    """Adaptive screener that ranks stocks by composite quality score.

    Replaces rigid threshold filters with a scoring system that adapts
    to market regime and evaluates stocks holistically.
    """

    # Default component weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        "piotroski": 0.20,
        "roic_spread": 0.15,
        "cash_flow_quality": 0.15,
        "balance_sheet": 0.15,
        "dcf_upside": 0.15,
        "blindspot": 0.10,
        "margin_trajectory": 0.10,
    }

    # Regime adjustments: multiply default weights by these factors
    REGIME_ADJUSTMENTS = {
        "risk_off": {
            "balance_sheet": 1.5,
            "cash_flow_quality": 1.3,
            "dcf_upside": 0.7,
            "margin_trajectory": 0.8,
        },
        "risk_on": {
            "dcf_upside": 1.4,
            "margin_trajectory": 1.3,
            "balance_sheet": 0.8,
            "blindspot": 0.8,
        },
    }

    def __init__(self, top_n: int = 100):
        self.top_n = top_n

    # ── Public API ────────────────────────────────────────────────────

    def screen(
        self,
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
        macro_regime: str = "neutral",
        vix_level: float | None = None,
    ) -> list[str]:
        """Screen and rank stocks, return top N tickers.

        Steps:
        1. Apply hard safety filters (can't score your way past these)
        2. Compute 7 component scores per stock (0-1 range, percentile-ranked)
        3. Apply regime-adjusted weights
        4. Sort by composite score, return top N
        """
        df = self.compute_composite_scores(
            deep_fundamentals, dcf_results, info_map, macro_regime
        )
        if df.empty:
            logger.warning("No stocks survived screening.")
            return []

        safe = df[df["passes_safety"]].copy()
        if safe.empty:
            logger.warning("All stocks failed safety filters.")
            return []

        safe = safe.sort_values("composite_score", ascending=False).head(self.top_n)
        return safe["ticker"].tolist()

    def compute_composite_scores(
        self,
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
        macro_regime: str = "neutral",
    ) -> pd.DataFrame:
        """Compute composite quality scores for all stocks.

        Returns DataFrame with columns:
        - ticker, composite_score, rank
        - piotroski_score, roic_spread_score, cash_flow_score,
          balance_sheet_score, dcf_score, blindspot_score, margin_score
        - passes_safety (bool)
        """
        tickers = sorted(
            set(deep_fundamentals) | set(dcf_results) | set(info_map)
        )
        if not tickers:
            return pd.DataFrame()

        # Extract raw values for each component
        raw = self._extract_raw_values(tickers, deep_fundamentals, dcf_results, info_map)

        # Percentile-rank each component across the universe
        scored = pd.DataFrame({"ticker": tickers})
        scored["piotroski_score"] = self._score_component(raw["piotroski"])
        scored["roic_spread_score"] = self._score_component(raw["roic_spread"])
        scored["cash_flow_score"] = self._score_component(raw["cash_flow_quality"])
        scored["balance_sheet_score"] = self._score_component(raw["balance_sheet"])
        scored["dcf_score"] = self._score_component(raw["dcf_upside"])
        scored["blindspot_score"] = self._score_component(raw["blindspot"])
        scored["margin_score"] = self._score_component(raw["margin_trajectory"])

        # Safety filter
        scored["passes_safety"] = [
            self._apply_safety_filters(
                info_map.get(t, {}), deep_fundamentals.get(t, {})
            )
            for t in tickers
        ]

        # Weighted composite
        weights = self._get_regime_weights(macro_regime)
        component_columns = {
            "piotroski": "piotroski_score",
            "roic_spread": "roic_spread_score",
            "cash_flow_quality": "cash_flow_score",
            "balance_sheet": "balance_sheet_score",
            "dcf_upside": "dcf_score",
            "blindspot": "blindspot_score",
            "margin_trajectory": "margin_score",
        }
        scored["composite_score"] = sum(
            weights[component] * scored[col]
            for component, col in component_columns.items()
        )

        scored["rank"] = (
            scored["composite_score"].rank(ascending=False, method="min").astype(int)
        )
        scored = scored.sort_values("composite_score", ascending=False).reset_index(
            drop=True
        )
        return scored

    # ── Safety Filters ────────────────────────────────────────────────

    def _apply_safety_filters(self, info: dict, deep_fund: dict) -> bool:
        """Hard safety filters that cannot be compensated:

        - altman_z_score > 1.81  (not in distress zone)
        - market_cap > 300_000_000  ($300M minimum)
        - At minimum 2 quarters of data available

        Returns True if stock passes all filters.
        """
        market_cap = _safe_float(info.get("marketCap"))
        if np.isnan(market_cap) or market_cap <= 300_000_000:
            return False

        altman_z = _safe_float(deep_fund.get("altman_z_score"))
        if np.isnan(altman_z) or altman_z <= 1.81:
            return False

        quarters = deep_fund.get("quarters_available", 0)
        if not isinstance(quarters, (int, float)) or quarters < 2:
            return False

        return True

    # ── Component Score Helpers ───────────────────────────────────────

    def _score_component(self, values: pd.Series) -> pd.Series:
        """Convert raw values to 0-1 percentile scores across the universe.

        Higher percentile = better.  NaN values get 0.5 (neutral).
        """
        filled = values.copy()
        mask = filled.isna()
        non_null = filled[~mask]

        if non_null.empty:
            return pd.Series(0.5, index=values.index)

        ranked = non_null.rank(pct=True, method="average")
        filled.loc[~mask] = ranked
        filled.loc[mask] = 0.5
        return filled

    def _get_regime_weights(self, macro_regime: str) -> dict:
        """Apply regime adjustments to default weights, then renormalize to sum=1."""
        weights = dict(self.DEFAULT_WEIGHTS)
        adjustments = self.REGIME_ADJUSTMENTS.get(macro_regime, {})

        for component, factor in adjustments.items():
            if component in weights:
                weights[component] *= factor

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    # ── Raw Value Extraction ──────────────────────────────────────────

    def _extract_raw_values(
        self,
        tickers: list[str],
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
    ) -> dict[str, pd.Series]:
        """Extract raw numeric values for each scoring component.

        Returns dict mapping component name to a Series indexed by position
        (aligned with *tickers* list).
        """
        piotroski_vals = []
        roic_spread_vals = []
        cash_flow_vals = []
        balance_sheet_vals = []
        dcf_upside_vals = []
        blindspot_vals = []
        margin_vals = []

        for ticker in tickers:
            df = deep_fundamentals.get(ticker, {})
            dcf = dcf_results.get(ticker, {})

            # Piotroski F-Score (0-9)
            piotroski_vals.append(_safe_float(df.get("piotroski_f_score")))

            # ROIC vs WACC spread
            roic_spread_vals.append(_safe_float(dcf.get("roic_vs_wacc_spread")))

            # Cash flow quality: average of inverted accruals_ratio and fcf_to_net_income
            accruals = _safe_float(df.get("accruals_ratio"))
            fcf_ni = _safe_float(df.get("fcf_to_net_income"))
            inverted_accruals = -accruals if not np.isnan(accruals) else np.nan
            cash_flow_vals.append(_nanmean([inverted_accruals, fcf_ni]))

            # Balance sheet: average of altman_z and interest_coverage
            altman = _safe_float(df.get("altman_z_score"))
            interest_cov = _safe_float(df.get("interest_coverage"))
            balance_sheet_vals.append(_nanmean([altman, interest_cov]))

            # DCF upside
            dcf_upside_vals.append(_safe_float(dcf.get("dcf_upside_pct")))

            # Blindspot (institutional analysis)
            blindspot_vals.append(_safe_float(df.get("blindspot_score")))

            # Margin trajectory: average of gross and operating margin trends
            gm_trend = _safe_float(df.get("gross_margin_4q_trend"))
            om_trend = _safe_float(df.get("operating_margin_4q_trend"))
            margin_vals.append(_nanmean([gm_trend, om_trend]))

        return {
            "piotroski": pd.Series(piotroski_vals),
            "roic_spread": pd.Series(roic_spread_vals),
            "cash_flow_quality": pd.Series(cash_flow_vals),
            "balance_sheet": pd.Series(balance_sheet_vals),
            "dcf_upside": pd.Series(dcf_upside_vals),
            "blindspot": pd.Series(blindspot_vals),
            "margin_trajectory": pd.Series(margin_vals),
        }


# ── Module-level helpers ──────────────────────────────────────────────


def _safe_float(value) -> float:
    """Convert *value* to float, returning np.nan on failure."""
    if value is None:
        return np.nan
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _nanmean(values: list[float]) -> float:
    """Mean of non-NaN values.  Returns np.nan if all are NaN."""
    finite = [v for v in values if not np.isnan(v)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


# ── Convenience function ──────────────────────────────────────────────


def rank_stocks_by_quality(
    deep_fundamentals: dict[str, dict],
    dcf_results: dict[str, dict],
    info_map: dict[str, dict],
    macro_regime: str = "neutral",
    top_n: int = 100,
) -> list[str]:
    """Convenience function: screen and return top N tickers."""
    screener = DynamicScreener(top_n=top_n)
    return screener.screen(deep_fundamentals, dcf_results, info_map, macro_regime)
