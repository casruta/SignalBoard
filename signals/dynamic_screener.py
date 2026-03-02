"""DCF-focused stock screener — ranks stocks purely by intrinsic value discount.

Stocks are scored across 3 DCF-derived dimensions:
  1. Margin of Safety (40%) — DCF intrinsic value vs market price
  2. FCF Yield (30%) — free cash flow relative to enterprise value
  3. ROIC vs WACC Spread (30%) — economic value creation

Safety filters enforce hard quality gates (Altman Z > 3, Piotroski >= 5,
ROIC > WACC, positive FCF, etc.) to ensure only fundamentally sound stocks
are ranked by their DCF discount.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DynamicScreener:
    """DCF-focused screener that ranks stocks by intrinsic value discount.

    Replaces the prior 9-dimension composite model with a pure DCF approach:
    stocks are ranked by how undervalued they are relative to a Damodaran-aligned
    discounted cash flow valuation.
    """

    DEFAULT_WEIGHTS = {
        "margin_of_safety": 0.40,
        "fcf_yield": 0.30,
        "roic_wacc_spread": 0.30,
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
        config: dict | None = None,
    ) -> list[str]:
        """Screen and rank stocks, return top N tickers.

        Steps:
        1. Apply hard safety filters (can't score your way past these)
        2. Compute 3 DCF component scores per stock (0-1 range, percentile-ranked)
        3. Sort by composite score, return top N
        """
        df = self.compute_composite_scores(
            deep_fundamentals, dcf_results, info_map, macro_regime, config=config
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
        config: dict | None = None,
    ) -> pd.DataFrame:
        """Compute DCF-based composite scores for all stocks.

        Returns DataFrame with columns:
        - ticker, composite_score, rank
        - dcf_upside_score, fcf_yield_score, roic_spread_score
        - passes_safety (bool)
        """
        tickers = sorted(
            set(deep_fundamentals) | set(dcf_results) | set(info_map)
        )
        if not tickers:
            return pd.DataFrame()

        # Extract raw DCF values for each component
        raw = self._extract_raw_values(tickers, deep_fundamentals, dcf_results, info_map)

        # Percentile-rank each component across the universe
        scored = pd.DataFrame({"ticker": tickers})
        scored["dcf_upside_score"] = self._score_component(raw["margin_of_safety"])
        scored["fcf_yield_score"] = self._score_component(raw["fcf_yield"])
        scored["roic_spread_score"] = self._score_component(raw["roic_wacc_spread"])

        # Safety filter
        scored["passes_safety"] = [
            self._apply_safety_filters(
                info_map.get(t, {}), deep_fundamentals.get(t, {}),
                dcf=dcf_results.get(t, {}), config=config,
            )
            for t in tickers
        ]

        # Data completeness penalty
        scored["data_completeness"] = self._compute_data_completeness(raw, len(tickers))

        # Weighted composite
        weights = self._get_weights(config=config)
        component_columns = {
            "margin_of_safety": "dcf_upside_score",
            "fcf_yield": "fcf_yield_score",
            "roic_wacc_spread": "roic_spread_score",
        }
        raw_composite = sum(
            weights[component] * scored[col]
            for component, col in component_columns.items()
        )
        scored["composite_score"] = raw_composite * scored["data_completeness"]

        scored["rank"] = (
            scored["composite_score"].rank(ascending=False, method="min").astype(int)
        )
        scored = scored.sort_values("composite_score", ascending=False).reset_index(
            drop=True
        )
        return scored

    # ── Safety Filters ────────────────────────────────────────────────

    def _apply_safety_filters(
        self, info: dict, deep_fund: dict, dcf: dict | None = None,
        config: dict | None = None,
    ) -> bool:
        """Hard safety filters — a stock cannot score its way past these.

        Damodaran-aligned quality gates:
         1. Market cap between $300M and $10B (configurable)
         2. Altman Z-Score > 3.0 (safe zone only)
         3. At least 4 quarters of financial data
         4. Revenue not declining (>= 0% YoY, configurable)
         5. Piotroski F-Score >= 5/9 (minimum financial quality)
         6. Positive free cash flow (FCF yield > 0)
         7. ROIC > WACC (must create economic value)
         8. Liquidity: current ratio >= 1.0
         9. Minimum trading volume (liquidity screen)
        10. Margin of safety: DCF upside >= 15% (Damodaran)

        Returns True if stock passes ALL filters.
        """
        cfg_s = config.get("screening", {}) if config else {}
        dcf = dcf or {}

        # 1. Market cap range
        market_cap = _safe_float(info.get("marketCap"))
        min_cap = cfg_s.get("min_market_cap", 300_000_000)
        max_cap = cfg_s.get("max_market_cap", 10_000_000_000)
        if np.isnan(market_cap) or market_cap < min_cap or market_cap > max_cap:
            return False

        # 2. Financial soundness — Altman Z safe zone
        min_z = cfg_s.get("min_altman_z", 3.0)
        altman_z = _safe_float(deep_fund.get("altman_z_score"))
        if np.isnan(altman_z) or altman_z <= min_z:
            return False

        # 3. Data completeness — need enough history for trend analysis
        quarters = deep_fund.get("quarters_available", 0)
        min_quarters = cfg_s.get("min_quarters", 4)
        if not isinstance(quarters, (int, float)) or quarters < min_quarters:
            return False

        # 4. Revenue trajectory — no declining businesses
        min_rev_growth = cfg_s.get("min_revenue_growth", 0.0)
        rev_growth = _safe_float(info.get("revenueGrowth"))
        if not np.isnan(rev_growth) and rev_growth < min_rev_growth:
            return False

        # 5. Piotroski quality floor — minimum financial health
        min_piotroski = cfg_s.get("min_piotroski", 5)
        piotroski = _safe_float(deep_fund.get("piotroski_f_score"))
        if np.isnan(piotroski) or piotroski < min_piotroski:
            return False

        # 6. Positive free cash flow — must generate cash
        fcf_yield = _safe_float(dcf.get("fcf_yield"))
        if not np.isnan(fcf_yield) and fcf_yield <= 0:
            return False

        # 7. Value creation — ROIC must exceed cost of capital
        roic_spread = _safe_float(dcf.get("roic_vs_wacc_spread"))
        if not np.isnan(roic_spread) and roic_spread <= 0:
            return False

        # 8. Liquidity gate — current ratio >= 1.0
        current_ratio = _safe_float(deep_fund.get("current_ratio"))
        if not np.isnan(current_ratio) and current_ratio < 1.0:
            return False

        # 9. Trading volume — enforce minimum daily volume (Damodaran: liquidity screen)
        min_volume = cfg_s.get("min_avg_daily_volume", 100_000)
        avg_volume = _safe_float(info.get("averageVolume"))
        if not np.isnan(avg_volume) and avg_volume < min_volume:
            return False

        # 10. Margin of safety — Damodaran: require discount to intrinsic value
        min_mos = cfg_s.get("min_margin_of_safety", 0.15)
        dcf_upside = _safe_float(dcf.get("dcf_upside_pct"))
        if not np.isnan(dcf_upside) and dcf_upside < min_mos:
            return False

        return True

    # ── Component Score Helpers ───────────────────────────────────────

    def _score_component(self, values: pd.Series) -> pd.Series:
        """Convert raw values to 0-1 percentile scores across the universe.

        Higher percentile = better.  NaN values get 0.25 (below-median penalty)
        to discourage data-sparse stocks from scoring well by default.
        """
        filled = values.copy()
        mask = filled.isna()
        non_null = filled[~mask]

        if non_null.empty:
            return pd.Series(0.25, index=values.index)

        ranked = non_null.rank(pct=True, method="average")
        filled.loc[~mask] = ranked
        filled.loc[mask] = 0.25  # penalise missing data instead of neutral 0.5
        return filled

    @staticmethod
    def _compute_data_completeness(
        raw: dict[str, pd.Series], n_stocks: int,
    ) -> pd.Series:
        """Fraction of non-NaN components per stock (0-1).

        3 components total.  A stock with all 3 populated gets 1.0.
        Sqrt scaling so partial data is penalised but not catastrophically.
        """
        n_components = len(raw)
        non_nan_count = pd.Series(0, index=range(n_stocks), dtype=float)
        for component_values in raw.values():
            non_nan_count += (~component_values.isna()).astype(float)
        return (non_nan_count / n_components).apply(np.sqrt)

    def _get_weights(self, config: dict | None = None) -> dict:
        """Get scoring weights from config or use defaults."""
        config_weights = (
            config.get("screening", {}).get("weights") if config else None
        )
        weights = dict(config_weights if config_weights else self.DEFAULT_WEIGHTS)

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
        """Extract raw numeric values for each DCF scoring component.

        Returns dict mapping component name to a Series indexed by position
        (aligned with *tickers* list).
        """
        mos_vals = []
        fcf_yield_vals = []
        roic_spread_vals = []

        for ticker in tickers:
            dcf = dcf_results.get(ticker, {})

            # Margin of safety (DCF upside %)
            mos_vals.append(_safe_float(dcf.get("dcf_upside_pct")))

            # FCF yield
            fcf_yield_vals.append(_safe_float(dcf.get("fcf_yield")))

            # ROIC vs WACC spread
            roic_spread_vals.append(_safe_float(dcf.get("roic_vs_wacc_spread")))

        return {
            "margin_of_safety": pd.Series(mos_vals),
            "fcf_yield": pd.Series(fcf_yield_vals),
            "roic_wacc_spread": pd.Series(roic_spread_vals),
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
