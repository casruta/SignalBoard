"""Dynamic screening: Damodaran-aligned composite ranking via 9-dimension model.

Stocks are scored across 9 dimensions (Piotroski, cash flow quality, ROIC spread,
balance sheet, DCF upside, income health, growth momentum, margin trajectory,
blindspot) with percentile-ranked 0-1 scores and regime-adjusted weights.

Safety filters enforce 10 hard quality gates following Damodaran's framework:
Altman Z > 3, Piotroski >= 5, ROIC > WACC, positive FCF, non-declining revenue,
current ratio >= 1.0, minimum trading volume, and a 15% margin-of-safety
requirement (Damodaran: never buy at fair value — require compensation for
estimation error in the DCF).
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

    # Default component weights (sum to 1.0) — fundamentals-first, numbers-driven
    DEFAULT_WEIGHTS = {
        "piotroski": 0.14,
        "cash_flow_quality": 0.16,    # critical for small caps
        "roic_spread": 0.14,          # direct value creation measure
        "balance_sheet": 0.10,
        "dcf_upside": 0.10,           # valuation upside
        "income_health": 0.16,        # revenue trajectory, earnings persistence, peer-relative
        "growth_momentum": 0.10,      # revenue + earnings growth
        "margin_trajectory": 0.05,
        "blindspot": 0.05,            # minor — coverage gap is noise
    }

    # Regime adjustments: multiply default weights by these factors
    REGIME_ADJUSTMENTS = {
        "risk_off": {
            "balance_sheet": 1.5,
            "cash_flow_quality": 1.3,
            "income_health": 1.3,
            "dcf_upside": 0.7,
            "growth_momentum": 0.7,
            "margin_trajectory": 0.8,
        },
        "risk_on": {
            "dcf_upside": 1.4,
            "growth_momentum": 1.5,
            "income_health": 1.1,
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
        config: dict | None = None,
    ) -> list[str]:
        """Screen and rank stocks, return top N tickers.

        Steps:
        1. Apply hard safety filters (can't score your way past these)
        2. Compute 8 component scores per stock (0-1 range, percentile-ranked)
        3. Apply regime-adjusted weights
        4. Sort by composite score, return top N
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
        """Compute composite quality scores for all stocks.

        Returns DataFrame with columns:
        - ticker, composite_score, rank
        - piotroski_score, roic_spread_score, cash_flow_score,
          balance_sheet_score, dcf_score, growth_score, blindspot_score, margin_score
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
        scored["income_health_score"] = self._score_component(raw["income_health"])
        scored["growth_score"] = self._score_component(raw["growth_momentum"])
        scored["blindspot_score"] = self._score_component(raw["blindspot"])
        scored["margin_score"] = self._score_component(raw["margin_trajectory"])

        # Safety filter — now includes DCF data for ROIC/FCF gates
        scored["passes_safety"] = [
            self._apply_safety_filters(
                info_map.get(t, {}), deep_fundamentals.get(t, {}),
                dcf=dcf_results.get(t, {}), config=config,
            )
            for t in tickers
        ]

        # Data completeness penalty — stocks with many NaN components
        # get penalised instead of hiding behind 0.5 neutral scores.
        # Count how many of the 9 raw components were NaN for each stock.
        scored["data_completeness"] = self._compute_data_completeness(raw, len(tickers))

        # Weighted composite — use config overrides if provided
        weights = self._get_regime_weights(macro_regime, config=config)
        component_columns = {
            "piotroski": "piotroski_score",
            "roic_spread": "roic_spread_score",
            "cash_flow_quality": "cash_flow_score",
            "balance_sheet": "balance_sheet_score",
            "dcf_upside": "dcf_score",
            "income_health": "income_health_score",
            "growth_momentum": "growth_score",
            "blindspot": "blindspot_score",
            "margin_trajectory": "margin_score",
        }
        raw_composite = sum(
            weights[component] * scored[col]
            for component, col in component_columns.items()
        )
        # Apply data completeness penalty: full data = 1.0x, missing dims = discount
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

        9 components total.  A stock with all 9 populated gets 1.0.
        A stock with 6/9 gets ~0.93 (mild penalty via sqrt scaling).
        A stock with 3/9 gets ~0.82 (meaningful penalty).
        """
        n_components = len(raw)
        non_nan_count = pd.Series(0, index=range(n_stocks), dtype=float)
        for component_values in raw.values():
            non_nan_count += (~component_values.isna()).astype(float)
        # Sqrt scaling: not as harsh as linear, but still meaningful
        return (non_nan_count / n_components).apply(np.sqrt)

    def _get_regime_weights(
        self, macro_regime: str, config: dict | None = None
    ) -> dict:
        """Apply regime adjustments to default weights, then renormalize to sum=1.

        If config["screening"]["weights"] exists, use those as the base weights
        instead of DEFAULT_WEIGHTS.
        """
        config_weights = (
            config.get("screening", {}).get("weights") if config else None
        )
        weights = dict(config_weights if config_weights else self.DEFAULT_WEIGHTS)
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
        income_health_vals = []
        growth_vals = []
        blindspot_vals = []
        margin_vals = []

        for ticker in tickers:
            df = deep_fundamentals.get(ticker, {})
            dcf = dcf_results.get(ticker, {})
            info = info_map.get(ticker, {})

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

            # Income statement health: revenue growth, consistency,
            # earnings persistence, operating trend, and peer-relative growth
            rev_growth = _safe_float(info.get("revenueGrowth"))
            rev_consistency = _safe_float(df.get("revenue_growth_consistency"))
            # Invert consistency (lower std = better) so higher is better
            inv_consistency = -rev_consistency if not np.isnan(rev_consistency) else np.nan
            earn_persist = _safe_float(df.get("earnings_persistence"))
            om_trend = _safe_float(df.get("operating_margin_4q_trend"))
            # Industry-relative revenue growth percentile (0-1, already oriented)
            rev_industry_pctl = _safe_float(df.get("revenue_growth_industry_pctl"))
            income_health_vals.append(
                _nanmean([rev_growth, inv_consistency, earn_persist, om_trend, rev_industry_pctl])
            )

            # Growth momentum: blend of revenue growth and earnings growth
            earn_growth = _safe_float(info.get("earningsGrowth"))
            growth_vals.append(_nanmean([rev_growth, earn_growth]))

            # Blindspot (institutional analysis)
            blindspot_vals.append(_safe_float(df.get("blindspot_score")))

            # Margin trajectory: average of gross and operating margin trends
            gm_trend = _safe_float(df.get("gross_margin_4q_trend"))
            margin_vals.append(_nanmean([gm_trend, om_trend]))

        return {
            "piotroski": pd.Series(piotroski_vals),
            "roic_spread": pd.Series(roic_spread_vals),
            "cash_flow_quality": pd.Series(cash_flow_vals),
            "balance_sheet": pd.Series(balance_sheet_vals),
            "dcf_upside": pd.Series(dcf_upside_vals),
            "income_health": pd.Series(income_health_vals),
            "growth_momentum": pd.Series(growth_vals),
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
