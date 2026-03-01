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
    # Revised: added price_momentum (most robust cross-sectional predictor,
    # Jegadeesh-Titman 1993); reduced income_health and growth_momentum to
    # compensate and reduce multicollinearity from shared revenue/growth inputs.
    DEFAULT_WEIGHTS = {
        "piotroski": 0.14,
        "cash_flow_quality": 0.16,    # critical for small caps
        "roic_spread": 0.12,          # direct value creation measure
        "balance_sheet": 0.10,
        "dcf_upside": 0.07,           # valuation upside (reduced — also a gate)
        "income_health": 0.11,        # earnings persistence, consistency, peer-relative
        "growth_momentum": 0.05,      # revenue + earnings growth
        "margin_trajectory": 0.05,
        "blindspot": 0.05,            # coverage gap / neglect premium
        "price_momentum": 0.10,       # 12-1 month price momentum
        "low_volatility": 0.05,       # 60-day realized vol (inverted)
    }

    # Regime adjustments: multiply default weights by these factors
    REGIME_ADJUSTMENTS = {
        "risk_off": {
            "balance_sheet": 1.5,
            "cash_flow_quality": 1.3,
            "income_health": 1.3,
            "low_volatility": 1.4,
            "dcf_upside": 0.7,
            "growth_momentum": 0.7,
            "margin_trajectory": 0.8,
            "price_momentum": 0.6,      # momentum crashes in risk-off
        },
        "risk_on": {
            "dcf_upside": 1.4,
            "growth_momentum": 1.5,
            "income_health": 1.1,
            "margin_trajectory": 1.3,
            "price_momentum": 1.3,
            "balance_sheet": 0.8,
            "blindspot": 0.8,
            "low_volatility": 0.7,
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
        price_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[str]:
        """Screen and rank stocks, return top N tickers.

        Steps:
        1. Apply hard safety filters (can't score your way past these)
        2. Compute component scores per stock (0-1 range, percentile-ranked)
        3. Apply regime-adjusted weights
        4. Sort by composite score, return top N
        """
        df = self.compute_composite_scores(
            deep_fundamentals, dcf_results, info_map, macro_regime,
            config=config, price_data=price_data,
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

    # Minimum populated dimensions to be scored (hard exclusion below this)
    MIN_POPULATED_DIMS = 6

    def compute_composite_scores(
        self,
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
        macro_regime: str = "neutral",
        config: dict | None = None,
        price_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Compute composite quality scores for all stocks.

        Returns DataFrame with columns:
        - ticker, composite_score, rank
        - piotroski_score, roic_spread_score, cash_flow_score,
          balance_sheet_score, dcf_score, growth_score, blindspot_score,
          margin_score, momentum_score, low_vol_score
        - passes_safety (bool)

        Parameters
        ----------
        price_data : optional dict mapping ticker -> DataFrame with 'Close'
            column and DatetimeIndex.  Used for price momentum and
            realized volatility dimensions.
        """
        tickers = sorted(
            set(deep_fundamentals) | set(dcf_results) | set(info_map)
        )
        if not tickers:
            return pd.DataFrame()

        # Extract raw values for each component
        raw = self._extract_raw_values(
            tickers, deep_fundamentals, dcf_results, info_map,
            price_data=price_data,
        )

        # Winsorize raw inputs at 1st/99th percentile before ranking
        # to stabilise rankings and reduce turnover from outlier-driven swaps
        for key in raw:
            raw[key] = self._winsorize(raw[key])

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
        scored["momentum_score"] = self._score_component(raw["price_momentum"])
        scored["low_vol_score"] = self._score_component(raw["low_volatility"])

        # Safety filter — now includes DCF data for ROIC/FCF gates
        scored["passes_safety"] = [
            self._apply_safety_filters(
                info_map.get(t, {}), deep_fundamentals.get(t, {}),
                dcf=dcf_results.get(t, {}), config=config,
            )
            for t in tickers
        ]

        # Data completeness: hard-exclude stocks with fewer than MIN_POPULATED_DIMS.
        # Stocks with sufficient data get no penalty (the old sqrt approach was
        # ad-hoc and biased toward large-cap well-covered stocks).
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
            "price_momentum": "momentum_score",
            "low_volatility": "low_vol_score",
        }
        raw_composite = sum(
            weights.get(component, 0.0) * scored[col]
            for component, col in component_columns.items()
        )
        # Apply data completeness: stocks below threshold get score zeroed out
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

        # 8. Liquidity gate — use quick ratio for non-manufacturing sectors
        # (current ratio includes inventory, which is irrelevant for SaaS/services)
        _INVENTORY_SECTORS = {"Industrials", "Materials", "Consumer Staples", "Energy"}
        sector = info.get("sector", "")
        if sector in _INVENTORY_SECTORS:
            current_ratio = _safe_float(deep_fund.get("current_ratio"))
            if not np.isnan(current_ratio) and current_ratio < 1.0:
                return False
        else:
            quick_ratio = _safe_float(deep_fund.get("quick_ratio"))
            if not np.isnan(quick_ratio) and quick_ratio < 0.8:
                return False
            elif np.isnan(quick_ratio):
                # Fallback to current ratio when quick ratio unavailable
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

    @staticmethod
    def _winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Winsorize at *lower*/*upper* percentiles to reduce outlier impact."""
        non_null = values.dropna()
        if len(non_null) < 5:
            return values
        lo = non_null.quantile(lower)
        hi = non_null.quantile(upper)
        return values.clip(lower=lo, upper=hi)

    def _score_component(self, values: pd.Series) -> pd.Series:
        """Convert raw values to 0-1 percentile scores across the universe.

        Higher percentile = better.  NaN values get 0.5 (neutral) rather
        than a penalty — data-sparse stocks are handled by the hard
        completeness exclusion at MIN_POPULATED_DIMS instead.
        """
        filled = values.copy()
        mask = filled.isna()
        non_null = filled[~mask]

        if non_null.empty:
            return pd.Series(0.5, index=values.index)

        ranked = non_null.rank(pct=True, method="average")
        filled.loc[~mask] = ranked
        filled.loc[mask] = 0.5  # neutral for missing — hard exclusion handles sparse data
        return filled

    def _compute_data_completeness(
        self, raw: dict[str, pd.Series], n_stocks: int,
    ) -> pd.Series:
        """Hard exclusion for stocks with fewer than MIN_POPULATED_DIMS.

        Stocks with sufficient data (>= MIN_POPULATED_DIMS) get 1.0 (no
        penalty).  Stocks below the threshold get 0.0 (excluded from ranking).
        This replaces the old sqrt scaling which was ad-hoc and let stocks
        with 3/9 dims still rank competitively.
        """
        n_components = len(raw)
        non_nan_count = pd.Series(0, index=range(n_stocks), dtype=float)
        for component_values in raw.values():
            non_nan_count += (~component_values.isna()).astype(float)
        min_dims = self.MIN_POPULATED_DIMS
        # Binary: 1.0 if enough data, 0.0 if not
        return (non_nan_count >= min_dims).astype(float)

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

    @staticmethod
    def compute_ic_weights(
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        base_weights: dict[str, float],
        shrinkage: float = 0.5,
    ) -> dict[str, float]:
        """IC-weighted factor allocation shrunk toward base weights.

        Parameters
        ----------
        factor_scores : DataFrame with columns matching base_weights keys,
            rows are stocks.
        forward_returns : Series of 1-period forward returns, aligned to rows.
        base_weights : dict of factor_name -> weight (e.g. DEFAULT_WEIGHTS).
        shrinkage : blend toward base_weights (0 = pure IC, 1 = pure base).

        Returns
        -------
        dict of factor_name -> weight, summing to 1.0.
        """
        ic_raw = {}
        for factor in base_weights:
            if factor not in factor_scores.columns:
                ic_raw[factor] = 0.0
                continue
            mask = factor_scores[factor].notna() & forward_returns.notna()
            if mask.sum() < 20:
                ic_raw[factor] = 0.0
                continue
            ic_raw[factor] = float(
                factor_scores.loc[mask, factor].corr(
                    forward_returns[mask], method="spearman"
                )
            )

        # Clamp negative ICs to zero (don't short-weight factors)
        ic_pos = {k: max(0.0, v) for k, v in ic_raw.items()}
        ic_sum = sum(ic_pos.values())
        if ic_sum <= 0:
            return dict(base_weights)

        ic_weights = {k: v / ic_sum for k, v in ic_pos.items()}

        # Shrink toward base weights
        blended = {}
        for k in base_weights:
            blended[k] = shrinkage * base_weights.get(k, 0) + (1 - shrinkage) * ic_weights.get(k, 0)

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    # ── Raw Value Extraction ──────────────────────────────────────────

    # DCF upside bounds: cap extreme values that are almost certainly noise
    DCF_UPSIDE_CAP = 2.0    # +200%
    DCF_UPSIDE_FLOOR = -0.5  # -50%

    def _extract_raw_values(
        self,
        tickers: list[str],
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
        price_data: dict[str, pd.DataFrame] | None = None,
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
        momentum_vals = []
        low_vol_vals = []

        price_data = price_data or {}

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

            # DCF upside — capped at [FLOOR, CAP] to reduce noise from
            # extreme valuations that are almost certainly input errors
            raw_dcf = _safe_float(dcf.get("dcf_upside_pct"))
            if not np.isnan(raw_dcf):
                raw_dcf = float(np.clip(raw_dcf, self.DCF_UPSIDE_FLOOR, self.DCF_UPSIDE_CAP))
            dcf_upside_vals.append(raw_dcf)

            # Income statement health: consistency, earnings persistence,
            # and peer-relative growth.  Revenue growth is handled by
            # Growth Momentum; operating margin trend by Margin Trajectory.
            # De-duplicated to reduce multicollinearity.
            rev_growth = _safe_float(info.get("revenueGrowth"))
            rev_consistency = _safe_float(df.get("revenue_growth_consistency"))
            inv_consistency = -rev_consistency if not np.isnan(rev_consistency) else np.nan
            earn_persist = _safe_float(df.get("earnings_persistence"))
            rev_industry_pctl = _safe_float(df.get("revenue_growth_industry_pctl"))
            income_health_vals.append(
                _nanmean([inv_consistency, earn_persist, rev_industry_pctl])
            )

            # Growth momentum: blend of revenue growth and earnings growth
            earn_growth = _safe_float(info.get("earningsGrowth"))
            growth_vals.append(_nanmean([rev_growth, earn_growth]))

            # Blindspot (institutional analysis)
            blindspot_vals.append(_safe_float(df.get("blindspot_score")))

            # Margin trajectory: average of gross and operating margin trends
            gm_trend = _safe_float(df.get("gross_margin_4q_trend"))
            om_trend = _safe_float(df.get("operating_margin_4q_trend"))
            margin_vals.append(_nanmean([gm_trend, om_trend]))

            # Price momentum: 12-1 month return (skip most recent month
            # to avoid short-term reversal, per Jegadeesh-Titman 1993)
            momentum_vals.append(
                _compute_price_momentum(price_data.get(ticker))
            )

            # Low volatility: inverted 60-day realized vol (lower vol = higher score)
            vol = _compute_realized_vol(price_data.get(ticker))
            low_vol_vals.append(-vol if not np.isnan(vol) else np.nan)

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
            "price_momentum": pd.Series(momentum_vals),
            "low_volatility": pd.Series(low_vol_vals),
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


def _compute_price_momentum(price_df: pd.DataFrame | None) -> float:
    """12-1 month price momentum (skip most recent 21 trading days).

    Returns the cumulative return from T-252 to T-21 trading days.
    The 1-month skip avoids short-term reversal (Jegadeesh-Titman 1993).
    """
    if price_df is None or price_df.empty:
        return np.nan
    close = price_df.get("Close")
    if close is None:
        close = price_df.get("Adj Close")
    if close is None or len(close) < 252:
        return np.nan
    try:
        recent = float(close.iloc[-21])   # 1 month ago
        older = float(close.iloc[-252])   # 12 months ago
        if older <= 0 or np.isnan(older) or np.isnan(recent):
            return np.nan
        return (recent / older) - 1.0
    except (IndexError, TypeError, ValueError):
        return np.nan


def _compute_realized_vol(price_df: pd.DataFrame | None, window: int = 60) -> float:
    """Annualised realized volatility from daily log returns over *window* days."""
    if price_df is None or price_df.empty:
        return np.nan
    close = price_df.get("Close")
    if close is None:
        close = price_df.get("Adj Close")
    if close is None or len(close) < window + 1:
        return np.nan
    try:
        log_ret = np.log(close.iloc[-window:] / close.iloc[-window:].shift(1)).dropna()
        if len(log_ret) < window - 5:
            return np.nan
        return float(log_ret.std() * np.sqrt(252))
    except (TypeError, ValueError):
        return np.nan


# ── Convenience function ──────────────────────────────────────────────


def rank_stocks_by_quality(
    deep_fundamentals: dict[str, dict],
    dcf_results: dict[str, dict],
    info_map: dict[str, dict],
    macro_regime: str = "neutral",
    top_n: int = 100,
    price_data: dict[str, pd.DataFrame] | None = None,
) -> list[str]:
    """Convenience function: screen and return top N tickers."""
    screener = DynamicScreener(top_n=top_n)
    return screener.screen(
        deep_fundamentals, dcf_results, info_map, macro_regime,
        price_data=price_data,
    )
