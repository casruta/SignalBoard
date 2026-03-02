"""Dynamic screening: DCF-only undervaluation ranking with Damodaran safety gates.

Stocks are filtered through 10 hard quality gates (Altman Z > 3, Piotroski >= 5,
ROIC > WACC, positive FCF, non-declining revenue, etc.) and then ranked solely
by DCF upside (margin of safety). Quality metrics serve as binary pass/fail
filters, not scored dimensions.

Safety filters follow Damodaran's framework: never buy at fair value — require
compensation for estimation error in the DCF.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DynamicScreener:
    """DCF-focused screener that ranks stocks by intrinsic value upside.

    Applies hard safety gates, then ranks survivors by DCF margin of safety.
    """

    # DCF upside bounds: cap extreme values that are almost certainly noise
    DCF_UPSIDE_CAP = 2.0    # +200%
    DCF_UPSIDE_FLOOR = -0.5  # -50%

    def __init__(self, top_n: int = 100):
        self.top_n = top_n

    # ── Public API ────────────────────────────────────────────────────

    def screen(
        self,
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
        config: dict | None = None,
    ) -> list[str]:
        """Screen and rank stocks, return top N tickers.

        Steps:
        1. Apply hard safety filters (can't score your way past these)
        2. Rank survivors by DCF upside (margin of safety)
        3. Return top N
        """
        df = self.compute_dcf_rankings(
            deep_fundamentals, dcf_results, info_map, config=config,
        )
        if df.empty:
            logger.warning("No stocks survived screening.")
            return []

        safe = df[df["passes_safety"]].copy()
        if safe.empty:
            logger.warning("All stocks failed safety filters.")
            return []

        safe = safe.sort_values("dcf_upside_pct", ascending=False).head(self.top_n)
        return safe["ticker"].tolist()

    def compute_dcf_rankings(
        self,
        deep_fundamentals: dict[str, dict],
        dcf_results: dict[str, dict],
        info_map: dict[str, dict],
        config: dict | None = None,
    ) -> pd.DataFrame:
        """Compute DCF-based rankings for all stocks.

        Returns DataFrame with columns:
        - ticker, dcf_upside_pct, margin_of_safety, intrinsic_value,
          current_price, bear_iv, base_iv, bull_iv, wacc, fcf_yield,
          roic_vs_wacc_spread, piotroski_f_score, altman_z_score,
          passes_safety, rank, calculation_details
        """
        tickers = sorted(
            set(deep_fundamentals) | set(dcf_results) | set(info_map)
        )
        if not tickers:
            return pd.DataFrame()

        rows = []
        for ticker in tickers:
            dcf = dcf_results.get(ticker, {})
            deep = deep_fundamentals.get(ticker, {})
            info = info_map.get(ticker, {})

            raw_upside = _safe_float(dcf.get("dcf_upside_pct"))
            capped_upside = raw_upside
            if not np.isnan(raw_upside):
                capped_upside = float(np.clip(
                    raw_upside, self.DCF_UPSIDE_FLOOR, self.DCF_UPSIDE_CAP
                ))

            rows.append({
                "ticker": ticker,
                "dcf_upside_pct": capped_upside,
                "margin_of_safety": _safe_float(dcf.get("margin_of_safety")),
                "intrinsic_value": _safe_float(dcf.get("intrinsic_value_per_share")),
                "current_price": _safe_float(dcf.get("current_price", info.get("currentPrice"))),
                "bear_iv": _safe_float(dcf.get("bear_iv")),
                "base_iv": _safe_float(dcf.get("base_iv")),
                "bull_iv": _safe_float(dcf.get("bull_iv")),
                "wacc": _safe_float(dcf.get("wacc")),
                "fcf_yield": _safe_float(dcf.get("fcf_yield")),
                "roic_vs_wacc_spread": _safe_float(dcf.get("roic_vs_wacc_spread")),
                "piotroski_f_score": _safe_float(deep.get("piotroski_f_score")),
                "altman_z_score": _safe_float(deep.get("altman_z_score")),
                "passes_safety": self._apply_safety_filters(
                    info, deep, dcf=dcf, config=config,
                ),
            })

        scored = pd.DataFrame(rows)

        # Rank by DCF upside descending (NaN gets worst rank)
        scored["rank"] = (
            scored["dcf_upside_pct"]
            .rank(ascending=False, method="min", na_option="bottom")
            .astype(int)
        )

        # Build per-ticker calculation details for the report layer
        scored["calculation_details"] = [
            self._build_calculation_details(
                ticker=rows[i]["ticker"],
                deep_fund=deep_fundamentals.get(rows[i]["ticker"], {}),
                dcf=dcf_results.get(rows[i]["ticker"], {}),
                info=info_map.get(rows[i]["ticker"], {}),
                config=config,
            )
            for i in range(len(rows))
        ]

        scored = scored.sort_values(
            "dcf_upside_pct", ascending=False, na_position="last"
        ).reset_index(drop=True)
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
                current_ratio = _safe_float(deep_fund.get("current_ratio"))
                if not np.isnan(current_ratio) and current_ratio < 1.0:
                    return False

        # 9. Trading volume — enforce minimum daily volume
        min_volume = cfg_s.get("min_avg_daily_volume", 100_000)
        avg_volume = _safe_float(info.get("averageVolume"))
        if not np.isnan(avg_volume) and avg_volume < min_volume:
            return False

        # 10. Margin of safety — require discount to intrinsic value
        min_mos = cfg_s.get("min_margin_of_safety", 0.15)
        dcf_upside = _safe_float(dcf.get("dcf_upside_pct"))
        if not np.isnan(dcf_upside) and dcf_upside < min_mos:
            return False

        return True

    # ── Calculation Detail Builders ──────────────────────────────────

    def _build_calculation_details(
        self,
        ticker: str,
        deep_fund: dict,
        dcf: dict,
        info: dict,
        config: dict | None,
    ) -> dict:
        """Build a detailed calculation breakdown dict for one stock.

        Returns a dict with DCF valuation inputs and safety gate results.
        """
        dcf_details = {
            "dcf_upside_pct": _fmt(dcf.get("dcf_upside_pct")),
            "margin_of_safety": _fmt(dcf.get("margin_of_safety")),
            "intrinsic_value_per_share": _fmt(dcf.get("intrinsic_value_per_share")),
            "current_price": _fmt(dcf.get("current_price", info.get("currentPrice"))),
            "wacc": _fmt(dcf.get("wacc")),
            "cost_of_equity": _fmt(dcf.get("cost_of_equity")),
            "cost_of_debt": _fmt(dcf.get("cost_of_debt")),
            "beta": _fmt(dcf.get("beta")),
            "terminal_growth": _fmt(dcf.get("terminal_growth")),
            "fcf_yield": _fmt(dcf.get("fcf_yield")),
            "ev_to_fcf": _fmt(dcf.get("ev_to_fcf")),
            "roic": _fmt(dcf.get("roic")),
            "roic_vs_wacc_spread": _fmt(dcf.get("roic_vs_wacc_spread")),
            "bear_iv": _fmt(dcf.get("bear_iv")),
            "base_iv": _fmt(dcf.get("base_iv")),
            "bull_iv": _fmt(dcf.get("bull_iv")),
            "tv_gordon": _fmt(dcf.get("tv_gordon")),
            "tv_exit_multiple": _fmt(dcf.get("tv_exit_multiple")),
            "tv_divergence_pct": _fmt(dcf.get("tv_divergence_pct")),
            "distress_adjusted_iv": _fmt(dcf.get("distress_adjusted_iv")),
            "implied_growth_rate": _fmt(dcf.get("implied_growth_rate")),
        }

        quality_context = {
            "piotroski_f_score": _fmt(deep_fund.get("piotroski_f_score")),
            "altman_z_score": _fmt(deep_fund.get("altman_z_score")),
            "interest_coverage": _fmt(deep_fund.get("interest_coverage")),
            "accruals_ratio": _fmt(deep_fund.get("accruals_ratio")),
            "fcf_to_net_income": _fmt(deep_fund.get("fcf_to_net_income")),
        }

        safety_gates = self._evaluate_safety_gates(info, deep_fund, dcf, config)

        return {
            "dcf": dcf_details,
            "quality_context": quality_context,
            "safety_gates": safety_gates,
        }

    def _evaluate_safety_gates(
        self, info: dict, deep_fund: dict, dcf: dict | None = None,
        config: dict | None = None,
    ) -> list[dict]:
        """Evaluate each safety gate individually and return pass/fail details."""
        cfg_s = config.get("screening", {}) if config else {}
        dcf = dcf or {}
        gates: list[dict] = []

        # 1. Market cap
        mc = _safe_float(info.get("marketCap"))
        min_cap = cfg_s.get("min_market_cap", 300_000_000)
        max_cap = cfg_s.get("max_market_cap", 10_000_000_000)
        gates.append({
            "gate": "Market Cap",
            "rule": f"${min_cap/1e6:.0f}M\u2013${max_cap/1e9:.0f}B",
            "value": f"${mc/1e6:.0f}M" if not np.isnan(mc) else "N/A",
            "pass": not (np.isnan(mc) or mc < min_cap or mc > max_cap),
        })

        # 2. Altman Z
        min_z = cfg_s.get("min_altman_z", 3.0)
        az = _safe_float(deep_fund.get("altman_z_score"))
        gates.append({
            "gate": "Altman Z-Score",
            "rule": f"> {min_z:.1f}",
            "value": f"{az:.2f}" if not np.isnan(az) else "N/A",
            "pass": not (np.isnan(az) or az <= min_z),
        })

        # 3. Quarters
        quarters = deep_fund.get("quarters_available", 0)
        min_q = cfg_s.get("min_quarters", 4)
        gates.append({
            "gate": "Data History",
            "rule": f"\u2265 {min_q} quarters",
            "value": str(quarters),
            "pass": isinstance(quarters, (int, float)) and quarters >= min_q,
        })

        # 4. Revenue
        min_rg = cfg_s.get("min_revenue_growth", 0.0)
        rg = _safe_float(info.get("revenueGrowth"))
        gates.append({
            "gate": "Revenue Growth",
            "rule": f"\u2265 {min_rg:.0%}",
            "value": f"{rg:.1%}" if not np.isnan(rg) else "N/A",
            "pass": np.isnan(rg) or rg >= min_rg,
        })

        # 5. Piotroski
        min_p = cfg_s.get("min_piotroski", 5)
        pf = _safe_float(deep_fund.get("piotroski_f_score"))
        gates.append({
            "gate": "Piotroski F-Score",
            "rule": f"\u2265 {min_p}/9",
            "value": f"{pf:.0f}" if not np.isnan(pf) else "N/A",
            "pass": not (np.isnan(pf) or pf < min_p),
        })

        # 6. FCF
        fy = _safe_float(dcf.get("fcf_yield"))
        gates.append({
            "gate": "Free Cash Flow",
            "rule": "FCF yield > 0",
            "value": f"{fy:.1%}" if not np.isnan(fy) else "N/A",
            "pass": np.isnan(fy) or fy > 0,
        })

        # 7. ROIC > WACC
        rs = _safe_float(dcf.get("roic_vs_wacc_spread"))
        gates.append({
            "gate": "ROIC vs WACC",
            "rule": "ROIC > WACC (spread > 0)",
            "value": f"{rs:+.1%}" if not np.isnan(rs) else "N/A",
            "pass": np.isnan(rs) or rs > 0,
        })

        # 8. Liquidity
        _INV = {"Industrials", "Materials", "Consumer Staples", "Energy"}
        sector = info.get("sector", "")
        if sector in _INV:
            cr = _safe_float(deep_fund.get("current_ratio"))
            gates.append({
                "gate": "Liquidity (Current Ratio)",
                "rule": "\u2265 1.0 (inventory sector)",
                "value": f"{cr:.2f}" if not np.isnan(cr) else "N/A",
                "pass": np.isnan(cr) or cr >= 1.0,
            })
        else:
            qr = _safe_float(deep_fund.get("quick_ratio"))
            if not np.isnan(qr):
                gates.append({
                    "gate": "Liquidity (Quick Ratio)",
                    "rule": "\u2265 0.8 (non-inventory sector)",
                    "value": f"{qr:.2f}",
                    "pass": qr >= 0.8,
                })
            else:
                cr = _safe_float(deep_fund.get("current_ratio"))
                gates.append({
                    "gate": "Liquidity (Current Ratio fallback)",
                    "rule": "\u2265 1.0",
                    "value": f"{cr:.2f}" if not np.isnan(cr) else "N/A",
                    "pass": np.isnan(cr) or cr >= 1.0,
                })

        # 9. Volume
        min_vol = cfg_s.get("min_avg_daily_volume", 100_000)
        av = _safe_float(info.get("averageVolume"))
        gates.append({
            "gate": "Avg Daily Volume",
            "rule": f"\u2265 {min_vol:,.0f}",
            "value": f"{av:,.0f}" if not np.isnan(av) else "N/A",
            "pass": np.isnan(av) or av >= min_vol,
        })

        # 10. Margin of safety
        min_mos = cfg_s.get("min_margin_of_safety", 0.15)
        du = _safe_float(dcf.get("dcf_upside_pct"))
        gates.append({
            "gate": "DCF Margin of Safety",
            "rule": f"\u2265 {min_mos:.0%}",
            "value": f"{du:.1%}" if not np.isnan(du) else "N/A",
            "pass": np.isnan(du) or du >= min_mos,
        })

        return gates


# ── Module-level helpers ──────────────────────────────────────────────


def _fmt(v) -> str | float | None:
    """Format a numeric value for JSON-safe display."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float:
    """Convert *value* to float, returning np.nan on failure."""
    if value is None:
        return np.nan
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ── Convenience function ──────────────────────────────────────────────


def rank_stocks_by_dcf(
    deep_fundamentals: dict[str, dict],
    dcf_results: dict[str, dict],
    info_map: dict[str, dict],
    top_n: int = 100,
) -> list[str]:
    """Convenience function: screen and return top N tickers by DCF upside."""
    screener = DynamicScreener(top_n=top_n)
    return screener.screen(deep_fundamentals, dcf_results, info_map)
