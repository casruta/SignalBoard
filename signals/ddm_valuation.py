"""Dividend Discount Model (Gordon Growth Model) intrinsic value computation.

Provides single-stage and two-stage DDM valuations for dividend-paying stocks,
along with dividend stability metrics and implied dividend growth rates.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Reuse the same sector terminal growth mapping as dcf_valuation
SECTOR_TERMINAL_GROWTH = {
    "Technology": 0.035, "Communication Services": 0.030,
    "Healthcare": 0.030, "Consumer Discretionary": 0.025,
    "Industrials": 0.025, "Financial Services": 0.025, "Financials": 0.025,
    "Consumer Staples": 0.020, "Real Estate": 0.020,
    "Energy": 0.015, "Materials": 0.020, "Utilities": 0.015,
}


def compute_ddm_valuation(
    statements: dict,
    info: dict,
    risk_free_rate: float,
    equity_risk_premium: float = 0.06,
) -> dict:
    """Compute DDM intrinsic value and related dividend metrics.

    Parameters
    ----------
    statements : dict
        Financial statements dict with keys ``quarterly_cashflow``,
        ``annual_cashflow``, ``quarterly_income``, etc.
    info : dict
        yfinance ``.info`` dict for the ticker.
    risk_free_rate : float
        10Y Treasury rate (e.g. 0.04 for 4 %).
    equity_risk_premium : float
        Equity risk premium used in CAPM (default 6 %).

    Returns
    -------
    dict
        Valuation metrics. Contains ``ddm_applicable: False`` when the stock
        does not pay a meaningful dividend.
    """
    empty_result = _empty_result()

    if info is None:
        info = {}

    # ------------------------------------------------------------------
    # 1. Get D0 — trailing annual dividend per share
    # ------------------------------------------------------------------
    d0 = _safe_float(info.get("trailingAnnualDividendRate"))
    if np.isnan(d0) or d0 <= 0:
        logger.info("No trailing dividend rate; DDM not applicable")
        return empty_result

    # ------------------------------------------------------------------
    # 2. Required return (r) via CAPM + illiquidity premium
    # ------------------------------------------------------------------
    r = _compute_required_return(info, risk_free_rate, equity_risk_premium)

    # ------------------------------------------------------------------
    # 3. Dividend growth rate (g)
    # ------------------------------------------------------------------
    roe = _safe_float(info.get("returnOnEquity"))
    payout_ratio = _safe_float(info.get("payoutRatio"))

    # Sustainable growth = ROE * (1 - payout)
    sustainable_g = np.nan
    if not np.isnan(roe) and not np.isnan(payout_ratio):
        retention = max(1.0 - payout_ratio, 0.0)
        sustainable_g = roe * retention

    # 3-year dividend CAGR from cashflow statements
    dividend_cagr_3yr = _dividend_cagr(statements, years=3)

    # Sector terminal growth
    sector = info.get("sector", "") if info else ""
    sector_terminal_g = SECTOR_TERMINAL_GROWTH.get(sector, 0.02)

    # Final g: most conservative of the available estimates
    g = _select_growth_rate(sustainable_g, dividend_cagr_3yr, sector_terminal_g)

    # Cap g so that r - g > 0 (model convergence)
    g = min(g, r - 0.01)

    if g < 0:
        logger.warning("Computed dividend growth rate is negative (%.4f); clamping to 0", g)
        g = 0.0

    d1 = d0 * (1 + g)

    # ------------------------------------------------------------------
    # 4 & 5. DDM valuation (single-stage or two-stage)
    # ------------------------------------------------------------------
    two_stage_used = g > 2 * sector_terminal_g

    if two_stage_used:
        intrinsic = _two_stage_ddm(d0, g, sector_terminal_g, r, stage1_years=5)
    else:
        intrinsic = _single_stage_ddm(d1, r, g)

    # ------------------------------------------------------------------
    # 6. Dividend stability (CV of last 8 quarterly dividends)
    # ------------------------------------------------------------------
    dividend_stability = _dividend_stability(statements)

    # ------------------------------------------------------------------
    # 7. Implied dividend growth
    # ------------------------------------------------------------------
    market_price = _safe_float(info.get("currentPrice"))
    implied_g = _implied_dividend_growth(d1, r, market_price)

    # ------------------------------------------------------------------
    # 8. Assemble result
    # ------------------------------------------------------------------
    upside_pct = np.nan
    margin_of_safety = np.nan
    if not np.isnan(market_price) and market_price > 0 and not np.isnan(intrinsic):
        upside_pct = (intrinsic - market_price) / market_price
        if intrinsic != 0:
            margin_of_safety = (intrinsic - market_price) / intrinsic

    return {
        "ddm_applicable": True,
        "ddm_intrinsic_value": intrinsic,
        "ddm_upside_pct": upside_pct,
        "ddm_margin_of_safety": margin_of_safety,
        "required_return": r,
        "sustainable_growth_rate": sustainable_g,
        "dividend_growth_rate_3yr": dividend_cagr_3yr,
        "implied_dividend_growth": implied_g,
        "dividend_stability": dividend_stability,
        "d0_per_share": d0,
        "d1_per_share": d1,
        "growth_rate_used": g,
        "two_stage_used": two_stage_used,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    """Return a result dict indicating DDM is not applicable."""
    return {
        "ddm_applicable": False,
        "ddm_intrinsic_value": np.nan,
        "ddm_upside_pct": np.nan,
        "ddm_margin_of_safety": np.nan,
        "required_return": np.nan,
        "sustainable_growth_rate": np.nan,
        "dividend_growth_rate_3yr": np.nan,
        "implied_dividend_growth": np.nan,
        "dividend_stability": np.nan,
        "d0_per_share": np.nan,
        "d1_per_share": np.nan,
        "growth_rate_used": np.nan,
        "two_stage_used": False,
    }


def _safe_float(val, default=np.nan) -> float:
    """Convert *val* to float, returning *default* on None / NaN / failure."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def _compute_required_return(
    info: dict,
    risk_free_rate: float,
    equity_risk_premium: float,
) -> float:
    """CAPM cost of equity with tiered illiquidity premiums, clamped to [5%, 25%]."""
    beta = _safe_float(info.get("beta"), default=1.0)
    r = risk_free_rate + beta * equity_risk_premium

    market_cap = _safe_float(info.get("marketCap"))
    if not np.isnan(market_cap):
        if market_cap < 500_000_000:
            r += 0.025   # micro-cap
        elif market_cap < 1_000_000_000:
            r += 0.020   # small-cap
        elif market_cap < 3_000_000_000:
            r += 0.010   # lower-mid
        elif market_cap < 10_000_000_000:
            r += 0.005   # mid-cap

    return float(np.clip(r, 0.05, 0.25))


def _select_growth_rate(
    sustainable_g: float,
    dividend_cagr: float,
    sector_terminal_g: float,
) -> float:
    """Pick the most conservative growth estimate from available candidates.

    Final g = min(sustainable_g, dividend_cagr, sector_terminal_g * 1.5).
    Falls back to sector_terminal_g if no other estimate is available.
    """
    candidates = []
    if not np.isnan(sustainable_g):
        candidates.append(sustainable_g)
    if not np.isnan(dividend_cagr):
        candidates.append(dividend_cagr)
    candidates.append(sector_terminal_g * 1.5)

    g = min(candidates)
    logger.debug(
        "Growth candidates: sustainable=%.4f, cagr=%.4f, sector_cap=%.4f -> g=%.4f",
        sustainable_g if not np.isnan(sustainable_g) else -1,
        dividend_cagr if not np.isnan(dividend_cagr) else -1,
        sector_terminal_g * 1.5,
        g,
    )
    return g


def _single_stage_ddm(d1: float, r: float, g: float) -> float:
    """Gordon Growth Model: P = D1 / (r - g)."""
    denominator = r - g
    if denominator <= 0:
        logger.warning("r - g <= 0 (%.4f); single-stage DDM undefined", denominator)
        return np.nan
    return d1 / denominator


def _two_stage_ddm(
    d0: float,
    g_stage1: float,
    g_terminal: float,
    r: float,
    stage1_years: int = 5,
) -> float:
    """Two-stage DDM: high growth for *stage1_years*, then perpetuity at *g_terminal*.

    PV = sum( D0*(1+g1)^t / (1+r)^t, t=1..N )
       + D0*(1+g1)^N * (1+g_terminal) / ((r - g_terminal) * (1+r)^N)
    """
    if r <= g_terminal:
        logger.warning(
            "r (%.4f) <= g_terminal (%.4f); two-stage DDM undefined", r, g_terminal
        )
        return np.nan

    pv_stage1 = 0.0
    for t in range(1, stage1_years + 1):
        dividend_t = d0 * (1 + g_stage1) ** t
        pv_stage1 += dividend_t / (1 + r) ** t

    # Terminal value at the end of stage 1
    d_n = d0 * (1 + g_stage1) ** stage1_years
    terminal_value = d_n * (1 + g_terminal) / (r - g_terminal)
    pv_terminal = terminal_value / (1 + r) ** stage1_years

    return pv_stage1 + pv_terminal


def _dividend_cagr(statements: dict, years: int = 3) -> float:
    """Compute CAGR of dividends paid over the last *years* annual periods.

    Uses the ``Cash Dividends Paid`` line item from the annual cashflow
    statement. Dividends paid are typically negative; we use absolute values.
    """
    annual_cf = statements.get("annual_cashflow")
    if annual_cf is None or annual_cf.empty:
        logger.info("Annual cashflow unavailable for dividend CAGR")
        return np.nan

    line_item = _find_dividend_line(annual_cf)
    if line_item is None:
        logger.info("Dividend line item not found in annual cashflow")
        return np.nan

    row = annual_cf.loc[line_item].dropna()
    if len(row) < 2:
        return np.nan

    n = min(years, len(row) - 1)
    recent = abs(float(row.iloc[0]))
    older = abs(float(row.iloc[n]))

    if older <= 0 or recent <= 0:
        return np.nan

    cagr = (recent / older) ** (1.0 / n) - 1.0
    return cagr


def _dividend_stability(statements: dict) -> float:
    """Coefficient of variation of the last 8 quarterly dividends paid.

    A lower CV indicates more stable dividend payments.
    Returns np.nan when insufficient data is available.
    """
    quarterly_cf = statements.get("quarterly_cashflow")
    if quarterly_cf is None or quarterly_cf.empty:
        return np.nan

    line_item = _find_dividend_line(quarterly_cf)
    if line_item is None:
        return np.nan

    row = quarterly_cf.loc[line_item].dropna()
    values = row.iloc[:8]  # most recent 8 quarters

    if len(values) < 4:
        logger.info("Fewer than 4 quarterly dividends available for stability calc")
        return np.nan

    amounts = np.abs(values.values.astype(float))
    mean_val = np.mean(amounts)
    if mean_val == 0:
        return np.nan

    cv = float(np.std(amounts, ddof=1) / mean_val)
    return cv


def _implied_dividend_growth(d1: float, r: float, market_price: float) -> float:
    """Solve for g in P = D1 / (r - g)  =>  g = r - D1 / P."""
    if np.isnan(market_price) or market_price <= 0:
        return np.nan
    if np.isnan(d1) or d1 <= 0:
        return np.nan

    g = r - d1 / market_price
    return g


def _find_dividend_line(cf_df: pd.DataFrame) -> str | None:
    """Locate the dividend line item in a cashflow DataFrame.

    yfinance labels vary across versions; try common names.
    """
    candidates = [
        "Cash Dividends Paid",
        "Common Stock Dividend Paid",
        "Payment Of Dividends",
        "Dividends Paid",
    ]
    for name in candidates:
        if name in cf_df.index:
            return name
    return None
