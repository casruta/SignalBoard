"""Deep fundamental analysis: ~45 features across 8 categories.

Designed to mirror how an advanced financial analyst reviews quarterly
earnings and financial statements.  Every function returns a dict of
feature_name -> float, defaulting to ``np.nan`` when data is missing.

yfinance financial-statement DataFrames have line items as rows (index)
and Timestamp columns sorted **descending** (most recent first).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────


def _safe_line_item(df: pd.DataFrame, item: str, col_idx: int = 0) -> float:
    """Safely extract a line item from a financial statement DataFrame.

    Parameters
    ----------
    df : DataFrame with line items as index, dates as columns.
    item : Row label to look up.
    col_idx : Column position (0 = most recent).

    Returns np.nan if item not found or column doesn't exist.
    """
    if df is None or df.empty:
        return np.nan
    if item not in df.index:
        return np.nan
    if col_idx >= df.shape[1]:
        return np.nan
    val = df.iloc[df.index.get_loc(item), col_idx]
    try:
        result = float(val)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _trailing_sum(df: pd.DataFrame, item: str, n_periods: int = 4) -> float:
    """Sum a line item across the last *n_periods* columns (most-recent first)."""
    if df is None or df.empty or item not in df.index:
        return np.nan
    n = min(n_periods, df.shape[1])
    values = [_safe_line_item(df, item, i) for i in range(n)]
    if all(np.isnan(v) for v in values):
        return np.nan
    return float(np.nansum(values))


def _yoy_change(df: pd.DataFrame, item: str) -> float:
    """Year-over-year change: (current - prior) / |prior|.

    For quarterly data compare col 0 to col 4 (same quarter last year).
    For annual data compare col 0 to col 1.
    """
    if df is None or df.empty or item not in df.index:
        return np.nan
    gap = 4 if df.shape[1] >= 5 else 1
    if df.shape[1] <= gap:
        return np.nan
    current = _safe_line_item(df, item, 0)
    prior = _safe_line_item(df, item, gap)
    if np.isnan(current) or np.isnan(prior) or prior == 0.0:
        return np.nan
    return (current - prior) / abs(prior)


def _compute_trend(values: list | np.ndarray) -> float:
    """OLS slope of *values* normalised by their mean.

    Returns np.nan if fewer than 3 non-NaN data points.
    """
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 3:
        return np.nan
    y = arr[mask]
    x = np.arange(len(y), dtype=float)
    mean_y = np.mean(y)
    if mean_y == 0.0:
        return np.nan
    # OLS slope: cov(x,y) / var(x)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / abs(mean_y))


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or np.nan when unsafe."""
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0.0:
        return np.nan
    result = numerator / denominator
    return result if np.isfinite(result) else np.nan


def _series_over_quarters(
    df: pd.DataFrame, item: str, n: int = 4
) -> list[float]:
    """Extract up to *n* most-recent quarterly values for *item*."""
    if df is None or df.empty or item not in df.index:
        return []
    count = min(n, df.shape[1])
    return [_safe_line_item(df, item, i) for i in range(count)]


def _ratio_series(
    df_num: pd.DataFrame,
    item_num: str,
    df_den: pd.DataFrame,
    item_den: str,
    n: int = 4,
) -> list[float]:
    """Build a list of ratio values across *n* quarters."""
    results: list[float] = []
    if df_num is None or df_den is None:
        return results
    count = min(n, df_num.shape[1], df_den.shape[1])
    for i in range(count):
        num = _safe_line_item(df_num, item_num, i)
        den = _safe_line_item(df_den, item_den, i)
        results.append(_safe_divide(num, den))
    return results


# ── Category 1: Profitability Trajectory ─────────────────────────────


def compute_profitability_trajectory(
    quarterly_income: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
    info: dict,
) -> dict:
    """7 features describing profitability trends over recent quarters."""
    feats: dict[str, float] = {}

    # --- ROE series (net_income / stockholders_equity) ---
    roe_vals = _ratio_series(
        quarterly_income, "Net Income",
        quarterly_balance_sheet, "Stockholders Equity",
    )
    feats["roe_4q_trend"] = _compute_trend(roe_vals)

    # --- ROIC latest & trend ---
    # ROIC = NOPAT / invested_capital
    # NOPAT ~ operating_income * (1 - effective_tax_rate)
    # invested_capital = total_assets - current_liabilities
    roic_vals: list[float] = []
    n = min(4, quarterly_income.shape[1] if not quarterly_income.empty else 0)
    for i in range(n):
        op_inc = _safe_line_item(quarterly_income, "Operating Income", i)
        tax = _safe_line_item(quarterly_income, "Tax Provision", i)
        rev = _safe_line_item(quarterly_income, "Total Revenue", i)
        ta = _safe_line_item(quarterly_balance_sheet, "Total Assets", i)
        cl = _safe_line_item(quarterly_balance_sheet, "Current Liabilities", i)

        if not np.isnan(op_inc) and not np.isnan(rev) and rev != 0:
            eff_tax = tax / op_inc if (not np.isnan(tax) and op_inc != 0) else 0.21
            eff_tax = max(0.0, min(eff_tax, 1.0))
        else:
            eff_tax = 0.21

        nopat = op_inc * (1 - eff_tax) if not np.isnan(op_inc) else np.nan
        invested = (ta - cl) if (not np.isnan(ta) and not np.isnan(cl)) else np.nan
        roic_vals.append(_safe_divide(nopat, invested))

    feats["roic_latest"] = roic_vals[0] if roic_vals else np.nan
    feats["roic_4q_trend"] = _compute_trend(roic_vals)

    # --- Margin trends ---
    gm_vals = _ratio_series(
        quarterly_income, "Gross Profit",
        quarterly_income, "Total Revenue",
    )
    feats["gross_margin_4q_trend"] = _compute_trend(gm_vals)

    om_vals = _ratio_series(
        quarterly_income, "Operating Income",
        quarterly_income, "Total Revenue",
    )
    feats["operating_margin_4q_trend"] = _compute_trend(om_vals)

    nm_vals = _ratio_series(
        quarterly_income, "Net Income",
        quarterly_income, "Total Revenue",
    )
    feats["net_margin_4q_trend"] = _compute_trend(nm_vals)

    # --- Revenue per employee ---
    rev_ttm = _trailing_sum(quarterly_income, "Total Revenue", 4)
    employees = info.get("fullTimeEmployees")
    if employees and employees > 0 and not np.isnan(rev_ttm):
        feats["revenue_per_employee"] = rev_ttm / employees
    else:
        feats["revenue_per_employee"] = np.nan

    return feats


# ── Category 2: Balance Sheet Fortress Score ─────────────────────────


def compute_balance_sheet_health(
    info: dict,
    quarterly_balance_sheet: pd.DataFrame,
    quarterly_income: pd.DataFrame,
    quarterly_cashflow: pd.DataFrame,
) -> dict:
    """6 features assessing the strength of the balance sheet."""
    feats: dict[str, float] = {}
    bs = quarterly_balance_sheet
    inc = quarterly_income

    # Shorthand for most-recent quarter values
    ta = _safe_line_item(bs, "Total Assets", 0)
    tl = _safe_line_item(bs, "Total Liabilities Net Minority Interest", 0)
    ca = _safe_line_item(bs, "Current Assets", 0)
    cl = _safe_line_item(bs, "Current Liabilities", 0)
    cash = _safe_line_item(bs, "Cash And Cash Equivalents", 0)
    ltd = _safe_line_item(bs, "Long Term Debt", 0)
    re = _safe_line_item(bs, "Retained Earnings", 0)
    equity = _safe_line_item(bs, "Stockholders Equity", 0)

    revenue_ttm = _trailing_sum(inc, "Total Revenue", 4)
    ebit_ttm = _trailing_sum(inc, "Operating Income", 4)
    ebitda_ttm = _trailing_sum(inc, "EBITDA", 4)
    interest = _trailing_sum(inc, "Interest Expense", 4)
    fcf_ttm = _trailing_sum(quarterly_cashflow, "Free Cash Flow", 4)

    mkt_cap = info.get("marketCap", np.nan)
    if mkt_cap is None:
        mkt_cap = np.nan

    # Altman Z-Score
    wc = (ca - cl) if (not np.isnan(ca) and not np.isnan(cl)) else np.nan
    total_debt = ltd  # simplified
    mve = float(mkt_cap) if not np.isnan(mkt_cap) else np.nan

    z_parts = []
    z_parts.append(1.2 * _safe_divide(wc, ta))
    z_parts.append(1.4 * _safe_divide(re, ta))
    z_parts.append(3.3 * _safe_divide(ebit_ttm, ta))
    z_parts.append(0.6 * _safe_divide(mve, tl))
    z_parts.append(1.0 * _safe_divide(revenue_ttm, ta))

    if all(not np.isnan(p) for p in z_parts):
        feats["altman_z_score"] = sum(z_parts)
    else:
        feats["altman_z_score"] = np.nan

    # Interest coverage
    if not np.isnan(interest) and interest < 0:
        interest = abs(interest)
    feats["interest_coverage"] = _safe_divide(ebit_ttm, interest)

    # Current ratio
    feats["current_ratio"] = _safe_divide(ca, cl)

    # Debt to FCF
    feats["debt_to_fcf"] = _safe_divide(total_debt, fcf_ttm)

    # Net debt to EBITDA
    net_debt = (total_debt - cash) if (not np.isnan(total_debt) and not np.isnan(cash)) else np.nan
    feats["net_debt_to_ebitda"] = _safe_divide(net_debt, ebitda_ttm)

    # Cash as pct of assets
    feats["cash_as_pct_of_assets"] = _safe_divide(cash, ta)

    return feats


# ── Category 3: Cash Flow Quality ────────────────────────────────────


def compute_cash_flow_quality(
    quarterly_income: pd.DataFrame,
    quarterly_cashflow: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
    info: dict,
) -> dict:
    """7 features evaluating the quality and sustainability of cash flows."""
    feats: dict[str, float] = {}
    inc = quarterly_income
    cf = quarterly_cashflow
    bs = quarterly_balance_sheet

    ni_ttm = _trailing_sum(inc, "Net Income", 4)
    ocf_ttm = _trailing_sum(cf, "Operating Cash Flow", 4)
    fcf_ttm = _trailing_sum(cf, "Free Cash Flow", 4)
    rev_ttm = _trailing_sum(inc, "Total Revenue", 4)
    ta = _safe_line_item(bs, "Total Assets", 0)
    capex_ttm = _trailing_sum(cf, "Capital Expenditure", 4)

    # Accruals ratio (negative is good — cash exceeds earnings)
    accruals = (ni_ttm - ocf_ttm) if (not np.isnan(ni_ttm) and not np.isnan(ocf_ttm)) else np.nan
    feats["accruals_ratio"] = _safe_divide(accruals, ta)

    # FCF to net income
    feats["fcf_to_net_income"] = _safe_divide(fcf_ttm, ni_ttm)

    # OCF margin
    feats["ocf_margin"] = _safe_divide(ocf_ttm, rev_ttm)

    # OCF margin trend over 4Q
    ocf_margin_vals = _ratio_series(cf, "Operating Cash Flow", inc, "Total Revenue")
    feats["ocf_margin_4q_trend"] = _compute_trend(ocf_margin_vals)

    # Capex intensity
    capex_abs = abs(capex_ttm) if not np.isnan(capex_ttm) else np.nan
    feats["capex_intensity"] = _safe_divide(capex_abs, rev_ttm)

    # Cash conversion cycle
    rev_latest = _safe_line_item(inc, "Total Revenue", 0)
    cogs_latest = _safe_line_item(inc, "Cost Of Revenue", 0)
    if np.isnan(cogs_latest):
        # Estimate COGS = Revenue - Gross Profit
        gp = _safe_line_item(inc, "Gross Profit", 0)
        cogs_latest = (rev_latest - gp) if (not np.isnan(rev_latest) and not np.isnan(gp)) else np.nan

    ar = _safe_line_item(bs, "Accounts Receivable", 0)
    inv = _safe_line_item(bs, "Inventory", 0)
    ap = _safe_line_item(bs, "Accounts Payable", 0)

    # Days in quarter ~ 90
    days_q = 90.0
    days_recv = _safe_divide(ar * days_q, rev_latest)
    days_inv = _safe_divide(inv * days_q, cogs_latest)
    days_pay = _safe_divide(ap * days_q, cogs_latest)

    if all(not np.isnan(d) for d in [days_recv, days_inv, days_pay]):
        feats["cash_conversion_cycle"] = days_recv + days_inv - days_pay
    else:
        feats["cash_conversion_cycle"] = np.nan

    # FCF 3-year CAGR (use annual cashflow from info if available, else TTM approximation)
    feats["fcf_growth_3yr_cagr"] = _compute_fcf_cagr(cf)

    return feats


def _compute_fcf_cagr(quarterly_cashflow: pd.DataFrame) -> float:
    """Approximate 3-year CAGR of FCF using quarterly data.

    Sums FCF for the most recent 4Q vs. quarters 8-11 (two years prior)
    and quarters 12-15 if available, to get annual figures.
    """
    if quarterly_cashflow is None or quarterly_cashflow.empty:
        return np.nan
    if "Free Cash Flow" not in quarterly_cashflow.index:
        return np.nan

    n_cols = quarterly_cashflow.shape[1]
    if n_cols < 12:
        return np.nan

    def _annual_fcf(start: int) -> float:
        vals = [_safe_line_item(quarterly_cashflow, "Free Cash Flow", i)
                for i in range(start, min(start + 4, n_cols))]
        if any(np.isnan(v) for v in vals) or len(vals) < 4:
            return np.nan
        return sum(vals)

    recent = _annual_fcf(0)
    older = _annual_fcf(8)
    if np.isnan(recent) or np.isnan(older) or older <= 0 or recent <= 0:
        return np.nan
    years = 2.0  # gap between period 0-3 and 8-11 is ~2 years
    return float((recent / older) ** (1.0 / years) - 1.0)


# ── Category 4: Earnings Quality & Persistence ──────────────────────


def compute_earnings_quality(
    quarterly_income: pd.DataFrame,
    quarterly_cashflow: pd.DataFrame,
) -> dict:
    """5 features assessing earnings quality and persistence."""
    feats: dict[str, float] = {}
    inc = quarterly_income

    # EPS series
    eps_vals = _series_over_quarters(inc, "Basic EPS", 8)
    eps_arr = np.array(eps_vals, dtype=float)
    valid = eps_arr[np.isfinite(eps_arr)]

    # Earnings persistence: lag-1 autocorrelation of EPS
    if len(valid) >= 4:
        series = pd.Series(valid)
        ac = series.autocorr(lag=1)
        feats["earnings_persistence"] = float(ac) if np.isfinite(ac) else np.nan
    else:
        feats["earnings_persistence"] = np.nan

    # Earnings volatility: std(EPS) / |mean(EPS)|
    if len(valid) >= 3 and np.mean(np.abs(valid)) > 0:
        feats["earnings_volatility"] = float(np.std(valid) / abs(np.mean(valid)))
    else:
        feats["earnings_volatility"] = np.nan

    # Revenue growth consistency: std of QoQ revenue growth rates
    rev_vals = _series_over_quarters(inc, "Total Revenue", 8)
    rev_arr = np.array(rev_vals, dtype=float)
    rev_valid = rev_arr[np.isfinite(rev_arr)]
    if len(rev_valid) >= 3:
        growth_rates = []
        for i in range(len(rev_valid) - 1):
            if rev_valid[i + 1] != 0:
                growth_rates.append(
                    (rev_valid[i] - rev_valid[i + 1]) / abs(rev_valid[i + 1])
                )
        if len(growth_rates) >= 2:
            feats["revenue_growth_consistency"] = float(np.std(growth_rates))
        else:
            feats["revenue_growth_consistency"] = np.nan
    else:
        feats["revenue_growth_consistency"] = np.nan

    # SG&A as pct of revenue (most recent quarter)
    sga = _safe_line_item(inc, "Selling General And Administration", 0)
    rev = _safe_line_item(inc, "Total Revenue", 0)
    feats["sga_as_pct_revenue"] = _safe_divide(sga, rev)

    # SG&A trend over 4Q (negative = improving)
    sga_ratio_vals = _ratio_series(
        inc, "Selling General And Administration",
        inc, "Total Revenue",
    )
    feats["sga_trend"] = _compute_trend(sga_ratio_vals)

    return feats


# ── Category 5: Capital Allocation ───────────────────────────────────


def compute_capital_allocation(
    info: dict,
    quarterly_cashflow: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
) -> dict:
    """5 features evaluating how management allocates capital."""
    feats: dict[str, float] = {}
    cf = quarterly_cashflow
    bs = quarterly_balance_sheet

    # Dividend yield (from info)
    div_yield = info.get("dividendYield")
    div_yield = float(div_yield) if div_yield is not None else 0.0

    # Buyback yield: trailing 12m buybacks / market cap
    buybacks_ttm = _trailing_sum(cf, "Repurchase Of Capital Stock", 4)
    mkt_cap = info.get("marketCap")
    if (
        not np.isnan(buybacks_ttm)
        and mkt_cap is not None
        and mkt_cap > 0
    ):
        # Repurchases are typically negative in statements
        buyback_yield = abs(buybacks_ttm) / mkt_cap
    else:
        buyback_yield = 0.0

    feats["total_shareholder_yield"] = div_yield + buyback_yield

    # Reinvestment rate: |capex| / depreciation
    capex_ttm = _trailing_sum(cf, "Capital Expenditure", 4)
    dep_ttm = _trailing_sum(cf, "Depreciation And Amortization", 4)
    capex_abs = abs(capex_ttm) if not np.isnan(capex_ttm) else np.nan
    feats["reinvestment_rate"] = _safe_divide(capex_abs, dep_ttm)

    # Acquisition intensity (may not be available)
    # yfinance sometimes has "Acquisitions Net" or similar
    acq = _trailing_sum(cf, "Acquisitions Net", 4)
    if np.isnan(acq):
        acq = _trailing_sum(cf, "Net Business Purchase And Sale", 4)
    ta = _safe_line_item(bs, "Total Assets", 0)
    feats["acquisition_intensity"] = _safe_divide(
        abs(acq) if not np.isnan(acq) else np.nan, ta
    )

    # Dividend payout ratio: |dividends_paid| / net_income
    div_paid_ttm = _trailing_sum(cf, "Cash Dividends Paid", 4)
    # Also try alternative label
    if np.isnan(div_paid_ttm):
        div_paid_ttm = _trailing_sum(cf, "Dividends Paid", 4)
    ni_ttm = _trailing_sum(cf, "Net Income", 4)
    if np.isnan(ni_ttm):
        # Net income might only be in income statement; approximate from CF
        ni_ttm = _safe_line_item(cf, "Net Income From Continuing Operations", 0)
    div_abs = abs(div_paid_ttm) if not np.isnan(div_paid_ttm) else np.nan
    feats["dividend_payout_ratio"] = _safe_divide(div_abs, ni_ttm)

    # Retained earnings growth (QoQ)
    re_curr = _safe_line_item(bs, "Retained Earnings", 0)
    re_prev = _safe_line_item(bs, "Retained Earnings", 1)
    if not np.isnan(re_curr) and not np.isnan(re_prev) and re_prev != 0:
        feats["retained_earnings_growth"] = (re_curr - re_prev) / abs(re_prev)
    else:
        feats["retained_earnings_growth"] = np.nan

    return feats


# ── Category 6: Piotroski F-Score ────────────────────────────────────


def compute_piotroski_f_score(
    quarterly_income: pd.DataFrame,
    quarterly_cashflow: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
    annual_income: pd.DataFrame,
    annual_balance_sheet: pd.DataFrame,
) -> dict:
    """1 composite score (0-9) plus 9 binary flags."""
    feats: dict[str, float] = {}
    flags: list[float] = []

    ai = annual_income
    ab = annual_balance_sheet
    qi = quarterly_income
    qcf = quarterly_cashflow
    qbs = quarterly_balance_sheet

    # Use annual data if available, fall back to quarterly TTM
    def _annual_or_ttm_income(item: str, year: int = 0) -> float:
        val = _safe_line_item(ai, item, year)
        if not np.isnan(val):
            return val
        if year == 0:
            return _trailing_sum(qi, item, 4)
        return np.nan

    def _annual_or_ttm_bs(item: str, year: int = 0) -> float:
        val = _safe_line_item(ab, item, year)
        if not np.isnan(val):
            return val
        return _safe_line_item(qbs, item, year * 4)

    # --- Profitability (4 points) ---

    # 1. ROA > 0 (current year)
    ni_curr = _annual_or_ttm_income("Net Income", 0)
    ta_curr = _annual_or_ttm_bs("Total Assets", 0)
    roa_curr = _safe_divide(ni_curr, ta_curr)
    flag1 = 1.0 if (not np.isnan(roa_curr) and roa_curr > 0) else 0.0
    flags.append(flag1)

    # 2. OCF > 0
    ocf_ttm = _trailing_sum(qcf, "Operating Cash Flow", 4)
    flag2 = 1.0 if (not np.isnan(ocf_ttm) and ocf_ttm > 0) else 0.0
    flags.append(flag2)

    # 3. ROA improving YoY
    ni_prev = _annual_or_ttm_income("Net Income", 1)
    ta_prev = _annual_or_ttm_bs("Total Assets", 1)
    roa_prev = _safe_divide(ni_prev, ta_prev)
    flag3 = 1.0 if (
        not np.isnan(roa_curr) and not np.isnan(roa_prev) and roa_curr > roa_prev
    ) else 0.0
    flags.append(flag3)

    # 4. OCF > Net Income (accruals quality)
    flag4 = 1.0 if (
        not np.isnan(ocf_ttm) and not np.isnan(ni_curr) and ocf_ttm > ni_curr
    ) else 0.0
    flags.append(flag4)

    # --- Leverage / Liquidity (3 points) ---

    # 5. Long-term debt ratio decreasing YoY
    ltd_curr = _annual_or_ttm_bs("Long Term Debt", 0)
    ltd_prev = _annual_or_ttm_bs("Long Term Debt", 1)
    ltd_ratio_curr = _safe_divide(ltd_curr, ta_curr)
    ltd_ratio_prev = _safe_divide(ltd_prev, ta_prev)
    flag5 = 1.0 if (
        not np.isnan(ltd_ratio_curr) and not np.isnan(ltd_ratio_prev)
        and ltd_ratio_curr < ltd_ratio_prev
    ) else 0.0
    flags.append(flag5)

    # 6. Current ratio improving YoY
    ca_curr = _annual_or_ttm_bs("Current Assets", 0)
    cl_curr = _annual_or_ttm_bs("Current Liabilities", 0)
    ca_prev = _annual_or_ttm_bs("Current Assets", 1)
    cl_prev = _annual_or_ttm_bs("Current Liabilities", 1)
    cr_curr = _safe_divide(ca_curr, cl_curr)
    cr_prev = _safe_divide(ca_prev, cl_prev)
    flag6 = 1.0 if (
        not np.isnan(cr_curr) and not np.isnan(cr_prev) and cr_curr > cr_prev
    ) else 0.0
    flags.append(flag6)

    # 7. No new share dilution (shares outstanding same or decreased)
    # Try "Ordinary Shares Number" or "Share Issued" from balance sheet
    shares_curr = _annual_or_ttm_bs("Ordinary Shares Number", 0)
    if np.isnan(shares_curr):
        shares_curr = _annual_or_ttm_bs("Share Issued", 0)
    shares_prev = _annual_or_ttm_bs("Ordinary Shares Number", 1)
    if np.isnan(shares_prev):
        shares_prev = _annual_or_ttm_bs("Share Issued", 1)
    flag7 = 1.0 if (
        not np.isnan(shares_curr) and not np.isnan(shares_prev)
        and shares_curr <= shares_prev
    ) else 0.0
    flags.append(flag7)

    # --- Operating Efficiency (2 points) ---

    # 8. Gross margin improving YoY
    gp_curr = _annual_or_ttm_income("Gross Profit", 0)
    rev_curr = _annual_or_ttm_income("Total Revenue", 0)
    gp_prev = _annual_or_ttm_income("Gross Profit", 1)
    rev_prev = _annual_or_ttm_income("Total Revenue", 1)
    gm_curr = _safe_divide(gp_curr, rev_curr)
    gm_prev = _safe_divide(gp_prev, rev_prev)
    flag8 = 1.0 if (
        not np.isnan(gm_curr) and not np.isnan(gm_prev) and gm_curr > gm_prev
    ) else 0.0
    flags.append(flag8)

    # 9. Asset turnover improving YoY
    at_curr = _safe_divide(rev_curr, ta_curr)
    at_prev = _safe_divide(rev_prev, ta_prev)
    flag9 = 1.0 if (
        not np.isnan(at_curr) and not np.isnan(at_prev) and at_curr > at_prev
    ) else 0.0
    flags.append(flag9)

    # Composite
    feats["piotroski_f_score"] = float(sum(flags))
    for i, f in enumerate(flags, start=1):
        feats[f"piotroski_flag_{i}"] = f

    return feats


# ── Category 7: Industry-Relative Metrics ────────────────────────────


_INDUSTRY_METRIC_KEYS = [
    ("pe_industry_pctl", "pe_ratio"),
    ("pb_industry_pctl", "pb_ratio"),
    ("roe_industry_pctl", "roe_4q_trend"),
    ("roic_industry_pctl", "roic_latest"),
    ("gross_margin_industry_pctl", "gross_margin_4q_trend"),
    ("operating_margin_industry_pctl", "operating_margin_4q_trend"),
    ("debt_to_equity_industry_pctl", "debt_to_equity"),
    ("fcf_yield_industry_pctl", "fcf_yield"),
    ("ev_to_ebitda_industry_pctl", "ev_to_ebitda"),
    ("revenue_growth_industry_pctl", "revenue_growth"),
    ("earnings_growth_industry_pctl", "earnings_growth"),
    ("dividend_yield_industry_pctl", "dividend_yield"),
    ("altman_z_industry_pctl", "altman_z_score"),
    ("piotroski_industry_pctl", "piotroski_f_score"),
    ("dcf_upside_industry_pctl", "dcf_upside"),
]


def compute_industry_relative_metrics(
    ticker: str,
    ticker_fundamentals: dict,
    all_fundamentals: dict[str, dict],
    industry_map: dict[str, str],
) -> dict:
    """16 features: percentile rank within the stock's industry group.

    Falls back to sector-level if fewer than 3 peers in industry,
    then full universe if fewer than 3 in sector.
    """
    feats: dict[str, float] = {}

    my_industry = industry_map.get(ticker, "")
    # Build sector map by extracting the first word / broader category
    # Convention: industry_map values are "Sector > Industry" or just industry
    sector_map: dict[str, str] = {}
    for t, ind in industry_map.items():
        sector_map[t] = ind.split(">")[0].strip() if ">" in ind else ind

    my_sector = sector_map.get(ticker, "")

    # Find peer groups
    industry_peers = [
        t for t, ind in industry_map.items()
        if ind == my_industry and t in all_fundamentals
    ]
    sector_peers = [
        t for t, sec in sector_map.items()
        if sec == my_sector and t in all_fundamentals
    ]
    universe = list(all_fundamentals.keys())

    # Choose comparison group: industry >= 3, else sector >= 3, else universe
    if len(industry_peers) >= 3:
        peers = industry_peers
    elif len(sector_peers) >= 3:
        peers = sector_peers
    else:
        peers = universe

    # Compute percentile for each metric
    composite_pctls: list[float] = []
    for feat_name, metric_key in _INDUSTRY_METRIC_KEYS:
        my_val = ticker_fundamentals.get(metric_key, np.nan)
        if np.isnan(my_val) if isinstance(my_val, float) else my_val is None:
            feats[feat_name] = np.nan
            continue

        peer_vals = []
        for p in peers:
            v = all_fundamentals[p].get(metric_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                peer_vals.append(v)

        if len(peer_vals) < 2:
            feats[feat_name] = np.nan
            continue

        # Percentile: fraction of peers with value <= my_val
        count_leq = sum(1 for v in peer_vals if v <= my_val)
        pctl = count_leq / len(peer_vals)
        feats[feat_name] = float(pctl)
        composite_pctls.append(pctl)

    # Composite industry rank
    if composite_pctls:
        feats["industry_fundamental_rank"] = float(np.mean(composite_pctls))
    else:
        feats["industry_fundamental_rank"] = np.nan

    return feats


# ── Category 8: Institutional Blind-Spot Detection ───────────────────


def compute_institutional_blindspot(
    info: dict,
    holders_df: pd.DataFrame,
    insider_df: pd.DataFrame,
) -> dict:
    """6 features detecting under-covered or under-owned stocks."""
    feats: dict[str, float] = {}

    # Analyst count
    analyst_count = info.get("numberOfAnalystOpinions")
    if analyst_count is None:
        analyst_count = info.get("numberOfAnalysts")
    feats["analyst_count"] = float(analyst_count) if analyst_count is not None else np.nan

    # Institutional ownership
    inst_pct = info.get("heldPercentInstitutions")
    feats["inst_ownership_pct"] = float(inst_pct) if inst_pct is not None else np.nan

    # Insider cluster buy: 3+ insider buys within 30 days
    feats["insider_cluster_buy"] = _detect_insider_cluster(insider_df)

    # Price vs analyst target
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    if price is not None and target is not None and target > 0:
        feats["price_vs_analyst_target"] = float(price) / float(target)
    else:
        feats["price_vs_analyst_target"] = np.nan

    # Short interest days to cover
    short_ratio = info.get("shortRatio")
    feats["short_interest_days"] = float(short_ratio) if short_ratio is not None else np.nan

    # Blindspot score: composite of low coverage, low ownership, insider buying
    _analyst = feats["analyst_count"]
    _inst = feats["inst_ownership_pct"]
    _cluster = feats["insider_cluster_buy"]

    score_parts: list[float] = []
    # Low analyst coverage (fewer than 5 is under-covered)
    if not np.isnan(_analyst):
        score_parts.append(1.0 if _analyst < 5 else (0.5 if _analyst < 10 else 0.0))
    # Low institutional ownership (< 30%)
    if not np.isnan(_inst):
        score_parts.append(1.0 if _inst < 0.30 else (0.5 if _inst < 0.60 else 0.0))
    # Insider buying
    if not np.isnan(_cluster):
        score_parts.append(_cluster)

    feats["blindspot_score"] = float(np.mean(score_parts)) if score_parts else np.nan

    return feats


def _detect_insider_cluster(insider_df: pd.DataFrame) -> float:
    """Return 1.0 if 3+ insider *buys* occurred within any 30-day window, else 0.0."""
    if insider_df is None or insider_df.empty:
        return np.nan

    # Insider transactions typically have columns like:
    # "Start Date", "Transaction", "Shares", "Value" or similar
    # yfinance insider_transactions columns vary; be defensive
    try:
        df = insider_df.copy()

        # Identify buy transactions
        tx_col = None
        for candidate in ["Transaction", "transaction", "Text", "text", "Action"]:
            if candidate in df.columns:
                tx_col = candidate
                break
        if tx_col is None:
            return np.nan

        buys = df[df[tx_col].astype(str).str.lower().str.contains("buy|purchase", na=False)]
        if buys.empty:
            return 0.0

        # Identify date column
        date_col = None
        for candidate in ["Start Date", "startDate", "Date", "date", "Filing Date"]:
            if candidate in buys.columns:
                date_col = candidate
                break
        if date_col is None and buys.index.name and "date" in buys.index.name.lower():
            buys = buys.reset_index()
            date_col = buys.columns[0]
        if date_col is None:
            return np.nan

        dates = pd.to_datetime(buys[date_col], errors="coerce").dropna().sort_values()
        if len(dates) < 3:
            return 0.0

        # Sliding window: check if any 30-day window has >= 3 buys
        for i in range(len(dates) - 2):
            window_end = dates.iloc[i] + pd.Timedelta(days=30)
            count_in_window = ((dates >= dates.iloc[i]) & (dates <= window_end)).sum()
            if count_in_window >= 3:
                return 1.0

        return 0.0

    except Exception:
        logger.debug("Insider cluster detection failed", exc_info=True)
        return np.nan


# ── Master Function ──────────────────────────────────────────────────


def compute_deep_fundamentals(
    statements: dict,
    info: dict,
) -> dict:
    """Compute all ~45 deep fundamental features for a single ticker.

    Parameters
    ----------
    statements : dict from ``FinancialStatementLoader.fetch_all_statements``
        Expected keys: quarterly_income, quarterly_balance_sheet,
        quarterly_cashflow, annual_income, annual_balance_sheet,
        annual_cashflow.
    info : yfinance ``.info`` dict.

    Returns
    -------
    dict of feature_name -> float.  Every value is np.nan when the
    underlying data is unavailable.
    """
    qi = statements.get("quarterly_income", pd.DataFrame())
    qbs = statements.get("quarterly_balance_sheet", pd.DataFrame())
    qcf = statements.get("quarterly_cashflow", pd.DataFrame())
    ai = statements.get("annual_income", pd.DataFrame())
    ab = statements.get("annual_balance_sheet", pd.DataFrame())

    features: dict[str, float] = {}

    # Category 1: Profitability Trajectory (7 features)
    try:
        features.update(compute_profitability_trajectory(qi, qbs, info))
    except Exception:
        logger.warning("Profitability trajectory computation failed", exc_info=True)

    # Category 2: Balance Sheet Health (6 features)
    try:
        features.update(compute_balance_sheet_health(info, qbs, qi, qcf))
    except Exception:
        logger.warning("Balance sheet health computation failed", exc_info=True)

    # Category 3: Cash Flow Quality (7 features)
    try:
        features.update(compute_cash_flow_quality(qi, qcf, qbs, info))
    except Exception:
        logger.warning("Cash flow quality computation failed", exc_info=True)

    # Category 4: Earnings Quality (5 features)
    try:
        features.update(compute_earnings_quality(qi, qcf))
    except Exception:
        logger.warning("Earnings quality computation failed", exc_info=True)

    # Category 5: Capital Allocation (5 features)
    try:
        features.update(compute_capital_allocation(info, qcf, qbs))
    except Exception:
        logger.warning("Capital allocation computation failed", exc_info=True)

    # Category 6: Piotroski F-Score (10 features: 1 composite + 9 flags)
    try:
        features.update(compute_piotroski_f_score(qi, qcf, qbs, ai, ab))
    except Exception:
        logger.warning("Piotroski F-Score computation failed", exc_info=True)

    # Categories 7 & 8 require cross-sectional data or holder data
    # and are computed externally via:
    #   compute_industry_relative_metrics(...)
    #   compute_institutional_blindspot(...)

    return features
