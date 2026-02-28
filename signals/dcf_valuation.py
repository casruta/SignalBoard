"""DCF (Discounted Cash Flow) intrinsic value computation and valuation features."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SECTOR_TERMINAL_GROWTH = {
    "Technology": 0.035, "Communication Services": 0.030,
    "Healthcare": 0.030, "Consumer Discretionary": 0.025,
    "Industrials": 0.025, "Financial Services": 0.025, "Financials": 0.025,
    "Consumer Staples": 0.020, "Real Estate": 0.020,
    "Energy": 0.015, "Materials": 0.020, "Utilities": 0.015,
}


def get_terminal_growth(sector: str, default: float = 0.02) -> float:
    """Return sector-appropriate terminal growth rate."""
    return SECTOR_TERMINAL_GROWTH.get(sector, default)


def compute_dcf_valuation(
    statements: dict,
    info: dict,
    risk_free_rate: float,
    projection_years: int = 5,
    terminal_growth: float | None = None,
    equity_risk_premium: float = 0.06,
    improving_fcf: bool = False,
) -> dict:
    """Compute DCF intrinsic value and related valuation metrics.

    Parameters
    ----------
    statements : dict from FinancialStatementLoader with keys:
        quarterly_income, quarterly_cashflow, quarterly_balance_sheet,
        annual_income, annual_cashflow, annual_balance_sheet, info
    info : yfinance .info dict
    risk_free_rate : 10Y Treasury rate (e.g., 0.04 for 4%)
    projection_years : number of years to project FCF forward
    terminal_growth : long-term FCF growth rate (default 2%)
    equity_risk_premium : ERP used in CAPM (default 6%)

    Returns
    -------
    dict with valuation metrics. Values are np.nan when data is insufficient.
    """
    result = {
        "intrinsic_value_per_share": np.nan,
        "margin_of_safety": np.nan,
        "dcf_upside_pct": np.nan,
        "wacc": np.nan,
        "roic": np.nan,
        "roic_vs_wacc_spread": np.nan,
        "fcf_yield": np.nan,
        "implied_growth_rate": np.nan,
        "ev_to_fcf": np.nan,
        "ev_to_revenue": np.nan,
    }

    # Sector-dependent terminal growth if not explicitly provided
    if terminal_growth is None:
        sector = info.get("sector", "") if info else ""
        terminal_growth = get_terminal_growth(sector)

    market_price = _safe_get(info, "currentPrice")
    shares_outstanding = _safe_get(info, "sharesOutstanding")
    market_cap = _safe_get(info, "marketCap")
    enterprise_value = _safe_get(info, "enterpriseValue")

    # --- WACC ---
    wacc = compute_wacc(info, statements, risk_free_rate, equity_risk_premium)
    result["wacc"] = wacc

    # --- ROIC ---
    roic = compute_roic(statements)
    result["roic"] = roic

    if not np.isnan(wacc) and not np.isnan(roic):
        result["roic_vs_wacc_spread"] = roic - wacc

    # --- Base FCF (trailing 4 quarters) ---
    quarterly_cf = statements.get("quarterly_cashflow")
    base_fcf = np.nan
    if quarterly_cf is not None and not quarterly_cf.empty:
        base_fcf = _trailing_4q_sum(quarterly_cf, "Free Cash Flow")
        if np.isnan(base_fcf):
            # Fall back to Operating Cash Flow - CapEx
            ocf = _trailing_4q_sum(quarterly_cf, "Operating Cash Flow")
            capex = _trailing_4q_sum(quarterly_cf, "Capital Expenditure")
            if not np.isnan(ocf) and not np.isnan(capex):
                base_fcf = ocf + capex  # capex is typically negative

    if np.isnan(base_fcf) or base_fcf <= 0:
        logger.warning("Cannot compute DCF: base FCF is missing or non-positive (%.2f)",
                        base_fcf if not np.isnan(base_fcf) else 0.0)
        # Still populate what we can
        _populate_non_dcf_metrics(result, base_fcf, market_cap, enterprise_value)

        # EV/Revenue for pre-profit companies
        quarterly_income = statements.get("quarterly_income")
        if quarterly_income is not None and not quarterly_income.empty:
            rev_ttm = _trailing_4q_sum(quarterly_income, "Total Revenue")
            if not np.isnan(rev_ttm) and rev_ttm > 0 and not np.isnan(enterprise_value):
                result["ev_to_revenue"] = enterprise_value / rev_ttm

        return result

    # --- FCF yield and EV/FCF ---
    _populate_non_dcf_metrics(result, base_fcf, market_cap, enterprise_value)

    # --- Revenue growth for projections ---
    annual_income = statements.get("annual_income")
    revenue_growth = _revenue_growth_cagr(annual_income, years=3)
    if np.isnan(revenue_growth):
        revenue_growth = terminal_growth
        logger.warning("Revenue growth unavailable; defaulting to terminal rate %.2f%%",
                        terminal_growth * 100)

    # --- Project FCF ---
    if np.isnan(wacc) or wacc <= terminal_growth:
        logger.warning("WACC (%.4f) <= terminal growth (%.4f); cannot compute DCF", wacc, terminal_growth)
        return result

    projected = project_fcf(base_fcf, revenue_growth, terminal_growth, projection_years,
                            improving_fcf=improving_fcf)

    # --- Terminal value ---
    tv = compute_terminal_value(projected[-1], terminal_growth, wacc)

    # --- Discount projected FCFs and terminal value ---
    pv_fcfs = sum(fcf / (1 + wacc) ** yr for yr, fcf in enumerate(projected, start=1))
    pv_tv = tv / (1 + wacc) ** projection_years

    enterprise_dcf = pv_fcfs + pv_tv

    # --- Equity value ---
    # Subtract net debt: total_debt - cash
    total_debt = _get_balance_sheet_item(statements, "Long Term Debt")
    cash = _get_balance_sheet_item(statements, "Cash And Cash Equivalents")

    net_debt = 0.0
    if not np.isnan(total_debt):
        net_debt += total_debt
    if not np.isnan(cash):
        net_debt -= cash

    equity_value = enterprise_dcf - net_debt

    if not np.isnan(shares_outstanding) and shares_outstanding > 0:
        intrinsic = equity_value / shares_outstanding
        result["intrinsic_value_per_share"] = intrinsic

        if not np.isnan(market_price) and market_price > 0:
            result["margin_of_safety"] = (intrinsic - market_price) / intrinsic
            result["dcf_upside_pct"] = intrinsic / market_price - 1
    else:
        logger.warning("Shares outstanding unavailable; cannot compute per-share intrinsic value")

    # --- Implied growth rate ---
    if not np.isnan(enterprise_value) and enterprise_value > 0:
        result["implied_growth_rate"] = compute_implied_growth_rate(
            base_fcf, enterprise_value, wacc, projection_years
        )

    return result


def compute_wacc(
    info: dict,
    statements: dict,
    risk_free_rate: float,
    equity_risk_premium: float = 0.06,
) -> float:
    """Compute WACC = E/(E+D) * Ke + D/(E+D) * Kd * (1-tax).

    Cost of equity via CAPM: risk_free + beta * equity_risk_premium.
    Cost of debt: interest_expense / total_debt.
    Adds tiered illiquidity premiums by market cap (2.5% micro, 2% small,
    1% lower-mid, 0.5% mid-cap).
    Clamps result to [0.05, 0.30].
    """
    market_cap = _safe_get(info, "marketCap")
    beta = _safe_get(info, "beta", default=1.0)

    # Cost of equity via CAPM
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Debt figures from balance sheet
    total_debt = _get_balance_sheet_item(statements, "Long Term Debt")
    if np.isnan(total_debt):
        total_debt = 0.0

    # Interest expense and tax rate from income statement
    annual_income = statements.get("annual_income")
    interest_expense = np.nan
    tax_rate = np.nan

    if annual_income is not None and not annual_income.empty:
        interest_expense = _get_latest_line_item(annual_income, "Interest Expense")
        tax_provision = _get_latest_line_item(annual_income, "Tax Provision")
        pretax_income = _get_latest_line_item(annual_income, "Net Income")

        # Try to compute effective tax rate
        if not np.isnan(tax_provision) and not np.isnan(pretax_income):
            # Pretax = net_income + tax_provision (approximate)
            pretax = pretax_income + tax_provision
            if pretax > 0:
                tax_rate = tax_provision / pretax

    if np.isnan(tax_rate) or tax_rate < 0 or tax_rate > 0.5:
        tax_rate = 0.21  # Default US corporate rate

    # Cost of debt
    cost_of_debt = 0.05  # Default
    if not np.isnan(interest_expense) and total_debt > 0:
        raw_cost = abs(interest_expense) / total_debt
        if 0.0 < raw_cost < 0.30:
            cost_of_debt = raw_cost

    # Capital structure weights
    equity = market_cap if not np.isnan(market_cap) and market_cap > 0 else 0.0
    total_capital = equity + total_debt

    if total_capital <= 0:
        logger.warning("Total capital is zero; cannot compute WACC")
        return np.nan

    weight_equity = equity / total_capital
    weight_debt = total_debt / total_capital

    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)

    # Tiered illiquidity premiums by market cap
    if not np.isnan(market_cap):
        if market_cap < 500_000_000:
            wacc += 0.025  # micro-cap
        elif market_cap < 1_000_000_000:
            wacc += 0.020  # small-cap
        elif market_cap < 3_000_000_000:
            wacc += 0.010  # lower mid-cap
        elif market_cap < 10_000_000_000:
            wacc += 0.005  # mid-cap

    # Clamp to sane range
    wacc = float(np.clip(wacc, 0.05, 0.30))
    return wacc


def compute_roic(statements: dict) -> float:
    """ROIC = NOPAT / Invested Capital.

    NOPAT = Operating Income * (1 - tax_rate)
    Invested Capital = Total Equity + Total Debt - Cash
    """
    annual_income = statements.get("annual_income")
    if annual_income is None or annual_income.empty:
        logger.warning("Annual income statement unavailable for ROIC")
        return np.nan

    operating_income = _get_latest_line_item(annual_income, "Operating Income")
    if np.isnan(operating_income):
        logger.warning("Operating income unavailable for ROIC")
        return np.nan

    # Tax rate
    tax_provision = _get_latest_line_item(annual_income, "Tax Provision")
    net_income = _get_latest_line_item(annual_income, "Net Income")
    tax_rate = 0.21
    if not np.isnan(tax_provision) and not np.isnan(net_income):
        pretax = net_income + tax_provision
        if pretax > 0:
            tax_rate = np.clip(tax_provision / pretax, 0.0, 0.5)

    nopat = operating_income * (1 - tax_rate)

    # Invested capital from balance sheet
    total_equity = _get_balance_sheet_item(statements, "Stockholders Equity")
    total_debt = _get_balance_sheet_item(statements, "Long Term Debt")
    cash = _get_balance_sheet_item(statements, "Cash And Cash Equivalents")

    if np.isnan(total_equity):
        logger.warning("Stockholders equity unavailable for ROIC")
        return np.nan

    invested_capital = total_equity
    if not np.isnan(total_debt):
        invested_capital += total_debt
    if not np.isnan(cash):
        invested_capital -= cash

    if invested_capital <= 0:
        logger.warning("Invested capital is non-positive (%.2f); cannot compute ROIC", invested_capital)
        return np.nan

    return nopat / invested_capital


def project_fcf(
    base_fcf: float,
    revenue_growth_rate: float,
    terminal_growth: float = 0.02,
    years: int = 5,
    improving_fcf: bool = False,
) -> list[float]:
    """Project FCF forward with growth decaying linearly toward terminal rate.

    Year 1 growth = revenue_growth_rate, Year N growth = terminal_growth.
    Intermediate years interpolate linearly. Each year's FCF is capped at
    a multiple of base_fcf (3x if improving_fcf, else 2x) to prevent
    hockey-stick projections.
    """
    if years < 1:
        return []

    cap_multiplier = 3.0 if improving_fcf else 2.0
    cap = cap_multiplier * abs(base_fcf)
    projected = []
    current_fcf = base_fcf

    for yr in range(years):
        if years == 1:
            growth = terminal_growth
        else:
            # Linear interpolation: yr=0 -> revenue_growth_rate, yr=years-1 -> terminal_growth
            fraction = yr / (years - 1)
            growth = revenue_growth_rate + fraction * (terminal_growth - revenue_growth_rate)

        current_fcf = current_fcf * (1 + growth)
        current_fcf = min(current_fcf, cap)
        projected.append(current_fcf)

    return projected


def compute_terminal_value(
    final_year_fcf: float,
    terminal_growth: float,
    wacc: float,
) -> float:
    """Gordon Growth Model: TV = FCF * (1+g) / (WACC - g)."""
    denominator = wacc - terminal_growth
    if denominator <= 0:
        logger.warning("WACC (%.4f) <= terminal growth (%.4f); terminal value undefined",
                        wacc, terminal_growth)
        return 0.0

    return final_year_fcf * (1 + terminal_growth) / denominator


def compute_implied_growth_rate(
    current_fcf: float,
    enterprise_value: float,
    wacc: float,
    projection_years: int = 5,
) -> float:
    """Solve for the growth rate that makes DCF = current enterprise value.

    Uses binary search between -0.10 and 0.30.
    """
    if current_fcf <= 0 or enterprise_value <= 0 or np.isnan(wacc):
        return np.nan

    def _dcf_at_growth(g: float) -> float:
        """Compute enterprise value given constant growth rate g."""
        pv = 0.0
        fcf = current_fcf
        for yr in range(1, projection_years + 1):
            fcf = fcf * (1 + g)
            pv += fcf / (1 + wacc) ** yr

        # Terminal value using the same growth rate capped at terminal
        terminal_g = min(g, wacc - 0.01)  # Ensure convergence
        if terminal_g < wacc:
            tv = fcf * (1 + terminal_g) / (wacc - terminal_g)
            pv += tv / (1 + wacc) ** projection_years

        return pv

    lo, hi = -0.10, 0.30
    for _ in range(100):
        mid = (lo + hi) / 2.0
        ev_mid = _dcf_at_growth(mid)

        if ev_mid < enterprise_value:
            lo = mid
        else:
            hi = mid

        if abs(hi - lo) < 1e-6:
            break

    return (lo + hi) / 2.0


def _safe_get(d: dict, key: str, default=np.nan):
    """Safely get a value from dict, returning default if missing/None/0."""
    if d is None:
        return default
    val = d.get(key)
    if val is None or val == 0:
        return default
    return val


def _trailing_4q_sum(quarterly_df: pd.DataFrame, line_item: str) -> float:
    """Sum the last 4 quarters of a line item.

    yfinance DataFrames have line items as rows and date columns sorted
    descending (most recent first). Returns np.nan if insufficient data.
    """
    if quarterly_df is None or quarterly_df.empty:
        return np.nan

    if line_item not in quarterly_df.index:
        return np.nan

    row = quarterly_df.loc[line_item]
    # Columns are dates in descending order; take the first 4 (most recent)
    values = row.iloc[:4].dropna()

    if len(values) < 4:
        logger.warning("Only %d quarters available for '%s'; need 4", len(values), line_item)
        if len(values) == 0:
            return np.nan

    return float(values.sum())


def _revenue_growth_cagr(annual_income: pd.DataFrame, years: int = 3) -> float:
    """Compute CAGR of revenue over the last N annual periods.

    Columns are dates in descending order, so column 0 is most recent.
    """
    if annual_income is None or annual_income.empty:
        return np.nan

    if "Total Revenue" not in annual_income.index:
        logger.warning("Total Revenue not found in annual income statement")
        return np.nan

    revenue_row = annual_income.loc["Total Revenue"].dropna()

    if len(revenue_row) < 2:
        return np.nan

    n = min(years, len(revenue_row) - 1)
    recent = float(revenue_row.iloc[0])    # Most recent
    older = float(revenue_row.iloc[n])      # N years ago

    if older <= 0 or recent <= 0:
        return np.nan

    cagr = (recent / older) ** (1 / n) - 1
    return cagr


def _get_latest_line_item(df: pd.DataFrame, line_item: str) -> float:
    """Get the most recent value of a line item from a financial statement DataFrame."""
    if df is None or df.empty:
        return np.nan
    if line_item not in df.index:
        return np.nan

    row = df.loc[line_item].dropna()
    if len(row) == 0:
        return np.nan

    return float(row.iloc[0])


def _get_balance_sheet_item(statements: dict, line_item: str) -> float:
    """Get the most recent value of a balance sheet item.

    Tries quarterly first for recency, then falls back to annual.
    """
    for key in ("quarterly_balance_sheet", "annual_balance_sheet"):
        bs = statements.get(key)
        if bs is not None and not bs.empty and line_item in bs.index:
            row = bs.loc[line_item].dropna()
            if len(row) > 0:
                return float(row.iloc[0])

    logger.warning("Balance sheet item '%s' not found", line_item)
    return np.nan


def _populate_non_dcf_metrics(result: dict, base_fcf: float, market_cap: float,
                               enterprise_value: float) -> None:
    """Fill in FCF-based metrics that don't require full DCF computation."""
    if not np.isnan(base_fcf) and base_fcf > 0:
        if not np.isnan(market_cap) and market_cap > 0:
            result["fcf_yield"] = base_fcf / market_cap
        if not np.isnan(enterprise_value) and enterprise_value > 0:
            result["ev_to_fcf"] = enterprise_value / base_fcf
