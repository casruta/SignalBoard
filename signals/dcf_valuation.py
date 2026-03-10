"""DCF (Discounted Cash Flow) intrinsic value computation — Damodaran-aligned.

Key methodology choices following Aswath Damodaran's framework:
- FCFF (Free Cash Flow to Firm) = EBIT(1-t) + D&A - CapEx - ΔWC
- Discount at WACC (not cost of equity)
- Bottom-up beta: sector unlevered beta relevered for company D/E
- Synthetic cost of debt: interest-coverage → default spread + Rf
- Marginal tax rate (21%) for WACC shield; effective for NOPAT
- Terminal growth capped at risk-free rate
- SBC subtracted from cash flows (real dilution)
- Size premium applied to cost of equity, not WACC
"""

import math

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Sector terminal growth — capped at Rf dynamically ────────────────
SECTOR_TERMINAL_GROWTH = {
    "Technology": 0.035, "Communication Services": 0.030,
    "Healthcare": 0.030, "Consumer Discretionary": 0.025,
    "Industrials": 0.025, "Financial Services": 0.025, "Financials": 0.025,
    "Consumer Staples": 0.020, "Real Estate": 0.020,
    "Energy": 0.015, "Materials": 0.020, "Utilities": 0.015,
}

# ── Damodaran bottom-up unlevered betas by sector ────────────────────
# Source: Damodaran Online, US sector averages (Jan 2025)
SECTOR_UNLEVERED_BETA = {
    "Technology": 1.10,
    "Communication Services": 0.90,
    "Healthcare": 0.95,
    "Consumer Discretionary": 0.95,
    "Consumer Staples": 0.55,
    "Industrials": 0.80,
    "Financial Services": 0.55,
    "Financials": 0.55,
    "Energy": 0.90,
    "Materials": 0.80,
    "Real Estate": 0.55,
    "Utilities": 0.35,
}

# ── Damodaran synthetic rating: interest coverage → default spread ───
# Source: Damodaran Online, ratings/default spreads (2024-2025)
COVERAGE_TO_SPREAD = [
    (12.50, 0.0063),   # AAA
    (8.50,  0.0078),   # AA
    (6.50,  0.0098),   # A+
    (5.50,  0.0108),   # A
    (4.25,  0.0122),   # A-
    (3.00,  0.0156),   # BBB
    (2.50,  0.0200),   # BB+
    (2.00,  0.0250),   # BB
    (1.50,  0.0325),   # B+
    (1.25,  0.0400),   # B
    (0.80,  0.0500),   # B-
    (0.50,  0.0800),   # CCC
]

US_MARGINAL_TAX_RATE = 0.25  # 21% federal + ~4% blended state

# ── Damodaran country risk premiums (Jan 2025) ──────────────────────
# Source: Damodaran Online, country risk premium estimates
COUNTRY_RISK_PREMIUM = {
    "United States": 0.0, "Canada": 0.005, "United Kingdom": 0.005,
    "Germany": 0.005, "France": 0.005, "Switzerland": 0.0,
    "Japan": 0.008, "Australia": 0.005, "Ireland": 0.005,
    "Netherlands": 0.005, "Sweden": 0.005, "Denmark": 0.005,
    "South Korea": 0.008, "Taiwan": 0.008, "Israel": 0.008,
    "Singapore": 0.005, "Hong Kong": 0.008,
    "China": 0.015, "India": 0.020, "Brazil": 0.030,
    "Mexico": 0.020, "South Africa": 0.025, "Russia": 0.040,
    "Turkey": 0.030, "Indonesia": 0.020, "Thailand": 0.015,
}

# ── Sector median EV/EBITDA exit multiples (Damodaran, Jan 2025) ────
SECTOR_EXIT_MULTIPLES = {
    "Technology": 15.0, "Communication Services": 12.0,
    "Healthcare": 13.0, "Consumer Discretionary": 12.0,
    "Consumer Staples": 12.0, "Industrials": 11.0,
    "Financial Services": 10.0, "Financials": 10.0,
    "Energy": 8.0, "Materials": 9.0,
    "Real Estate": 14.0, "Utilities": 10.0,
}

# ── Cyclical sectors for normalized earnings (Phase 1.3) ─────────────
CYCLICAL_SECTORS = {"Energy", "Materials", "Industrials", "Consumer Discretionary"}

# ── Multi-stage DCF growth classification (Phase 2.1) ────────────────
# Durations for high-growth, transition, and stable stages (total = 10 years)
GROWTH_STAGE_PARAMS = {
    "high": {"high_years": 3, "transition_years": 4, "stable_years": 3},    # >20% growth
    "medium": {"high_years": 2, "transition_years": 3, "stable_years": 5},  # 10-20%
    "low": {"high_years": 0, "transition_years": 2, "stable_years": 3},     # <10%
}

# ── R&D capitalization useful life by sector (Damodaran) (Phase 2.2) ─
RD_USEFUL_LIFE = {
    "Technology": 5,
    "Healthcare": 10,
    "Communication Services": 5,
}

# ── ROIC competitive advantage half-lives by sector (Damodaran) (Phase 2.4) ─
ROIC_HALFLIFE = {
    "Technology": 5,
    "Healthcare": 7,
    "Communication Services": 5,
    "Consumer Discretionary": 7,
    "Consumer Staples": 10,
    "Industrials": 10,
    "Financial Services": 7,
    "Financials": 7,
    "Energy": 7,
    "Materials": 10,
    "Real Estate": 10,
    "Utilities": 10,
}


def get_terminal_growth(
    sector: str, default: float = 0.02, risk_free_rate: float | None = None,
) -> float:
    """Return sector-appropriate terminal growth rate, capped at Rf."""
    raw = SECTOR_TERMINAL_GROWTH.get(sector, default)
    if risk_free_rate is not None:
        return min(raw, risk_free_rate)
    return raw


def compute_dcf_valuation(
    statements: dict,
    info: dict,
    risk_free_rate: float,
    projection_years: int = 10,
    terminal_growth: float | None = None,
    equity_risk_premium: float = 0.06,
    improving_fcf: bool = False,
    altman_z: float | None = None,
    capitalize_rd: bool = True,
    normalize_earnings: bool | None = None,
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
    capitalize_rd : capitalize R&D for tech/pharma sectors (Damodaran)
    normalize_earnings : use mid-cycle EBIT for cyclical sectors; None = auto-detect

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
        "delta_wc_estimated": False,
        "implied_reinvestment_rate": np.nan,
        "bear_iv": np.nan,
        "base_iv": np.nan,
        "bull_iv": np.nan,
        "scenario_range_pct": np.nan,
        "tv_gordon": np.nan,
        "tv_exit_multiple": np.nan,
        "tv_divergence_pct": np.nan,
        "distress_probability": np.nan,
        "distress_adjusted_iv": np.nan,
        "mc_median_iv": np.nan,
        "mc_p10_iv": np.nan,
        "mc_p90_iv": np.nan,
        "mc_std_iv": np.nan,
        "rd_adjustment_ebit": 0.0,
        "rd_asset": 0.0,
        "normalized_ebit_used": False,
    }

    # Sector-dependent terminal growth — capped at risk-free rate (Damodaran)
    sector = info.get("sector", "") if info else ""
    if terminal_growth is None:
        terminal_growth = get_terminal_growth(sector, risk_free_rate=risk_free_rate)
    else:
        terminal_growth = min(terminal_growth, risk_free_rate)

    market_price = _safe_get(info, "currentPrice")
    market_cap = _safe_get(info, "marketCap")
    enterprise_value = _safe_get(info, "enterpriseValue")

    # Prefer diluted share count (Damodaran: accounts for SBC dilution)
    shares_outstanding = _safe_get(info, "impliedSharesOutstanding")
    if np.isnan(shares_outstanding):
        shares_outstanding = _safe_get(info, "sharesOutstanding")

    # --- WACC ---
    wacc = compute_wacc(info, statements, risk_free_rate, equity_risk_premium)
    result["wacc"] = wacc

    # --- R&D capitalization (Phase 2.2) ---
    rd_info = {"rd_adjustment_ebit": 0.0, "rd_asset": 0.0, "rd_current": 0.0}
    if capitalize_rd and sector in RD_USEFUL_LIFE:
        rd_info = _capitalize_rd(statements, sector)
        result["rd_adjustment_ebit"] = rd_info["rd_adjustment_ebit"]
        result["rd_asset"] = rd_info["rd_asset"]

    # --- ROIC (with R&D adjustment when applicable) ---
    roic = compute_roic(statements, sector=sector, rd_info=rd_info)
    result["roic"] = roic

    if not np.isnan(wacc) and not np.isnan(roic):
        result["roic_vs_wacc_spread"] = roic - wacc

    # --- Normalized earnings for cyclical sectors (Phase 1.3) ---
    if normalize_earnings is None:
        normalize_earnings = sector in CYCLICAL_SECTORS

    # --- Base FCFF (Damodaran: EBIT(1-t) + D&A - CapEx - ΔWC) -------
    quarterly_cf = statements.get("quarterly_cashflow")
    quarterly_income = statements.get("quarterly_income")
    base_fcf, delta_wc_estimated = _compute_fcff(
        statements, normalize=normalize_earnings, sector=sector, rd_info=rd_info,
    )
    result["delta_wc_estimated"] = delta_wc_estimated
    if normalize_earnings and not np.isnan(base_fcf):
        result["normalized_ebit_used"] = True

    # Fallback: OCF→FCFF bridge (Damodaran reconciliation).
    # OCF already reflects NOPAT + D&A - ΔWC - after-tax interest (paid to
    # debtholders). To recover FCFF we add CapEx (negative, so OCF+CapEx = FCFE)
    # then add back after-tax interest to convert FCFE→FCFF.
    # Sign conventions: CapEx is negative in yfinance; Interest Expense
    # is typically positive (cash outflow reported as positive).
    if np.isnan(base_fcf) and quarterly_cf is not None and not quarterly_cf.empty:
        ocf = _trailing_4q_sum(quarterly_cf, "Operating Cash Flow")
        capex = _trailing_4q_sum(quarterly_cf, "Capital Expenditure")
        if not np.isnan(ocf) and not np.isnan(capex):
            base_fcf = ocf + capex  # capex negative
            # Add back after-tax interest to convert FCFE→FCFF
            annual_income = statements.get("annual_income")
            if annual_income is not None and not annual_income.empty:
                interest = _get_latest_line_item(annual_income, "Interest Expense")
                if not np.isnan(interest) and abs(interest) > 0:
                    base_fcf += abs(interest) * (1 - US_MARGINAL_TAX_RATE)

    # Subtract SBC from FCF (Damodaran: SBC is real dilution expense)
    if not np.isnan(base_fcf) and quarterly_cf is not None and not quarterly_cf.empty:
        sbc = _trailing_4q_sum(quarterly_cf, "Stock Based Compensation")
        if not np.isnan(sbc):
            base_fcf -= abs(sbc)

    if np.isnan(base_fcf) or base_fcf <= 0:
        logger.warning("Cannot compute DCF: base FCFF is missing or non-positive (%.2f)",
                        base_fcf if not np.isnan(base_fcf) else 0.0)
        _populate_non_dcf_metrics(result, base_fcf, market_cap, enterprise_value)

        # EV/Revenue for pre-profit companies
        if quarterly_income is not None and not quarterly_income.empty:
            rev_ttm = _trailing_4q_sum(quarterly_income, "Total Revenue")
            if not np.isnan(rev_ttm) and rev_ttm > 0 and not np.isnan(enterprise_value):
                result["ev_to_revenue"] = enterprise_value / rev_ttm

        return result

    # --- FCF yield and EV/FCF ---
    _populate_non_dcf_metrics(result, base_fcf, market_cap, enterprise_value)

    # --- Revenue growth for projections (multi-horizon median) -------
    annual_income = statements.get("annual_income")
    g1 = _revenue_growth_cagr(annual_income, years=1)
    g3 = _revenue_growth_cagr(annual_income, years=3)
    g5 = _revenue_growth_cagr(annual_income, years=5)
    revenue_growth = float(np.nanmedian([g1, g3, g5]))
    if np.isnan(revenue_growth):
        revenue_growth = terminal_growth
        logger.warning("Revenue growth unavailable; defaulting to terminal rate %.2f%%",
                        terminal_growth * 100)

    # --- Reinvestment rate consistency check (Damodaran: g = reinvestment × ROIC)
    if not np.isnan(roic) and roic > 0 and not np.isnan(revenue_growth):
        implied_reinvestment = revenue_growth / roic
        result["implied_reinvestment_rate"] = implied_reinvestment
        if implied_reinvestment > 1.0:
            logger.warning(
                "Implied reinvestment rate %.1f%% > 100%% — growth %.1f%% "
                "inconsistent with ROIC %.1f%%",
                implied_reinvestment * 100, revenue_growth * 100, roic * 100,
            )

    # --- Project FCF ---
    if np.isnan(wacc) or wacc <= terminal_growth:
        logger.warning("WACC (%.4f) <= terminal growth (%.4f); cannot compute DCF",
                        wacc, terminal_growth)
        return result

    projected = project_fcf_multistage(
        base_fcf, revenue_growth, terminal_growth,
        improving_fcf=improving_fcf,
        roic_latest=roic if not np.isnan(roic) else None,
        wacc=wacc,
        sector=sector,
    )
    projection_years = len(projected)

    # --- Terminal value (Gordon Growth + exit multiple cross-check) ---
    tv = compute_terminal_value(projected[-1], terminal_growth, wacc)
    result["tv_gordon"] = tv

    # Exit multiple cross-check using sector EV/EBITDA (Damodaran)
    tv_exit = _compute_exit_multiple_tv(statements, sector, projected[-1], terminal_growth)
    result["tv_exit_multiple"] = tv_exit
    if tv > 0 and not np.isnan(tv_exit) and tv_exit > 0:
        divergence = abs(tv - tv_exit) / min(tv, tv_exit)
        result["tv_divergence_pct"] = divergence
        if divergence > 0.50:
            logger.warning(
                "Terminal value divergence %.0f%%: Gordon=%.0fM vs Exit Multiple=%.0fM "
                "— using blended average",
                divergence * 100, tv / 1e6, tv_exit / 1e6,
            )
            tv = (tv + tv_exit) / 2.0

    # --- Discount projected FCFs and terminal value ---
    pv_fcfs = sum(fcf / (1 + wacc) ** yr for yr, fcf in enumerate(projected, start=1))
    pv_tv = tv / (1 + wacc) ** projection_years

    enterprise_dcf = pv_fcfs + pv_tv

    # --- Equity value (total debt incl. short-term) ---
    total_debt = _get_total_debt(statements)
    cash = _get_balance_sheet_item(statements, "Cash And Cash Equivalents")

    net_debt = total_debt
    if not np.isnan(cash):
        net_debt -= cash

    equity_value = enterprise_dcf - net_debt

    if not np.isnan(shares_outstanding) and shares_outstanding > 0:
        intrinsic = equity_value / shares_outstanding
        result["intrinsic_value_per_share"] = intrinsic
        result["base_iv"] = intrinsic

        if not np.isnan(market_price) and market_price > 0:
            result["current_price"] = market_price
            if intrinsic > 0:
                result["margin_of_safety"] = (intrinsic - market_price) / intrinsic
                result["dcf_upside_pct"] = intrinsic / market_price - 1
            else:
                # Intrinsic value <= 0 means equity is worthless (high debt);
                # margin of safety is undefined, upside is -100%.
                logger.warning(
                    "Intrinsic value per share is non-positive (%.2f); "
                    "margin of safety undefined",
                    intrinsic,
                )
                result["margin_of_safety"] = -1.0
                result["dcf_upside_pct"] = -1.0

        # --- Distress probability discount (Damodaran) ---
        if altman_z is not None and not np.isnan(altman_z):
            p_distress = _distress_probability(altman_z)
            result["distress_probability"] = p_distress
            if p_distress > 0:
                total_assets = _get_balance_sheet_item(statements, "Total Assets")
                liquidation_per_share = 0.0
                if not np.isnan(total_assets) and total_assets > 0:
                    liquidation_per_share = 0.3 * total_assets / shares_outstanding
                adjusted = intrinsic * (1 - p_distress) + liquidation_per_share * p_distress
                result["distress_adjusted_iv"] = adjusted
            else:
                result["distress_adjusted_iv"] = intrinsic

        # --- Monte Carlo terminal value sensitivity (Phase 2.3) ---
        mc_results = _monte_carlo_dcf(
            base_fcf, wacc, revenue_growth, terminal_growth,
            projection_years, net_debt, shares_outstanding,
            improving_fcf=improving_fcf,
            roic_latest=roic if not np.isnan(roic) else None,
            sector=sector,
        )
        if mc_results:
            result.update(mc_results)
            result["bear_iv"] = mc_results.get("mc_p10_iv", np.nan)
            result["bull_iv"] = mc_results.get("mc_p90_iv", np.nan)
            mc_median = mc_results.get("mc_median_iv", np.nan)
            if not np.isnan(mc_median) and mc_median > 0:
                p10 = mc_results.get("mc_p10_iv", np.nan)
                p90 = mc_results.get("mc_p90_iv", np.nan)
                if not np.isnan(p10) and not np.isnan(p90):
                    result["scenario_range_pct"] = (p90 - p10) / mc_median
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
    """Compute WACC following Damodaran methodology.

    Improvements over naive WACC:
    - Bottom-up beta (sector unlevered, relevered for company D/E)
    - Synthetic cost of debt (interest coverage → default spread + Rf)
    - Marginal tax rate (25% = 21% fed + 4% state) for debt shield
    - Total debt includes short-term + long-term
    - Size premium added to cost of equity (not WACC)
    """
    market_cap = _safe_get(info, "marketCap")
    sector = info.get("sector", "") if info else ""

    # ── Total debt: long-term + short-term ───────────────────────────
    total_debt = _get_total_debt(statements)

    # ── Bottom-up beta: sector unlevered → relever for company D/E ──
    equity_value = market_cap if not np.isnan(market_cap) and market_cap > 0 else 0.0
    beta = _compute_bottom_up_beta(sector, total_debt, equity_value)

    # ── Size premium on cost of equity ───────────────────────────────
    size_premium = 0.0
    if not np.isnan(market_cap):
        if market_cap < 500_000_000:
            size_premium = 0.025
        elif market_cap < 1_000_000_000:
            size_premium = 0.020
        elif market_cap < 3_000_000_000:
            size_premium = 0.010
        elif market_cap < 10_000_000_000:
            size_premium = 0.005

    # Country risk premium (Damodaran: domicile-based equity risk adjustment)
    country = info.get("country", "United States") if info else "United States"
    crp = COUNTRY_RISK_PREMIUM.get(country, 0.01)  # 1% default for unknown

    cost_of_equity = risk_free_rate + beta * equity_risk_premium + size_premium + crp

    # ── Synthetic cost of debt (Damodaran coverage → spread) ─────────
    annual_income = statements.get("annual_income")
    interest_expense = np.nan
    ebit = np.nan

    if annual_income is not None and not annual_income.empty:
        interest_expense = _get_latest_line_item(annual_income, "Interest Expense")
        ebit = _get_latest_line_item(annual_income, "Operating Income")

    cost_of_debt = _synthetic_cost_of_debt(ebit, interest_expense, risk_free_rate)

    # ── Capital structure weights ────────────────────────────────────
    total_capital = equity_value + total_debt
    if total_capital <= 0:
        logger.warning("Total capital is zero; cannot compute WACC")
        return np.nan

    weight_equity = equity_value / total_capital
    weight_debt = total_debt / total_capital

    # Marginal tax rate for WACC debt shield (Damodaran: always marginal)
    wacc = (weight_equity * cost_of_equity
            + weight_debt * cost_of_debt * (1 - US_MARGINAL_TAX_RATE))

    # Clamp to sane range — log warning if inputs are suspect
    # Warning range aligned with clamp range [4%, 30%] to avoid silent clamping
    if wacc < 0.04 or wacc > 0.30:
        logger.warning("WACC %.4f outside normal range [4%%, 30%%] — check inputs "
                       "(beta=%.2f, Kd=%.4f, D/E=%.2f)",
                       wacc, beta, cost_of_debt,
                       total_debt / equity_value if equity_value > 0 else float("inf"))
    wacc = float(np.clip(wacc, 0.04, 0.30))
    return wacc


def compute_roic(
    statements: dict,
    sector: str = "",
    rd_info: dict | None = None,
) -> float:
    """ROIC = NOPAT / Invested Capital.

    NOPAT = Operating Income * (1 - tax_rate)
    Invested Capital = Total Equity + Total Debt - Cash

    When R&D capitalization is active (rd_info provided), NOPAT is adjusted
    by adding rd_adjustment_ebit*(1-tax), and invested_capital gains rd_asset.

    Tax rate note (Damodaran): Uses *effective* tax rate for NOPAT (reflects
    actual taxes paid), while ``compute_wacc`` uses *marginal* tax rate (25%)
    for the debt shield. This is correct — the debt tax shield is valued at the
    marginal rate because each incremental dollar of interest saves taxes at
    that rate, while NOPAT reflects actual cash flows after actual taxes paid.
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

    # R&D capitalization adjustment (Phase 2.2)
    adjusted_ebit = operating_income
    if rd_info is not None and rd_info.get("rd_adjustment_ebit", 0.0) != 0.0:
        adjusted_ebit += rd_info["rd_adjustment_ebit"]

    nopat = adjusted_ebit * (1 - tax_rate)

    # Invested capital: Equity + Total Debt (LT + ST) - Cash  (Damodaran)
    total_equity = _get_balance_sheet_item(statements, "Stockholders Equity")
    total_debt = _get_total_debt(statements)
    cash = _get_balance_sheet_item(statements, "Cash And Cash Equivalents")

    if np.isnan(total_equity):
        logger.warning("Stockholders equity unavailable for ROIC")
        return np.nan

    invested_capital = total_equity + total_debt
    if not np.isnan(cash):
        invested_capital -= cash

    # Add unamortized R&D asset to invested capital (Phase 2.2)
    if rd_info is not None and rd_info.get("rd_asset", 0.0) > 0:
        invested_capital += rd_info["rd_asset"]

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
            # Convex decay: faster fade early, flattens near terminal rate.
            # Damodaran competitive fade — high growth attracts competition,
            # eroding advantages non-linearly (base effect + competitive entry).
            fraction = yr / (years - 1)
            convex_fraction = 1 - (1 - fraction) ** 2
            growth = revenue_growth_rate + convex_fraction * (terminal_growth - revenue_growth_rate)

        current_fcf = current_fcf * (1 + growth)
        current_fcf = min(current_fcf, cap)
        projected.append(current_fcf)

    return projected


def project_fcf_multistage(
    base_fcf: float,
    revenue_growth_rate: float,
    terminal_growth: float = 0.02,
    improving_fcf: bool = False,
    roic_latest: float | None = None,
    wacc: float | None = None,
    sector: str = "",
) -> list[float]:
    """Project FCF using a 3-stage model (Damodaran high-growth firm valuation).

    Stage 1 (High growth): Revenue growth rate with minimal decay
    Stage 2 (Transition): Linear fade from high growth to stable growth
    Stage 3 (Stable): Terminal growth rate

    Total projection is 10 years.
    """
    abs_growth = abs(revenue_growth_rate)
    if abs_growth > 0.20:
        params = GROWTH_STAGE_PARAMS["high"]
    elif abs_growth > 0.10:
        params = GROWTH_STAGE_PARAMS["medium"]
    else:
        params = GROWTH_STAGE_PARAMS["low"]

    high_years = params["high_years"]
    transition_years = params["transition_years"]
    stable_years = params["stable_years"]
    total_years = high_years + transition_years + stable_years

    cap_multiplier = 3.0 if improving_fcf else 2.5
    cap = cap_multiplier * abs(base_fcf)

    projected = []
    current_fcf = base_fcf

    for yr in range(total_years):
        if yr < high_years:
            # Stage 1: High growth with minimal 5% annual decay
            decay = 0.95 ** yr
            growth = revenue_growth_rate * decay
        elif yr < high_years + transition_years:
            # Stage 2: Linear fade from last high-growth rate to terminal
            t_yr = yr - high_years
            t_fraction = t_yr / max(transition_years - 1, 1)
            last_high_growth = revenue_growth_rate * (0.95 ** max(high_years - 1, 0))
            growth = last_high_growth + t_fraction * (terminal_growth - last_high_growth)
        else:
            # Stage 3: Stable growth
            growth = terminal_growth

        # Apply ROIC mean reversion if available (Phase 2.4)
        if roic_latest is not None and wacc is not None:
            growth = _apply_roic_mean_reversion(
                growth, roic_latest, wacc, yr, sector
            )

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
    """Safely get a value from dict, returning default if missing or None.

    Zero is a valid value (e.g. beta=0) and is NOT replaced by default.
    """
    if d is None:
        return default
    val = d.get(key)
    if val is None:
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


def _get_total_debt(statements: dict) -> float:
    """Total debt = Long-term + Short-term (Damodaran: all interest-bearing).

    Priority: prefer "Current Debt" when available (excludes lease obligations).
    Falls back to "Current Debt And Capital Lease Obligation" only when
    "Current Debt" is missing — avoids double-counting lease obligations
    that yfinance sometimes bundles into the combined line item.
    """
    ltd = _get_balance_sheet_item(statements, "Long Term Debt")
    std = _get_balance_sheet_item(statements, "Current Debt")

    total = 0.0
    if not np.isnan(ltd):
        total += ltd
    if not np.isnan(std):
        total += std
    else:
        # Fallback: combined line item (includes capital lease obligations)
        cp = _get_balance_sheet_item(statements, "Current Debt And Capital Lease Obligation")
        if not np.isnan(cp):
            total += cp

    return total


def _compute_bottom_up_beta(
    sector: str, total_debt: float, market_cap_equity: float,
    marginal_tax: float = US_MARGINAL_TAX_RATE,
) -> float:
    """Damodaran bottom-up beta: unlevered sector beta relevered for D/E.

    beta_levered = beta_unlevered × (1 + (1-t) × D/E)
    """
    unlevered = SECTOR_UNLEVERED_BETA.get(sector, 0.85)
    if market_cap_equity <= 0:
        return unlevered  # no leverage info → return unlevered
    de_ratio = total_debt / market_cap_equity
    return unlevered * (1 + (1 - marginal_tax) * de_ratio)


def _synthetic_cost_of_debt(
    ebit: float, interest_expense: float, risk_free_rate: float,
) -> float:
    """Damodaran synthetic rating: interest coverage → default spread + Rf."""
    # Debt-free companies: near risk-free cost of debt
    if np.isnan(interest_expense) or abs(interest_expense) < 1:
        return risk_free_rate + 0.0063  # AAA-equivalent spread

    if np.isnan(ebit):
        return risk_free_rate + 0.0250  # BB-equivalent when EBIT unknown

    coverage = ebit / abs(interest_expense)
    for threshold, spread in COVERAGE_TO_SPREAD:
        if coverage >= threshold:
            return risk_free_rate + spread
    return risk_free_rate + 0.12  # D-rated / distressed


def _compute_fcff(
    statements: dict,
    normalize: bool = False,
    sector: str = "",
    rd_info: dict | None = None,
) -> tuple[float, bool]:
    """Compute FCFF = EBIT(1-t) + D&A - CapEx - ΔWC (Damodaran).

    Returns (fcff, delta_wc_estimated) where delta_wc_estimated is True
    when Change In Working Capital data is missing and defaults to 0.
    For growth companies, missing ΔWC inflates FCFF because working capital
    investment (inventory build, receivables growth) is ignored.

    When normalize=True, uses mid-cycle EBIT (5-year median margin × TTM revenue)
    instead of raw TTM EBIT — appropriate for cyclical sectors.

    When rd_info is provided, applies R&D capitalization adjustment to EBIT.
    """
    quarterly_income = statements.get("quarterly_income")
    quarterly_cf = statements.get("quarterly_cashflow")

    if quarterly_income is None or quarterly_income.empty:
        return np.nan, False
    if quarterly_cf is None or quarterly_cf.empty:
        return np.nan, False

    # Use normalized EBIT for cyclical sectors (Phase 1.3)
    if normalize:
        ebit = _compute_normalized_ebit(statements)
        if np.isnan(ebit):
            # Fallback to raw TTM EBIT if normalization fails
            ebit = _trailing_4q_sum(quarterly_income, "Operating Income")
    else:
        ebit = _trailing_4q_sum(quarterly_income, "Operating Income")

    if np.isnan(ebit):
        return np.nan, False

    # R&D capitalization adjustment (Phase 2.2)
    if rd_info is not None and rd_info.get("rd_adjustment_ebit", 0.0) != 0.0:
        ebit += rd_info["rd_adjustment_ebit"]

    da = _trailing_4q_sum(quarterly_cf, "Depreciation And Amortization")
    capex = _trailing_4q_sum(quarterly_cf, "Capital Expenditure")
    delta_wc = _trailing_4q_sum(quarterly_cf, "Change In Working Capital")

    delta_wc_estimated = False
    if np.isnan(da):
        da = 0.0
    if np.isnan(capex):
        return np.nan, False  # can't compute without capex
    if np.isnan(delta_wc):
        delta_wc = 0.0
        delta_wc_estimated = True

    # FCFF = EBIT(1-t) + D&A + CapEx - ΔWC
    # Note: capex is typically negative in yfinance; delta_wc sign varies
    nopat = ebit * (1 - US_MARGINAL_TAX_RATE)
    fcff = nopat + da + capex - delta_wc

    return fcff, delta_wc_estimated


def _populate_non_dcf_metrics(result: dict, base_fcf: float, market_cap: float,
                               enterprise_value: float) -> None:
    """Fill in FCF-based metrics that don't require full DCF computation."""
    if not np.isnan(base_fcf) and base_fcf > 0:
        if not np.isnan(market_cap) and market_cap > 0:
            result["fcf_yield"] = base_fcf / market_cap
        if not np.isnan(enterprise_value) and enterprise_value > 0:
            result["ev_to_fcf"] = enterprise_value / base_fcf


def _distress_probability(altman_z: float) -> float:
    """Map Altman Z-Score to probability of distress (Damodaran).

    Provides continuous discounting rather than binary gate. Companies
    in the safe zone (Z > 3.0) get 0% distress, while grey zone firms
    get a proportional haircut reflecting survival uncertainty.
    """
    if np.isnan(altman_z):
        return 0.0
    if altman_z > 3.0:
        return 0.0
    if altman_z > 2.7:
        return 0.02
    if altman_z > 1.8:
        return 0.10
    if altman_z > 1.0:
        return 0.25
    return 0.50


def _compute_exit_multiple_tv(
    statements: dict,
    sector: str,
    final_year_fcf: float,
    terminal_growth: float,
) -> float:
    """Cross-check terminal value using sector EV/EBITDA exit multiple.

    Gordon Growth terminal values are extremely sensitive to the WACC-g
    spread. A sector-appropriate exit multiple provides a sanity bound.
    Uses final-year EBITDA (estimated from TTM EBITDA grown at terminal rate
    over the projection horizon) × sector median EV/EBITDA.
    """
    multiple = SECTOR_EXIT_MULTIPLES.get(sector)
    if multiple is None:
        return np.nan

    # Use TTM EBITDA as base, grown at terminal rate to match projection endpoint
    quarterly_income = statements.get("quarterly_income")
    if quarterly_income is None or quarterly_income.empty:
        return np.nan

    if "EBITDA" not in quarterly_income.index:
        return np.nan

    ebitda_row = quarterly_income.loc["EBITDA"]
    values = ebitda_row.iloc[:4].dropna()
    if len(values) < 4:
        return np.nan

    ttm_ebitda = float(values.sum())
    if ttm_ebitda <= 0:
        return np.nan

    return ttm_ebitda * multiple


def _compute_scenarios(
    base_fcf: float,
    wacc: float,
    revenue_growth: float,
    terminal_growth: float,
    projection_years: int,
    net_debt: float,
    shares_outstanding: float,
    improving_fcf: bool,
) -> dict:
    """Compute bear/base/bull DCF scenarios (Damodaran: scenario analysis).

    A single-point DCF creates false precision. Three scenarios reveal
    the value range and which assumptions drive the valuation most.
    """
    result = {"bear_iv": np.nan, "bull_iv": np.nan, "scenario_range_pct": np.nan}

    scenarios = {
        "bear_iv": (max(0.0, revenue_growth * 0.5), wacc + 0.01, max(0.005, terminal_growth - 0.005)),
        "bull_iv": (revenue_growth * 1.3, max(0.04, wacc - 0.005), terminal_growth),
    }

    for key, (g, w, tg) in scenarios.items():
        if w <= tg:
            continue
        projected = project_fcf(base_fcf, g, tg, projection_years, improving_fcf=improving_fcf)
        if not projected:
            continue
        tv = compute_terminal_value(projected[-1], tg, w)
        pv_fcfs = sum(fcf / (1 + w) ** yr for yr, fcf in enumerate(projected, start=1))
        pv_tv = tv / (1 + w) ** projection_years
        ev = pv_fcfs + pv_tv
        equity = ev - net_debt
        if shares_outstanding > 0:
            result[key] = equity / shares_outstanding

    bear = result.get("bear_iv", np.nan)
    bull = result.get("bull_iv", np.nan)
    base = base_fcf  # use base_iv from caller
    if not np.isnan(bear) and not np.isnan(bull) and not np.isnan(bear):
        mid = (bear + bull) / 2.0
        if mid > 0:
            result["scenario_range_pct"] = (bull - bear) / mid

    return result


# ── Phase 1.3: Normalized earnings for cyclical companies ────────────

def _compute_normalized_ebit(statements: dict) -> float:
    """Compute mid-cycle EBIT using 5-year median operating margin applied to TTM revenue."""
    annual_income = statements.get("annual_income")
    if annual_income is None or annual_income.empty:
        return np.nan

    if "Operating Income" not in annual_income.index or "Total Revenue" not in annual_income.index:
        return np.nan

    op_income_row = annual_income.loc["Operating Income"].dropna()
    revenue_row = annual_income.loc["Total Revenue"].dropna()

    # Need at least 3 years for meaningful median
    n = min(len(op_income_row), len(revenue_row))
    if n < 3:
        return np.nan

    margins = []
    for i in range(min(n, 5)):
        rev = float(revenue_row.iloc[i])
        oi = float(op_income_row.iloc[i])
        if rev > 0:
            margins.append(oi / rev)

    if not margins:
        return np.nan

    median_margin = float(np.median(margins))

    # Apply median margin to TTM revenue
    quarterly_income = statements.get("quarterly_income")
    if quarterly_income is not None and not quarterly_income.empty:
        ttm_revenue = _trailing_4q_sum(quarterly_income, "Total Revenue")
        if not np.isnan(ttm_revenue) and ttm_revenue > 0:
            return ttm_revenue * median_margin

    # Fallback to most recent annual revenue
    latest_rev = float(revenue_row.iloc[0])
    if latest_rev > 0:
        return latest_rev * median_margin

    return np.nan


# ── Phase 2.2: R&D capitalization for tech/pharma ────────────────────

def _capitalize_rd(statements: dict, sector: str) -> dict:
    """Capitalize R&D following Damodaran methodology.

    Returns dict with:
    - rd_adjustment_ebit: amount to add to EBIT (current R&D - amortization)
    - rd_asset: unamortized R&D asset to add to invested capital
    - rd_current: current year R&D expense
    """
    result = {"rd_adjustment_ebit": 0.0, "rd_asset": 0.0, "rd_current": 0.0}

    useful_life = RD_USEFUL_LIFE.get(sector)
    if useful_life is None:
        return result

    annual_income = statements.get("annual_income")
    if annual_income is None or annual_income.empty:
        return result

    # Try "Research And Development" or "Research Development" line items
    rd_row_name = None
    for name in ["Research And Development", "Research Development"]:
        if name in annual_income.index:
            rd_row_name = name
            break

    if rd_row_name is None:
        return result

    rd_row = annual_income.loc[rd_row_name].dropna()
    if len(rd_row) == 0:
        return result

    current_rd = abs(float(rd_row.iloc[0]))
    result["rd_current"] = current_rd

    # Amortize past R&D over useful life using straight-line
    rd_values = [abs(float(rd_row.iloc[i])) for i in range(min(len(rd_row), useful_life))]

    total_amortization = 0.0
    unamortized_asset = 0.0

    for age, rd in enumerate(rd_values):
        if age == 0:
            # Current year R&D: add back to EBIT entirely
            continue

        # Fraction still unamortized
        remaining_life = useful_life - age
        if remaining_life > 0:
            annual_amort = rd / useful_life
            total_amortization += annual_amort
            unamortized_asset += rd * (remaining_life / useful_life)

    # EBIT adjustment: +current R&D (add back expense), -amortization
    result["rd_adjustment_ebit"] = current_rd - total_amortization
    result["rd_asset"] = unamortized_asset

    return result


# ── Phase 2.3: Monte Carlo terminal value sensitivity ────────────────

def _monte_carlo_dcf(
    base_fcf: float,
    wacc: float,
    revenue_growth: float,
    terminal_growth: float,
    projection_years: int,
    net_debt: float,
    shares_outstanding: float,
    improving_fcf: bool = False,
    n_simulations: int = 1000,
    roic_latest: float | None = None,
    sector: str = "",
) -> dict:
    """Monte Carlo simulation varying WACC, terminal growth, and revenue growth.

    Returns dict with:
    - mc_median_iv: median intrinsic value
    - mc_p10_iv: 10th percentile (conservative)
    - mc_p90_iv: 90th percentile (optimistic)
    - mc_std_iv: standard deviation of IV estimates
    """
    rng = np.random.default_rng(42)  # reproducible

    # Parameter distributions
    wacc_samples = rng.normal(wacc, 0.01, n_simulations)  # std=1%
    tg_samples = rng.uniform(
        max(0.005, terminal_growth - 0.01),
        min(wacc - 0.01, terminal_growth + 0.01),
        n_simulations,
    )
    # Triangular: bear=0.5x, base=1x, bull=1.3x
    growth_multipliers = rng.triangular(0.5, 1.0, 1.3, n_simulations)

    ivs = []
    for i in range(n_simulations):
        w = float(np.clip(wacc_samples[i], 0.04, 0.30))
        tg = float(tg_samples[i])
        g = revenue_growth * growth_multipliers[i]

        if w <= tg:
            continue

        projected = project_fcf_multistage(
            base_fcf, g, tg, improving_fcf=improving_fcf,
            roic_latest=roic_latest, wacc=w, sector=sector,
        )
        if not projected:
            continue

        n_years = len(projected)
        tv = compute_terminal_value(projected[-1], tg, w)
        pv_fcfs = sum(fcf / (1 + w) ** yr for yr, fcf in enumerate(projected, start=1))
        pv_tv = tv / (1 + w) ** n_years
        ev = pv_fcfs + pv_tv
        equity = ev - net_debt

        if shares_outstanding > 0:
            ivs.append(equity / shares_outstanding)

    if not ivs:
        return {}

    iv_arr = np.array(ivs)
    result = {
        "mc_median_iv": float(np.median(iv_arr)),
        "mc_p10_iv": float(np.percentile(iv_arr, 10)),
        "mc_p90_iv": float(np.percentile(iv_arr, 90)),
        "mc_std_iv": float(np.std(iv_arr)),
    }

    return result


# ── Phase 2.4: ROIC mean reversion ──────────────────────────────────

def _apply_roic_mean_reversion(
    growth_rate: float,
    roic_latest: float,
    wacc: float,
    year: int,
    sector: str,
) -> float:
    """Fade growth rate as ROIC converges toward WACC+2% over time.

    High ROIC attracts competition; the reinvestment rate should decrease
    as excess returns decay. Growth = reinvestment_rate * ROIC, so as ROIC
    fades, achievable growth fades proportionally.
    """
    target_roic = wacc + 0.02  # long-run sustainable ROIC
    if roic_latest <= target_roic:
        return growth_rate  # already at or below equilibrium

    halflife = ROIC_HALFLIFE.get(sector, 7)
    decay = 0.5 ** (year / halflife)

    # ROIC at this projection year
    projected_roic = target_roic + (roic_latest - target_roic) * decay

    # Scale growth rate by ratio of projected vs current ROIC
    if roic_latest > 0:
        roic_ratio = projected_roic / roic_latest
        return growth_rate * roic_ratio

    return growth_rate
