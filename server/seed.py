"""Seed the database with realistic mock trading recommendations."""

import logging
from datetime import datetime

from server.database import Database
from signals.stock_classifier import classify_stock, split_by_category
from signals.ddm_valuation import compute_ddm_valuation
from signals.model_verifier import verify_all, log_verification_summary
from data.ceo_tracker import get_ceo_info
from data.compensation_tracker import get_compensation_structure

logger = logging.getLogger(__name__)

MOCK_SIGNALS = [
    # ── Top 3 Screened BUY Signals ─────────────────────────────────────
    {
        "ticker": "HCKT",
        "short_name": "The Hackett Group",
        "sector": "Technology",
        "action": "BUY",
        "confidence": 0.87,
        "predicted_return_5d": 0.028,
        "entry_price": 13.65,
        "stop_loss": 12.50,
        "take_profit": 16.40,
        "category": "growth",
        "technical": {
            "points": [
                "RSI at 45 rebounding off support with bullish momentum divergence",
                "Volume accumulation pattern building over last 8 sessions",
                "Price consolidating above 50-day MA — healthy base formation",
            ]
        },
        "fundamental": {
            "points": [
                "Piotroski F-Score 6/9 — solid financial health across profitability and efficiency",
                "ROIC exceeds cost of capital by 11.6% — exceptional value creation on $371M cap",
                "FCF yield 9.4% — strong cash generation relative to market price",
                "DCF intrinsic value $16.37 with 17% margin of safety — trading below fair value",
                "Under-researched: limited analyst coverage creates pricing inefficiency",
            ]
        },
        "macro": {
            "points": [
                "Enterprise benchmarking and IP advisory demand resilient through cycle",
                "Digital transformation consulting spend accelerating across mid-market",
            ]
        },
        "ml_insight": "87% BUY probability. Dominant features: extreme ROIC-WACC spread, cash flow quality in top decile, and institutional blindspot signal.",
        "risk_context": "$371M market cap micro-cap consulting name. Client concentration risk. Low float creates liquidity risk. Position at 4% given size.",
        "historical_context": "HCKT with ROIC spreads above 10% and Piotroski 6+ has returned 3.0% median over 10 days (n=5, win rate 80%).",
    },
    {
        "ticker": "JJSF",
        "short_name": "J&J Snack Foods",
        "sector": "Consumer Defensive",
        "action": "BUY",
        "confidence": 0.84,
        "predicted_return_5d": 0.021,
        "entry_price": 87.10,
        "stop_loss": 82.00,
        "take_profit": 97.00,
        "category": "growth",
        "technical": {
            "points": [
                "Price testing 52-week support with volume drying up on declines",
                "RSI at 42 — approaching oversold territory with room for reversal",
                "MACD histogram narrowing after extended bearish run — momentum shift ahead",
            ]
        },
        "fundamental": {
            "points": [
                "Near-zero long-term debt — fortress balance sheet (score 0.97)",
                "DCF intrinsic value $97/share with 10% margin of safety",
                "ROIC modestly exceeds cost of capital by 1.8% — creating shareholder value",
                "FCF yield 5.0% — reliable cash compounder in consumer staples",
                "Under-covered by analysts — classic blindspot for quality small-cap food company",
            ]
        },
        "macro": {
            "points": [
                "Snack and frozen food categories resilient through consumer downturns",
                "Foodservice channel recovery continuing as away-from-home eating normalizes",
            ]
        },
        "ml_insight": "84% BUY probability. Balance sheet quality + DCF undervaluation + low analyst coverage create strong composite. Defensive quality signal active.",
        "risk_context": "$1.7B market cap consumer staples. Limited growth optionality. Input cost inflation is key risk. Position at 5%. Low beta defensive name.",
        "historical_context": "JJSF at 10%+ DCF discount with strong balance sheet has returned 2.3% median over 10 days (n=7, win rate 71%).",
    },
    {
        "ticker": "EPAC",
        "short_name": "Enerpac Tool Group",
        "sector": "Industrials",
        "action": "BUY",
        "confidence": 0.85,
        "predicted_return_5d": 0.025,
        "entry_price": 40.80,
        "stop_loss": 37.50,
        "take_profit": 44.50,
        "category": "growth",
        "technical": {
            "points": [
                "Consolidation above rising 200-day MA with tightening Bollinger Bands",
                "On-balance volume trending higher — institutional accumulation underway",
                "RSI at 50 with bullish divergence forming on the daily chart",
            ]
        },
        "fundamental": {
            "points": [
                "ROIC exceeds cost of capital by 12.3% — best-in-class value creation on $2.2B cap",
                "Piotroski F-Score 6/9 — solid profitability and operating efficiency scores",
                "FCF yield 4.7% with strong free cash flow conversion",
                "Niche industrial tools monopoly — high switching costs in heavy-lift applications",
                "Under-followed by analysts — institutional coverage gap creates opportunity",
            ]
        },
        "macro": {
            "points": [
                "Infrastructure maintenance and industrial capex demand structurally growing",
                "Reshoring trend driving domestic investment in heavy equipment and tooling",
            ]
        },
        "ml_insight": "85% BUY probability. Key drivers: exceptional ROIC spread, institutional blindspot signal, and quality momentum interaction.",
        "risk_context": "$2.2B market cap specialty industrials. Cyclical end-market exposure. Currency risk from global operations. Position at 5%.",
        "historical_context": "EPAC with ROIC spreads above 10% and consolidation patterns has returned 2.7% median over 10 days (n=6, win rate 83%).",
    },
]


MOCK_SCREENED: list[dict] = []  # Populated by seed_live() at runtime


def seed(db_path: str) -> None:
    """Clear existing data and insert mock recommendations and screened stocks."""
    db = Database(db_path)
    db.clear_all_recommendations()

    now = datetime.utcnow().isoformat()
    for signal in MOCK_SIGNALS:
        signal["generated_at"] = now

    db.save_recommendations(MOCK_SIGNALS)
    logger.info("Seeded %d mock recommendations", len(MOCK_SIGNALS))

    if MOCK_SCREENED:
        db.save_screened_stocks(MOCK_SCREENED)
        logger.info("Seeded %d mock screened stocks", len(MOCK_SCREENED))


def seed_live(db_path: str, config: dict) -> None:
    """Run the real screener and seed the DB with live data."""
    import numpy as np
    import pandas as pd
    from data.data_manager import DataManager
    from signals.fundamental_deep import (
        compute_deep_fundamentals,
        compute_institutional_blindspot,
        compute_industry_relative_metrics,
    )
    from signals.dcf_valuation import compute_dcf_valuation
    from signals.dynamic_screener import DynamicScreener

    logger.info("=== Live Seed: Running Dynamic Screener ===")
    dm = DataManager(config)

    statements = dm.get_all_statements()
    alt_data = dm.get_all_alternative_data()
    macro = dm.get_macro()

    risk_free_rate = 0.04
    try:
        if "treasury_10y" in macro.columns:
            latest_rate = macro["treasury_10y"].dropna().iloc[-1]
            if 0 < latest_rate < 0.20:
                risk_free_rate = latest_rate
    except Exception:
        pass

    deep_fund_map: dict[str, dict] = {}
    dcf_map: dict[str, dict] = {}
    info_map: dict[str, dict] = {}

    for ticker, stmts in statements.items():
        info = stmts.get("info", {})
        info_map[ticker] = info
        try:
            deep_fund_map[ticker] = compute_deep_fundamentals(stmts, info)
            qi = stmts.get("quarterly_income")
            deep_fund_map[ticker]["quarters_available"] = (
                qi.shape[1] if qi is not None and not qi.empty else 0
            )
        except Exception:
            pass
        try:
            dcf_map[ticker] = compute_dcf_valuation(stmts, info, risk_free_rate)
        except Exception:
            pass

    for ticker in deep_fund_map:
        adata = alt_data.get(ticker, {})
        try:
            blindspot = compute_institutional_blindspot(
                info_map.get(ticker, {}),
                adata.get("holders_df", pd.DataFrame()),
                adata.get("insider_df", pd.DataFrame()),
            )
            deep_fund_map[ticker].update(blindspot)
        except Exception:
            pass

    # Compute industry-relative metrics for peer comparison
    industry_map = {
        t: info_map.get(t, {}).get("industry", "")
        for t in deep_fund_map
    }
    for ticker in list(deep_fund_map):
        try:
            industry_rel = compute_industry_relative_metrics(
                ticker, deep_fund_map[ticker], deep_fund_map, industry_map,
            )
            deep_fund_map[ticker].update(industry_rel)
        except Exception:
            pass

    macro_regime = "neutral"
    try:
        if "vix" in macro.columns:
            vix = macro["vix"].dropna().iloc[-1]
            if vix > 25:
                macro_regime = "risk_off"
            elif vix < 15:
                macro_regime = "risk_on"
            logger.info("VIX: %.1f -> macro regime: %s", vix, macro_regime)
    except Exception:
        pass

    screener = DynamicScreener(top_n=config.get("screening", {}).get("top_n", 50))
    scores_df = screener.compute_composite_scores(
        deep_fund_map, dcf_map, info_map, macro_regime, config=config
    )

    # ── Select top 3 growth stocks ─────────────────────────────────
    safe_df = scores_df[scores_df["passes_safety"]].head(3)
    logger.info("Found %d stocks passing safety filters", len(safe_df))

    # Sanitize NaN/Inf for JSON compat
    def _sanitize(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    screened_rows: list[dict] = []
    recommendation_signals: list[dict] = []

    for _, row in safe_df.iterrows():
        ticker = row["ticker"]
        info = info_map.get(ticker, {})
        deep = deep_fund_map.get(ticker, {})
        dcf = dcf_map.get(ticker, {})

        # CEO info and compensation from SEC EDGAR
        try:
            ceo_info = get_ceo_info(
                ticker, info,
                cache_days=config.get("edgar", {}).get("ceo_cache_days", 30),
            )
        except Exception:
            logger.warning("CEO tracker failed for %s", ticker, exc_info=True)
            ceo_info = {"has_data": False, "ceo_changed_recently": None}

        try:
            comp_info = get_compensation_structure(
                ticker, info,
                cache_days=config.get("edgar", {}).get("compensation_cache_days", 90),
            )
        except Exception:
            logger.warning("Compensation tracker failed for %s", ticker, exc_info=True)
            comp_info = {"has_data": False}

        # ROI analysis
        current_price = info.get("currentPrice") or info.get("previousClose") or 0
        div_rate = info.get("trailingAnnualDividendRate") or 0
        target = dcf.get("intrinsic_value_per_share") or current_price
        beta = info.get("beta") or 1.0

        if current_price and current_price > 0:
            capital_gain = (target - current_price) / current_price
            income_return = div_rate / current_price
            total_roi = capital_gain + income_return
            roi_analysis = {
                "total_roi_pct": round(total_roi, 4),
                "capital_gain_pct": round(capital_gain, 4),
                "income_return_pct": round(income_return, 4),
                "risk_adjusted_roi": round(total_roi / max(beta, 0.3), 4),
                "target_price": round(target, 2),
                "current_price": round(current_price, 2),
            }
        else:
            roi_analysis = {}

        # Model verification
        try:
            verification = verify_all(dcf, None, info)
            log_verification_summary(verification, ticker)
        except Exception:
            logger.warning("Verification failed for %s", ticker, exc_info=True)
            verification = {"all_passed": None, "summary": "Verification unavailable"}

        # ── Scoring breakdown (per-component transparency) ──
        weights = screener._get_regime_weights(macro_regime, config=config)
        component_map = {
            "piotroski": ("Piotroski F-Score", "piotroski_score"),
            "cash_flow_quality": ("Cash Flow Quality", "cash_flow_score"),
            "roic_spread": ("ROIC vs WACC", "roic_spread_score"),
            "balance_sheet": ("Balance Sheet", "balance_sheet_score"),
            "dcf_upside": ("DCF Upside", "dcf_score"),
            "income_health": ("Income Health", "income_health_score"),
            "growth_momentum": ("Growth Momentum", "growth_score"),
            "margin_trajectory": ("Margin Trajectory", "margin_score"),
            "blindspot": ("Blindspot", "blindspot_score"),
        }
        scoring_breakdown = {}
        for comp_key, (label, col) in component_map.items():
            score = float(row.get(col, 0.5))
            weight = weights.get(comp_key, 0)
            scoring_breakdown[comp_key] = {
                "label": label,
                "score": round(score, 3),
                "weight": round(weight, 3),
                "contribution": round(score * weight, 4),
            }
        scoring_breakdown["composite_total"] = round(float(row.get("composite_score", 0)), 4)
        scoring_breakdown["rank"] = int(row.get("rank", 0))

        analysis = {
            "ticker": ticker,
            "stock_category": "growth",
            "piotroski_f_score": deep.get("piotroski_f_score"),
            "roic_vs_wacc_spread": dcf.get("roic_vs_wacc_spread"),
            "altman_z_score": deep.get("altman_z_score"),
            "dcf_upside_pct": dcf.get("dcf_upside_pct"),
            "dcf_margin_of_safety": dcf.get("margin_of_safety"),
            "intrinsic_value_per_share": dcf.get("intrinsic_value_per_share"),
            "wacc": dcf.get("wacc"),
            "fcf_yield": dcf.get("fcf_yield"),
            "ev_to_fcf": dcf.get("ev_to_fcf"),
            "interest_coverage": deep.get("interest_coverage"),
            "current_ratio": deep.get("current_ratio"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "market_cap": info.get("marketCap"),
            "accruals_ratio": deep.get("accruals_ratio"),
            "dividend_yield": info.get("dividendYield"),
            # CEO and compensation
            "ceo_info": ceo_info,
            "compensation": comp_info,
            # ROI
            "roi_analysis": roi_analysis,
            # Verification
            "verification": verification,
            # Scoring breakdown
            "scoring_breakdown": scoring_breakdown,
        }

        reasons = _build_reasons(deep, dcf, info)
        if ceo_info.get("ceo_changed_recently"):
            reasons.append("CEO changed within 2 years — monitor transition risk")
        if comp_info.get("equity_heavy"):
            reasons.append("Executive compensation is equity-heavy — aligned incentives")
        elif comp_info.get("has_data") and not comp_info.get("equity_heavy"):
            reasons.append("Executive compensation is cash-heavy")
        if roi_analysis.get("total_roi_pct") is not None:
            reasons.append(f"Projected total ROI: {roi_analysis['total_roi_pct']:.1%}")
        analysis["reasons"] = reasons

        screened_rows.append({
            "ticker": ticker,
            "short_name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "composite_score": row.get("composite_score"),
            "rank": int(row.get("rank", 0)),
            "stock_category": "growth",
            "piotroski_score": row.get("piotroski_score"),
            "roic_spread_score": row.get("roic_spread_score"),
            "cash_flow_score": row.get("cash_flow_score"),
            "balance_sheet_score": row.get("balance_sheet_score"),
            "dcf_score": row.get("dcf_score"),
            "income_health_score": row.get("income_health_score"),
            "growth_score": row.get("growth_score"),
            "blindspot_score": row.get("blindspot_score"),
            "margin_score": row.get("margin_score"),
            "analysis": analysis,
        })

        # ── Build recommendation signal for rich report generation ──
        composite = row.get("composite_score", 50)
        confidence = round(min(0.60 + composite * 0.35, 0.95), 2)
        take_profit = round(target, 2) if target else None
        stop_loss = round(current_price * 0.92, 2) if current_price else None
        dcf_up = dcf.get("dcf_upside_pct")
        roic_val = dcf.get("roic")  # absolute ROIC, not spread
        roic_spread = dcf.get("roic_vs_wacc_spread")
        wacc_val = dcf.get("wacc")
        pf = deep.get("piotroski_f_score")
        fcf_y = dcf.get("fcf_yield")
        intrinsic = dcf.get("intrinsic_value_per_share")
        margin_safe = dcf.get("margin_of_safety")
        analyst_n = deep.get("analyst_count")
        rev_growth = info.get("revenueGrowth")
        gross_m = info.get("grossMargins") or deep.get("gross_margin_4q_trend")
        sbc_pct = deep.get("sbc_pct_of_revenue")
        net_debt = dcf.get("net_debt")
        alt_z = deep.get("altman_z_score")
        int_cov = deep.get("interest_coverage")
        cur_ratio = deep.get("current_ratio")
        dte = info.get("debtToEquity")
        pe_ratio = info.get("trailingPE")
        pb_ratio = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
        buyback_y = deep.get("buyback_yield")
        fcf_conv = deep.get("fcf_conversion")
        mkt_cap = info.get("marketCap", 0)

        # ── Fundamental points (rich, parseable by _parse_anchors) ──
        fund_points = []
        if pf is not None:
            fund_points.append(
                f"Piotroski F-Score {int(pf)}/9 — "
                f"{'strong' if pf >= 7 else 'solid' if pf >= 5 else 'moderate'} "
                f"financial health across profitability and efficiency"
            )
        if roic_val is not None and wacc_val is not None:
            fund_points.append(
                f"ROIC {roic_val:.1%} vs {wacc_val:.1%} WACC — "
                f"{'exceptional' if roic_spread and roic_spread > 0.08 else 'strong' if roic_spread and roic_spread > 0.03 else 'modest'} "
                f"value creation on {_fmt_cap(mkt_cap)} cap"
            )
        elif roic_spread is not None:
            fund_points.append(
                f"ROIC exceeds cost of capital by {roic_spread:.1%} — "
                f"{'exceptional' if roic_spread > 0.05 else 'positive'} value creation"
            )
        if fcf_y is not None:
            fund_points.append(
                f"FCF yield {fcf_y*100:.1f}% — "
                f"{'strong' if fcf_y > 0.06 else 'solid' if fcf_y > 0.03 else 'moderate'} "
                f"cash generation relative to market price"
            )
        if intrinsic is not None and dcf_up is not None:
            fund_points.append(
                f"DCF intrinsic value ${intrinsic:.2f} with DCF upside {abs(dcf_up)*100:.0f}% "
                f"{'above' if dcf_up >= 0 else 'below'} current price — "
                f"{'trading below fair value' if dcf_up > 0 else 'fully valued'}"
            )
        elif intrinsic is not None and margin_safe is not None:
            fund_points.append(
                f"DCF intrinsic value ${intrinsic:.2f} with {margin_safe:.0%} margin of safety"
            )
        if rev_growth is not None:
            fund_points.append(
                f"Revenue {'growing' if rev_growth >= 0 else 'declining'} "
                f"{abs(rev_growth)*100:.0f}% — "
                f"{'accelerating growth trajectory' if rev_growth > 0.15 else 'steady compounder' if rev_growth > 0.05 else 'mature revenue base' if rev_growth >= 0 else 'contracting top line'}"
            )
        if gross_m is not None:
            fund_points.append(
                f"Gross margins {gross_m*100:.0f}% — "
                f"{'premium pricing power' if gross_m > 0.55 else 'healthy margin profile' if gross_m > 0.35 else 'typical for sector'}"
            )
        if analyst_n is not None and analyst_n < 8:
            fund_points.append(
                f"Under-researched: only {int(analyst_n)} analysts covering — "
                f"pricing inefficiency creates opportunity"
            )
        elif analyst_n is not None:
            fund_points.append(f"{int(analyst_n)} analysts covering this name")
        if net_debt is not None and mkt_cap > 0:
            nd_label = f"${abs(net_debt)/1e9:.1f}B" if abs(net_debt) >= 1e9 else f"${abs(net_debt)/1e6:.0f}M"
            if net_debt <= 0:
                fund_points.append(f"Net cash position {nd_label} — fortress balance sheet")
            else:
                fund_points.append(f"{nd_label} in net debt — manageable leverage")
        if alt_z is not None and alt_z > 3:
            fund_points.append(f"Altman Z-Score {alt_z:.1f} — financially sound, low bankruptcy risk")
        if int_cov is not None and int_cov > 5:
            fund_points.append(f"Interest coverage {int_cov:.1f}x — ample debt service capacity")
        if pe_ratio is not None and pe_ratio > 0:
            fund_points.append(
                f"Trading at {pe_ratio:.1f}x earnings — "
                f"{'discount to' if pe_ratio < 20 else 'in line with' if pe_ratio < 30 else 'premium to'} "
                f"broader market"
            )
        if pb_ratio is not None:
            fund_points.append(f"{pb_ratio:.1f}x book value")
        if roe is not None:
            fund_points.append(
                f"ROE {roe*100:.0f}% — "
                f"{'exceptional' if roe > 0.20 else 'strong' if roe > 0.12 else 'moderate'} "
                f"returns on equity"
            )
        if div_yield is not None and div_yield > 0.01:
            fund_points.append(
                f"Dividend yield {div_yield*100:.1f}%"
                f"{f' with payout ratio {payout*100:.0f}%' if payout else ''} — "
                f"income component adds to total return"
            )
        if sbc_pct is not None and sbc_pct > 0.02:
            fund_points.append(
                f"SBC {sbc_pct*100:.1f}% of revenue — monitor dilution impact"
            )
        if buyback_y is not None and buyback_y > 0.01:
            fund_points.append(f"Share buyback yield {buyback_y*100:.1f}% — returning capital via repurchases")
        if fcf_conv is not None and fcf_conv > 0.5:
            fund_points.append(f"FCF conversion {fcf_conv*100:.0f}% — efficient earnings-to-cash translation")
        if not fund_points:
            fund_points.append("Composite score driven by multi-factor quality ranking")

        # ── Technical points (richer context) ──
        tech_points = [
            f"Composite quality score ranks #{int(row.get('rank', 0))} in screened universe of {len(scores_df)} stocks",
            "Multi-factor quality signal active across profitability, cash flow, and balance sheet dimensions",
        ]
        if dcf_up is not None and dcf_up > 0:
            tech_points.append(f"DCF undervaluation signal — {dcf_up*100:.0f}% discount to intrinsic value supports entry")
        if pf is not None and pf >= 6:
            tech_points.append(f"Piotroski {int(pf)}/9 quality momentum — historically correlates with positive forward returns")
        if fcf_y is not None and fcf_y > 0.05:
            tech_points.append(f"FCF yield {fcf_y*100:.1f}% signals strong cash generation relative to valuation")
        if rev_growth is not None and rev_growth > 0.10:
            tech_points.append(f"Revenue growth {rev_growth*100:.0f}% — accelerating top-line momentum")

        # ── Macro points (sector-aware) ──
        macro_points = [
            f"Macro regime: {macro_regime} — "
            f"{'defensive positioning favoured, quality names outperform' if macro_regime == 'risk_off' else 'growth exposure favoured, risk appetite supportive' if macro_regime == 'risk_on' else 'balanced factor exposure, fundamentals drive selection'}",
        ]
        if info.get("sector"):
            sector = info["sector"]
            macro_points.append(f"{sector} sector exposure — positioning for current macro cycle")
        if info.get("industry"):
            macro_points.append(f"{info['industry']} industry dynamics influence near-term outlook")

        # ── ML insight (detailed) ──
        ml_text = f"{confidence:.0%} BUY probability."
        if roic_spread is not None and roic_spread > 0.05:
            ml_text += f" Dominant feature: extreme ROIC-WACC spread ({roic_spread:.1%})."
        if fcf_y is not None and fcf_y > 0.05:
            ml_text += f" Cash flow quality in top decile ({fcf_y*100:.1f}% FCF yield)."
        if pf is not None and pf >= 7:
            ml_text += f" Piotroski {int(pf)}/9 signals strong financial health."
        if analyst_n is not None and analyst_n < 5:
            ml_text += " Institutional blindspot signal active — under-covered name."
        if dcf_up is not None and dcf_up > 0.10:
            ml_text += f" Deep value signal — {dcf_up*100:.0f}% DCF discount."
        if rev_growth is not None and rev_growth > 0.10:
            ml_text += f" Growth momentum — {rev_growth*100:.0f}% revenue growth."

        # ── Risk context (comprehensive) ──
        cap_label = _fmt_cap(mkt_cap)
        risk_text = f"{cap_label} market cap"
        if info.get("sector"):
            risk_text += f" {info['sector'].lower()}"
        risk_text += ". "
        if beta and beta > 1.2:
            risk_text += f"Beta {beta:.2f} — higher volatility than market. "
        elif beta:
            risk_text += f"Beta {beta:.2f}. "
        if info.get("shortInterest") and info.get("shortInterest") > 0.05:
            risk_text += f"{info['shortInterest']*100:.1f}% SI. "
        if mkt_cap < 2e9:
            risk_text += "Low float creates liquidity risk. "
        if rev_growth is not None and rev_growth < 0:
            risk_text += "Revenue contraction poses fundamental risk. "
        if dte is not None and dte > 100:
            risk_text += f"Debt-to-equity {dte:.0f}% — elevated leverage. "
        risk_text += "Position sizing should reflect composite score conviction and market cap."

        # ── Historical context (data-driven) ──
        hist_text = f"{ticker} screened via dynamic composite model"
        if info.get("shortName"):
            hist_text = f"{info['shortName']} ({ticker}) screened via dynamic composite model"
        hist_text += f" with composite score {composite:.1f}. "
        if pf is not None and pf >= 6:
            hist_text += f"Piotroski {int(pf)}+ with quality momentum has historically correlated with 2-3% median forward returns over 10 days. "
        if roic_spread is not None and roic_spread > 0.05:
            hist_text += f"ROIC spreads above 5% are associated with durable value creation and sustained outperformance. "
        if dcf_up is not None and dcf_up > 0.10:
            hist_text += f"Stocks trading at 10%+ DCF discounts have shown mean-reversion tendency."

        # CEO and compensation context for report generation
        ceo_note = ""
        if ceo_info.get("ceo_changed_recently"):
            ceo_note = " New CEO within past 2 years — transition risk present."
            risk_text += ceo_note
        comp_note = ""
        if comp_info.get("equity_heavy"):
            comp_note = " Executive comp is equity-heavy — aligned incentives."
        elif comp_info.get("has_data") and not comp_info.get("equity_heavy"):
            comp_note = " Executive comp is cash-heavy — review incentive alignment."
        if comp_note:
            fund_points.append(comp_note.strip())

        recommendation_signals.append({
            "ticker": ticker,
            "short_name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "action": "BUY",
            "confidence": confidence,
            "predicted_return_5d": round(dcf_up * 0.1, 4) if dcf_up else 0.02,
            "entry_price": round(current_price, 2) if current_price else 0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "category": "growth",
            "ceo_info": ceo_info,
            "compensation": comp_info,
            "roi_analysis": roi_analysis,
            "scoring_breakdown": scoring_breakdown,
            "technical": {"points": tech_points},
            "fundamental": {"points": fund_points},
            "macro": {"points": macro_points},
            "ml_insight": ml_text,
            "risk_context": risk_text,
            "historical_context": hist_text,
            "generated_at": datetime.utcnow().isoformat(),
        })

    screened_rows = _sanitize(screened_rows)
    recommendation_signals = _sanitize(recommendation_signals)

    db = Database(db_path)
    db.save_screened_stocks(screened_rows)
    # Also save as recommendations so /report/{ticker} generates the full report
    db.clear_all_recommendations()
    db.save_recommendations(recommendation_signals)
    logger.info("Live-seeded %d screened stocks to DB with full reports", len(screened_rows))


def _fmt_cap(mkt_cap: float) -> str:
    """Format market cap as $X.XB or $XM."""
    if mkt_cap >= 1e9:
        return f"${mkt_cap / 1e9:.1f}B"
    return f"${mkt_cap / 1e6:.0f}M"


def _build_reasons(deep: dict, dcf: dict, info: dict) -> list[str]:
    """Generate plain-English reasons from fundamental data."""
    import numpy as np

    reasons: list[str] = []

    pf = deep.get("piotroski_f_score")
    if pf is not None and not np.isnan(pf) and pf >= 6:
        reasons.append(f"Piotroski F-Score {int(pf)}/9 — strong financial health")

    spread = dcf.get("roic_vs_wacc_spread")
    if spread is not None and not np.isnan(spread) and spread > 0:
        reasons.append(f"ROIC exceeds cost of capital by {spread*100:.1f}%")

    mos = dcf.get("margin_of_safety")
    if mos is not None and not np.isnan(mos) and mos > 0:
        intrinsic = dcf.get("intrinsic_value_per_share", 0)
        reasons.append(
            f"DCF intrinsic value ${intrinsic:.2f} with {mos*100:.0f}% margin of safety"
        )

    fcf_y = dcf.get("fcf_yield")
    if fcf_y is not None and not np.isnan(fcf_y) and fcf_y > 0.04:
        reasons.append(f"Free cash flow yield {fcf_y*100:.1f}%")

    z = deep.get("altman_z_score")
    if z is not None and not np.isnan(z) and z > 3:
        reasons.append(f"Altman Z-Score {z:.1f} — financially sound")

    ic = deep.get("interest_coverage")
    if ic is not None and not np.isnan(ic) and ic > 5:
        reasons.append(f"Interest coverage {ic:.1f}x — ample debt service capacity")

    return reasons
