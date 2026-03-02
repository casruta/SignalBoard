"""Daily pipeline scheduler — runs the ML pipeline after market close."""

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config_loader import get_config
from data.data_manager import DataManager
from signals.combiner import SignalCombiner
from signals.dynamic_screener import DynamicScreener
from signals.dcf_valuation import compute_dcf_valuation
from signals.fundamental_deep import compute_deep_fundamentals, compute_institutional_blindspot
from models.features import add_target, prepare_train_data
from models.predict import predict_latest
from models.registry import ModelRegistry
from explainability.decomposer import compute_shap_values, decompose_by_category, top_contributing_features
from explainability.narrator import build_explanation
from explainability.historical import find_similar_signals
from explainability.schema import (
    Recommendation, ExplanationSection, MLInsight, RiskContext,
)
from server.database import Database
from server.push import send_push_notifications

logger = logging.getLogger(__name__)


async def run_daily_pipeline(config: dict | None = None) -> list[dict]:
    """Execute the full daily signal generation pipeline.

    Steps:
    1. Fetch latest data
    2. Compute signals
    3. Run ML predictions
    4. Generate explainability
    5. Store in database
    6. Send push notifications
    """
    config = config or get_config()
    db = Database(config["server"]["database_path"])

    logger.info("Starting daily pipeline at %s", datetime.now().isoformat())

    # ── 1. Fetch data ────────────────────────────────────────────
    logger.info("Step 1: Fetching data...")
    dm = DataManager(config)
    prices = dm.get_all_prices()
    fundamentals = dm.get_all_fundamentals()
    macro = dm.get_macro()

    if not prices:
        logger.error("No price data fetched, aborting pipeline")
        return []

    # ── 2. Compute signals ───────────────────────────────────────
    logger.info("Step 2: Computing signals...")
    combiner = SignalCombiner()
    feature_matrix = combiner.build_feature_matrix(prices, fundamentals, macro)
    feature_matrix = add_target(feature_matrix, prices, horizon_days=5)

    # ── 3. Run predictions ───────────────────────────────────────
    logger.info("Step 3: Running ML predictions...")
    registry = ModelRegistry()
    try:
        model = registry.load()
    except FileNotFoundError:
        logger.error("No trained model found. Run training first.")
        return []

    predictions = predict_latest(model, feature_matrix)

    # Filter to actionable signals only
    min_conf = config["strategy"]["min_confidence_threshold"]
    signals = [p for p in predictions if p.action != "HOLD" and p.confidence >= min_conf]

    logger.info("Found %d actionable signals", len(signals))

    # ── 4. Generate explainability ───────────────────────────────
    logger.info("Step 4: Generating explanations...")
    recommendations = []

    # Get latest date features for SHAP
    dates = feature_matrix.index.get_level_values("date")
    latest_date = dates.max()
    latest_features = feature_matrix.loc[latest_date]

    target_cols = ["target_return", "target_class"]
    feature_cols = [
        c for c in latest_features.columns
        if c not in target_cols
        and latest_features[c].dtype in [np.float64, np.int64, float, int]
    ]
    X_latest = latest_features[feature_cols].fillna(latest_features[feature_cols].median())

    # Compute SHAP for all latest predictions
    shap_df = compute_shap_values(model, X_latest)

    failed_tickers = []
    for pred in signals:
        ticker = pred.ticker
        if ticker not in X_latest.index:
            continue

        try:
            # Get current price
            current_price = float(prices[ticker]["Close"].iloc[-1])
            atr = _get_latest_atr(prices, ticker)

            # Stop/target levels
            stop_pct = config["strategy"]["stop_loss_pct"] / 100
            tp_pct = config["strategy"]["take_profit_pct"] / 100
            trail_pct = config["strategy"]["trailing_stop_trigger_pct"] / 100

            stop_loss = current_price * (1 - stop_pct)
            take_profit = current_price * (1 + tp_pct)
            trail_trigger = current_price * (1 + trail_pct)

            # SHAP decomposition
            categories = decompose_by_category(shap_df, ticker)
            feature_values = X_latest.loc[ticker].to_dict()
            fund_dict = fundamentals.loc[ticker].to_dict() if ticker in fundamentals.index else {}

            explanation = build_explanation(categories, feature_values, fund_dict)
            top_feats = top_contributing_features(shap_df, ticker, n=3)

            # Historical context
            if ticker in X_latest.index:
                hist = find_similar_signals(
                    feature_matrix,
                    X_latest.loc[ticker],
                    [f for f, _ in top_feats],
                    prices,
                    ticker,
                )
            else:
                hist = {"summary": "No historical data available"}

            # Sector info
            sector = str(fund_dict.get("sector", "Unknown"))
            short_name = str(fund_dict.get("short_name", ticker))

            rec = Recommendation(
                ticker=ticker,
                action=pred.action,
                confidence=pred.confidence,
                predicted_return_5d=pred.probabilities.get("BUY", 0) - pred.probabilities.get("SELL", 0),
                entry_price=current_price,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                trailing_stop_trigger=round(trail_trigger, 2),
                time_stop_days=config["strategy"]["time_stop_days"],
                position_size_pct=0.0,  # Calculated by portfolio constructor
                sector=sector,
                short_name=short_name,
                technical=ExplanationSection(explanation.get("technical", [])),
                fundamental=ExplanationSection(explanation.get("fundamental", [])),
                macro=ExplanationSection(explanation.get("macro", [])),
                ml_insight=MLInsight(
                    predicted_return=f"{pred.probabilities.get('BUY', 0) - pred.probabilities.get('SELL', 0):+.1%}",
                    confidence_percentile=f"Top {(1 - pred.confidence) * 100:.0f}%",
                    top_features=[f for f, _ in top_feats],
                ),
                historical_context=[hist.get("summary", "")],
            )
            recommendations.append(rec.to_dict())
        except Exception as e:
            failed_tickers.append(ticker)
            logger.error("Signal computation failed for %s: %s", ticker, e)

    succeeded = len(signals) - len(failed_tickers)
    logger.info(
        "Signal computation complete: %d/%d tickers succeeded",
        succeeded,
        len(signals),
    )

    # ── 4b. Run dynamic screener ─────────────────────────────────
    logger.info("Step 4b: Running dynamic screener...")
    screened_results = _run_screener(config, dm, prices, fundamentals, macro)
    if screened_results:
        logger.info("Screener found %d quality stocks", len(screened_results))
        db.save_screened_stocks(screened_results)
    else:
        logger.warning("Screener returned no results")

    # ── 5. Store in database ─────────────────────────────────────
    logger.info("Step 5: Storing %d recommendations...", len(recommendations))
    db.save_recommendations(recommendations)

    # ── 6. Push notifications ────────────────────────────────────
    logger.info("Step 6: Sending push notifications...")
    tokens = db.get_device_tokens()
    await send_push_notifications(recommendations, tokens, config)

    logger.info("Pipeline complete. %d recommendations generated.", len(recommendations))
    return recommendations


def _run_screener(
    config: dict,
    dm: DataManager,
    prices: dict[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    macro: pd.DataFrame,
) -> list[dict]:
    """Run the dynamic screener and build per-stock analysis payloads."""
    try:
        statements = dm.get_all_statements()
        alt_data = dm.get_all_alternative_data()
    except Exception as e:
        logger.warning("Could not fetch statements/alt data for screener: %s", e)
        return []

    # Build info_map from fundamentals
    info_map = {}
    for ticker in fundamentals.index:
        row = fundamentals.loc[ticker]
        info_map[ticker] = {
            "marketCap": row.get("market_cap"),
            "sector": row.get("sector", "Unknown"),
            "industry": row.get("industry", "Unknown"),
            "shortName": row.get("short_name", ticker),
            "trailingPE": row.get("pe_ratio"),
            "priceToBook": row.get("pb_ratio"),
            "returnOnEquity": row.get("roe"),
            "debtToEquity": row.get("debt_to_equity"),
            "dividendYield": row.get("dividend_yield"),
            "trailingEps": row.get("eps"),
            "forwardEps": row.get("forward_eps"),
            "revenueGrowth": row.get("revenue_growth"),
        }

    # Get risk-free rate from macro data
    risk_free_rate = 0.04
    if macro is not None and not macro.empty and "treasury_10y" in macro.columns:
        val = macro["treasury_10y"].dropna()
        if not val.empty:
            risk_free_rate = float(val.iloc[-1]) / 100.0

    # Compute deep fundamentals and DCF per ticker
    deep_fund_map = {}
    dcf_map = {}
    for ticker, stmts in statements.items():
        try:
            deep = compute_deep_fundamentals(stmts, info_map.get(ticker, {}))
            deep_fund_map[ticker] = deep
        except Exception:
            pass
        try:
            dcf = compute_dcf_valuation(stmts, info_map.get(ticker, {}), risk_free_rate)
            dcf_map[ticker] = dcf
        except Exception:
            pass

    # Compute blindspot signals
    for ticker in list(deep_fund_map):
        try:
            alt = alt_data.get(ticker, {})
            blindspot = compute_institutional_blindspot(
                info_map.get(ticker, {}), alt.get("holders"), alt.get("insiders")
            )
            deep_fund_map[ticker].update(blindspot)
        except Exception:
            pass

    # Run screener
    screener_cfg = config.get("screening", {})
    top_n = screener_cfg.get("top_n", 20)
    screener = DynamicScreener(top_n=top_n)

    macro_regime = "neutral"
    scored_df = screener.compute_composite_scores(
        deep_fund_map, dcf_map, info_map, macro_regime, config=config
    )
    if scored_df.empty:
        return []

    safe = scored_df[scored_df["passes_safety"]].copy()
    if safe.empty:
        safe = scored_df.head(top_n)  # Fallback: show top even if safety filters fail

    safe = safe.sort_values("composite_score", ascending=False).head(top_n)

    # Build result list with analysis payloads
    results = []
    for _, row in safe.iterrows():
        ticker = row["ticker"]
        info = info_map.get(ticker, {})
        deep = deep_fund_map.get(ticker, {})
        dcf = dcf_map.get(ticker, {})

        analysis = _build_analysis_payload(ticker, info, deep, dcf)

        # Extract key DCF metrics for display columns
        intrinsic = dcf.get("intrinsic_value_per_share")
        current_price = info.get("currentPrice") or info.get("previousClose")

        results.append({
            "ticker": ticker,
            "short_name": info.get("shortName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "composite_score": float(row["composite_score"]),
            "rank": int(row["rank"]),
            "dcf_upside_score": float(row.get("dcf_upside_score", 0)),
            "fcf_yield_score": float(row.get("fcf_yield_score", 0)),
            "roic_spread_score": float(row.get("roic_spread_score", 0)),
            "intrinsic_value": intrinsic,
            "market_price": current_price,
            "margin_of_safety_pct": dcf.get("dcf_upside_pct"),
            "fcf_yield_pct": dcf.get("fcf_yield"),
            "roic_spread_pct": dcf.get("roic_vs_wacc_spread"),
            "wacc_pct": dcf.get("wacc"),
            "analysis": analysis,
        })

    return results


def _build_analysis_payload(
    ticker: str, info: dict, deep: dict, dcf: dict
) -> dict:
    """Build the detailed DCF-focused analysis JSON for a screened stock."""
    payload = {"ticker": ticker}

    reasons = []

    # ── Core DCF Valuation ────────────────────────────────────────
    intrinsic = dcf.get("intrinsic_value_per_share")
    upside = dcf.get("dcf_upside_pct")
    mos = dcf.get("margin_of_safety")
    wacc = dcf.get("wacc")
    roic = dcf.get("roic")
    roic_spread = dcf.get("roic_vs_wacc_spread")
    fcf_yield = dcf.get("fcf_yield")

    if intrinsic is not None:
        payload["intrinsic_value_per_share"] = intrinsic
    if upside is not None:
        payload["dcf_upside_pct"] = upside
    if mos is not None:
        payload["dcf_margin_of_safety"] = mos
    if wacc is not None:
        payload["wacc"] = wacc
    if roic is not None:
        payload["roic"] = roic
    if roic_spread is not None:
        payload["roic_vs_wacc_spread"] = roic_spread
    if fcf_yield is not None:
        payload["fcf_yield"] = fcf_yield

    # Scenario analysis (bear/base/bull)
    for key in ("bear_iv", "base_iv", "bull_iv", "scenario_range_pct"):
        val = dcf.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            payload[key] = val

    # EV/FCF and implied growth
    ev_fcf = dcf.get("ev_to_fcf")
    if ev_fcf is not None and not (isinstance(ev_fcf, float) and np.isnan(ev_fcf)):
        payload["ev_to_fcf"] = ev_fcf

    ig = dcf.get("implied_growth_rate")
    if ig is not None and not (isinstance(ig, float) and np.isnan(ig)):
        payload["implied_growth_rate"] = ig

    net_debt = dcf.get("net_debt")
    if net_debt is not None and not (isinstance(net_debt, float) and np.isnan(net_debt)):
        payload["net_debt"] = net_debt

    # ── DCF-focused Reasons ───────────────────────────────────────
    current_price = info.get("currentPrice") or info.get("previousClose")
    if intrinsic is not None and upside is not None and current_price:
        reasons.append(
            f"DCF intrinsic value ${intrinsic:.2f} vs ${current_price:.2f} market price "
            f"— {upside:+.0%} upside with {mos:.0%} margin of safety"
        )

    if fcf_yield is not None and fcf_yield > 0:
        reasons.append(f"Free cash flow yield {fcf_yield:.1%} — strong cash generation relative to enterprise value")

    if roic_spread is not None and roic_spread > 0:
        reasons.append(f"ROIC exceeds WACC by {roic_spread:.1%} — creating economic value above cost of capital")

    if wacc is not None and roic is not None:
        reasons.append(f"WACC {wacc:.1%}, ROIC {roic:.1%} — value creation spread of {roic_spread:.1%}" if roic_spread else f"WACC {wacc:.1%}")

    # Supplementary quality context (kept brief)
    f_score = deep.get("piotroski_f_score")
    if f_score is not None:
        payload["piotroski_f_score"] = f_score

    altman = deep.get("altman_z_score")
    if altman is not None:
        payload["altman_z_score"] = altman

    payload["reasons"] = reasons
    payload["market_cap"] = info.get("marketCap")

    return payload


def _get_latest_atr(prices: dict[str, pd.DataFrame], ticker: str) -> float:
    if ticker not in prices:
        return 0.0
    df = prices[ticker].tail(15)
    if len(df) < 2:
        return 0.0
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())
