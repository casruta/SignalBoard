"""Daily pipeline scheduler — runs the ML pipeline after market close."""

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config_loader import get_config
from data.data_manager import DataManager
from signals.combiner import SignalCombiner
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

    for pred in signals:
        ticker = pred.ticker
        if ticker not in X_latest.index:
            continue

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

    # ── 5. Store in database ─────────────────────────────────────
    logger.info("Step 5: Storing %d recommendations...", len(recommendations))
    db.save_recommendations(recommendations)

    # ── 6. Push notifications ────────────────────────────────────
    logger.info("Step 6: Sending push notifications...")
    tokens = db.get_device_tokens()
    await send_push_notifications(recommendations, tokens, config)

    logger.info("Pipeline complete. %d recommendations generated.", len(recommendations))
    return recommendations


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
