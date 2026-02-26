"""SignalBoard — main orchestrator.

Usage:
    python main.py train       Train/retrain the ML model
    python main.py backtest    Run a full backtest
    python main.py predict     Generate predictions for today
    python main.py serve       Start the API server
    python main.py pipeline    Run the full daily pipeline once
"""

import argparse
import asyncio
import logging
import sys

from config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signalboard")


def cmd_train(config: dict):
    """Train the ML model using walk-forward validation."""
    from data.data_manager import DataManager
    from signals.combiner import SignalCombiner
    from models.features import add_target, add_lag_features, add_rolling_features, prepare_train_data
    from models.trainer import WalkForwardTrainer

    logger.info("=== Training Pipeline ===")

    # 1. Load data
    logger.info("Loading data...")
    dm = DataManager(config)
    prices = dm.get_all_prices()
    fundamentals = dm.get_all_fundamentals()
    macro = dm.get_macro()

    logger.info("Loaded prices for %d tickers", len(prices))

    # 2. Build feature matrix
    logger.info("Building feature matrix...")
    combiner = SignalCombiner()
    fm = combiner.build_feature_matrix(prices, fundamentals, macro)
    fm = add_target(fm, prices, horizon_days=config["model"]["target_horizon_days"])

    # 3. Add derived features
    key_signals = ["rsi_14", "macd_histogram", "momentum_5", "zscore_20"]
    fm = add_lag_features(fm, key_signals, lags=[1, 2, 5])
    fm = add_rolling_features(fm, key_signals, windows=[5, 10])

    logger.info("Feature matrix shape: %s", fm.shape)

    # 4. Prepare training data
    X, y_return, y_class = prepare_train_data(fm)
    logger.info("Training samples: %d, features: %d", len(X), len(X.columns))

    # 5. Train
    trainer = WalkForwardTrainer(
        train_window_years=config["model"]["train_window_years"],
        val_window_months=config["model"]["validation_window_months"],
    )
    model, results = trainer.walk_forward_train(X, y_class)

    if results:
        logger.info("Training complete: %d folds", len(results))
        for r in results[-3:]:  # Show last 3 folds
            logger.info(
                "  Fold %d: acc=%.3f, f1=%.3f, prec_up=%.3f [%s → %s]",
                r.fold, r.accuracy, r.f1, r.precision_up,
                r.val_start, r.val_end,
            )
    else:
        logger.warning("No training folds completed. Need more data.")


def cmd_backtest(config: dict):
    """Run a full backtest."""
    from data.data_manager import DataManager
    from signals.combiner import SignalCombiner
    from models.features import add_target, add_lag_features, add_rolling_features
    from models.registry import ModelRegistry
    from backtest.engine import BacktestEngine
    from backtest.report import generate_report

    logger.info("=== Backtest ===")

    # Load data
    dm = DataManager(config)
    prices = dm.get_all_prices()
    fundamentals = dm.get_all_fundamentals()
    macro = dm.get_macro()

    # Build features
    combiner = SignalCombiner()
    fm = combiner.build_feature_matrix(prices, fundamentals, macro)
    fm = add_target(fm, prices, horizon_days=config["model"]["target_horizon_days"])
    key_signals = ["rsi_14", "macd_histogram", "momentum_5", "zscore_20"]
    fm = add_lag_features(fm, key_signals, lags=[1, 2, 5])
    fm = add_rolling_features(fm, key_signals, windows=[5, 10])

    # Load model
    registry = ModelRegistry()
    model = registry.load()

    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run(model, fm, prices, fundamentals)

    # Report
    metrics = results["metrics"]
    logger.info("=== Backtest Results ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info("  %s: %.4f", key, val)
        else:
            logger.info("  %s: %s", key, val)

    report_dir = generate_report(results)
    logger.info("Report saved to: %s", report_dir)


def cmd_predict(config: dict):
    """Generate predictions for the latest available date."""
    from data.data_manager import DataManager
    from signals.combiner import SignalCombiner
    from models.features import add_target, add_lag_features, add_rolling_features
    from models.registry import ModelRegistry
    from models.predict import predict_latest

    logger.info("=== Generating Predictions ===")

    dm = DataManager(config)
    prices = dm.get_all_prices()
    fundamentals = dm.get_all_fundamentals()
    macro = dm.get_macro()

    combiner = SignalCombiner()
    fm = combiner.build_feature_matrix(prices, fundamentals, macro)
    fm = add_target(fm, prices, horizon_days=config["model"]["target_horizon_days"])
    key_signals = ["rsi_14", "macd_histogram", "momentum_5", "zscore_20"]
    fm = add_lag_features(fm, key_signals, lags=[1, 2, 5])
    fm = add_rolling_features(fm, key_signals, windows=[5, 10])

    registry = ModelRegistry()
    model = registry.load()

    predictions = predict_latest(model, fm)
    min_conf = config["strategy"]["min_confidence_threshold"]

    logger.info("=== Signals (confidence >= %.0f%%) ===", min_conf * 100)
    for p in predictions:
        if p.action != "HOLD" and p.confidence >= min_conf:
            logger.info(
                "  %s  %-5s  %s  confidence=%.0f%%  probs=[BUY:%.0f%% HOLD:%.0f%% SELL:%.0f%%]",
                p.date, p.ticker, p.action, p.confidence * 100,
                p.probabilities["BUY"] * 100,
                p.probabilities["HOLD"] * 100,
                p.probabilities["SELL"] * 100,
            )


def cmd_serve(config: dict):
    """Start the FastAPI server."""
    from server.app import main as run_server
    run_server()


def cmd_pipeline(config: dict):
    """Run the full daily pipeline once."""
    from server.scheduler import run_daily_pipeline
    results = asyncio.run(run_daily_pipeline(config))
    logger.info("Pipeline produced %d recommendations", len(results))


def main():
    parser = argparse.ArgumentParser(description="SignalBoard — Algorithmic Trading Signals")
    parser.add_argument(
        "command",
        choices=["train", "backtest", "predict", "serve", "pipeline"],
        help="Command to run",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    commands = {
        "train": cmd_train,
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "serve": cmd_serve,
        "pipeline": cmd_pipeline,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
