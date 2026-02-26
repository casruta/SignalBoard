"""SignalBoard — main orchestrator.

Usage:
    python main.py train       Train/retrain the ML model
    python main.py backtest    Run a full backtest
    python main.py predict     Generate predictions for today
    python main.py serve       Start the API server
    python main.py pipeline    Run the full daily pipeline once
    python main.py analyze     Run IC analysis on all signals
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


def _build_features(config: dict):
    """Shared feature-building pipeline used by train/backtest/predict."""
    from data.data_manager import DataManager
    from signals.combiner import SignalCombiner
    from models.features import add_target, add_lag_features, add_rolling_features

    dm = DataManager(config)
    prices = dm.get_all_prices()
    fundamentals = dm.get_all_fundamentals()
    macro = dm.get_macro()

    logger.info("Loaded prices for %d tickers", len(prices))

    combiner = SignalCombiner()
    fm = combiner.build_feature_matrix(prices, fundamentals, macro)
    fm = add_target(fm, prices, horizon_days=config["model"]["target_horizon_days"])

    key_signals = ["rsi_14", "macd_histogram", "momentum_5", "zscore_20"]
    fm = add_lag_features(fm, key_signals, lags=[1, 2, 5])
    fm = add_rolling_features(fm, key_signals, windows=[5, 10])

    logger.info("Feature matrix shape: %s", fm.shape)
    return dm, prices, fundamentals, macro, fm


def cmd_train(config: dict):
    """Train the ML model using walk-forward validation with optional feature selection."""
    from models.features import prepare_train_data
    from models.trainer import WalkForwardTrainer
    from models.feature_selection import select_features, feature_importance_report
    from models.calibration import ProbabilityCalibrator
    from models.regime_detection import adversarial_validation

    logger.info("=== Training Pipeline ===")

    # 1. Load data and build features
    logger.info("Loading data and building features...")
    dm, prices, fundamentals, macro, fm = _build_features(config)

    # 2. Prepare training data
    X, y_return, y_class = prepare_train_data(fm)
    logger.info("Training samples: %d, features: %d", len(X), len(X.columns))

    # 3. Feature selection (MI + correlation pruning)
    logger.info("Running feature selection...")
    selected_features = select_features(X, y_class, mi_threshold=0.005, corr_threshold=0.90)
    logger.info("Selected %d / %d features", len(selected_features), len(X.columns))

    # Log top features
    report = feature_importance_report(X[selected_features], y_class)
    for _, row in report.head(10).iterrows():
        logger.info("  Feature: %-30s  MI=%.4f  Corr=%.4f", row["feature"], row["mi_score"], row["abs_corr_target"])

    X_selected = X[selected_features]

    # 4. Train with walk-forward
    trainer = WalkForwardTrainer(
        train_window_years=config["model"]["train_window_years"],
        val_window_months=config["model"]["validation_window_months"],
    )
    models, results = trainer.walk_forward_train(X_selected, y_class)

    if results:
        logger.info("Training complete: %d folds", len(results))
        for r in results[-3:]:
            logger.info(
                "  Fold %d: acc=%.3f, f1=%.3f, prec_up=%.3f [%s -> %s]",
                r.fold, r.accuracy, r.f1, r.precision_up,
                r.val_start, r.val_end,
            )
    else:
        logger.warning("No training folds completed. Need more data.")


def cmd_backtest(config: dict):
    """Run a full backtest."""
    from models.features import add_target, add_lag_features, add_rolling_features
    from models.registry import ModelRegistry
    from backtest.engine import BacktestEngine
    from backtest.report import generate_report

    logger.info("=== Backtest ===")
    dm, prices, fundamentals, macro, fm = _build_features(config)

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
    from models.registry import ModelRegistry
    from models.predict import predict_latest

    logger.info("=== Generating Predictions ===")
    dm, prices, fundamentals, macro, fm = _build_features(config)

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


def cmd_analyze(config: dict):
    """Run IC analysis on all signals to evaluate predictive power."""
    from models.features import prepare_train_data
    from signals.ic_analysis import ic_analysis_report, turnover_analysis

    logger.info("=== Signal IC Analysis ===")
    dm, prices, fundamentals, macro, fm = _build_features(config)

    X, y_return, y_class = prepare_train_data(fm)

    # IC against forward returns
    report = ic_analysis_report(X, y_return, method="rank")

    logger.info("Top 20 features by Information Coefficient:")
    for _, row in report.head(20).iterrows():
        ic_ir_str = f"IR={row['ic_ir']:.2f}" if row['ic_ir'] else "IR=N/A"
        logger.info(
            "  %-35s  IC=%+.4f  t=%.2f  p=%.4f  %s",
            row["feature"], row["ic"], row["t_stat"], row["p_value"], ic_ir_str,
        )

    # Signal turnover for top features
    logger.info("\nSignal Turnover (top 10):")
    for feat in report.head(10)["feature"]:
        if feat in X.columns:
            to = turnover_analysis(X[feat])
            logger.info("  %-35s  turnover=%.4f", feat, to if to == to else 0)


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
        choices=["train", "backtest", "predict", "serve", "pipeline", "analyze"],
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
        "analyze": cmd_analyze,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
