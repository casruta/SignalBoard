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

import pandas as pd

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

        # 5. Probability calibration on last fold's validation predictions
        if config.get("model", {}).get("calibrate_probabilities", True):
            try:
                last_result = results[-1]
                if last_result.val_probs is not None and last_result.val_y is not None:
                    calibrator = ProbabilityCalibrator()
                    calibrator.fit(last_result.val_y, last_result.val_probs)
                    logger.info("Probability calibrator fitted on last fold validation data")
            except Exception as e:
                logger.warning("Calibration skipped: %s", e)

        # 6. Adversarial validation for regime shift detection
        if config.get("model", {}).get("check_regime_shift", True):
            try:
                dates = X_selected.index.get_level_values("date").unique().sort_values()
                midpoint = len(dates) // 2
                early_dates = dates[:midpoint]
                recent_dates = dates[midpoint:]
                X_early = X_selected[X_selected.index.get_level_values("date").isin(early_dates)]
                X_recent = X_selected[X_selected.index.get_level_values("date").isin(recent_dates)]
                av_result = adversarial_validation(X_early, X_recent)
                logger.info("Adversarial validation AUC: %.3f", av_result["auc"])
                if av_result["regime_shift_detected"]:
                    logger.warning("Regime shift detected! Distribution has changed significantly.")
                    for feat, imp in list(av_result["top_shifting_features"].items())[:5]:
                        logger.warning("  Shifting feature: %-30s importance=%.4f", feat, imp)
            except Exception as e:
                logger.warning("Regime detection skipped: %s", e)
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
    results = engine.run(model, fm, prices, fundamentals, macro_df=macro)

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
    """Generate predictions for the latest available date with uncertainty quantification."""
    import numpy as np
    from models.registry import ModelRegistry
    from models.predict import predict_latest
    from models.uncertainty import ConformalPredictor, UncertaintyDecomposer

    logger.info("=== Generating Predictions ===")
    dm, prices, fundamentals, macro, fm = _build_features(config)

    registry = ModelRegistry()
    model = registry.load()

    predictions = predict_latest(model, fm)
    min_conf = config["strategy"]["min_confidence_threshold"]

    # Uncertainty quantification on latest predictions
    uncertainty_factors = {}
    try:
        latest_date = fm.index.get_level_values("date").max()
        target_cols = ["target_return", "target_class"]
        feature_cols = [
            c for c in fm.columns
            if c not in target_cols and fm[c].dtype in [np.float64, np.int64, float, int]
        ]
        latest = fm.loc[latest_date]
        if not isinstance(latest, pd.Series):
            X_latest = latest[feature_cols].fillna(latest[feature_cols].median())
            raw_probs = model.predict(X_latest)

            # Use earlier data as calibration set for conformal predictor
            dates = fm.index.get_level_values("date").unique().sort_values()
            cal_dates = dates[-60:-1] if len(dates) > 60 else dates[:-1]
            cal_mask = fm.index.get_level_values("date").isin(cal_dates)
            cal_data = fm[cal_mask]
            X_cal = cal_data[feature_cols].fillna(cal_data[feature_cols].median())
            if "target_class" in cal_data.columns:
                y_cal = cal_data["target_class"].map({-1: 0, 0: 1, 1: 2}).values
                cal_probs = model.predict(X_cal)

                conformal = ConformalPredictor(confidence=0.90)
                conformal.fit(y_cal, cal_probs)
                uf = conformal.uncertainty_factor(raw_probs)
                tickers = X_latest.index if not isinstance(X_latest.index, pd.MultiIndex) else X_latest.index.get_level_values("ticker")
                for i, ticker in enumerate(tickers):
                    uncertainty_factors[ticker] = float(uf[i])
    except Exception as e:
        logger.debug("Uncertainty quantification skipped: %s", e)

    logger.info("=== Signals (confidence >= %.0f%%) ===", min_conf * 100)
    for p in predictions:
        if p.action != "HOLD" and p.confidence >= min_conf:
            uf = uncertainty_factors.get(p.ticker, 1.0)
            logger.info(
                "  %s  %-5s  %s  confidence=%.0f%%  uncertainty=%.2f  probs=[BUY:%.0f%% HOLD:%.0f%% SELL:%.0f%%]",
                p.date, p.ticker, p.action, p.confidence * 100, uf,
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


def cmd_seed(config: dict):
    """Seed the database with mock trading recommendations."""
    from server.seed import seed
    db_path = config["server"]["database_path"]
    seed(db_path)
    logger.info("Database seeded successfully")


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
        choices=["train", "backtest", "predict", "serve", "pipeline", "analyze", "seed"],
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
        "seed": cmd_seed,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
