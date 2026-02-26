"""End-to-end integration test: signals -> train -> backtest -> predict.

Uses synthetic data fixtures to verify the full pipeline works
without any API keys or network access.
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import make_config, make_prices, make_fundamentals, make_macro


@pytest.fixture
def pipeline_data():
    """Build shared data for integration tests."""
    config = make_config()
    prices = make_prices(n_days=600)
    fundamentals = make_fundamentals()
    macro = make_macro(n_days=600)
    return config, prices, fundamentals, macro


class TestIntegration:

    def test_signal_combiner_produces_features(self, pipeline_data):
        config, prices, fundamentals, macro = pipeline_data
        from signals.combiner import SignalCombiner

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)

        assert isinstance(fm.index, pd.MultiIndex)
        assert fm.index.names == ["date", "ticker"]
        assert len(fm.columns) > 30
        assert len(fm) > 100

        # Network features should be present
        net_cols = [c for c in fm.columns if c.startswith("net_")]
        assert len(net_cols) > 0, "Network features not integrated"

    def test_feature_engineering_pipeline(self, pipeline_data):
        config, prices, fundamentals, macro = pipeline_data
        from signals.combiner import SignalCombiner
        from models.features import (
            add_target,
            add_lag_features,
            add_rolling_features,
            prepare_train_data,
        )

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)
        fm = add_target(fm, prices, horizon_days=5)

        key_signals = ["rsi_14", "macd_histogram", "momentum_5"]
        available = [s for s in key_signals if s in fm.columns]
        assert len(available) > 0, "Expected signals missing from feature matrix"

        fm = add_lag_features(fm, available, lags=[1, 2, 5])
        fm = add_rolling_features(fm, available, windows=[5, 10])

        X, y_return, y_class = prepare_train_data(fm)
        assert len(X) > 50
        assert y_class.isin([-1, 0, 1]).all()

    def test_train_produces_model(self, pipeline_data):
        config, prices, fundamentals, macro = pipeline_data
        from signals.combiner import SignalCombiner
        from models.features import (
            add_target,
            add_lag_features,
            add_rolling_features,
            prepare_train_data,
        )
        from models.feature_selection import select_features
        from models.trainer import WalkForwardTrainer

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)
        fm = add_target(fm, prices, horizon_days=5)

        key_signals = ["rsi_14", "macd_histogram", "momentum_5"]
        available = [s for s in key_signals if s in fm.columns]
        fm = add_lag_features(fm, available, lags=[1, 2])
        fm = add_rolling_features(fm, available, windows=[5])

        X, _, y_class = prepare_train_data(fm)
        selected = select_features(X, y_class, mi_threshold=0.001, corr_threshold=0.95)
        X_sel = X[selected]

        trainer = WalkForwardTrainer(
            train_window_years=1,
            val_window_months=1,
            num_boost_rounds=50,
            early_stopping_rounds=10,
        )
        models, results = trainer.walk_forward_train(X_sel, y_class)

        assert len(models) >= 1, "Training produced no models"
        assert len(results) >= 1, "Training produced no fold results"
        assert results[-1].accuracy > 0.20, "Accuracy worse than random"
        # Validation predictions should be captured
        assert results[-1].val_probs is not None
        assert results[-1].val_y is not None

    def test_backtest_end_to_end(self, pipeline_data):
        config, prices, fundamentals, macro = pipeline_data
        from signals.combiner import SignalCombiner
        from models.features import (
            add_target,
            add_lag_features,
            add_rolling_features,
            prepare_train_data,
        )
        from models.feature_selection import select_features
        from models.trainer import WalkForwardTrainer
        from backtest.engine import BacktestEngine

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)
        fm = add_target(fm, prices, horizon_days=5)

        key_signals = ["rsi_14", "macd_histogram", "momentum_5"]
        available = [s for s in key_signals if s in fm.columns]
        fm = add_lag_features(fm, available, lags=[1, 2])
        fm = add_rolling_features(fm, available, windows=[5])

        X, _, y_class = prepare_train_data(fm)
        selected = select_features(X, y_class, mi_threshold=0.001, corr_threshold=0.95)
        X_sel = X[selected]

        trainer = WalkForwardTrainer(
            train_window_years=1,
            val_window_months=1,
            num_boost_rounds=50,
            early_stopping_rounds=10,
        )
        models, _ = trainer.walk_forward_train(X_sel, y_class)
        model = models[-1]

        engine = BacktestEngine(config)
        results = engine.run(model, fm, prices, fundamentals, macro_df=macro)

        assert "equity_curve" in results
        assert "metrics" in results
        assert "trades" in results
        assert results["metrics"]["sharpe_ratio"] is not None
        assert isinstance(results["equity_curve"], pd.DataFrame)
        assert len(results["equity_curve"]) > 0
