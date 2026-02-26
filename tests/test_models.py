"""Unit tests for ML pipeline modules."""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import make_prices, make_fundamentals, make_macro, make_config


def _build_small_feature_matrix():
    """Build a small feature matrix for testing."""
    from signals.combiner import SignalCombiner
    from models.features import add_target, add_lag_features, add_rolling_features

    prices = make_prices(n_days=600)
    fundamentals = make_fundamentals()
    macro = make_macro(n_days=600)

    combiner = SignalCombiner()
    fm = combiner.build_feature_matrix(prices, fundamentals, macro)
    fm = add_target(fm, prices, horizon_days=5)

    key_signals = ["rsi_14", "macd_histogram", "momentum_5"]
    available = [s for s in key_signals if s in fm.columns]
    if available:
        fm = add_lag_features(fm, available, lags=[1, 2])
        fm = add_rolling_features(fm, available, windows=[5])

    return fm


class TestFeatureEngineering:
    def test_add_target(self):
        from signals.combiner import SignalCombiner
        from models.features import add_target

        prices = make_prices(n_days=100)
        fundamentals = make_fundamentals()
        macro = make_macro(n_days=100)

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)
        fm = add_target(fm, prices, horizon_days=5)

        assert "target_return" in fm.columns
        assert "target_class" in fm.columns
        # Target class should be -1, 0, or 1
        valid = fm["target_class"].dropna()
        assert valid.isin([-1, 0, 1]).all()

    def test_prepare_train_data(self):
        from models.features import prepare_train_data

        fm = _build_small_feature_matrix()
        X, y_return, y_class = prepare_train_data(fm)

        assert len(X) > 0
        assert len(y_return) == len(X)
        assert len(y_class) == len(X)
        # No NaN targets
        assert not y_class.isna().any()
        assert not y_return.isna().any()
        # target columns should not be in X
        assert "target_return" not in X.columns
        assert "target_class" not in X.columns


class TestFeatureSelection:
    def test_select_features(self):
        from models.features import prepare_train_data
        from models.feature_selection import select_features

        fm = _build_small_feature_matrix()
        X, _, y_class = prepare_train_data(fm)

        selected = select_features(X, y_class, mi_threshold=0.001, corr_threshold=0.95)
        assert isinstance(selected, list)
        assert len(selected) > 0
        assert len(selected) <= len(X.columns)


class TestProbabilityCalibrator:
    def test_calibrate(self):
        from models.calibration import ProbabilityCalibrator

        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 3, n)
        y_prob = rng.dirichlet([1, 1, 1], n)

        cal = ProbabilityCalibrator()
        cal.fit(y_true, y_prob)
        calibrated = cal.calibrate(y_prob)

        assert calibrated.shape == y_prob.shape
        # Should sum to ~1
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=0.01)

    def test_uncalibrated_passthrough(self):
        from models.calibration import ProbabilityCalibrator

        cal = ProbabilityCalibrator()
        rng = np.random.default_rng(42)
        y_prob = rng.dirichlet([1, 1, 1], 50)
        result = cal.calibrate(y_prob)
        np.testing.assert_array_equal(result, y_prob)


class TestAdversarialValidation:
    def test_adversarial_validation(self):
        from models.regime_detection import adversarial_validation

        rng = np.random.default_rng(42)
        X_train = pd.DataFrame(rng.normal(0, 1, (200, 10)), columns=[f"f{i}" for i in range(10)])
        X_test = pd.DataFrame(rng.normal(0, 1, (100, 10)), columns=[f"f{i}" for i in range(10)])

        result = adversarial_validation(X_train, X_test)
        assert "auc" in result
        assert "regime_shift_detected" in result
        assert "top_shifting_features" in result
        assert 0 <= result["auc"] <= 1

    def test_detects_shift(self):
        from models.regime_detection import adversarial_validation

        rng = np.random.default_rng(42)
        # Train: mean=0, Test: mean=5 — very different distributions
        X_train = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=[f"f{i}" for i in range(5)])
        X_test = pd.DataFrame(rng.normal(5, 1, (100, 5)), columns=[f"f{i}" for i in range(5)])

        result = adversarial_validation(X_train, X_test)
        assert result["regime_shift_detected"] == True
        assert result["auc"] > 0.8
