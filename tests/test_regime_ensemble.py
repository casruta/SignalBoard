"""Tests for models.regime_ensemble module."""

import numpy as np
import pandas as pd
import pytest

from models.regime_ensemble import HMMRegimeClassifier, RegimeConditionalEnsemble


# ── Synthetic Data Builders ──────────────────────────────────────────


def _make_market_returns(n_days: int = 500) -> pd.Series:
    """Generate synthetic daily market returns with regime-like behavior."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    # Create returns with varying mean/vol to simulate regimes
    returns = np.concatenate([
        rng.normal(0.001, 0.01, n_days // 3),   # bull
        rng.normal(-0.001, 0.02, n_days // 3),   # bear
        rng.normal(0.0, 0.008, n_days - 2 * (n_days // 3)),  # sideways
    ])
    return pd.Series(returns, index=dates, name="market_return")


def _make_training_data(n_train: int = 600, n_val: int = 150, n_features: int = 8):
    """Build synthetic training data with date-indexed MultiIndex."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n_train + n_val)
    train_dates = dates[:n_train]
    val_dates = dates[n_train:]

    X_train = pd.DataFrame(
        rng.standard_normal((n_train, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=train_dates,
    )
    X_val = pd.DataFrame(
        rng.standard_normal((n_val, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=val_dates,
    )

    # Create targets correlated to features
    signal_train = X_train.sum(axis=1)
    signal_val = X_val.sum(axis=1)

    def _to_class(s):
        q33 = s.quantile(0.33)
        q66 = s.quantile(0.66)
        return s.apply(lambda x: 0 if x < q33 else (2 if x > q66 else 1))

    y_train = _to_class(signal_train)
    y_val = _to_class(signal_val)

    # Market returns covering the full date range
    market_returns = pd.Series(
        rng.normal(0.0005, 0.012, len(dates)),
        index=dates,
    )

    return X_train, y_train, X_val, y_val, market_returns


# ── Tests: HMMRegimeClassifier.fit ───────────────────────────────────


class TestHMMRegimeClassifierFit:
    def test_fit_does_not_raise(self):
        returns = _make_market_returns()
        clf = HMMRegimeClassifier(lookback=60)
        clf.fit(returns)
        assert clf._fitted is True

    def test_fit_with_short_data_raises(self):
        returns = pd.Series([0.001] * 10)
        clf = HMMRegimeClassifier(lookback=60)
        with pytest.raises(ValueError, match="Not enough data"):
            clf.fit(returns)


# ── Tests: HMMRegimeClassifier.predict ───────────────────────────────


class TestHMMRegimeClassifierPredict:
    def test_predict_values_in_expected_set(self):
        returns = _make_market_returns()
        clf = HMMRegimeClassifier(lookback=60)
        clf.fit(returns)
        regimes = clf.predict(returns)

        assert set(regimes.unique()).issubset({0, 1, 2})

    def test_predict_same_length_as_input(self):
        returns = _make_market_returns()
        clf = HMMRegimeClassifier(lookback=60)
        clf.fit(returns)
        regimes = clf.predict(returns)
        assert len(regimes) == len(returns)

    def test_predict_before_fit_raises(self):
        clf = HMMRegimeClassifier()
        returns = pd.Series([0.001] * 100)
        with pytest.raises(RuntimeError, match="Call fit"):
            clf.predict(returns)

    def test_predict_latest_returns_int(self):
        returns = _make_market_returns()
        clf = HMMRegimeClassifier(lookback=60)
        clf.fit(returns)
        latest = clf.predict_latest(returns)
        assert isinstance(latest, int)
        assert latest in {0, 1, 2}


# ── Tests: HMMRegimeClassifier.regime_confidence ─────────────────────


class TestRegimeConfidence:
    def test_confidence_between_0_and_1(self):
        returns = _make_market_returns()
        clf = HMMRegimeClassifier(lookback=60)
        clf.fit(returns)
        confidence = clf.regime_confidence(returns)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_before_fit_raises(self):
        clf = HMMRegimeClassifier()
        returns = pd.Series([0.001] * 100)
        with pytest.raises(RuntimeError):
            clf.regime_confidence(returns)


# ── Tests: RegimeConditionalEnsemble.fit ─────────────────────────────


class TestRegimeConditionalEnsembleFit:
    def test_trains_fallback_model(self):
        X_train, y_train, X_val, y_val, market_returns = _make_training_data()
        ensemble = RegimeConditionalEnsemble(
            min_samples_per_regime=50,
            lgb_params={
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "verbose": -1,
                "num_leaves": 15,
                "n_estimators": 20,
            },
        )
        metrics = ensemble.fit(
            X_train, y_train, X_val, y_val, market_returns,
            num_boost_round=20, early_stopping=10,
        )

        assert ensemble.fallback_model is not None
        assert "fallback" in metrics

    def test_fit_returns_metrics(self):
        X_train, y_train, X_val, y_val, market_returns = _make_training_data()
        ensemble = RegimeConditionalEnsemble(min_samples_per_regime=50)
        metrics = ensemble.fit(
            X_train, y_train, X_val, y_val, market_returns,
            num_boost_round=20, early_stopping=10,
        )
        assert isinstance(metrics, dict)
        assert "fallback" in metrics


# ── Tests: RegimeConditionalEnsemble.predict_proba ───────────────────


class TestRegimeConditionalEnsemblePredictProba:
    def test_returns_correct_shape(self):
        X_train, y_train, X_val, y_val, market_returns = _make_training_data()
        ensemble = RegimeConditionalEnsemble(min_samples_per_regime=50)
        ensemble.fit(
            X_train, y_train, X_val, y_val, market_returns,
            num_boost_round=20, early_stopping=10,
        )

        proba = ensemble.predict_proba(X_val, market_returns)
        assert proba.shape == (len(X_val), 3)

    def test_probabilities_sum_to_one(self):
        X_train, y_train, X_val, y_val, market_returns = _make_training_data()
        ensemble = RegimeConditionalEnsemble(min_samples_per_regime=50)
        ensemble.fit(
            X_train, y_train, X_val, y_val, market_returns,
            num_boost_round=20, early_stopping=10,
        )

        proba = ensemble.predict_proba(X_val, market_returns)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_raises_before_fit(self):
        ensemble = RegimeConditionalEnsemble()
        X = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
        returns = pd.Series([0.001] * 100)
        with pytest.raises(RuntimeError):
            ensemble.predict_proba(X, returns)


# ── Tests: get_current_regime ────────────────────────────────────────


class TestGetCurrentRegime:
    def test_returns_expected_keys(self):
        X_train, y_train, X_val, y_val, market_returns = _make_training_data()
        ensemble = RegimeConditionalEnsemble(min_samples_per_regime=50)
        ensemble.fit(
            X_train, y_train, X_val, y_val, market_returns,
            num_boost_round=20, early_stopping=10,
        )

        info = ensemble.get_current_regime(market_returns)
        assert "regime_id" in info
        assert "label" in info
        assert "confidence" in info
        assert info["regime_id"] in {0, 1, 2}
        assert 0.0 <= info["confidence"] <= 1.0
