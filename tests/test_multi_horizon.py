"""Tests for models.multi_horizon module."""

import numpy as np
import pandas as pd
import pytest

from models.multi_horizon import MultiHorizonConsensus


# ── Helpers ──────────────────────────────────────────────────────────


def _make_feature_data(n_train: int = 500, n_val: int = 100, n_features: int = 10):
    """Build synthetic train/val feature matrices and multi-horizon targets."""
    rng = np.random.default_rng(42)

    X_train = pd.DataFrame(
        rng.standard_normal((n_train, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    X_val = pd.DataFrame(
        rng.standard_normal((n_val, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    # Create correlated targets (class 0, 1, 2)
    # Use feature sum to create a signal so the model can learn something
    signal_train = X_train.sum(axis=1)
    signal_val = X_val.sum(axis=1)

    def _make_targets(signal: pd.Series) -> pd.Series:
        q33 = signal.quantile(0.33)
        q66 = signal.quantile(0.66)
        return signal.apply(lambda x: 0 if x < q33 else (2 if x > q66 else 1))

    y_train_dict = {}
    y_val_dict = {}
    for h in [5, 10, 20]:
        # Add slight noise per horizon so they are not identical
        noise_tr = rng.standard_normal(n_train) * 0.5
        noise_val = rng.standard_normal(n_val) * 0.5
        y_train_dict[h] = _make_targets(signal_train + noise_tr)
        y_val_dict[h] = _make_targets(signal_val + noise_val)

    return X_train, X_val, y_train_dict, y_val_dict


# ── Tests: fit ───────────────────────────────────────────────────────


class TestMultiHorizonFit:
    def test_trains_models_for_all_horizons(self):
        X_train, X_val, y_dict, y_val_dict = _make_feature_data()
        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        metrics = model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=20)

        assert len(model.models) == 3
        for h in [5, 10, 20]:
            assert h in model.models
            assert h in metrics
            assert "accuracy" in metrics[h]

    def test_skips_horizon_with_missing_target(self):
        X_train, X_val, y_dict, y_val_dict = _make_feature_data()
        # Remove horizon 20 target
        del y_dict[20]
        del y_val_dict[20]
        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=20)

        assert 5 in model.models
        assert 10 in model.models
        assert 20 not in model.models


# ── Tests: predict_proba ─────────────────────────────────────────────


class TestMultiHorizonPredictProba:
    def test_returns_correct_shape(self):
        X_train, X_val, y_dict, y_val_dict = _make_feature_data()
        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=20)

        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 3)

    def test_probabilities_sum_to_one(self):
        X_train, X_val, y_dict, y_val_dict = _make_feature_data()
        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=20)

        proba = model.predict_proba(X_val)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_raises_before_fit(self):
        model = MultiHorizonConsensus()
        X = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="No models"):
            model.predict_proba(X)


# ── Tests: consensus_signal ──────────────────────────────────────────


class TestConsensusSignal:
    def test_agreement_score_range(self):
        X_train, X_val, y_dict, y_val_dict = _make_feature_data()
        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=20)

        consensus = model.consensus_signal(X_val)
        assert (consensus["agreement_score"] >= 0).all()
        assert (consensus["agreement_score"] <= 1).all()

    def test_returns_expected_columns(self):
        X_train, X_val, y_dict, y_val_dict = _make_feature_data()
        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=20)

        consensus = model.consensus_signal(X_val)
        expected_cols = [
            "predicted_class", "agreement_score", "avg_confidence",
            "buy_agreement", "sell_agreement",
        ]
        for col in expected_cols:
            assert col in consensus.columns

    def test_full_agreement_when_all_horizons_agree(self):
        """When all horizon models predict the same class, agreement should be 1.0."""
        X_train, X_val, y_dict, y_val_dict = _make_feature_data(n_train=500, n_val=50)
        # Make all horizons have identical targets so models learn the same thing
        base_target_train = y_dict[5].copy()
        base_target_val = y_val_dict[5].copy()
        for h in [5, 10, 20]:
            y_dict[h] = base_target_train.copy()
            y_val_dict[h] = base_target_val.copy()

        model = MultiHorizonConsensus(horizons=[5, 10, 20])
        model.fit(X_train, y_dict, X_val, y_val_dict, num_boost_round=50)

        consensus = model.consensus_signal(X_val)
        # With identical training targets, all horizons should agree on most samples
        high_agreement = (consensus["agreement_score"] == 1.0).sum()
        # At least 50% of samples should have full agreement
        assert high_agreement >= len(X_val) * 0.5


# ── Tests: add_multi_horizon_targets ─────────────────────────────────


class TestAddMultiHorizonTargets:
    def test_creates_correct_columns(self):
        from models.features import add_multi_horizon_targets

        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = {
            "AAPL": pd.DataFrame(
                {"Close": 150 + np.arange(100, dtype=float) * 0.1},
                index=dates,
            )
        }
        idx = pd.MultiIndex.from_tuples(
            [(d, "AAPL") for d in dates[:80]], names=["date", "ticker"]
        )
        features = pd.DataFrame(
            np.random.randn(80, 3), index=idx, columns=["f1", "f2", "f3"]
        )

        result = add_multi_horizon_targets(features, prices, horizons=[5, 10])

        assert "target_return_5d" in result.columns
        assert "target_class_5d" in result.columns
        assert "target_return_10d" in result.columns
        assert "target_class_10d" in result.columns


# ── Tests: init validation ───────────────────────────────────────────


class TestMultiHorizonInit:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            MultiHorizonConsensus(horizons=[5, 10], horizon_weights=[0.5])
