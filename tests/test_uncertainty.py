"""Unit tests for uncertainty quantification modules."""

import numpy as np
import pytest

from models.uncertainty import (
    BayesianModelAveraging,
    ConformalPredictor,
    UncertaintyDecomposer,
    ThompsonSamplingEnsemble,
    ExpectedValueOfInformation,
)


class TestBayesianModelAveraging:
    def test_predict_proba(self):
        bma = BayesianModelAveraging(n_models=3)
        rng = np.random.default_rng(42)

        model_preds = [rng.dirichlet([1, 1, 1], 50) for _ in range(3)]
        result = bma.predict_proba(model_preds)

        assert result.shape == (50, 3)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_update_from_validation(self):
        bma = BayesianModelAveraging(n_models=2)
        rng = np.random.default_rng(42)

        y_true = rng.integers(0, 3, 100)
        model_preds = [rng.dirichlet([1, 1, 1], 100) for _ in range(2)]

        bma.update_from_validation(y_true, model_preds)
        weights = bma.get_weights()
        assert len(weights) == 2
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


class TestConformalPredictor:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 3, n)
        y_probs = rng.dirichlet([2, 1, 1], n)

        cp = ConformalPredictor(confidence=0.90)
        cp.fit(y_true, y_probs)

        assert cp.threshold is not None

    def test_uncertainty_factor_range(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 3, n)
        y_probs = rng.dirichlet([2, 1, 1], n)

        cp = ConformalPredictor(confidence=0.90)
        cp.fit(y_true, y_probs)

        new_probs = rng.dirichlet([2, 1, 1], 50)
        factors = cp.uncertainty_factor(new_probs)

        assert factors.shape == (50,)
        assert (factors >= 0.3).all()
        assert (factors <= 1.0).all()


class TestUncertaintyDecomposer:
    def test_decompose(self):
        rng = np.random.default_rng(42)
        model_probs = [rng.dirichlet([2, 1, 1], 100) for _ in range(3)]

        result = UncertaintyDecomposer.decompose(model_probs)

        assert "epistemic" in result
        assert "aleatoric" in result
        assert "total" in result
        assert "confidence" in result
        assert len(result["epistemic"]) == 100
        assert (result["confidence"] >= 0).all()
        assert (result["confidence"] <= 1).all()


class TestThompsonSamplingEnsemble:
    def test_sample_allocation(self):
        ts = ThompsonSamplingEnsemble(n_models=3)
        weights = ts.sample_allocation()

        assert len(weights) == 3
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_record_outcome(self):
        ts = ThompsonSamplingEnsemble(n_models=2)
        ts.record_outcome(0, 0.05)
        ts.record_outcome(1, -0.02)

        mean = ts.get_posterior_mean()
        # Model 0 had a win, model 1 had a loss — model 0 should have higher weight
        assert mean[0] > mean[1]


class TestExpectedValueOfInformation:
    def test_should_trade_positive_return(self):
        evi = ExpectedValueOfInformation()
        result = evi.should_trade(
            expected_return=0.02,
            variance=0.01,
            epistemic_uncertainty=0.3,
        )

        assert "recommendation" in result
        assert result["recommendation"] in ("trade_now", "wait", "skip")
        assert "kelly" in result
        assert result["kelly"] >= 0

    def test_skip_negative_return(self):
        evi = ExpectedValueOfInformation()
        result = evi.should_trade(
            expected_return=-0.01,
            variance=0.01,
            epistemic_uncertainty=0.3,
        )
        assert result["recommendation"] == "skip"

    def test_event_triggers_wait(self):
        evi = ExpectedValueOfInformation()
        result = evi.should_trade(
            expected_return=0.01,
            variance=0.01,
            epistemic_uncertainty=0.5,
            upcoming_events=["earnings_report"],
            wait_days=2,
        )
        # With high epistemic uncertainty and upcoming earnings, should lean toward waiting
        assert "evi" in result
