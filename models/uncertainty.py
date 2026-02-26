"""Bayesian uncertainty quantification: BMA, conformal prediction, epistemic/aleatoric decomposition."""

import numpy as np
import pandas as pd


class BayesianModelAveraging:
    """Online Bayesian ensemble weight updates using Beta-Bernoulli conjugate priors.

    Tracks per-model accuracy posteriors and dynamically adjusts ensemble
    weights. Detects regime shifts by monitoring weight dispersion.
    """

    def __init__(self, n_models: int, alpha_init: float = 1.0):
        self.n_models = n_models
        self.alphas = np.ones(n_models) * alpha_init
        self.betas = np.ones(n_models) * alpha_init
        self.regime_change_threshold = 3.0

    def update_from_validation(
        self,
        y_true: np.ndarray,
        y_preds: list[np.ndarray],
    ):
        """Update weights from a validation window.

        Parameters
        ----------
        y_true : (n,) true class labels
        y_preds : list of (n, 3) probability arrays from each model
        """
        for m, pred_probs in enumerate(y_preds):
            y_pred_hard = np.argmax(pred_probs, axis=1)
            correct = int((y_pred_hard == y_true).sum())
            total = len(y_true)
            self.alphas[m] += correct
            self.betas[m] += total - correct

        # Detect regime shift: if one model dominates, reset priors for adaptation
        posterior_accs = self.alphas / (self.alphas + self.betas)
        if posterior_accs.min() > 0:
            ratio = posterior_accs.max() / posterior_accs.min()
            if ratio > self.regime_change_threshold:
                self.alphas *= 0.5
                self.betas *= 0.5

    def get_weights(self) -> np.ndarray:
        """Return normalized softmax weights from posterior accuracies."""
        posterior_accs = np.clip(self.alphas / (self.alphas + self.betas), 0.01, 0.99)
        log_odds = np.log(posterior_accs / (1 - posterior_accs))
        weights = np.exp(log_odds - log_odds.max())
        return weights / weights.sum()

    def predict_proba(self, y_preds: list[np.ndarray]) -> np.ndarray:
        """Weighted ensemble prediction using posterior weights.

        Returns (n_samples, 3) probability array.
        """
        weights = self.get_weights()
        ensemble = np.zeros_like(y_preds[0])
        for m, pred in enumerate(y_preds):
            ensemble += weights[m] * pred
        return ensemble


class ConformalPredictor:
    """Distribution-free prediction intervals for 3-class classification.

    Uses split conformal prediction to output interval widths
    that can be used to scale position sizes (wider = less certain).
    """

    def __init__(self, confidence: float = 0.90):
        self.confidence = confidence
        self.threshold = None

    def fit(self, y_true: np.ndarray, y_probs: np.ndarray):
        """Compute conformity threshold on a calibration set.

        Parameters
        ----------
        y_true : (n,) true class labels
        y_probs : (n, 3) predicted probabilities
        """
        predicted_class = np.argmax(y_probs, axis=1)
        pred_prob = y_probs[np.arange(len(y_true)), predicted_class]

        # Max probability among incorrect classes
        max_incorrect = np.zeros(len(y_true))
        for i in range(len(y_true)):
            probs_wrong = y_probs[i].copy()
            probs_wrong[predicted_class[i]] = -np.inf
            max_incorrect[i] = np.max(probs_wrong)

        # Residual: margin to misclassification (negative = well separated)
        residuals = max_incorrect - pred_prob

        # Compute quantile
        n = len(residuals)
        q_level = (n + 1) * (1 - self.confidence) / n
        q_idx = min(int(np.ceil(q_level * n)), n) - 1
        q_idx = max(q_idx, 0)

        self.threshold = float(np.sort(residuals)[q_idx])

    def prediction_intervals(self, y_probs: np.ndarray) -> np.ndarray:
        """Compute interval widths for each prediction.

        Smaller width = more confident (larger gap between top-2 classes).

        Returns (n,) array of interval widths.
        """
        if self.threshold is None:
            raise RuntimeError("Not fitted. Call fit() first.")

        intervals = np.zeros(len(y_probs))
        for i in range(len(y_probs)):
            sorted_p = np.sort(y_probs[i])
            intervals[i] = sorted_p[2] - sorted_p[1]  # gap between top-2
        return intervals

    def uncertainty_factor(self, y_probs: np.ndarray) -> np.ndarray:
        """Compute position sizing multiplier based on conformal intervals.

        Returns (n,) array in [0.3, 1.0] — multiply with position size.
        """
        intervals = self.prediction_intervals(y_probs)
        if self.threshold is None or self.threshold == 0:
            return np.ones(len(y_probs))

        factors = np.clip(intervals / abs(self.threshold), 0.3, 1.0)
        return factors


class UncertaintyDecomposer:
    """Decompose ensemble uncertainty into epistemic and aleatoric components.

    Epistemic (model disagreement): reducible with more data/training.
    Aleatoric (inherent noise): irreducible — data is fundamentally uncertain.
    """

    @staticmethod
    def decompose(model_probs_list: list[np.ndarray]) -> dict:
        """Decompose uncertainty from base learner predictions.

        Parameters
        ----------
        model_probs_list : list of (n_samples, 3) probability arrays

        Returns
        -------
        dict with 'epistemic', 'aleatoric', 'total', 'confidence' arrays
        """
        all_probs = np.array(model_probs_list)  # (M, n, 3)
        mean_probs = all_probs.mean(axis=0)     # (n, 3)

        # Epistemic: average KL divergence from each model to the mean
        epistemic = np.zeros(mean_probs.shape[0])
        for m in range(len(model_probs_list)):
            p = np.clip(all_probs[m], 1e-7, 1)
            q = np.clip(mean_probs, 1e-7, 1)
            epistemic += np.sum(p * (np.log(p) - np.log(q)), axis=-1)
        epistemic /= len(model_probs_list)

        # Aleatoric: average Shannon entropy across models
        aleatoric = np.zeros(mean_probs.shape[0])
        for m in range(len(model_probs_list)):
            p = np.clip(all_probs[m], 1e-7, 1)
            aleatoric += -np.sum(p * np.log(p), axis=-1)
        aleatoric /= len(model_probs_list)

        total = epistemic + aleatoric
        # Normalize by max possible entropy for 3-class
        max_entropy = np.log(3)
        confidence = 1.0 - np.clip(total / (max_entropy + 0.1), 0, 1)

        return {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": total,
            "confidence": confidence,
        }


class ThompsonSamplingEnsemble:
    """Dynamic model allocation using Thompson sampling.

    Maintains Beta posteriors on each model's "win rate" and samples
    allocations that balance exploration and exploitation.
    """

    def __init__(self, n_models: int, alpha_0: float = 1.0, beta_0: float = 1.0):
        self.n_models = n_models
        self.alphas = np.ones(n_models) * alpha_0
        self.betas = np.ones(n_models) * beta_0

    def record_outcome(self, model_idx: int, daily_return: float):
        """Update posterior from a trading day outcome."""
        if daily_return > 0:
            self.alphas[model_idx] += 1
        else:
            self.betas[model_idx] += 1

    def sample_allocation(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Thompson-sample model weights.

        Returns (n_models,) weights summing to 1.
        """
        rng = rng or np.random.default_rng()
        samples = np.array([
            rng.beta(self.alphas[m], self.betas[m])
            for m in range(self.n_models)
        ])
        exp_s = np.exp(samples - samples.max())
        return exp_s / exp_s.sum()

    def get_posterior_mean(self) -> np.ndarray:
        """Deterministic weights from posterior means."""
        means = self.alphas / (self.alphas + self.betas)
        return means / means.sum()


class ExpectedValueOfInformation:
    """Compute whether to trade now or wait for more data.

    Compares the expected value of trading immediately (with current
    uncertainty) vs. waiting for new information (e.g., earnings report).
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        daily_epistemic_decay: float = 0.10,
    ):
        self.txn_cost = transaction_cost
        self.daily_decay = daily_epistemic_decay

    def should_trade(
        self,
        expected_return: float,
        variance: float,
        epistemic_uncertainty: float,
        upcoming_events: list[str] | None = None,
        wait_days: int = 1,
    ) -> dict:
        """Decide: trade now or wait.

        Parameters
        ----------
        expected_return : predicted return for the asset
        variance : return variance
        epistemic_uncertainty : current model uncertainty (0-1)
        upcoming_events : e.g. ['earnings', 'fed_announcement']
        wait_days : how many days we'd wait

        Returns
        -------
        dict with 'recommendation' ('trade_now' or 'wait'), 'evi', 'kelly'
        """
        if variance <= 0 or expected_return <= 0:
            return {"recommendation": "skip", "evi": 0.0, "kelly": 0.0}

        # Kelly fraction (half-Kelly for safety)
        kelly = min(expected_return / variance / 2.0, 1.0)

        # Value of trading now
        value_now = kelly * expected_return - self.txn_cost

        # Projected uncertainty reduction from waiting
        event_reduction = 0.0
        if upcoming_events:
            if any("earn" in e.lower() for e in upcoming_events):
                event_reduction = 0.30
            if any("fed" in e.lower() or "fomc" in e.lower() for e in upcoming_events):
                event_reduction = 0.15

        total_reduction = min(
            self.daily_decay * wait_days + event_reduction,
            epistemic_uncertainty,
        )

        # With lower uncertainty, Kelly fraction improves
        if epistemic_uncertainty > 0:
            improvement = total_reduction / epistemic_uncertainty
        else:
            improvement = 0

        value_wait = kelly * (1 + improvement) * expected_return - self.txn_cost - 0.001 * wait_days

        evi = value_wait - value_now
        recommendation = "wait" if evi > 0.001 else "trade_now"

        return {
            "recommendation": recommendation,
            "evi": float(evi),
            "kelly": float(kelly),
            "value_now": float(value_now),
            "value_wait": float(value_wait),
        }
