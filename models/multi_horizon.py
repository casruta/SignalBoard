"""Multi-horizon consensus: train separate models per horizon, combine via weighted agreement."""

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


class MultiHorizonConsensus:
    """Train separate models per time horizon, combine predictions.

    When all horizons agree on BUY/SELL, that's a much stronger signal.
    """

    def __init__(
        self,
        horizons: list[int] = [5, 10, 20],
        horizon_weights: list[float] = [0.3, 0.4, 0.3],
    ):
        if len(horizons) != len(horizon_weights):
            raise ValueError("horizons and horizon_weights must have the same length")
        self.horizons = horizons
        self.weights = horizon_weights
        self.models: dict[int, lgb.Booster] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_dict: dict[int, pd.Series],
        X_val: pd.DataFrame,
        y_val_dict: dict[int, pd.Series],
        lgb_params: dict | None = None,
        num_boost_round: int = 500,
        early_stopping: int = 50,
        sample_weight: np.ndarray | None = None,
    ) -> dict[int, dict]:
        """Train one LightGBM model per horizon. Returns metrics per horizon.

        Parameters
        ----------
        X_train : training features
        y_dict : {horizon: target class series} with labels 0, 1, 2
        X_val : validation features
        y_val_dict : {horizon: validation target class series}
        lgb_params : LightGBM parameters (defaults provided if None)
        num_boost_round : maximum boosting rounds
        early_stopping : early stopping patience
        sample_weight : optional sample weights for training

        Returns
        -------
        {horizon: {"accuracy": float, "best_iteration": int}} per horizon
        """
        default_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }
        params = {**default_params, **(lgb_params or {})}
        metrics = {}

        for horizon in self.horizons:
            if horizon not in y_dict:
                logger.warning("No target data for horizon %d, skipping", horizon)
                continue

            y_train = y_dict[horizon]
            y_val = y_val_dict[horizon]

            # Drop rows with NaN targets
            train_mask = y_train.notna()
            val_mask = y_val.notna()

            X_tr = X_train[train_mask]
            y_tr = y_train[train_mask]
            X_v = X_val[val_mask]
            y_v = y_val[val_mask]

            if len(X_tr) < 100 or len(X_v) < 20:
                logger.warning(
                    "Horizon %dd: insufficient data (train=%d, val=%d), skipping",
                    horizon, len(X_tr), len(X_v),
                )
                continue

            w = sample_weight[train_mask] if sample_weight is not None else None
            train_set = lgb.Dataset(X_tr, label=y_tr, weight=w)
            val_set = lgb.Dataset(X_v, label=y_v, reference=train_set)

            logger.info("Training model for %d-day horizon (%d samples)", horizon, len(X_tr))

            model = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(early_stopping, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            self.models[horizon] = model

            # Compute validation accuracy
            y_pred = np.argmax(model.predict(X_v), axis=1)
            accuracy = float((y_pred == y_v.values).mean())
            best_iter = model.best_iteration if model.best_iteration else num_boost_round

            metrics[horizon] = {
                "accuracy": accuracy,
                "best_iteration": best_iter,
            }
            logger.info("Horizon %dd: accuracy=%.3f, best_iter=%d", horizon, accuracy, best_iter)

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of per-horizon predictions. Shape: (n_samples, 3)."""
        if not self.models:
            raise ValueError("No models have been trained yet. Call fit() first.")

        all_probs = []
        active_weights = []
        for horizon, weight in zip(self.horizons, self.weights):
            if horizon not in self.models:
                continue
            probs = self.models[horizon].predict(X)
            all_probs.append(probs)
            active_weights.append(weight)

        if not all_probs:
            raise ValueError("No trained models available for prediction")

        # Normalize weights in case some horizons were skipped
        weight_arr = np.array(active_weights)
        weight_arr = weight_arr / weight_arr.sum()

        stacked = np.stack(all_probs)  # (n_horizons, n_samples, 3)
        return np.average(stacked, axis=0, weights=weight_arr)

    def consensus_signal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return per-sample consensus analysis.

        Columns:
        - predicted_class: majority vote direction (-1, 0, 1)
        - agreement_score: fraction of horizons agreeing on direction (0-1)
        - avg_confidence: weighted average max probability
        - buy_agreement: fraction voting BUY
        - sell_agreement: fraction voting SELL
        """
        per_horizon = self.predict_per_horizon(X)
        n_samples = len(X)
        n_horizons = len(per_horizon)

        # Per-horizon predicted classes: remap 0->-1, 1->0, 2->1
        remap = {0: -1, 1: 0, 2: 1}
        horizon_classes = np.zeros((n_horizons, n_samples), dtype=int)
        horizon_confidences = np.zeros((n_horizons, n_samples))
        active_weights = []

        for i, (horizon, probs) in enumerate(per_horizon.items()):
            predicted = np.argmax(probs, axis=1)
            horizon_classes[i] = np.vectorize(remap.get)(predicted)
            horizon_confidences[i] = probs.max(axis=1)
            # Find the weight for this horizon
            idx = self.horizons.index(horizon)
            active_weights.append(self.weights[idx])

        weight_arr = np.array(active_weights)
        weight_arr = weight_arr / weight_arr.sum()

        # Majority vote (weighted)
        predicted_class = np.zeros(n_samples, dtype=int)
        agreement_score = np.zeros(n_samples)
        buy_agreement = np.zeros(n_samples)
        sell_agreement = np.zeros(n_samples)

        for j in range(n_samples):
            votes = horizon_classes[:, j]
            buy_frac = float((votes == 1).sum()) / n_horizons
            sell_frac = float((votes == -1).sum()) / n_horizons
            flat_frac = float((votes == 0).sum()) / n_horizons

            buy_agreement[j] = buy_frac
            sell_agreement[j] = sell_frac

            # Majority vote
            fracs = {1: buy_frac, -1: sell_frac, 0: flat_frac}
            winner = max(fracs, key=fracs.get)
            predicted_class[j] = winner
            agreement_score[j] = fracs[winner]

        # Weighted average confidence
        avg_confidence = np.average(horizon_confidences, axis=0, weights=weight_arr)

        return pd.DataFrame(
            {
                "predicted_class": predicted_class,
                "agreement_score": agreement_score,
                "avg_confidence": avg_confidence,
                "buy_agreement": buy_agreement,
                "sell_agreement": sell_agreement,
            },
            index=X.index,
        )

    def predict_per_horizon(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        """Return raw probabilities per horizon (for analysis)."""
        if not self.models:
            raise ValueError("No models have been trained yet. Call fit() first.")

        result = {}
        for horizon in self.horizons:
            if horizon in self.models:
                result[horizon] = self.models[horizon].predict(X)
        return result
