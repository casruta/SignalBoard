"""Stacking ensemble: LightGBM + XGBoost + Ridge with logistic meta-learner."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler


def _get_xgb():
    """Lazy import for xgboost."""
    import xgboost as xgb
    return xgb


class StackingEnsemble:
    """Two-layer stacking ensemble for 3-class prediction.

    Layer 1 (base learners):
        - LightGBM (tree-based, handles non-linear)
        - XGBoost (tree-based, different regularization)
        - Ridge classifier (linear, captures monotonic relationships)

    Layer 2 (meta-learner):
        - Logistic regression on base learner probability outputs
    """

    def __init__(
        self,
        lgb_params: dict | None = None,
        xgb_params: dict | None = None,
        num_boost_rounds: int = 300,
        early_stopping_rounds: int = 30,
    ):
        self.num_rounds = num_boost_rounds
        self.early_stop = early_stopping_rounds

        self.lgb_params = lgb_params or {
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

        self.xgb_params = xgb_params or {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
            "nthread": -1,
        }

        self.lgb_model = None
        self.xgb_model = None
        self.ridge_model = None
        self.ridge_scaler = None
        self.meta_model = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: np.ndarray | None = None,
    ):
        """Train all base learners and the meta-learner.

        The meta-learner is trained on the validation set predictions
        from each base learner (stacking with blending).
        """
        xgb = _get_xgb()

        # ── Train LightGBM ──────────────────────────────────────
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        self.lgb_model = lgb.train(
            self.lgb_params,
            lgb_train,
            num_boost_round=self.num_rounds,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(self.early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # ── Train XGBoost ───────────────────────────────────────
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val)
        self.xgb_model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.num_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=self.early_stop,
            verbose_eval=False,
        )

        # ── Train Ridge ─────────────────────────────────────────
        self.ridge_scaler = StandardScaler()
        X_train_scaled = self.ridge_scaler.fit_transform(X_train.fillna(0))
        self.ridge_model = RidgeClassifier(alpha=1.0)
        self.ridge_model.fit(X_train_scaled, y_train)

        # ── Train Meta-Learner ──────────────────────────────────
        meta_features = self._get_meta_features(X_val)
        self.meta_model = LogisticRegression(
            max_iter=1000, multi_class="multinomial", C=1.0,
        )
        self.meta_model.fit(meta_features, y_val)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the stacking ensemble.

        Returns array of shape (n_samples, 3).
        """
        if self.meta_model is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base learner predictions."""
        xgb = _get_xgb()

        # LightGBM probabilities (n, 3)
        lgb_probs = self.lgb_model.predict(X)

        # XGBoost probabilities (n, 3)
        dmat = xgb.DMatrix(X)
        xgb_probs = self.xgb_model.predict(dmat)

        # Ridge decision function (n, 3)
        X_scaled = self.ridge_scaler.transform(X.fillna(0))
        ridge_decision = self.ridge_model.decision_function(X_scaled)
        if ridge_decision.ndim == 1:
            ridge_decision = np.column_stack([-ridge_decision, np.zeros(len(X)), ridge_decision])

        # Stack all as meta-features: 9 features total (3 learners x 3 classes)
        return np.hstack([lgb_probs, xgb_probs, ridge_decision])
