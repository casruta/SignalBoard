"""Walk-forward model training pipeline with purged CV, sample weighting, and temporal ensemble."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, f1_score

from models.registry import ModelRegistry


@dataclass
class TrainResult:
    """Results from a single training fold."""
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    accuracy: float
    precision_up: float
    precision_down: float
    f1: float
    feature_importance: dict = field(default_factory=dict)
    val_probs: np.ndarray | None = None
    val_y: np.ndarray | None = None


def compute_sample_weights(
    X_train: pd.DataFrame,
    recency_halflife_days: int = 126,
) -> np.ndarray:
    """Compute sample weights combining recency and inverse-volatility.

    Recency: exponential decay with half-life of ~6 months.
    Inverse-vol: downweight samples from extreme volatility periods.
    """
    dates = X_train.index.get_level_values("date")
    max_date = dates.max()

    days_ago = np.array((max_date - dates).days, dtype=float)
    recency_weight = np.exp(-np.log(2) * days_ago / recency_halflife_days)

    vol_weight = np.ones(len(X_train))
    if "atr_14" in X_train.columns:
        atr = X_train["atr_14"].values
        atr_median = np.nanmedian(atr)
        if atr_median > 0:
            vol_ratio = atr / atr_median
            vol_weight = 1.0 / np.clip(vol_ratio, 0.5, 3.0)

    weights = recency_weight * vol_weight
    weights = weights / weights.mean()
    return weights


class WalkForwardTrainer:
    """Train LightGBM models using walk-forward validation.

    Includes purged/embargo cross-validation, sample weighting,
    and temporal ensemble (keeps last N models).
    """

    DEFAULT_PARAMS = {
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

    def __init__(
        self,
        train_window_years: int = 2,
        val_window_months: int = 3,
        target_horizon_days: int = 5,
        embargo_days: int = 5,
        num_boost_rounds: int = 500,
        early_stopping_rounds: int = 50,
        keep_last_n_models: int = 3,
        use_sample_weights: bool = True,
        params: dict | None = None,
    ):
        self.train_years = train_window_years
        self.val_months = val_window_months
        self.target_horizon = target_horizon_days
        self.embargo = embargo_days
        self.num_rounds = num_boost_rounds
        self.early_stop = early_stopping_rounds
        self.keep_last_n = keep_last_n_models
        self.use_sample_weights = use_sample_weights
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.registry = ModelRegistry()

    def walk_forward_train(
        self,
        X: pd.DataFrame,
        y_class: pd.Series,
    ) -> tuple[list[lgb.Booster], list[TrainResult]]:
        """Run walk-forward training with purged CV and temporal ensemble.

        Returns a list of the last N models (for temporal ensemble) + fold results.
        X and y_class must have a MultiIndex (date, ticker).
        Target classes: 0=down, 1=flat, 2=up (remapped from -1,0,1).
        """
        # Remap target: -1→0, 0→1, 1→2 for LightGBM multiclass
        y = y_class.map({-1: 0, 0: 1, 1: 2})

        dates = X.index.get_level_values("date").unique().sort_values()
        results = []
        recent_models = []
        fold = 0

        # Slide through time
        train_start_idx = 0
        train_days = self.train_years * 252  # approx trading days
        val_days = self.val_months * 21
        # Purge gap: target horizon + embargo to prevent information leakage
        gap_days = self.target_horizon + self.embargo

        while train_start_idx + train_days + gap_days + val_days <= len(dates):
            train_dates = dates[train_start_idx : train_start_idx + train_days]
            # Skip gap_days between train end and val start to prevent leakage
            val_start = train_start_idx + train_days + gap_days
            val_dates = dates[val_start : val_start + val_days]

            train_mask = X.index.get_level_values("date").isin(train_dates)
            val_mask = X.index.get_level_values("date").isin(val_dates)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(X_train) < 100 or len(X_val) < 20:
                train_start_idx += val_days
                continue

            # Sample weighting: recency + inverse-volatility
            weights = None
            if self.use_sample_weights:
                weights = compute_sample_weights(X_train)

            train_set = lgb.Dataset(X_train, label=y_train, weight=weights)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

            model = lgb.train(
                self.params,
                train_set,
                num_boost_round=self.num_rounds,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(self.early_stop, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            # Evaluate
            y_pred_prob = model.predict(X_val)
            y_pred = np.argmax(y_pred_prob, axis=1)

            acc = accuracy_score(y_val, y_pred)
            prec_up = precision_score(y_val, y_pred, labels=[2], average="micro", zero_division=0)
            prec_down = precision_score(y_val, y_pred, labels=[0], average="micro", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            # Feature importance
            importance = dict(
                zip(X_train.columns, model.feature_importance(importance_type="gain"))
            )

            result = TrainResult(
                fold=fold,
                train_start=str(train_dates[0].date()),
                train_end=str(train_dates[-1].date()),
                val_start=str(val_dates[0].date()),
                val_end=str(val_dates[-1].date()),
                accuracy=acc,
                precision_up=prec_up,
                precision_down=prec_down,
                f1=f1,
                feature_importance=importance,
                val_probs=y_pred_prob,
                val_y=y_val.values,
            )
            results.append(result)

            # Temporal ensemble: keep last N models
            recent_models.append(model)
            if len(recent_models) > self.keep_last_n:
                recent_models.pop(0)

            fold += 1
            train_start_idx += val_days

        if recent_models:
            self.registry.save(
                recent_models[-1],
                metadata={
                    "trained_at": datetime.now().isoformat(),
                    "folds": fold,
                    "ensemble_size": len(recent_models),
                    "final_accuracy": results[-1].accuracy if results else None,
                    "final_f1": results[-1].f1 if results else None,
                    "features": list(X.columns),
                    "purge_gap_days": gap_days,
                },
            )

        return recent_models, results


def predict_temporal_ensemble(
    models: list[lgb.Booster],
    X: pd.DataFrame,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Average predictions from multiple models trained on different windows.

    More recent models get higher weight by default (exponential decay).
    Returns probability array of shape (n_samples, 3).
    """
    n = len(models)
    if n == 0:
        raise ValueError("No models provided")
    if n == 1:
        return models[0].predict(X)

    if weights is None:
        raw_weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights = raw_weights / raw_weights.sum()

    all_probs = np.stack([m.predict(X) for m in models])
    return np.average(all_probs, axis=0, weights=weights)
