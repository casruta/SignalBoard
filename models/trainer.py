"""Walk-forward model training pipeline with purged CV, sample weighting, and temporal ensemble."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, f1_score

from models.registry import ModelRegistry

logger = logging.getLogger(__name__)

NUM_CLASSES = 3


def focal_loss_lgb(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Focal loss as a custom LightGBM objective for multiclass (3 classes).

    Focal loss: FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    y_true : 1-D array of integer labels (0, 1, 2)
    y_pred : flattened raw predictions, shape (n_samples * num_classes,)
    gamma : focusing parameter — higher values down-weight easy examples more
    alpha : per-class weights array of length num_classes (e.g. inverse frequency)

    Returns
    -------
    gradient, hessian : arrays of shape (n_samples * num_classes,)
    """
    n_samples = len(y_true)
    y_true = y_true.astype(int)

    # Reshape raw preds to (n_samples, num_classes) and apply softmax
    raw = y_pred.reshape(n_samples, NUM_CLASSES)
    exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-9, 1 - 1e-9)

    # One-hot encode true labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), y_true] = 1.0

    if alpha is None:
        alpha = np.ones(NUM_CLASSES)
    alpha_matrix = alpha[np.newaxis, :]  # (1, num_classes)

    # p_t for true class, broadcast to all class columns
    p_t = (probs * one_hot).sum(axis=1, keepdims=True)  # (n, 1)

    # Focal weight
    focal_weight = alpha_matrix * (1 - p_t) ** gamma  # (n, num_classes)

    # Gradient: focal_weight * (probs - one_hot)
    gradient = focal_weight * (probs - one_hot)

    # Hessian (diagonal approximation): focal_weight * probs * (1 - probs)
    hessian = focal_weight * probs * (1 - probs)
    hessian = np.maximum(hessian, 1e-9)

    return gradient.flatten(), hessian.flatten()


def focal_loss_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: np.ndarray | None = None,
) -> tuple[str, float, bool]:
    """Compute focal loss as a LightGBM evaluation metric.

    Returns
    -------
    (metric_name, metric_value, is_higher_better)
    """
    n_samples = len(y_true)
    y_true = y_true.astype(int)

    raw = y_pred.reshape(n_samples, NUM_CLASSES)
    exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-9, 1 - 1e-9)

    if alpha is None:
        alpha = np.ones(NUM_CLASSES)

    p_t = probs[np.arange(n_samples), y_true]
    alpha_t = alpha[y_true]
    loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t)

    return "focal_loss", float(loss.mean()), False


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
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
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
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
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

            # Focal loss setup
            focal_kwargs = {}
            train_params = dict(self.params)
            if self.use_focal_loss:
                # Compute class weights from training distribution (inverse frequency)
                class_counts = np.bincount(y_train.values.astype(int), minlength=NUM_CLASSES)
                class_weights = len(y_train) / (NUM_CLASSES * np.maximum(class_counts, 1))
                class_weights = class_weights / class_weights.sum() * NUM_CLASSES

                gamma = self.focal_gamma

                def focal_loss_objective(y_true, y_pred):
                    return focal_loss_lgb(y_true, y_pred, gamma=gamma, alpha=class_weights)

                def focal_loss_metric(y_true, y_pred):
                    return focal_loss_eval(y_true, y_pred, gamma=gamma, alpha=class_weights)

                # Remove conflicting params for custom objective
                train_params.pop("objective", None)
                train_params.pop("metric", None)

                focal_kwargs["fobj"] = focal_loss_objective
                focal_kwargs["feval"] = focal_loss_metric
                logger.info(
                    "Fold %d: using focal loss (gamma=%.1f, class_weights=%s)",
                    fold, gamma, class_weights.round(3),
                )

            model = lgb.train(
                train_params,
                train_set,
                num_boost_round=self.num_rounds,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(self.early_stop, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
                **focal_kwargs,
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
