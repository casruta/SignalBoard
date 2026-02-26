"""Walk-forward model training pipeline."""

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


class WalkForwardTrainer:
    """Train LightGBM models using walk-forward validation."""

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
        num_boost_rounds: int = 500,
        early_stopping_rounds: int = 50,
        params: dict | None = None,
    ):
        self.train_years = train_window_years
        self.val_months = val_window_months
        self.num_rounds = num_boost_rounds
        self.early_stop = early_stopping_rounds
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.registry = ModelRegistry()

    def walk_forward_train(
        self,
        X: pd.DataFrame,
        y_class: pd.Series,
    ) -> tuple[lgb.Booster, list[TrainResult]]:
        """Run walk-forward training and return the final model + fold results.

        X and y_class must have a MultiIndex (date, ticker).
        Target classes: 0=down, 1=flat, 2=up (remapped from -1,0,1).
        """
        # Remap target: -1→0, 0→1, 1→2 for LightGBM multiclass
        y = y_class.map({-1: 0, 0: 1, 1: 2})

        dates = X.index.get_level_values("date").unique().sort_values()
        results = []
        final_model = None
        fold = 0

        # Slide through time
        train_start_idx = 0
        train_days = self.train_years * 252  # approx trading days
        val_days = self.val_months * 21

        while train_start_idx + train_days + val_days <= len(dates):
            train_dates = dates[train_start_idx : train_start_idx + train_days]
            val_dates = dates[
                train_start_idx + train_days :
                train_start_idx + train_days + val_days
            ]

            train_mask = X.index.get_level_values("date").isin(train_dates)
            val_mask = X.index.get_level_values("date").isin(val_dates)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(X_train) < 100 or len(X_val) < 20:
                train_start_idx += val_days
                continue

            train_set = lgb.Dataset(X_train, label=y_train)
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
            )
            results.append(result)
            final_model = model
            fold += 1

            # Slide forward by validation window
            train_start_idx += val_days

        if final_model is not None:
            self.registry.save(
                final_model,
                metadata={
                    "trained_at": datetime.now().isoformat(),
                    "folds": fold,
                    "final_accuracy": results[-1].accuracy if results else None,
                    "final_f1": results[-1].f1 if results else None,
                    "features": list(X.columns),
                },
            )

        return final_model, results
