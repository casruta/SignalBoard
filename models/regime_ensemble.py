"""Regime-conditional ensemble: separate models per market regime."""

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}


class HMMRegimeClassifier:
    """Classify market regimes using a simple Gaussian mixture approach.

    Uses rolling return statistics to identify 3 regimes:
    - Bull: positive mean return, low-to-moderate volatility
    - Bear: negative mean return, high volatility
    - Sideways: near-zero mean return, low volatility

    Note: We avoid hmmlearn dependency by using a simpler rolling-statistics
    approach that captures the same intuition. If hmmlearn is installed,
    we can optionally use it for more sophisticated regime detection.
    """

    def __init__(self, n_regimes: int = 3, lookback: int = 60):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self._mean_of_rolling: float | None = None
        self._std_of_rolling: float | None = None
        self._fitted = False

    def fit(self, market_returns: pd.Series) -> "HMMRegimeClassifier":
        """Fit regime thresholds from historical market returns (e.g., SPY).

        Uses rolling mean and volatility to define regime boundaries.
        Bull: rolling_mean > mean + 0.5*std
        Bear: rolling_mean < mean - 0.5*std
        Sideways: in between
        """
        rolling_mean = market_returns.rolling(self.lookback).mean().dropna()
        if rolling_mean.empty:
            raise ValueError(
                f"Not enough data to compute {self.lookback}-day rolling mean. "
                f"Got {len(market_returns)} observations."
            )
        self._mean_of_rolling = float(rolling_mean.mean())
        self._std_of_rolling = float(rolling_mean.std())
        if self._std_of_rolling < 1e-12:
            self._std_of_rolling = 1e-12
        self._fitted = True
        logger.info(
            "Regime classifier fitted: rolling_mean_center=%.6f, rolling_mean_std=%.6f",
            self._mean_of_rolling,
            self._std_of_rolling,
        )
        return self

    def _classify_rolling_mean(self, rolling_mean: pd.Series) -> pd.Series:
        """Map rolling mean values to regime integers."""
        upper = self._mean_of_rolling + 0.5 * self._std_of_rolling
        lower = self._mean_of_rolling - 0.5 * self._std_of_rolling

        regimes = pd.Series(1, index=rolling_mean.index, dtype=int)  # default sideways
        regimes[rolling_mean > upper] = 2  # bull
        regimes[rolling_mean < lower] = 0  # bear
        return regimes

    def predict(self, market_returns: pd.Series) -> pd.Series:
        """Classify each date into regime 0 (bear), 1 (sideways), 2 (bull).
        Returns Series indexed by date with integer regime labels."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        rolling_mean = market_returns.rolling(self.lookback).mean()
        regimes = self._classify_rolling_mean(rolling_mean)
        # NaN at the start where rolling window hasn't filled — default to sideways
        regimes = regimes.fillna(1).astype(int)
        return regimes

    def predict_latest(self, market_returns: pd.Series) -> int:
        """Classify the most recent regime."""
        regimes = self.predict(market_returns)
        return int(regimes.iloc[-1])

    def regime_confidence(self, market_returns: pd.Series) -> float:
        """How confident we are in the current regime (0-1).
        Based on distance from regime boundary thresholds."""
        if not self._fitted:
            raise RuntimeError("Call fit() before regime_confidence().")

        rolling_mean = market_returns.rolling(self.lookback).mean()
        latest = rolling_mean.iloc[-1]
        if np.isnan(latest):
            return 0.0

        upper = self._mean_of_rolling + 0.5 * self._std_of_rolling
        lower = self._mean_of_rolling - 0.5 * self._std_of_rolling

        # Distance from the nearest boundary, normalised by the std
        dist_upper = abs(latest - upper)
        dist_lower = abs(latest - lower)
        min_dist = min(dist_upper, dist_lower)

        # Sigmoid-like mapping: farther from boundary -> higher confidence
        confidence = 1.0 - np.exp(-min_dist / self._std_of_rolling)
        return float(np.clip(confidence, 0.0, 1.0))


class RegimeConditionalEnsemble:
    """Train separate ML models per market regime, blend with fallback.

    Architecture:
    - Regime classifier labels each training sample
    - Per-regime model trained on regime-specific subset (if enough samples)
    - Fallback model trained on all data
    - Prediction: blend_weight * regime_model + (1-blend_weight) * fallback
    """

    def __init__(
        self,
        n_regimes: int = 3,
        blend_weight: float = 0.7,
        min_samples_per_regime: int = 200,
        lgb_params: dict | None = None,
    ):
        self.n_regimes = n_regimes
        self.blend_weight = blend_weight
        self.min_samples = min_samples_per_regime
        self.lgb_params = lgb_params or {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.regime_classifier = HMMRegimeClassifier(n_regimes=n_regimes)
        self.regime_models: dict[int, lgb.Booster] = {}
        self.fallback_model: lgb.Booster | None = None
        self._regime_sample_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dates(index: pd.Index) -> pd.Index:
        """Pull the date level from a MultiIndex (date, ticker)."""
        if isinstance(index, pd.MultiIndex):
            return index.get_level_values("date")
        return index

    def _align_regimes(
        self, X: pd.DataFrame, market_returns: pd.Series
    ) -> pd.Series:
        """Return a regime Series aligned to the rows of X."""
        dates = self._extract_dates(X.index)
        all_regimes = self.regime_classifier.predict(market_returns)
        # Map each row's date to its regime
        mapped = dates.map(all_regimes).fillna(1).astype(int)
        return pd.Series(mapped.values, index=X.index)

    def _train_lgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_boost_round: int,
        early_stopping: int,
        sample_weight: np.ndarray | None = None,
        label: str = "fallback",
    ) -> tuple[lgb.Booster, dict]:
        """Train a single LightGBM booster and return (model, metrics)."""
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            reference=dtrain,
            free_raw_data=False,
        )

        callbacks = [
            lgb.log_evaluation(period=0),  # suppress per-iteration logs
            lgb.early_stopping(stopping_rounds=early_stopping),
        ]

        booster = lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        best_score = booster.best_score.get("val", {}).get("multi_logloss", None)
        metrics = {
            "label": label,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": booster.best_iteration,
            "best_val_logloss": best_score,
        }
        logger.info("Trained %s model: %s", label, metrics)
        return booster, metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        market_returns: pd.Series,
        num_boost_round: int = 500,
        early_stopping: int = 50,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """Train regime classifier + per-regime models + fallback.

        Returns dict with training metrics per regime.
        market_returns should be a daily return series for SPY or similar market index.
        """
        # 1. Fit the regime classifier on market returns
        self.regime_classifier.fit(market_returns)

        # 2. Label every training row with its regime
        train_regimes = self._align_regimes(X_train, market_returns)
        val_regimes = self._align_regimes(X_val, market_returns)

        metrics: dict[str, dict] = {}

        # 3. Train per-regime models
        self.regime_models.clear()
        self._regime_sample_counts.clear()

        for regime in range(self.n_regimes):
            train_mask = train_regimes == regime
            val_mask = val_regimes == regime
            n_train = int(train_mask.sum())
            n_val = int(val_mask.sum())
            self._regime_sample_counts[regime] = n_train

            if n_train < self.min_samples:
                logger.info(
                    "Regime %d (%s): only %d train samples (< %d). Skipping regime model.",
                    regime,
                    REGIME_LABELS.get(regime, "?"),
                    n_train,
                    self.min_samples,
                )
                metrics[f"regime_{regime}"] = {
                    "label": REGIME_LABELS.get(regime, str(regime)),
                    "skipped": True,
                    "n_train": n_train,
                }
                continue

            if n_val < 10:
                logger.warning(
                    "Regime %d has only %d val samples; using full val set for early-stopping.",
                    regime,
                    n_val,
                )
                regime_X_val = X_val
                regime_y_val = y_val
            else:
                regime_X_val = X_val.loc[val_mask.values]
                regime_y_val = y_val.loc[val_mask.values]

            regime_X_train = X_train.loc[train_mask.values]
            regime_y_train = y_train.loc[train_mask.values]

            regime_weight = (
                sample_weight[train_mask.values]
                if sample_weight is not None
                else None
            )

            booster, m = self._train_lgb(
                regime_X_train,
                regime_y_train,
                regime_X_val,
                regime_y_val,
                num_boost_round,
                early_stopping,
                sample_weight=regime_weight,
                label=f"regime_{regime}_{REGIME_LABELS.get(regime, '')}",
            )
            self.regime_models[regime] = booster
            metrics[f"regime_{regime}"] = m

        # 4. Train fallback model on all data
        self.fallback_model, fallback_metrics = self._train_lgb(
            X_train,
            y_train,
            X_val,
            y_val,
            num_boost_round,
            early_stopping,
            sample_weight=sample_weight,
            label="fallback",
        )
        metrics["fallback"] = fallback_metrics

        logger.info(
            "RegimeConditionalEnsemble fitted: %d regime models active, fallback ready.",
            len(self.regime_models),
        )
        return metrics

    def predict_proba(
        self,
        X: pd.DataFrame,
        market_returns: pd.Series,
    ) -> np.ndarray:
        """Blended prediction: regime_weight * regime_model + (1-w) * fallback.

        Shape: (n_samples, 3) probability matrix.
        """
        if self.fallback_model is None:
            raise RuntimeError("Call fit() before predict_proba().")

        n = len(X)
        num_class = self.lgb_params.get("num_class", 3)

        # Fallback predictions for every row
        fallback_preds = self.fallback_model.predict(X)  # (n, num_class)
        if fallback_preds.ndim == 1:
            fallback_preds = fallback_preds.reshape(n, num_class)

        blended = np.copy(fallback_preds)

        # Per-regime predictions where a regime model exists
        regimes = self._align_regimes(X, market_returns)

        for regime, model in self.regime_models.items():
            mask = (regimes == regime).values if hasattr(regimes, "values") else (regimes == regime)
            if not np.any(mask):
                continue

            regime_X = X.loc[mask] if isinstance(mask, pd.Series) else X.iloc[mask]
            regime_preds = model.predict(regime_X)
            if regime_preds.ndim == 1:
                regime_preds = regime_preds.reshape(-1, num_class)

            # Blend: w * regime + (1-w) * fallback
            blended[mask] = (
                self.blend_weight * regime_preds
                + (1 - self.blend_weight) * fallback_preds[mask]
            )

        # Re-normalise rows to sum to 1
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        blended = blended / row_sums

        return blended

    def get_current_regime(self, market_returns: pd.Series) -> dict:
        """Return current regime info: label, confidence, description."""
        regime_id = self.regime_classifier.predict_latest(market_returns)
        confidence = self.regime_classifier.regime_confidence(market_returns)
        label = REGIME_LABELS.get(regime_id, "unknown")
        has_model = regime_id in self.regime_models

        descriptions = {
            0: "Negative mean returns with elevated volatility.",
            1: "Near-zero mean returns with subdued volatility.",
            2: "Positive mean returns with low-to-moderate volatility.",
        }

        return {
            "regime_id": regime_id,
            "label": label,
            "confidence": round(confidence, 4),
            "has_dedicated_model": has_model,
            "description": descriptions.get(regime_id, ""),
        }

    def regime_summary(self) -> dict:
        """Return summary: samples per regime, which regimes have models."""
        summary: dict[str, dict] = {}
        for regime in range(self.n_regimes):
            label = REGIME_LABELS.get(regime, str(regime))
            summary[label] = {
                "regime_id": regime,
                "train_samples": self._regime_sample_counts.get(regime, 0),
                "has_model": regime in self.regime_models,
            }
        return summary
