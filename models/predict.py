"""Generate predictions from a trained model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb


# Class labels: 0=down, 1=flat, 2=up
CLASS_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
CLASS_MAP_REVERSE = {0: -1, 1: 0, 2: 1}  # back to original encoding


@dataclass
class Prediction:
    """A single ticker prediction."""
    ticker: str
    date: str
    action: str           # BUY, SELL, HOLD
    confidence: float     # 0-1, probability of predicted class
    predicted_class: int  # -1, 0, 1
    probabilities: dict   # {SELL: p, HOLD: p, BUY: p}
    predicted_return: float | None = None


def predict_batch(
    model: lgb.Booster,
    X: pd.DataFrame,
) -> list[Prediction]:
    """Generate predictions for all rows in X (MultiIndex: date, ticker).

    Returns a list of Prediction objects sorted by confidence descending.
    """
    probs = model.predict(X)  # shape: (n_samples, 3)
    predicted_classes = np.argmax(probs, axis=1)

    predictions = []
    for i, (idx, row) in enumerate(X.iterrows()):
        date, ticker = idx
        cls = predicted_classes[i]
        prob_dict = {
            "SELL": float(probs[i][0]),
            "HOLD": float(probs[i][1]),
            "BUY": float(probs[i][2]),
        }
        predictions.append(
            Prediction(
                ticker=ticker,
                date=str(date.date()) if hasattr(date, "date") else str(date),
                action=CLASS_LABELS[cls],
                confidence=float(probs[i][cls]),
                predicted_class=CLASS_MAP_REVERSE[cls],
                probabilities=prob_dict,
            )
        )

    # Sort: actionable signals first (BUY/SELL), then by confidence
    predictions.sort(key=lambda p: (-int(p.action != "HOLD"), -p.confidence))
    return predictions


def predict_latest(
    model: lgb.Booster,
    feature_matrix: pd.DataFrame,
) -> list[Prediction]:
    """Predict only the most recent date in the feature matrix."""
    dates = feature_matrix.index.get_level_values("date")
    latest_date = dates.max()
    latest = feature_matrix.loc[latest_date]

    # Ensure we only pass numeric feature columns
    non_feature = ["target_return", "target_class"]
    feature_cols = [
        c for c in latest.columns
        if c not in non_feature and latest[c].dtype in [np.float64, np.int64, float, int]
    ]
    X = latest[feature_cols].fillna(latest[feature_cols].median())

    probs = model.predict(X)
    predicted_classes = np.argmax(probs, axis=1)

    predictions = []
    tickers = latest.index if isinstance(latest.index, pd.Index) else [latest.name]

    for i, ticker in enumerate(tickers):
        cls = predicted_classes[i]
        prob_dict = {
            "SELL": float(probs[i][0]),
            "HOLD": float(probs[i][1]),
            "BUY": float(probs[i][2]),
        }
        predictions.append(
            Prediction(
                ticker=ticker,
                date=str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date),
                action=CLASS_LABELS[cls],
                confidence=float(probs[i][cls]),
                predicted_class=CLASS_MAP_REVERSE[cls],
                probabilities=prob_dict,
            )
        )

    predictions.sort(key=lambda p: (-int(p.action != "HOLD"), -p.confidence))
    return predictions
