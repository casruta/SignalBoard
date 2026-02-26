"""Adversarial validation and regime detection for distribution shift detection."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def adversarial_validation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.60,
) -> dict:
    """Detect distribution shift between train and test using adversarial validation.

    Trains a classifier to distinguish train from test samples.
    If AUC >> 0.5, there's a significant regime shift.

    Parameters
    ----------
    X_train : training features
    X_test : test/live features
    threshold : AUC above this triggers a regime shift warning

    Returns
    -------
    dict with 'auc', 'regime_shift_detected', 'top_shifting_features'
    """
    X_train_filled = X_train.fillna(0)
    X_test_filled = X_test.fillna(0)

    # Label: 0=train, 1=test
    X_combined = pd.concat([X_train_filled, X_test_filled], axis=0).reset_index(drop=True)
    y_combined = np.concatenate([
        np.zeros(len(X_train_filled)),
        np.ones(len(X_test_filled)),
    ])

    # Train a lightweight classifier
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
    )

    scores = cross_val_score(clf, X_combined, y_combined, cv=3, scoring="roc_auc")
    mean_auc = scores.mean()

    # Identify which features shift the most
    clf.fit(X_combined, y_combined)
    importances = pd.Series(
        clf.feature_importances_, index=X_combined.columns
    ).sort_values(ascending=False)
    top_features = importances.head(10).to_dict()

    return {
        "auc": float(mean_auc),
        "regime_shift_detected": mean_auc > threshold,
        "top_shifting_features": top_features,
    }


def detect_regime_changes(
    feature_matrix: pd.DataFrame,
    window: int = 60,
    step: int = 21,
) -> pd.DataFrame:
    """Detect regime changes by comparing rolling windows of feature distributions.

    Uses the PSI (Population Stability Index) to measure how much
    the feature distribution has changed from the reference period.

    Parameters
    ----------
    feature_matrix : DataFrame with MultiIndex (date, ticker)
    window : rolling window size in trading days
    step : step size for comparison

    Returns
    -------
    DataFrame with columns: date, psi_score, regime_change
    """
    dates = feature_matrix.index.get_level_values("date").unique().sort_values()
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns

    results = []
    ref_start = 0

    for i in range(window + step, len(dates), step):
        ref_dates = dates[ref_start : ref_start + window]
        test_dates = dates[i - step : i]

        ref_mask = feature_matrix.index.get_level_values("date").isin(ref_dates)
        test_mask = feature_matrix.index.get_level_values("date").isin(test_dates)

        ref_data = feature_matrix.loc[ref_mask, numeric_cols]
        test_data = feature_matrix.loc[test_mask, numeric_cols]

        if len(ref_data) < 20 or len(test_data) < 10:
            continue

        psi = _compute_psi(ref_data, test_data)
        results.append({
            "date": dates[i - 1],
            "psi_score": psi,
            "regime_change": psi > 0.25,  # PSI > 0.25 = significant shift
        })

        ref_start += step

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["date", "psi_score", "regime_change"]
    )


def _compute_psi(
    reference: pd.DataFrame, test: pd.DataFrame, n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions."""
    psi_values = []

    for col in reference.columns:
        ref_vals = reference[col].dropna().values
        test_vals = test[col].dropna().values
        if len(ref_vals) < 10 or len(test_vals) < 5:
            continue

        # Create bins from reference distribution
        edges = np.percentile(ref_vals, np.linspace(0, 100, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Remove duplicate edges
        edges = np.unique(edges)
        if len(edges) < 3:
            continue

        ref_counts = np.histogram(ref_vals, bins=edges)[0] + 1  # add-1 smoothing
        test_counts = np.histogram(test_vals, bins=edges)[0] + 1

        ref_pct = ref_counts / ref_counts.sum()
        test_pct = test_counts / test_counts.sum()

        psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        psi_values.append(psi)

    return float(np.mean(psi_values)) if psi_values else 0.0
