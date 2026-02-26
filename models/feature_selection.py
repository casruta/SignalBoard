"""Feature selection: mutual information scoring + pairwise correlation removal."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    mi_threshold: float = 0.01,
    corr_threshold: float = 0.90,
    max_features: int | None = None,
) -> list[str]:
    """Select features using mutual information + correlation pruning.

    1. Score each feature by mutual information with the target.
    2. Remove features with MI below threshold (uninformative).
    3. Among highly-correlated feature pairs (>corr_threshold),
       keep the one with higher MI score.

    Parameters
    ----------
    X : feature DataFrame
    y : target Series (class labels)
    mi_threshold : minimum mutual information score to keep a feature
    corr_threshold : max allowed pairwise correlation before pruning
    max_features : optional cap on total feature count

    Returns
    -------
    List of selected feature names.
    """
    X_filled = X.fillna(0)

    # Step 1: Compute mutual information scores
    mi_scores = mutual_info_classif(
        X_filled, y, discrete_features=False, random_state=42, n_neighbors=5,
    )
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Step 2: Filter by MI threshold
    selected = mi_series[mi_series >= mi_threshold].index.tolist()
    if not selected:
        # Keep at least the top 10 if nothing passes threshold
        selected = mi_series.head(10).index.tolist()

    # Step 3: Pairwise correlation pruning
    selected = _prune_correlated(X_filled[selected], mi_series, corr_threshold)

    # Step 4: Optional cap
    if max_features and len(selected) > max_features:
        # Keep top-MI features
        ranked = mi_series[mi_series.index.isin(selected)].head(max_features)
        selected = ranked.index.tolist()

    return selected


def _prune_correlated(
    X: pd.DataFrame,
    mi_scores: pd.Series,
    threshold: float,
) -> list[str]:
    """Remove features that are highly correlated, keeping the higher-MI one."""
    corr_matrix = X.corr().abs()
    selected = set(X.columns)

    for i in range(len(corr_matrix)):
        if corr_matrix.columns[i] not in selected:
            continue
        for j in range(i + 1, len(corr_matrix)):
            col_j = corr_matrix.columns[j]
            if col_j not in selected:
                continue
            if corr_matrix.iloc[i, j] >= threshold:
                col_i = corr_matrix.columns[i]
                # Drop the one with lower MI
                if mi_scores.get(col_i, 0) < mi_scores.get(col_j, 0):
                    selected.discard(col_i)
                else:
                    selected.discard(col_j)

    return list(selected)


def feature_importance_report(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Generate a report of feature importance metrics.

    Returns DataFrame with columns: feature, mi_score, abs_correlation_with_target.
    """
    X_filled = X.fillna(0)

    # Mutual information
    mi_scores = mutual_info_classif(
        X_filled, y, discrete_features=False, random_state=42,
    )

    # Point-biserial correlation (target as numeric)
    y_numeric = y.astype(float)
    corr_with_target = X_filled.corrwith(y_numeric).abs()

    report = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores,
        "abs_corr_target": corr_with_target.values,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    return report
