"""SHAP-based signal decomposition — attribute predictions to features."""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute SHAP values for each prediction.

    Parameters
    ----------
    model : trained LightGBM booster
    X : feature DataFrame

    Returns
    -------
    DataFrame of SHAP values, same shape as X, for the predicted class.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values is a list of 3 arrays (one per class)
    # Pick the SHAP values for each sample's predicted class
    probs = model.predict(X)
    predicted_classes = np.argmax(probs, axis=1)

    result = np.zeros_like(X.values, dtype=float)
    for i in range(len(X)):
        cls = predicted_classes[i]
        result[i] = shap_values[cls][i]

    return pd.DataFrame(result, index=X.index, columns=X.columns)


def top_contributing_features(
    shap_df: pd.DataFrame,
    row_idx: int | str,
    n: int = 5,
) -> list[tuple[str, float]]:
    """Return the top N features by absolute SHAP value for a given row.

    Returns list of (feature_name, shap_value) sorted by |shap_value| desc.
    """
    if isinstance(row_idx, int):
        row = shap_df.iloc[row_idx]
    else:
        row = shap_df.loc[row_idx]

    abs_vals = row.abs().sort_values(ascending=False)
    top_features = abs_vals.head(n).index.tolist()
    return [(f, float(row[f])) for f in top_features]


def categorize_features(feature_name: str) -> str:
    """Map a feature name to its signal category."""
    if feature_name.startswith("macro_"):
        return "macro"
    if feature_name.startswith("fund_"):
        return "fundamental"
    # Everything else is technical (from technical.py)
    return "technical"


def decompose_by_category(
    shap_df: pd.DataFrame,
    row_idx: int | str,
) -> dict[str, list[tuple[str, float]]]:
    """Decompose SHAP contributions by signal category.

    Returns {category: [(feature, shap_value), ...]} sorted by |shap_value|.
    """
    if isinstance(row_idx, int):
        row = shap_df.iloc[row_idx]
    else:
        row = shap_df.loc[row_idx]

    categories = {"technical": [], "fundamental": [], "macro": []}

    for feat in row.index:
        cat = categorize_features(feat)
        categories[cat].append((feat, float(row[feat])))

    # Sort each category by absolute value
    for cat in categories:
        categories[cat].sort(key=lambda x: abs(x[1]), reverse=True)

    return categories
