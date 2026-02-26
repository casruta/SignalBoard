"""Feature engineering — build ML-ready features and targets from the signal matrix."""

import numpy as np
import pandas as pd


def add_target(
    feature_matrix: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    horizon_days: int = 5,
) -> pd.DataFrame:
    """Add forward-return target columns to the feature matrix.

    Parameters
    ----------
    feature_matrix : MultiIndex DataFrame (date, ticker) from SignalCombiner
    prices : {ticker: OHLCV DataFrame}
    horizon_days : number of trading days ahead for target

    Returns
    -------
    feature_matrix with additional columns:
        - target_return: raw forward return
        - target_class: -1 (down > 1%), 0 (flat), 1 (up > 1%)
    """
    targets = []
    for ticker, group in feature_matrix.groupby(level="ticker"):
        if ticker not in prices:
            continue
        close = prices[ticker]["Close"]
        # Align to feature matrix dates
        close_aligned = close.reindex(group.index.get_level_values("date"))
        forward = close_aligned.shift(-horizon_days)
        ret = (forward - close_aligned) / close_aligned

        ticker_targets = pd.DataFrame(
            {"target_return": ret.values, "target_class": _classify(ret.values)},
            index=group.index,
        )
        targets.append(ticker_targets)

    if not targets:
        return feature_matrix

    target_df = pd.concat(targets)
    return feature_matrix.join(target_df)


def add_lag_features(
    df: pd.DataFrame, columns: list[str], lags: list[int] = [1, 2, 5]
) -> pd.DataFrame:
    """Add lagged versions of specified columns per ticker."""
    result = df.copy()
    for ticker, group in result.groupby(level="ticker"):
        for col in columns:
            if col not in group.columns:
                continue
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                result.loc[group.index, lag_col] = group[col].shift(lag).values
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = [5, 10],
) -> pd.DataFrame:
    """Add rolling mean and std for specified columns per ticker."""
    result = df.copy()
    for ticker, group in result.groupby(level="ticker"):
        for col in columns:
            if col not in group.columns:
                continue
            series = group[col]
            for w in windows:
                result.loc[group.index, f"{col}_rmean{w}"] = (
                    series.rolling(w).mean().values
                )
                result.loc[group.index, f"{col}_rstd{w}"] = (
                    series.rolling(w).std().values
                )
    return result


def prepare_train_data(
    feature_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split feature matrix into X, y_return, y_class, dropping NaN targets.

    Returns
    -------
    X : feature DataFrame (numeric columns only, no targets)
    y_return : continuous target
    y_class : categorical target (-1, 0, 1)
    """
    df = feature_matrix.dropna(subset=["target_return", "target_class"])

    target_cols = ["target_return", "target_class"]
    non_feature_cols = target_cols + ["ticker"]
    feature_cols = [
        c for c in df.columns
        if c not in non_feature_cols and df[c].dtype in [np.float64, np.int64, float, int]
    ]

    X = df[feature_cols].copy()
    # Fill remaining NaN features with column median
    X = X.fillna(X.median())

    y_return = df["target_return"]
    y_class = df["target_class"].astype(int)

    return X, y_return, y_class


def _classify(returns: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Classify returns into -1 (down), 0 (flat), 1 (up)."""
    result = np.zeros(len(returns))
    result[returns > threshold] = 1
    result[returns < -threshold] = -1
    result[np.isnan(returns)] = np.nan
    return result
