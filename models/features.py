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
        - target_class: -1 (down), 0 (flat), 1 (up) using volatility-adjusted thresholds
        - target_threshold: per-row volatility-adjusted threshold used for classification
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

        # Compute rolling 20-day volatility for adaptive thresholds
        daily_ret = close.pct_change()
        vol_20 = daily_ret.rolling(20).std()
        vol_aligned = vol_20.reindex(group.index.get_level_values("date")).values

        classes, thresholds = _classify_adaptive(ret.values, vol_aligned)

        ticker_targets = pd.DataFrame(
            {
                "target_return": ret.values,
                "target_class": classes,
                "target_threshold": thresholds,
            },
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


def add_multi_horizon_targets(
    feature_matrix: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    horizons: list[int] = [5, 10, 20],
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Add target columns for multiple horizons.

    For each horizon h, adds:
    - target_return_{h}d: raw forward return
    - target_class_{h}d: -1/0/1 classification

    Also keeps the default target_return and target_class columns
    (set to the middle horizon for backward compatibility).

    Parameters
    ----------
    feature_matrix : MultiIndex DataFrame (date, ticker)
    prices : {ticker: OHLCV DataFrame}
    horizons : list of forward-looking windows in trading days
    threshold : return threshold for classification

    Returns
    -------
    feature_matrix with additional target columns per horizon
    """
    all_targets = []
    middle_horizon = horizons[len(horizons) // 2]

    for ticker, group in feature_matrix.groupby(level="ticker"):
        if ticker not in prices:
            continue
        close = prices[ticker]["Close"]
        close_aligned = close.reindex(group.index.get_level_values("date"))

        # Compute rolling 20-day volatility for adaptive thresholds
        daily_ret = close.pct_change()
        vol_20 = daily_ret.rolling(20).std()
        vol_aligned = vol_20.reindex(group.index.get_level_values("date")).values

        ticker_data = {}
        for h in horizons:
            forward = close_aligned.shift(-h)
            ret = (forward - close_aligned) / close_aligned
            ticker_data[f"target_return_{h}d"] = ret.values
            classes, thresholds = _classify_adaptive(ret.values, vol_aligned)
            ticker_data[f"target_class_{h}d"] = classes
            ticker_data[f"target_threshold_{h}d"] = thresholds

        # Backward-compatible default columns use the middle horizon
        ticker_data["target_return"] = ticker_data[f"target_return_{middle_horizon}d"]
        ticker_data["target_class"] = ticker_data[f"target_class_{middle_horizon}d"]
        ticker_data["target_threshold"] = ticker_data[f"target_threshold_{middle_horizon}d"]

        ticker_targets = pd.DataFrame(ticker_data, index=group.index)
        all_targets.append(ticker_targets)

    if not all_targets:
        return feature_matrix

    target_df = pd.concat(all_targets)
    return feature_matrix.join(target_df)


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
    non_feature_cols = target_cols + ["ticker", "target_threshold"]
    feature_cols = [
        c for c in df.columns
        if c not in non_feature_cols
        and not c.startswith("target_")
        and df[c].dtype in [np.float64, np.int64, float, int]
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


def _classify_adaptive(
    returns: np.ndarray,
    volatilities: np.ndarray,
    sigma_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify returns using volatility-adjusted thresholds.

    threshold_per_stock = sigma_threshold * rolling_volatility
    Large-cap (1.2% vol) -> ~1.0% threshold
    Small-cap (2.5% vol) -> ~2.0% threshold

    Parameters
    ----------
    returns : forward returns array
    volatilities : rolling volatility array (same length as returns)
    sigma_threshold : multiplier for volatility to set threshold

    Returns
    -------
    (classes, thresholds) : classification array and per-row threshold used
    """
    thresholds = sigma_threshold * np.abs(volatilities)
    # Fallback to 1% when volatility is NaN or zero
    thresholds = np.where(
        np.isnan(thresholds) | (thresholds <= 0), 0.01, thresholds
    )

    result = np.zeros(len(returns))
    result[returns > thresholds] = 1
    result[returns < -thresholds] = -1
    result[np.isnan(returns)] = np.nan
    return result, thresholds
