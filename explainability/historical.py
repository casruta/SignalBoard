"""Historical context — how similar signal patterns performed in the past."""

import numpy as np
import pandas as pd


def find_similar_signals(
    feature_matrix: pd.DataFrame,
    target_row: pd.Series,
    key_features: list[str],
    prices: dict[str, pd.DataFrame],
    ticker: str,
    forward_days: int = 10,
    n_similar: int = 50,
) -> dict:
    """Find historically similar signal patterns and their outcomes.

    Parameters
    ----------
    feature_matrix : full historical feature matrix (MultiIndex: date, ticker)
    target_row : the current signal's feature values
    key_features : the most important features to match on
    prices : {ticker: OHLCV}
    ticker : the ticker to analyze
    forward_days : how many days forward to measure outcome
    n_similar : max number of similar historical instances

    Returns
    -------
    dict with: count, avg_return, win_rate, summary_text
    """
    # Filter to just this ticker's history
    if ticker not in feature_matrix.index.get_level_values("ticker"):
        return _empty_result()

    ticker_data = feature_matrix.xs(ticker, level="ticker")

    # Use only the key features
    available = [f for f in key_features if f in ticker_data.columns and f in target_row.index]
    if not available:
        return _empty_result()

    # Compute distance to current signal (normalized Euclidean)
    current_vals = target_row[available].values.astype(float)
    hist_vals = ticker_data[available].values.astype(float)

    # Z-score normalize for distance computation
    mean = np.nanmean(hist_vals, axis=0)
    std = np.nanstd(hist_vals, axis=0)
    std[std == 0] = 1

    current_norm = (current_vals - mean) / std
    hist_norm = (hist_vals - mean) / std

    # Euclidean distance
    distances = np.sqrt(np.nansum((hist_norm - current_norm) ** 2, axis=1))

    # Get the closest N instances
    sorted_idx = np.argsort(distances)
    top_indices = sorted_idx[:n_similar]

    # Calculate forward returns for those dates
    if ticker not in prices:
        return _empty_result()

    close = prices[ticker]["Close"]
    similar_dates = ticker_data.index[top_indices]

    forward_returns = []
    for date in similar_dates:
        if date not in close.index:
            continue
        future_idx = close.index.get_loc(date)
        if future_idx + forward_days >= len(close):
            continue
        entry = close.iloc[future_idx]
        exit_ = close.iloc[future_idx + forward_days]
        forward_returns.append((exit_ - entry) / entry)

    if not forward_returns:
        return _empty_result()

    returns = np.array(forward_returns)
    avg_return = float(np.mean(returns))
    win_rate = float(np.mean(returns > 0))
    count = len(returns)

    summary = (
        f"In the last {count} similar setups for {ticker}, "
        f"the stock {'gained' if avg_return > 0 else 'lost'} "
        f"an average of {abs(avg_return):.1%} over {forward_days} days "
        f"({win_rate:.0%} win rate)"
    )

    return {
        "count": count,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "forward_days": forward_days,
        "summary": summary,
    }


def _empty_result() -> dict:
    return {
        "count": 0,
        "avg_return": 0.0,
        "win_rate": 0.0,
        "forward_days": 0,
        "summary": "Insufficient historical data for comparison",
    }
