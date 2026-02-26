"""Network/graph analysis: correlation networks, lead-lag, sector rotation, spillovers."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_correlation_network(
    returns: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.50,
) -> dict:
    """Build a correlation network from rolling return correlations.

    Parameters
    ----------
    returns : DataFrame with tickers as columns, daily returns as rows
    window : rolling window for correlation
    threshold : minimum |correlation| to form an edge

    Returns
    -------
    dict with 'adjacency' (matrix), 'degree_centrality', 'avg_correlation'
    """
    n = len(returns.columns)
    if n < 2 or len(returns) < window:
        return {"adjacency": np.zeros((n, n)), "degree_centrality": {}, "avg_correlation": 0.0}

    # Use the last `window` days
    recent = returns.tail(window).dropna(axis=1, how="all")
    corr = recent.corr()
    tickers = corr.columns.tolist()

    # Adjacency matrix (edges where |corr| > threshold)
    adj = (corr.abs() > threshold).astype(int).values.copy()
    np.fill_diagonal(adj, 0)

    # Degree centrality: fraction of connections
    degrees = adj.sum(axis=1)
    max_degree = n - 1 if n > 1 else 1
    degree_centrality = {
        tickers[i]: float(degrees[i] / max_degree) for i in range(len(tickers))
    }

    # Average correlation (excluding diagonal)
    mask = ~np.eye(len(corr), dtype=bool)
    avg_corr = float(corr.values[mask].mean()) if mask.sum() > 0 else 0.0

    return {
        "adjacency": adj,
        "tickers": tickers,
        "degree_centrality": degree_centrality,
        "avg_correlation": avg_corr,
    }


def compute_network_features(
    prices: dict[str, pd.DataFrame],
    window: int = 60,
) -> pd.DataFrame:
    """Compute network-based features for the entire universe.

    Features per ticker:
    - net_degree: how connected to other assets
    - net_avg_corr: average correlation with other assets
    - net_lead_lag_score: does this ticker lead or lag the market

    Returns DataFrame indexed by ticker.
    """
    # Build returns DataFrame
    close_dict = {}
    for ticker, df in prices.items():
        if "Close" in df.columns:
            close_dict[ticker] = df["Close"]

    if not close_dict:
        return pd.DataFrame()

    closes = pd.DataFrame(close_dict)
    returns = closes.pct_change().dropna()

    if len(returns) < window:
        return pd.DataFrame(index=list(prices.keys()))

    tickers = returns.columns.tolist()
    network = build_correlation_network(returns, window=window)

    # Lead-lag: cross-correlation at lag 1
    lead_lag_scores = {}
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        # Average correlation of this ticker's returns with other tickers' next-day returns
        leads = []
        for other in tickers:
            if other == ticker or other not in returns.columns:
                continue
            # Does ticker lead other? Corr(ticker_t, other_{t+1})
            shifted = returns[other].shift(-1)
            valid = returns[ticker].notna() & shifted.notna()
            if valid.sum() > 20:
                corr = np.corrcoef(returns[ticker][valid].tail(window), shifted[valid].tail(window))[0, 1]
                if not np.isnan(corr):
                    leads.append(corr)
        lead_lag_scores[ticker] = float(np.mean(leads)) if leads else 0.0

    result = pd.DataFrame(index=tickers)
    result["net_degree"] = pd.Series(network["degree_centrality"])
    result["net_avg_corr"] = network["avg_correlation"]
    result["net_lead_lag_score"] = pd.Series(lead_lag_scores)

    return result


def compute_sector_rotation_signals(
    prices: dict[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
) -> dict[str, float]:
    """Compute sector momentum for rotation signals.

    Returns {sector: momentum_score} based on trailing returns.
    """
    if "sector" not in fundamentals.columns:
        return {}

    sector_returns = {}
    for ticker, df in prices.items():
        if ticker not in fundamentals.index or "Close" not in df.columns:
            continue
        sector = fundamentals.loc[ticker, "sector"]
        ret_20d = df["Close"].pct_change(20).iloc[-1] if len(df) > 20 else np.nan
        if not np.isnan(ret_20d):
            sector_returns.setdefault(sector, []).append(ret_20d)

    return {
        sector: float(np.mean(rets))
        for sector, rets in sector_returns.items()
        if rets
    }


def compute_spillover_index(
    returns: pd.DataFrame,
    lag_order: int = 2,
    forecast_horizon: int = 5,
) -> dict:
    """Compute a simplified Diebold-Yilmaz spillover index.

    Uses pairwise Granger-causal relationships (OLS-based) as a
    lightweight proxy for the full VAR forecast error variance decomposition.

    Parameters
    ----------
    returns : DataFrame with tickers as columns
    lag_order : number of lags in the VAR-like regression
    forecast_horizon : not used directly in this simplified version

    Returns
    -------
    dict with 'total_spillover', 'directional_to', 'directional_from'
    """
    tickers = returns.columns.tolist()
    n = len(tickers)
    if n < 2 or len(returns) < lag_order + 20:
        return {"total_spillover": 0.0, "directional_to": {}, "directional_from": {}}

    recent = returns.dropna().tail(252)  # Use last year of data
    spillover_matrix = np.zeros((n, n))

    for i, target in enumerate(tickers):
        y = recent[target].values[lag_order:]
        # Regressors: own lags + other ticker lags
        X_own = np.column_stack([
            recent[target].shift(lag).values[lag_order:]
            for lag in range(1, lag_order + 1)
        ])

        for j, source in enumerate(tickers):
            if i == j:
                continue
            X_source = np.column_stack([
                recent[source].shift(lag).values[lag_order:]
                for lag in range(1, lag_order + 1)
            ])

            # Full model: own lags + source lags
            X_full = np.column_stack([X_own, X_source])
            valid = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y)
            if valid.sum() < lag_order * 2 + 10:
                continue

            X_f = X_full[valid]
            X_r = X_own[valid]
            y_v = y[valid]

            # R² improvement from adding source
            try:
                r2_full = _ols_r2(X_f, y_v)
                r2_own = _ols_r2(X_r, y_v)
                spillover_matrix[i, j] = max(0, r2_full - r2_own)
            except Exception:
                pass

    # Normalize
    total = spillover_matrix.sum()
    max_total = n * (n - 1)
    total_spillover = float(total / max_total) if max_total > 0 else 0.0

    directional_to = {tickers[i]: float(spillover_matrix[i, :].sum()) for i in range(n)}
    directional_from = {tickers[j]: float(spillover_matrix[:, j].sum()) for j in range(n)}

    return {
        "total_spillover": total_spillover,
        "directional_to": directional_to,
        "directional_from": directional_from,
    }


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS R-squared."""
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        y_hat = X_with_const @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        return 0.0
