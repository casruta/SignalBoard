"""Portfolio optimization: Hierarchical Risk Parity (HRP) with Ledoit-Wolf covariance."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf constant-correlation shrinkage estimator for covariance.

    More stable than sample covariance for portfolio optimization,
    especially when n_samples / n_assets is small.

    Parameters
    ----------
    returns : DataFrame of asset returns (rows=dates, cols=assets)

    Returns
    -------
    Shrunk covariance matrix as numpy array.
    """
    X = returns.values
    n, p = X.shape
    if n < 2 or p < 2:
        return np.cov(X.T) if n >= 2 else np.eye(p)

    # Sample covariance
    S = np.cov(X.T, ddof=1)

    # Target: constant-correlation model
    var = np.diag(S)
    std = np.sqrt(var)
    # Average correlation
    corr_matrix = S / np.outer(std, std)
    np.fill_diagonal(corr_matrix, 1.0)
    avg_corr = (corr_matrix.sum() - p) / (p * (p - 1))
    F = avg_corr * np.outer(std, std)
    np.fill_diagonal(F, var)

    # Optimal shrinkage intensity
    # Simplified Ledoit-Wolf formula
    X_centered = X - X.mean(axis=0)
    sum_sq = 0.0
    for i in range(n):
        xi = X_centered[i].reshape(-1, 1)
        prod = xi @ xi.T
        sum_sq += np.sum((prod - S) ** 2)
    pi = sum_sq / (n ** 2)

    gamma = np.sum((F - S) ** 2)
    kappa = pi / gamma if gamma > 0 else 1.0
    delta = max(0.0, min(1.0, kappa / n))

    return delta * F + (1 - delta) * S


def hrp_portfolio_weights(
    returns: pd.DataFrame,
    use_shrinkage: bool = True,
) -> pd.Series:
    """Compute Hierarchical Risk Parity (HRP) portfolio weights.

    HRP uses hierarchical clustering on the correlation matrix to build
    a diversified portfolio without requiring covariance matrix inversion
    (more stable than Markowitz).

    Parameters
    ----------
    returns : DataFrame of asset returns (rows=dates, cols=tickers)
    use_shrinkage : whether to use Ledoit-Wolf shrinkage

    Returns
    -------
    Series of portfolio weights indexed by ticker, summing to 1.0.
    """
    assets = returns.columns.tolist()
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series({assets[0]: 1.0})

    # Step 1: Covariance and correlation
    if use_shrinkage:
        cov = ledoit_wolf_shrinkage(returns)
    else:
        cov = np.cov(returns.values.T, ddof=1)

    std = np.sqrt(np.diag(cov))
    std = np.where(std > 0, std, 1e-8)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Step 2: Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)

    # Step 3: Hierarchical clustering
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method="single")
    sort_idx = leaves_list(link).tolist()

    # Step 4: Recursive bisection
    weights = np.ones(n)
    cluster_items = [sort_idx]

    while cluster_items:
        next_items = []
        for cluster in cluster_items:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Cluster variance (inverse-variance allocation)
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)

            total_var = left_var + right_var
            if total_var > 0:
                alpha = 1 - left_var / total_var
            else:
                alpha = 0.5

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)

            next_items.append(left)
            next_items.append(right)

        cluster_items = next_items

    # Normalize
    weights = weights / weights.sum()
    return pd.Series(weights, index=assets)


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Compute the variance of an equal-weight portfolio of cluster members."""
    if not indices:
        return 0.0
    sub_cov = cov[np.ix_(indices, indices)]
    n = len(indices)
    w = np.ones(n) / n
    return float(w @ sub_cov @ w)


def black_litterman_returns(
    market_cap_weights: pd.Series,
    cov: np.ndarray,
    views: dict[str, float],
    tau: float = 0.05,
    view_confidence: float = 0.5,
) -> pd.Series:
    """Compute Black-Litterman expected returns.

    Blends market-implied equilibrium returns with investor views.

    Parameters
    ----------
    market_cap_weights : market-cap-weighted portfolio (prior)
    cov : covariance matrix
    views : {ticker: expected_return} absolute views
    tau : uncertainty scaling of the prior
    view_confidence : confidence in views (0-1, higher = more confident)

    Returns
    -------
    Series of posterior expected returns.
    """
    assets = market_cap_weights.index.tolist()
    n = len(assets)
    w_mkt = market_cap_weights.values

    # Risk aversion parameter (typical value)
    delta = 2.5

    # Equilibrium returns: pi = delta * Sigma * w_mkt
    pi = delta * cov @ w_mkt

    if not views:
        return pd.Series(pi, index=assets)

    # Build P (pick matrix) and Q (view vector)
    view_tickers = [t for t in views if t in assets]
    if not view_tickers:
        return pd.Series(pi, index=assets)

    k = len(view_tickers)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    for i, ticker in enumerate(view_tickers):
        j = assets.index(ticker)
        P[i, j] = 1.0
        Q[i] = views[ticker]

    # View uncertainty: omega = diag(P * tau * Sigma * P')
    omega = np.diag(np.diag(P @ (tau * cov) @ P.T)) / view_confidence

    # Posterior returns
    tau_cov = tau * cov
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(omega)

    posterior_cov = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
    posterior_returns = posterior_cov @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

    return pd.Series(posterior_returns, index=assets)


def kelly_fraction(
    expected_return: float,
    variance: float,
    max_leverage: float = 1.0,
) -> float:
    """Compute the Kelly criterion fraction for position sizing.

    Kelly fraction = expected_return / variance (continuous approximation).
    Capped at max_leverage for risk management.

    Parameters
    ----------
    expected_return : expected return of the asset
    variance : variance of the asset's returns
    max_leverage : maximum allowed fraction (1.0 = no leverage)

    Returns
    -------
    Optimal fraction of portfolio to allocate.
    """
    if variance <= 0 or expected_return <= 0:
        return 0.0

    # Full Kelly
    f_star = expected_return / variance

    # Half Kelly (standard practice for robustness)
    f_half = f_star / 2.0

    return min(f_half, max_leverage)
