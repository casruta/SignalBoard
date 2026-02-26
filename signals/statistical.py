"""Statistical regime features: Hurst exponent, variance ratio, autocorrelation."""

import numpy as np
import pandas as pd


def compute_statistical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical features that detect market regime characteristics.

    Parameters
    ----------
    prices : DataFrame with columns Open, High, Low, Close, Volume.

    Returns
    -------
    DataFrame with statistical regime columns.
    """
    df = pd.DataFrame(index=prices.index)
    close = prices["Close"]
    log_ret = np.log(close / close.shift(1))

    # ── Hurst Exponent (rolling) ────────────────────────────────
    # H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending
    df["hurst_100"] = _rolling_hurst(log_ret, window=100)

    # ── Variance Ratio ──────────────────────────────────────────
    # VR(q) = Var(q-period return) / (q * Var(1-period return))
    # VR > 1: momentum/trending, VR < 1: mean-reverting
    df["var_ratio_5"] = _rolling_variance_ratio(log_ret, q=5, window=60)
    df["var_ratio_20"] = _rolling_variance_ratio(log_ret, q=20, window=120)

    # ── Autocorrelation ─────────────────────────────────────────
    df["autocorr_1_20"] = log_ret.rolling(20, min_periods=15).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    df["autocorr_5_60"] = log_ret.rolling(60, min_periods=30).apply(
        lambda x: _safe_autocorr(x, 5), raw=False
    )

    # ── Realized Volatility (different windows) ─────────────────
    df["realized_vol_5"] = log_ret.rolling(5).std() * np.sqrt(252)
    df["realized_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["realized_vol_60"] = log_ret.rolling(60).std() * np.sqrt(252)

    # ── Volatility of Volatility (vol clustering) ───────────────
    rv20 = df["realized_vol_20"]
    df["vol_of_vol_20"] = rv20.rolling(20, min_periods=10).std()

    # ── Skewness and Kurtosis ───────────────────────────────────
    df["skew_20"] = log_ret.rolling(20, min_periods=15).skew()
    df["kurtosis_20"] = log_ret.rolling(20, min_periods=15).kurt()

    # ── Regime Indicator (composite) ────────────────────────────
    # Combine Hurst + VR into a single trending/reverting score
    # Positive = trending regime, Negative = mean-reverting
    hurst_centered = df["hurst_100"] - 0.5
    vr_centered = df["var_ratio_5"] - 1.0
    valid_mask = hurst_centered.notna() & vr_centered.notna()
    df["regime_score"] = np.where(
        valid_mask,
        0.5 * _clip_zscore(hurst_centered) + 0.5 * _clip_zscore(vr_centered),
        np.nan,
    )

    return df


def _rolling_hurst(returns: pd.Series, window: int = 100) -> pd.Series:
    """Estimate Hurst exponent using the R/S method over a rolling window."""
    def _hurst_rs(x):
        x = x.dropna()
        n = len(x)
        if n < 20:
            return np.nan
        # Use two sub-periods for a quick R/S estimate
        max_k = min(n // 2, 50)
        if max_k < 4:
            return np.nan

        rs_list = []
        for k in [max_k // 4, max_k // 2, max_k]:
            if k < 4:
                continue
            num_blocks = n // k
            rs_vals = []
            for b in range(num_blocks):
                block = x.iloc[b * k : (b + 1) * k].values
                mean_block = np.mean(block)
                deviations = np.cumsum(block - mean_block)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(block, ddof=1)
                if s > 0:
                    rs_vals.append(r / s)
            if rs_vals:
                rs_list.append((np.log(k), np.log(np.mean(rs_vals))))

        if len(rs_list) < 2:
            return np.nan

        # Linear regression slope = Hurst exponent
        log_n = np.array([p[0] for p in rs_list])
        log_rs = np.array([p[1] for p in rs_list])
        slope = np.polyfit(log_n, log_rs, 1)[0]
        return np.clip(slope, 0.0, 1.0)

    return returns.rolling(window, min_periods=window // 2).apply(_hurst_rs, raw=False)


def _rolling_variance_ratio(
    returns: pd.Series, q: int = 5, window: int = 60
) -> pd.Series:
    """Rolling variance ratio VR(q)."""
    def _vr(x):
        x = x.dropna().values
        n = len(x)
        if n < q + 10:
            return np.nan
        var_1 = np.var(x, ddof=1)
        if var_1 == 0:
            return np.nan
        # q-period returns
        q_rets = np.array([x[i:i+q].sum() for i in range(n - q + 1)])
        var_q = np.var(q_rets, ddof=1)
        return var_q / (q * var_1)

    return returns.rolling(window, min_periods=window // 2).apply(_vr, raw=False)


def _safe_autocorr(x, lag):
    """Autocorrelation with safety checks."""
    if len(x) <= lag + 1:
        return np.nan
    return x.autocorr(lag=lag) if hasattr(x, "autocorr") else np.nan


def _clip_zscore(s: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score clipped to [-3, 3]."""
    mean = s.rolling(window, min_periods=20).mean()
    std = s.rolling(window, min_periods=20).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.clip(-3, 3)
