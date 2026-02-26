"""Information Coefficient (IC) analysis for signal evaluation."""

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "rank",
) -> float:
    """Compute Information Coefficient between a signal and forward returns.

    Parameters
    ----------
    signal : signal values aligned with forward_returns
    forward_returns : realized returns over the prediction horizon
    method : 'rank' for Spearman rank IC (default), 'pearson' for linear IC

    Returns
    -------
    IC value (correlation between signal and returns).
    """
    valid = signal.notna() & forward_returns.notna()
    if valid.sum() < 20:
        return np.nan

    s = signal[valid]
    r = forward_returns[valid]

    if method == "rank":
        ic, _ = stats.spearmanr(s, r)
    else:
        ic, _ = stats.pearsonr(s, r)

    return float(ic)


def compute_rolling_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 60,
    method: str = "rank",
) -> pd.Series:
    """Compute rolling IC over a sliding window."""
    ic_values = []
    dates = signal.index

    for i in range(window, len(dates)):
        s = signal.iloc[i - window : i]
        r = forward_returns.iloc[i - window : i]
        ic_values.append(compute_ic(s, r, method=method))

    return pd.Series(
        ic_values,
        index=dates[window:],
        name=f"ic_{signal.name}" if hasattr(signal, "name") else "ic",
    )


def ic_analysis_report(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    method: str = "rank",
) -> pd.DataFrame:
    """Generate a comprehensive IC report for all features.

    Parameters
    ----------
    feature_matrix : DataFrame with signal columns
    forward_returns : Series of forward returns aligned by index

    Returns
    -------
    DataFrame with columns: feature, ic, ic_abs, t_stat, p_value, ic_ir
    sorted by absolute IC descending.
    """
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    results = []

    for col in numeric_cols:
        signal = feature_matrix[col]
        valid = signal.notna() & forward_returns.notna()
        if valid.sum() < 30:
            continue

        s = signal[valid]
        r = forward_returns[valid]

        if method == "rank":
            ic, p_val = stats.spearmanr(s, r)
        else:
            ic, p_val = stats.pearsonr(s, r)

        # IC Information Ratio: mean(IC) / std(IC) estimated from rolling
        rolling_ic = []
        window = min(60, len(s) // 4)
        if window >= 20:
            for i in range(window, len(s), window // 2):
                sub_s = s.iloc[i - window : i]
                sub_r = r.iloc[i - window : i]
                if method == "rank":
                    sub_ic, _ = stats.spearmanr(sub_s, sub_r)
                else:
                    sub_ic, _ = stats.pearsonr(sub_s, sub_r)
                rolling_ic.append(sub_ic)

        ic_ir = np.nan
        if len(rolling_ic) > 2:
            ic_std = np.std(rolling_ic)
            if ic_std > 0:
                ic_ir = np.mean(rolling_ic) / ic_std

        # T-statistic
        n = valid.sum()
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-10))

        results.append({
            "feature": col,
            "ic": float(ic),
            "ic_abs": abs(float(ic)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "ic_ir": float(ic_ir) if not np.isnan(ic_ir) else None,
        })

    report = pd.DataFrame(results).sort_values("ic_abs", ascending=False).reset_index(drop=True)
    return report


def turnover_analysis(
    signal: pd.Series,
    groupby_date: bool = True,
) -> float:
    """Compute signal turnover (how much rankings change day-to-day).

    High turnover = unstable signal = higher transaction costs.
    """
    if groupby_date and isinstance(signal.index, pd.MultiIndex):
        # Cross-sectional rank turnover
        ranked = signal.groupby(level="date").rank(pct=True)
        shifted = ranked.groupby(level="ticker").shift(1)
        valid = ranked.notna() & shifted.notna()
        if valid.sum() < 20:
            return np.nan
        return float(np.abs(ranked[valid] - shifted[valid]).mean())
    else:
        shifted = signal.shift(1)
        valid = signal.notna() & shifted.notna()
        if valid.sum() < 20:
            return np.nan
        return float(np.abs(signal[valid] - shifted[valid]).mean())
