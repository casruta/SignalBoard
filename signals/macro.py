"""Macro regime signals derived from economic indicators."""

import numpy as np
import pandas as pd


def compute_macro_signals(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro regime signals from a FRED macro DataFrame.

    Parameters
    ----------
    macro_df : DataFrame with columns from macro_loader.fetch_all_macro():
               treasury_10y, treasury_2y, vix, oil_wti, etc.

    Returns
    -------
    DataFrame with macro signal columns, same index.
    """
    df = pd.DataFrame(index=macro_df.index)

    # ── Yield Curve Slope (10Y - 2Y) ────────────────────────────
    if "treasury_10y" in macro_df and "treasury_2y" in macro_df:
        df["yield_curve_slope"] = macro_df["treasury_10y"] - macro_df["treasury_2y"]
        df["yield_curve_inverted"] = (df["yield_curve_slope"] < 0).astype(float)
    else:
        df["yield_curve_slope"] = np.nan
        df["yield_curve_inverted"] = np.nan

    # ── VIX Regime ───────────────────────────────────────────────
    if "vix" in macro_df:
        vix = macro_df["vix"]
        df["vix"] = vix
        df["vix_sma_20"] = vix.rolling(20).mean()
        # Regime buckets: low < 15, normal 15-25, high > 25
        df["vix_regime"] = pd.cut(
            vix,
            bins=[0, 15, 25, 100],
            labels=[0, 1, 2],  # 0=low, 1=normal, 2=high
        ).astype(float)
        df["vix_trend"] = vix.pct_change(10)  # 10-day VIX momentum
    else:
        df["vix"] = np.nan
        df["vix_regime"] = np.nan
        df["vix_trend"] = np.nan

    # ── Oil Price Trend (WTI) ────────────────────────────────────
    if "oil_wti" in macro_df:
        oil = macro_df["oil_wti"]
        df["oil_wti"] = oil
        df["oil_momentum_20"] = oil.pct_change(20)
        df["oil_sma_50"] = oil.rolling(50).mean()
        df["oil_above_sma50"] = (oil > df["oil_sma_50"]).astype(float)
    else:
        df["oil_wti"] = np.nan
        df["oil_momentum_20"] = np.nan
        df["oil_above_sma50"] = np.nan

    # ── Risk-On / Risk-Off Composite ─────────────────────────────
    # Simple composite: positive = risk-on, negative = risk-off
    risk_components = []
    if "vix_trend" in df:
        # Declining VIX = risk-on (invert sign)
        risk_components.append(-_zscore(df["vix_trend"]))
    if "yield_curve_slope" in df:
        # Positive slope = risk-on
        risk_components.append(_zscore(df["yield_curve_slope"]))
    if "oil_momentum_20" in df:
        # Rising oil = risk-on (commodity demand)
        risk_components.append(_zscore(df["oil_momentum_20"]))

    if risk_components:
        df["risk_on_off"] = sum(risk_components) / len(risk_components)
        df["risk_regime"] = pd.cut(
            df["risk_on_off"],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[-1, 0, 1],  # -1=risk-off, 0=neutral, 1=risk-on
        ).astype(float)
    else:
        df["risk_on_off"] = np.nan
        df["risk_regime"] = np.nan

    return df


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score normalization."""
    mean = series.rolling(window, min_periods=20).mean()
    std = series.rolling(window, min_periods=20).std()
    return (series - mean) / std.replace(0, np.nan)
