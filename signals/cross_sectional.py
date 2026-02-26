"""Cross-sectional features: relative ranking across the ticker universe."""

import numpy as np
import pandas as pd


def compute_cross_sectional_features(
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cross-sectional (relative) features for each date.

    For each date, ranks each ticker's signals against the full universe
    so the model sees *relative* strength, not just absolute values.

    Parameters
    ----------
    feature_matrix : DataFrame with MultiIndex (date, ticker) and numeric signal columns.

    Returns
    -------
    DataFrame with new cross-sectional columns appended.
    """
    df = feature_matrix.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Key signals to rank cross-sectionally
    rank_signals = [
        "momentum_5", "momentum_10", "momentum_20",
        "rsi_14", "volume_ratio", "zscore_20",
        "atr_14", "bb_pct",
    ]
    rank_signals = [s for s in rank_signals if s in numeric_cols]

    for signal in rank_signals:
        # Percentile rank within each date (0 = weakest, 1 = strongest)
        df[f"xs_rank_{signal}"] = df.groupby(level="date")[signal].rank(pct=True)

    # Sector-relative momentum (if sector info available)
    if "fund_sector" in df.columns and "momentum_5" in df.columns:
        df["sector_rel_mom_5"] = df.groupby(
            [df.index.get_level_values("date"), "fund_sector"]
        )["momentum_5"].transform(lambda x: x - x.mean())

    # Universe-wide z-scores (cross-sectional standardization)
    for signal in ["momentum_5", "rsi_14", "volume_ratio"]:
        if signal not in numeric_cols:
            continue
        grouped = df.groupby(level="date")[signal]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        df[f"xs_zscore_{signal}"] = (df[signal] - mean) / std

    # Dispersion features (market-wide breadth)
    if "momentum_5" in numeric_cols:
        # Cross-sectional volatility of momentum = market dispersion
        disp = df.groupby(level="date")["momentum_5"].transform("std")
        df["xs_momentum_dispersion"] = disp

    # Breadth: fraction of universe with positive momentum
    if "momentum_5" in numeric_cols:
        breadth = df.groupby(level="date")["momentum_5"].transform(
            lambda x: (x > 0).mean()
        )
        df["xs_market_breadth"] = breadth

    return df
