"""Cross-sectional features: relative ranking across the ticker universe."""

import numpy as np
import pandas as pd


def _assign_cap_tier(market_cap: pd.Series) -> pd.Series:
    """Assign market-cap tier labels based on dollar value.

    <2B  = 'small'
    2B-10B = 'mid'
    >10B = 'large'
    """
    tiers = pd.Series("large", index=market_cap.index)
    tiers[market_cap < 2e9] = "small"
    tiers[(market_cap >= 2e9) & (market_cap < 10e9)] = "mid"
    tiers[market_cap.isna()] = np.nan
    return tiers


def compute_cross_sectional_features(
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cross-sectional (relative) features for each date.

    If ``fund_market_cap`` is present, rankings are computed within
    market-cap tiers (small / mid / large) so that a 2% return is
    evaluated against peers of similar size. Falls back to full-
    universe ranking when market cap data is unavailable.

    Parameters
    ----------
    feature_matrix : DataFrame with MultiIndex (date, ticker) and numeric signal columns.

    Returns
    -------
    DataFrame with new cross-sectional columns appended.
    """
    df = feature_matrix.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Determine grouping keys: cap-tier-aware or full universe
    has_cap = "fund_market_cap" in df.columns
    if has_cap:
        df["cap_tier"] = _assign_cap_tier(df["fund_market_cap"])
        rank_group = [df.index.get_level_values("date"), df["cap_tier"]]
    else:
        rank_group = "date"  # groupby level name

    # Key signals to rank cross-sectionally
    rank_signals = [
        "momentum_5", "momentum_10", "momentum_20",
        "rsi_14", "volume_ratio", "zscore_20",
        "atr_14", "bb_pct",
    ]
    rank_signals = [s for s in rank_signals if s in numeric_cols]

    for signal in rank_signals:
        if has_cap:
            # Rank within [date, cap_tier]
            df[f"xs_rank_{signal}"] = df.groupby(rank_group)[signal].rank(pct=True)
            # Also keep full-universe rank as fallback / complementary feature
            df[f"xs_rank_{signal}_univ"] = df.groupby(level="date")[signal].rank(pct=True)
        else:
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
        if has_cap:
            # Per-tier dispersion
            disp_tier = df.groupby(rank_group)["momentum_5"].transform("std")
            df["xs_momentum_dispersion_tier"] = disp_tier
        # Full-universe dispersion (always useful)
        disp = df.groupby(level="date")["momentum_5"].transform("std")
        df["xs_momentum_dispersion"] = disp

    # Breadth: fraction of universe with positive momentum
    if "momentum_5" in numeric_cols:
        breadth = df.groupby(level="date")["momentum_5"].transform(
            lambda x: (x > 0).mean()
        )
        df["xs_market_breadth"] = breadth

    return df
