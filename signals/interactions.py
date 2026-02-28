"""Interaction features: non-linear combinations of base signals."""

import numpy as np
import pandas as pd


def compute_interaction_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features from existing signals.

    These capture non-linear relationships that tree models might not
    efficiently learn on their own (especially at shallow depths).

    Parameters
    ----------
    feature_matrix : DataFrame with MultiIndex (date, ticker) and signal columns.

    Returns
    -------
    DataFrame with new interaction columns appended.
    """
    df = feature_matrix.copy()

    # ── RSI x Momentum ──────────────────────────────────────────
    # Oversold + positive momentum = stronger reversal signal
    if "rsi_14" in df.columns and "momentum_5" in df.columns:
        rsi_norm = (df["rsi_14"] - 50) / 50  # [-1, 1]
        df["rsi_x_mom5"] = rsi_norm * df["momentum_5"]

    # ── Volume x Price Movement ─────────────────────────────────
    # High volume + large move = conviction; low volume + large move = suspicious
    if "volume_ratio" in df.columns and "momentum_5" in df.columns:
        df["vol_x_mom5"] = df["volume_ratio"] * df["momentum_5"]

    # ── Bollinger %B x RSI ──────────────────────────────────────
    # Both oversold = strong mean reversion candidate
    if "bb_pct" in df.columns and "rsi_14" in df.columns:
        df["bb_x_rsi"] = df["bb_pct"] * (df["rsi_14"] / 100)

    # ── Trend x Volatility ──────────────────────────────────────
    # Trending in low vol = sustainable; trending in high vol = risky
    if "sma_crossover" in df.columns and "atr_14" in df.columns:
        # Normalize ATR by price to get a vol percentage
        if "sma_10" in df.columns:
            atr_pct = df["atr_14"] / df["sma_10"].replace(0, np.nan)
            df["trend_x_vol"] = df["sma_crossover"] * (1 / atr_pct.clip(0.001))

    # ── MACD x Volume ───────────────────────────────────────────
    # MACD crossover confirmed by volume = stronger signal
    if "macd_histogram" in df.columns and "volume_ratio" in df.columns:
        df["macd_x_vol"] = df["macd_histogram"] * df["volume_ratio"]

    # ── Stochastic x Bollinger ──────────────────────────────────
    # Both at extremes = double confirmation of overbought/oversold
    if "stoch_k" in df.columns and "bb_pct" in df.columns:
        stoch_norm = (df["stoch_k"] - 50) / 50  # [-1, 1]
        bb_norm = (df["bb_pct"] - 0.5) * 2      # [-1, 1]
        df["stoch_x_bb"] = stoch_norm * bb_norm

    # ── Momentum Divergence ─────────────────────────────────────
    # Short-term vs long-term momentum divergence
    if "momentum_5" in df.columns and "momentum_20" in df.columns:
        df["mom_divergence"] = df["momentum_5"] - df["momentum_20"]

    # ── Z-score x Volume ────────────────────────────────────────
    # Extreme z-score + high volume = potential breakout/breakdown
    if "zscore_20" in df.columns and "volume_ratio" in df.columns:
        df["zscore_x_vol"] = df["zscore_20"] * df["volume_ratio"]

    # ── Macro x Momentum ────────────────────────────────────────
    # Risk-on regime + positive momentum = aligned trade
    if "macro_risk_on_off" in df.columns and "momentum_5" in df.columns:
        df["risk_x_mom"] = df["macro_risk_on_off"] * df["momentum_5"]

    # ── VIX Regime x RSI ────────────────────────────────────────
    # Oversold in high VIX = potential bargain; oversold in low VIX = different regime
    if "macro_vix_regime" in df.columns and "rsi_14" in df.columns:
        df["vix_regime_x_rsi"] = df["macro_vix_regime"] * (df["rsi_14"] / 100)

    # ── Fundamental x Technical ─────────────────────────────────
    # Value stock + oversold = deep value signal
    if "fund_value_score" in df.columns and "rsi_14" in df.columns:
        oversold_indicator = np.where(df["rsi_14"] < 30, 1.0, 0.0)
        df["value_x_oversold"] = df["fund_value_score"] * oversold_indicator

    # ── Small-Cap Alpha Interactions ──────────────────────────────────

    # Insider buying + volume spike = institutional front-running signal
    if "fund_insider_cluster_buy" in df.columns and "volume_ratio" in df.columns:
        df["insider_x_volume"] = df["fund_insider_cluster_buy"] * df["volume_ratio"]

    # Short squeeze setup: high short interest + oversold RSI
    if "fund_short_squeeze_score" in df.columns and "rsi_14" in df.columns:
        oversold = np.where(df["rsi_14"] < 35, (35 - df["rsi_14"]) / 35, 0.0)
        df["short_squeeze_setup"] = df["fund_short_squeeze_score"] * oversold

    # Undiscovered value: high DCF upside + low analyst coverage
    if "fund_dcf_upside_pct" in df.columns and "fund_analyst_count" in df.columns:
        low_coverage = np.where(df["fund_analyst_count"] < 8, 1.0, 0.5)
        df["undiscovered_value"] = df["fund_dcf_upside_pct"].clip(lower=0) * low_coverage

    # Quality momentum: strong Piotroski + positive price momentum
    if "fund_piotroski_f_score" in df.columns and "momentum_5" in df.columns:
        quality = df["fund_piotroski_f_score"] / 9.0  # normalize to 0-1
        df["quality_momentum"] = quality * df["momentum_5"].clip(lower=0)

    return df
