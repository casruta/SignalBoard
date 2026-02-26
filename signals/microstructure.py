"""Microstructure features: liquidity, price efficiency, and volatility estimators."""

import numpy as np
import pandas as pd


def compute_microstructure_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute microstructure features from OHLCV data.

    Parameters
    ----------
    prices : DataFrame with columns Open, High, Low, Close, Volume.

    Returns
    -------
    DataFrame with microstructure signal columns.
    """
    df = pd.DataFrame(index=prices.index)
    close = prices["Close"]
    high = prices["High"]
    low = prices["Low"]
    open_ = prices["Open"]
    volume = prices["Volume"]

    # ── Amihud Illiquidity ──────────────────────────────────────
    # |return| / dollar volume — higher = less liquid
    dollar_volume = close * volume
    daily_return = close.pct_change().abs()
    raw_amihud = daily_return / dollar_volume.replace(0, np.nan)
    df["amihud_illiq_20"] = raw_amihud.rolling(20, min_periods=10).mean()

    # ── Close Location Value (CLV) ──────────────────────────────
    # Where the close sits within the day's range: -1 (near low) to +1 (near high)
    hl_range = high - low
    df["clv"] = np.where(
        hl_range > 0,
        ((close - low) - (high - close)) / hl_range,
        0.0,
    )
    df["clv_sma_10"] = df["clv"].rolling(10).mean()

    # ── Garman-Klass Volatility ─────────────────────────────────
    # More efficient volatility estimator than close-to-close
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df["garman_klass_vol_20"] = np.sqrt(gk_daily.rolling(20, min_periods=10).mean() * 252)

    # ── Parkinson Volatility ────────────────────────────────────
    # Uses high-low range only
    log_hl_sq = np.log(high / low) ** 2
    df["parkinson_vol_20"] = np.sqrt(
        log_hl_sq.rolling(20, min_periods=10).mean() / (4 * np.log(2)) * 252
    )

    # ── Volume-Weighted Price Momentum ──────────────────────────
    # Dollar-volume-weighted return over 5 days
    ret = close.pct_change()
    dvol = close * volume
    df["vwap_momentum_5"] = (
        (ret * dvol).rolling(5).sum() / dvol.rolling(5).sum()
    )

    # ── Kyle's Lambda Proxy ─────────────────────────────────────
    # Approximation of price impact: |return| / sqrt(volume)
    sqrt_vol = np.sqrt(volume.replace(0, np.nan))
    df["kyle_lambda_20"] = (daily_return / sqrt_vol).rolling(20, min_periods=10).mean()

    # ── Intraday Range Ratio ────────────────────────────────────
    # Ratio of true range to ATR — spikes indicate unusual volatility
    true_range = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ),
    )
    atr_14 = true_range.rolling(14).mean()
    df["range_ratio"] = np.where(atr_14 > 0, true_range / atr_14, 1.0)

    return df
