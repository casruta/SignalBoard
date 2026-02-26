"""Technical analysis signals computed from price data."""

import numpy as np
import pandas as pd
import ta


def compute_all_technical(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical signals from an OHLCV DataFrame.

    Parameters
    ----------
    prices : DataFrame with columns Open, High, Low, Close, Volume

    Returns
    -------
    DataFrame with technical signal columns, same index as *prices*.
    """
    df = pd.DataFrame(index=prices.index)

    close = prices["Close"]
    high = prices["High"]
    low = prices["Low"]
    volume = prices["Volume"]

    # ── Trend / Moving Averages ──────────────────────────────────
    df["sma_10"] = ta.trend.sma_indicator(close, window=10)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["sma_crossover"] = (df["sma_10"] > df["sma_50"]).astype(float)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)

    # ── RSI ──────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(close, window=14)

    # ── MACD ─────────────────────────────────────────────────────
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # ── Bollinger Bands ──────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()  # %B: where price is within bands

    # ── ATR (volatility) ─────────────────────────────────────────
    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # ── Momentum (N-day returns) ─────────────────────────────────
    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)
    df["momentum_20"] = close.pct_change(20)

    # ── Volume Profile ───────────────────────────────────────────
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"]

    # ── Mean Reversion (Z-score) ─────────────────────────────────
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    df["zscore_20"] = (close - rolling_mean) / rolling_std

    # ── Stochastic Oscillator ────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    return df


def get_signal_names() -> list[str]:
    """Return list of all technical signal column names."""
    return [
        "sma_10", "sma_50", "sma_crossover", "ema_12", "ema_26",
        "rsi_14",
        "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_lower", "bb_width", "bb_pct",
        "atr_14",
        "momentum_5", "momentum_10", "momentum_20",
        "volume_sma_20", "volume_ratio",
        "zscore_20",
        "stoch_k", "stoch_d",
    ]
