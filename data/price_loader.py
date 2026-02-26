"""Fetch and cache daily OHLCV price data via yfinance."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache" / "prices"


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def fetch_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    lookback_years: int = 10,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for *ticker*, using Parquet cache when possible.

    Columns: Open, High, Low, Close, Volume (adjusted prices).
    Index: DatetimeIndex named 'Date'.
    """
    cache = _cache_path(ticker)

    if not force_refresh and cache.exists():
        cached = pd.read_parquet(cache)
        last_date = cached.index.max()
        today = pd.Timestamp.now().normalize()
        # If cache is fresh (updated today or market hasn't closed yet), return it
        if last_date >= today - pd.Timedelta(days=1):
            return _filter(cached, start, end)
        # Otherwise, fetch only the missing tail
        new_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        fresh = _download(ticker, start=new_start, end=end)
        if not fresh.empty:
            cached = pd.concat([cached, fresh])
            cached = cached[~cached.index.duplicated(keep="last")]
            cached.sort_index(inplace=True)
        _save(cached, cache)
        return _filter(cached, start, end)

    # Full download
    if start is None:
        start = (datetime.now() - timedelta(days=365 * lookback_years)).strftime(
            "%Y-%m-%d"
        )
    df = _download(ticker, start=start, end=end)
    if not df.empty:
        _save(df, cache)
    return df


def fetch_prices_bulk(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    lookback_years: int = 10,
) -> dict[str, pd.DataFrame]:
    """Fetch prices for multiple tickers. Returns {ticker: DataFrame}."""
    results = {}
    for t in tickers:
        df = fetch_prices(t, start=start, end=end, lookback_years=lookback_years)
        if not df.empty:
            results[t] = df
    return results


def _download(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Download from yfinance."""
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        return df
    # Keep only OHLCV columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]
    df.index.name = "Date"
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _filter(
    df: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df
