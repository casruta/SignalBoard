"""Fetch and cache daily OHLCV price data via yfinance."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

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
        if cached.index.tz is not None:
            cached.index = cached.index.tz_localize(None)
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
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Fetch prices for multiple tickers in parallel."""
    results = {}

    def _fetch_one(t):
        return t, fetch_prices(t, start=start, end=end, lookback_years=lookback_years)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, df = future.result()
                if not df.empty:
                    results[t] = df
            except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
                logger.warning("Price fetch failed for %s: %s", ticker, e)

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
    # Normalize to tz-naive so cached parquet and Timestamp.now() comparisons work
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
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
