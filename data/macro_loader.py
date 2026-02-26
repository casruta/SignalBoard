"""Fetch macroeconomic indicators from FRED API."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache" / "macro"

# FRED series IDs for key macro indicators
SERIES = {
    "treasury_10y": "DGS10",       # 10-Year Treasury Yield
    "treasury_2y": "DGS2",         # 2-Year Treasury Yield
    "fed_funds": "FEDFUNDS",       # Federal Funds Rate
    "vix": "VIXCLS",               # CBOE Volatility Index
    "unemployment": "UNRATE",      # Unemployment Rate
    "cpi": "CPIAUCSL",             # Consumer Price Index
    "oil_wti": "DCOILWTICO",       # WTI Crude Oil Price
}


def _get_fred(api_key: str):
    """Lazy import and instantiate Fred client."""
    from fredapi import Fred
    return Fred(api_key=api_key)


def fetch_macro_series(
    series_name: str,
    api_key: str,
    start: str | None = None,
    lookback_years: int = 10,
    force_refresh: bool = False,
) -> pd.Series:
    """Fetch a single FRED series by friendly name (e.g. 'vix', 'treasury_10y').

    Returns a pd.Series with DatetimeIndex.
    """
    if series_name not in SERIES:
        raise ValueError(
            f"Unknown series '{series_name}'. Available: {list(SERIES.keys())}"
        )
    fred_id = SERIES[series_name]
    cache = CACHE_DIR / f"{series_name}.parquet"

    if not force_refresh and cache.exists():
        cached = pd.read_parquet(cache).squeeze()
        last_date = cached.index.max()
        if last_date >= pd.Timestamp.now().normalize() - pd.Timedelta(days=2):
            return _filter_start(cached, start)

    if start is None:
        start = (datetime.now() - timedelta(days=365 * lookback_years)).strftime(
            "%Y-%m-%d"
        )

    fred = _get_fred(api_key)
    data = fred.get_series(fred_id, observation_start=start)
    data.name = series_name
    data.index.name = "Date"
    data = data.dropna()

    # Cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    data.to_frame().to_parquet(cache)

    return data


def fetch_all_macro(
    api_key: str,
    start: str | None = None,
    lookback_years: int = 10,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch all macro series and return as a single DataFrame.

    Columns: treasury_10y, treasury_2y, fed_funds, vix, unemployment, cpi, oil_wti.
    Index: DatetimeIndex (daily, forward-filled for alignment).
    """
    frames = {}
    for name in SERIES:
        try:
            s = fetch_macro_series(
                name,
                api_key=api_key,
                start=start,
                lookback_years=lookback_years,
                force_refresh=force_refresh,
            )
            frames[name] = s
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.sort_index(inplace=True)
    # Forward-fill to align different release frequencies
    df.ffill(inplace=True)
    return df


def compute_yield_curve_slope(macro_df: pd.DataFrame) -> pd.Series:
    """10Y - 2Y treasury spread."""
    if "treasury_10y" in macro_df and "treasury_2y" in macro_df:
        return macro_df["treasury_10y"] - macro_df["treasury_2y"]
    return pd.Series(dtype=float)


def _filter_start(s: pd.Series, start: str | None) -> pd.Series:
    if start:
        s = s[s.index >= start]
    return s
