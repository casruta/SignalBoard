"""Fetch fundamental / financial data via yfinance."""

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache" / "fundamentals"


def fetch_fundamentals(ticker: str, force_refresh: bool = False) -> dict:
    """Return a dict of fundamental metrics for *ticker*.

    Keys: pe_ratio, forward_pe, eps, eps_growth, revenue_growth,
          debt_to_equity, roe, dividend_yield, pb_ratio, sector, industry,
          market_cap, short_name.
    """
    cache = CACHE_DIR / f"{ticker}.parquet"

    if not force_refresh and cache.exists():
        cached = pd.read_parquet(cache)
        # Refresh if older than 7 days
        if not cached.empty:
            age = pd.Timestamp.now() - cached.attrs.get(
                "fetched_at", pd.Timestamp("2000-01-01")
            )
            if age < pd.Timedelta(days=7):
                return cached.iloc[0].to_dict()

    info = yf.Ticker(ticker).info
    data = _extract(info)

    # Save as single-row DataFrame for Parquet compat
    df = pd.DataFrame([data])
    df.attrs["fetched_at"] = pd.Timestamp.now()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)

    return data


def fetch_fundamentals_bulk(
    tickers: list[str], force_refresh: bool = False
) -> pd.DataFrame:
    """Return a DataFrame of fundamentals, one row per ticker."""
    rows = []
    for t in tickers:
        try:
            row = fetch_fundamentals(t, force_refresh=force_refresh)
            row["ticker"] = t
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")


def fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    """Return quarterly income statement data."""
    t = yf.Ticker(ticker)
    fin = t.quarterly_financials
    if fin is None or fin.empty:
        return pd.DataFrame()
    return fin.T.sort_index()


def _extract(info: dict) -> dict:
    """Pull relevant fields from yfinance info dict."""
    def _get(key, default=None):
        v = info.get(key, default)
        return v if v and v != "N/A" else default

    return {
        "pe_ratio": _get("trailingPE"),
        "forward_pe": _get("forwardPE"),
        "eps": _get("trailingEps"),
        "forward_eps": _get("forwardEps"),
        "eps_growth": _compute_eps_growth(info),
        "revenue_growth": _get("revenueGrowth"),
        "debt_to_equity": _get("debtToEquity"),
        "roe": _get("returnOnEquity"),
        "dividend_yield": _get("dividendYield"),
        "pb_ratio": _get("priceToBook"),
        "sector": _get("sector", "Unknown"),
        "industry": _get("industry", "Unknown"),
        "market_cap": _get("marketCap"),
        "short_name": _get("shortName", ""),
    }


def _compute_eps_growth(info: dict) -> float | None:
    trailing = info.get("trailingEps")
    forward = info.get("forwardEps")
    if trailing and forward and trailing != 0:
        return (forward - trailing) / abs(trailing)
    return None
