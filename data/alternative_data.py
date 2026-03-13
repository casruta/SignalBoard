"""Alternative data loaders: SEC EDGAR filings, insider transactions, short interest."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache" / "alt_data"
EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index?q="

# SEC EDGAR requires a user-agent header
EDGAR_HEADERS = {
    "User-Agent": "SignalBoard Research bot@signalboard.local",
    "Accept-Encoding": "gzip, deflate",
}


def fetch_insider_transactions(
    ticker: str,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Fetch insider buying/selling from SEC EDGAR Form 4 filings via yfinance.

    Returns DataFrame with columns: date, insider_name, transaction_type, shares, value.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is None or insider.empty:
            return pd.DataFrame()
        return insider
    except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
        logger.warning("Failed to fetch insider data for %s: %s", ticker, e)
        return pd.DataFrame()


def compute_insider_signals(
    insider_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute insider transaction signals.

    Features:
    - insider_buy_ratio_30d: buys / (buys + sells) over trailing 30 days
    - insider_net_shares_30d: net shares bought (positive = bullish)
    - insider_transaction_count_30d: total insider transactions
    """
    df = pd.DataFrame(index=dates)
    df["insider_buy_ratio_30d"] = np.nan
    df["insider_net_shares_30d"] = 0.0
    df["insider_transaction_count_30d"] = 0

    if insider_df.empty:
        return df

    # Try to parse insider data (yfinance format varies)
    try:
        if "Shares" in insider_df.columns and "Start Date" in insider_df.columns:
            insider_df = insider_df.copy()
            insider_df["date"] = pd.to_datetime(insider_df["Start Date"])
            insider_df["shares"] = pd.to_numeric(insider_df["Shares"], errors="coerce").fillna(0)

            for i, date in enumerate(dates):
                window_start = date - pd.Timedelta(days=30)
                mask = (insider_df["date"] >= window_start) & (insider_df["date"] <= date)
                window = insider_df[mask]

                if len(window) == 0:
                    continue

                buys = (window["shares"] > 0).sum()
                sells = (window["shares"] < 0).sum()
                total = buys + sells
                df.iloc[i, df.columns.get_loc("insider_buy_ratio_30d")] = buys / total if total > 0 else np.nan
                df.iloc[i, df.columns.get_loc("insider_net_shares_30d")] = float(window["shares"].sum())
                df.iloc[i, df.columns.get_loc("insider_transaction_count_30d")] = total
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Error computing insider signals: %s", e)

    return df


def fetch_short_interest_proxy(ticker: str) -> pd.DataFrame:
    """Fetch short interest proxy data from yfinance.

    yfinance provides 'sharesShort' and 'shortRatio' in the info dict.
    These update roughly every 2 weeks (FINRA reporting dates).
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        return pd.DataFrame([{
            "short_interest": info.get("sharesShort", np.nan),
            "short_ratio": info.get("shortRatio", np.nan),  # days to cover
            "shares_outstanding": info.get("sharesOutstanding", np.nan),
            "short_pct_float": info.get("shortPercentOfFloat", np.nan),
        }])
    except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
        logger.warning("Failed to fetch short interest for %s: %s", ticker, e)
        return pd.DataFrame()


def compute_short_interest_signals(
    short_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Broadcast short interest data across dates (static, updates bi-weekly)."""
    df = pd.DataFrame(index=dates)
    df["short_ratio"] = np.nan
    df["short_pct_float"] = np.nan

    if not short_df.empty:
        row = short_df.iloc[0]
        df["short_ratio"] = row.get("short_ratio", np.nan)
        df["short_pct_float"] = row.get("short_pct_float", np.nan)

    return df


def fetch_institutional_holders(ticker: str) -> pd.DataFrame:
    """Fetch top institutional holders from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders
        if holders is None or holders.empty:
            return pd.DataFrame()
        return holders
    except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
        logger.warning("Failed to fetch institutional holders for %s: %s", ticker, e)
        return pd.DataFrame()


def compute_institutional_signals(holders_df: pd.DataFrame) -> dict:
    """Compute signals from institutional holdings.

    Returns dict of static features per ticker.
    """
    if holders_df.empty:
        return {
            "inst_holder_count": np.nan,
            "inst_top10_pct": np.nan,
        }

    try:
        count = len(holders_df)
        if "pctHeld" in holders_df.columns:
            top10_pct = holders_df["pctHeld"].head(10).sum()
        elif "% Out" in holders_df.columns:
            top10_pct = pd.to_numeric(
                holders_df["% Out"].head(10).str.rstrip("%"), errors="coerce"
            ).sum() / 100
        else:
            top10_pct = np.nan

        return {
            "inst_holder_count": count,
            "inst_top10_pct": float(top10_pct) if not pd.isna(top10_pct) else np.nan,
        }
    except (ValueError, KeyError, TypeError):
        logger.warning("Error computing institutional signals")
        return {"inst_holder_count": np.nan, "inst_top10_pct": np.nan}
