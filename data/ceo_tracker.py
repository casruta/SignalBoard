"""Track CEO changes using SEC EDGAR full-text search (EFTS) API.

Searches for 8-K filings mentioning Item 5.02 (Departure/Election of
Directors or Principal Officers) to detect recent CEO turnover.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache" / "ceo"
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_HEADERS = {"User-Agent": "SignalApp/1.0 (signalapp@example.com)"}
LOOKBACK_YEARS = 2


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


def _read_cache(ticker: str, cache_days: int) -> dict | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
        if datetime.now() - cached_at < timedelta(days=cache_days):
            return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.debug("Cache read failed for %s: %s", ticker, exc)
    return None


def _write_cache(ticker: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**data, "_cached_at": datetime.now().isoformat()}
    _cache_path(ticker).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _query_efts(ticker: str) -> dict:
    """Query EDGAR EFTS for 8-K filings mentioning Item 5.02."""
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")

    params = {
        "q": '"5.02"',
        "forms": "8-K",
        "tickers": ticker.upper(),
        "dateRange": "custom",
        "startdt": start,
        "enddt": today,
    }

    resp = requests.get(EDGAR_EFTS_URL, params=params, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_efts_response(data: dict) -> dict:
    """Extract CEO-change information from EFTS JSON response."""
    hits = data.get("hits", {}).get("hits", [])
    filings_found = len(hits)

    if filings_found == 0:
        return {
            "ceo_changed_recently": False,
            "change_date": None,
            "filing_url": None,
            "filings_found": 0,
            "has_data": True,
            "source": "sec_edgar",
        }

    # Sort by file_date descending to get the most recent filing first.
    hits.sort(
        key=lambda h: h.get("_source", {}).get("file_date", ""),
        reverse=True,
    )
    latest = hits[0].get("_source", {})
    file_date = latest.get("file_date")
    file_num = latest.get("file_num", "")
    accession = latest.get("accession_no", "")

    filing_url = None
    if accession:
        clean = accession.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{clean[:10]}/{accession}.txt"
        )

    return {
        "ceo_changed_recently": True,
        "change_date": file_date,
        "filing_url": filing_url,
        "filings_found": filings_found,
        "has_data": True,
        "source": "sec_edgar",
    }


def get_ceo_info(
    ticker: str, info: dict | None = None, cache_days: int = 30
) -> dict:
    """Return CEO-change data for *ticker* from SEC EDGAR 8-K filings.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    info : dict | None
        Optional pre-fetched company info dict (currently unused, reserved
        for future enrichment).
    cache_days : int
        Number of days to consider cached results valid.

    Returns
    -------
    dict
        Keys: ``ceo_changed_recently``, ``change_date``, ``filing_url``,
        ``filings_found``, ``has_data``, ``source``.
    """
    cached = _read_cache(ticker, cache_days)
    if cached is not None:
        logger.debug("Returning cached CEO data for %s", ticker)
        return cached

    try:
        raw = _query_efts(ticker)
        time.sleep(0.15)  # respect SEC rate limits
        result = _parse_efts_response(raw)
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("EDGAR query failed for %s: %s", ticker, exc)
        return {
            "ceo_changed_recently": None,
            "change_date": None,
            "filing_url": None,
            "filings_found": 0,
            "has_data": False,
            "source": "sec_edgar",
        }

    _write_cache(ticker, result)
    return result
