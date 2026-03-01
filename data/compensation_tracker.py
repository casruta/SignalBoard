"""Track executive compensation structure using SEC EDGAR DEF 14A proxy statements.

Queries EDGAR full-text search for proxy statements, then attempts to parse
the compensation summary table to estimate the equity-vs-cash split of CEO pay.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache" / "compensation"
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_HEADERS = {"User-Agent": "SignalApp/1.0 (signalapp@example.com)"}
LOOKBACK_YEARS = 3

# Keywords used to identify compensation components in proxy filing text.
EQUITY_KEYWORDS = ("stock awards", "option awards")
CASH_KEYWORDS = ("salary", "bonus", "non-equity incentive")


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


def _empty_result() -> dict:
    return {
        "equity_heavy": None,
        "equity_pct": None,
        "cash_pct": None,
        "total_ceo_compensation": None,
        "latest_proxy_date": None,
        "filing_url": None,
        "has_data": False,
        "source": "sec_edgar_def14a",
        "parsing_confidence": "none",
    }


def _query_efts(ticker: str) -> dict:
    """Query EDGAR EFTS for DEF 14A filings mentioning executive compensation."""
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")

    params = {
        "q": '"executive compensation"',
        "forms": "DEF 14A",
        "tickers": ticker.upper(),
        "dateRange": "custom",
        "startdt": start,
        "enddt": today,
    }

    resp = requests.get(EDGAR_EFTS_URL, params=params, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _build_filing_url(hit: dict) -> str | None:
    """Construct an SEC filing URL from an EFTS hit."""
    source = hit.get("_source", {})
    accession = source.get("accession_no", "")
    if not accession:
        return None
    clean = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{clean[:10]}/{accession}.txt"


def _fetch_filing_text(url: str, max_bytes: int = 2_000_000) -> str:
    """Download the first *max_bytes* of a filing for text analysis."""
    resp = requests.get(
        url, headers={**SEC_HEADERS, "Range": f"bytes=0-{max_bytes}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def _extract_dollar_amount(text: str) -> float | None:
    """Parse a dollar string like '$1,234,567' or '1234567' into a float."""
    cleaned = re.sub(r"[$,\s]", "", text.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_compensation(text: str) -> dict:
    """Best-effort extraction of compensation components from filing text.

    Scans for keywords like 'Stock Awards', 'Salary', etc. and tries to
    pick up nearby dollar amounts.  Returns partial results when full
    parsing is not possible.
    """
    text_lower = text.lower()

    # Pattern: keyword followed by a dollar amount within a short window.
    dollar_pattern = r"[\$]?\s*[\d,]+(?:\.\d+)?"

    def _find_amount(keyword: str) -> float | None:
        pattern = rf"{re.escape(keyword)}\s*(?:[:\s|]+)\s*({dollar_pattern})"
        match = re.search(pattern, text_lower)
        if match:
            return _extract_dollar_amount(match.group(1))
        return None

    equity_total = 0.0
    cash_total = 0.0
    found_any = False

    for kw in EQUITY_KEYWORDS:
        amt = _find_amount(kw)
        if amt is not None and amt > 0:
            equity_total += amt
            found_any = True

    for kw in CASH_KEYWORDS:
        amt = _find_amount(kw)
        if amt is not None and amt > 0:
            cash_total += amt
            found_any = True

    if not found_any:
        return {"confidence": "none"}

    total = equity_total + cash_total
    if total == 0:
        return {"confidence": "none"}

    # Also look for an explicit total compensation figure.
    total_comp = _find_amount("total")
    if total_comp is not None and total_comp > total:
        total = total_comp

    equity_pct = equity_total / total if total > 0 else None
    cash_pct = cash_total / total if total > 0 else None

    confidence = "high" if (equity_total > 0 and cash_total > 0) else "low"

    return {
        "equity_pct": round(equity_pct, 4) if equity_pct is not None else None,
        "cash_pct": round(cash_pct, 4) if cash_pct is not None else None,
        "total_ceo_compensation": total,
        "confidence": confidence,
    }


def get_compensation_structure(
    ticker: str, info: dict | None = None, cache_days: int = 90
) -> dict:
    """Return executive compensation breakdown for *ticker*.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    info : dict | None
        Optional pre-fetched company info dict (currently unused).
    cache_days : int
        Number of days to consider cached results valid.

    Returns
    -------
    dict
        Keys: ``equity_heavy``, ``equity_pct``, ``cash_pct``,
        ``total_ceo_compensation``, ``latest_proxy_date``, ``filing_url``,
        ``has_data``, ``source``, ``parsing_confidence``.
    """
    cached = _read_cache(ticker, cache_days)
    if cached is not None:
        logger.debug("Returning cached compensation data for %s", ticker)
        return cached

    # -- Step 1: find the most recent DEF 14A via EFTS ----------------------
    try:
        raw = _query_efts(ticker)
        time.sleep(0.15)  # respect SEC rate limits
    except (requests.RequestException, ValueError) as exc:
        logger.warning("EDGAR EFTS query failed for %s: %s", ticker, exc)
        return _empty_result()

    hits = raw.get("hits", {}).get("hits", [])
    if not hits:
        result = _empty_result()
        result["has_data"] = True  # query succeeded, just no filings
        _write_cache(ticker, result)
        return result

    # Take the most recent filing.
    hits.sort(
        key=lambda h: h.get("_source", {}).get("file_date", ""),
        reverse=True,
    )
    latest = hits[0]
    source = latest.get("_source", {})
    filing_url = _build_filing_url(latest)
    proxy_date = source.get("file_date")

    # -- Step 2: fetch and parse the filing text ----------------------------
    comp_data: dict = {"confidence": "none"}
    if filing_url:
        try:
            time.sleep(0.15)  # respect SEC rate limits
            filing_text = _fetch_filing_text(filing_url)
            comp_data = _parse_compensation(filing_text)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Failed to fetch filing for %s: %s", ticker, exc)

    equity_pct = comp_data.get("equity_pct")
    result = {
        "equity_heavy": equity_pct > 0.5 if equity_pct is not None else None,
        "equity_pct": equity_pct,
        "cash_pct": comp_data.get("cash_pct"),
        "total_ceo_compensation": comp_data.get("total_ceo_compensation"),
        "latest_proxy_date": proxy_date,
        "filing_url": filing_url,
        "has_data": True,
        "source": "sec_edgar_def14a",
        "parsing_confidence": comp_data.get("confidence", "none"),
    }

    _write_cache(ticker, result)
    return result
