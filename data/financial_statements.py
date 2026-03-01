"""Fetch & cache quarterly and annual financial statements.

Primary source: yfinance.  Fallback: Financial Modeling Prep (FMP) API.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache" / "statements"

# Mapping from FMP camelCase keys to yfinance-style title-case names.
_FMP_INCOME_MAP: dict[str, str] = {
    "revenue": "Total Revenue",
    "grossProfit": "Gross Profit",
    "operatingIncome": "Operating Income",
    "netIncome": "Net Income",
    "ebitda": "EBITDA",
    "interestExpense": "Interest Expense",
    "incomeTaxExpense": "Income Tax Expense",
    "eps": "EPS",
    "epsdiluted": "EPS",
}

_FMP_BALANCE_MAP: dict[str, str] = {
    "totalAssets": "Total Assets",
    "totalLiabilities": "Total Liabilities",
    "totalStockholdersEquity": "Total Stockholders Equity",
    "cashAndCashEquivalents": "Cash And Cash Equivalents",
    "totalCurrentAssets": "Current Assets",
    "totalCurrentLiabilities": "Current Liabilities",
    "longTermDebt": "Long Term Debt",
    "retainedEarnings": "Retained Earnings",
}

_FMP_CASHFLOW_MAP: dict[str, str] = {
    "operatingCashFlow": "Operating Cash Flow",
    "capitalExpenditure": "Capital Expenditure",
    "freeCashFlow": "Free Cash Flow",
    "dividendsPaid": "Dividends Paid",
    "commonStockRepurchased": "Repurchase Of Capital Stock",
    "depreciationAndAmortization": "Depreciation And Amortization",
}

_FMP_COLUMN_MAPS: dict[str, dict[str, str]] = {
    "income-statement": _FMP_INCOME_MAP,
    "balance-sheet-statement": _FMP_BALANCE_MAP,
    "cash-flow-statement": _FMP_CASHFLOW_MAP,
}

# Statement keys used throughout the module.
_STATEMENT_KEYS = [
    "quarterly_income",
    "quarterly_balance_sheet",
    "quarterly_cashflow",
    "annual_income",
    "annual_balance_sheet",
    "annual_cashflow",
]

# Maps statement key to (yf attribute name, FMP endpoint, period).
_STATEMENT_SPEC: dict[str, tuple[str, str, str]] = {
    "quarterly_income": ("quarterly_financials", "income-statement", "quarter"),
    "quarterly_balance_sheet": ("quarterly_balance_sheet", "balance-sheet-statement", "quarter"),
    "quarterly_cashflow": ("quarterly_cashflow", "cash-flow-statement", "quarter"),
    "annual_income": ("financials", "income-statement", "annual"),
    "annual_balance_sheet": ("balance_sheet", "balance-sheet-statement", "annual"),
    "annual_cashflow": ("cashflow", "cash-flow-statement", "annual"),
}


class FinancialStatementLoader:
    """Fetch & cache 3 financial statements.  Primary: yfinance.  Fallback: FMP API."""

    def __init__(self, fmp_api_key: str | None = None):
        self.fmp_key = fmp_api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_statements(
        self, ticker: str, force_refresh: bool = False
    ) -> dict:
        """Return a dict with keys for each statement, ``info``, and ``source``.

        Keys: quarterly_income, quarterly_balance_sheet, quarterly_cashflow,
        annual_income, annual_balance_sheet, annual_cashflow, info, source.

        Uses cached parquet files when fresh.  Falls back to FMP for any
        statement that yfinance returns empty.
        """
        ticker = ticker.upper()

        # Try loading everything from cache first.
        if not force_refresh:
            cached = self._load_all_from_cache(ticker)
            if cached is not None:
                return cached

        # Fetch from yfinance.
        result = self._fetch_yfinance(ticker)
        source = "yfinance"

        # Fall back to FMP for any empty statements.
        if self.fmp_key:
            for key in _STATEMENT_KEYS:
                if result[key].empty:
                    _, fmp_endpoint, period = _STATEMENT_SPEC[key]
                    fmp_df = self._fetch_fmp_statement(ticker, fmp_endpoint, period)
                    if not fmp_df.empty:
                        result[key] = fmp_df
                        source = "fmp" if source == "fmp" else "mixed"

        # If everything still came from yfinance (no FMP needed), keep source.
        if source == "yfinance":
            # Double-check: if *all* are empty, mark source accordingly.
            all_empty = all(result[k].empty for k in _STATEMENT_KEYS)
            if all_empty:
                logger.warning("All statements empty for %s", ticker)

        result["source"] = source

        # Persist to cache.
        for key in _STATEMENT_KEYS:
            if not result[key].empty:
                self._save_cache(result[key], self._cache_path(ticker, key))

        return result

    def fetch_all_statements_bulk(
        self, tickers: list[str], force_refresh: bool = False, max_workers: int = 8
    ) -> dict[str, dict]:
        """Bulk fetch for a list of tickers in parallel."""
        results: dict[str, dict] = {}

        def _fetch_one(t):
            return t.upper(), self.fetch_all_statements(t, force_refresh=force_refresh)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    key, data = future.result()
                    results[key] = data
                except Exception:
                    logger.exception("Failed to fetch statements for %s", ticker)

        return results

    # ------------------------------------------------------------------
    # yfinance fetcher
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, ticker: str) -> dict:
        """Fetch all 6 statements + info from yfinance."""
        result: dict = {k: pd.DataFrame() for k in _STATEMENT_KEYS}
        result["info"] = {}

        try:
            t = yf.Ticker(ticker)

            for key, (yf_attr, _, _) in _STATEMENT_SPEC.items():
                try:
                    raw = getattr(t, yf_attr, None)
                    if raw is not None and not raw.empty:
                        # yfinance returns line items as rows and dates as
                        # columns (most-recent first).  Keep that layout —
                        # downstream modules (dcf_valuation, fundamental_deep)
                        # expect line items in the index.
                        result[key] = raw
                except Exception:
                    logger.debug("yfinance %s failed for %s", yf_attr, ticker)

            try:
                result["info"] = t.info or {}
            except Exception:
                logger.debug("yfinance info failed for %s", ticker)

        except Exception:
            logger.exception("yfinance fetch failed entirely for %s", ticker)

        return result

    # ------------------------------------------------------------------
    # FMP fetcher
    # ------------------------------------------------------------------

    def _fetch_fmp_statement(
        self,
        ticker: str,
        statement_type: str,
        period: str = "quarter",
    ) -> pd.DataFrame:
        """Fetch a single statement from the FMP API.

        *statement_type*: ``'income-statement'``, ``'balance-sheet-statement'``,
        or ``'cash-flow-statement'``.
        """
        if not self.fmp_key:
            return pd.DataFrame()

        url = f"https://financialmodelingprep.com/api/v3/{statement_type}/{ticker}"
        params: dict[str, str | int] = {"apikey": self.fmp_key, "limit": 20}
        if period == "quarter":
            params["period"] = "quarter"

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("FMP request failed for %s %s", ticker, statement_type)
            return pd.DataFrame()

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Use the filing/report date as the index.
        date_col = "date" if "date" in df.columns else None
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()

        # Rename columns to match yfinance conventions.
        col_map = _FMP_COLUMN_MAPS.get(statement_type, {})
        rename = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=rename)

        # Transpose to match yfinance layout: line items as rows,
        # dates as columns (most recent first).
        df = df.T
        df = df[df.columns[::-1]]  # most recent first

        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, statement_name: str) -> Path:
        """Return the parquet cache file path for a specific statement."""
        return CACHE_DIR / ticker.upper() / f"{statement_name}.parquet"

    def _is_cache_fresh(self, path: Path, max_age_days: int = 90) -> bool:
        """Return ``True`` if *path* exists and is less than *max_age_days* old."""
        if not path.exists():
            return False
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds < max_age_days * 86_400

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Persist *df* to parquet, creating parent directories as needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            logger.debug("Cached %s", path)
        except Exception:
            logger.debug("Failed to cache %s", path, exc_info=True)

    def _load_cache(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from a parquet cache file."""
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_from_cache(self, ticker: str) -> dict | None:
        """Try to load every statement from cache.

        Returns the full result dict if all caches are fresh, else ``None``.
        """
        result: dict = {}
        for key in _STATEMENT_KEYS:
            path = self._cache_path(ticker, key)
            if not self._is_cache_fresh(path):
                logger.debug("Cache miss for %s/%s", ticker, key)
                return None
            result[key] = self._load_cache(path)

        # Info is not cached separately; re-fetch from yfinance since it's
        # cheap relative to statements and required for market cap, beta, etc.
        try:
            result["info"] = yf.Ticker(ticker).info or {}
        except Exception:
            logger.debug("yfinance info re-fetch failed for %s (cache hit)", ticker)
            result["info"] = {}
        result["source"] = "cache"
        logger.debug("Cache hit for %s (all statements)", ticker)
        return result
