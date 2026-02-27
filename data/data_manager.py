"""Unified data interface that coordinates all loaders."""

import logging

import pandas as pd

from config_loader import get_config
from data.price_loader import fetch_prices, fetch_prices_bulk
from data.fundamental_loader import (
    fetch_fundamentals,
    fetch_fundamentals_bulk,
)
from data.macro_loader import fetch_all_macro, fetch_macro_series
from data.financial_statements import FinancialStatementLoader
from data.alternative_data import (
    fetch_insider_transactions,
    fetch_institutional_holders,
    fetch_short_interest_proxy,
)

logger = logging.getLogger(__name__)


class DataManager:
    """Single entry point for all data access."""

    def __init__(self, config: dict | None = None):
        self.config = config or get_config()
        self._tickers = self.config["universe"]["tickers"]
        self._lookback = self.config["universe"]["lookback_years"]
        self._fred_key = self.config["fred"]["api_key"]
        self._fmp_key = self.config.get("fmp", {}).get("api_key")
        self._stmt_loader = FinancialStatementLoader(fmp_api_key=self._fmp_key)

    # ── Prices ──────────────────────────────────────────────────────

    def get_prices(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        return fetch_prices(
            ticker, start=start, end=end, lookback_years=self._lookback
        )

    def get_all_prices(
        self, start: str | None = None, end: str | None = None
    ) -> dict[str, pd.DataFrame]:
        return fetch_prices_bulk(
            self._tickers, start=start, end=end, lookback_years=self._lookback
        )

    # ── Fundamentals ────────────────────────────────────────────────

    def get_fundamentals(self, ticker: str) -> dict:
        return fetch_fundamentals(ticker)

    def get_all_fundamentals(self) -> pd.DataFrame:
        return fetch_fundamentals_bulk(self._tickers)

    # ── Macro ───────────────────────────────────────────────────────

    def get_macro(
        self, start: str | None = None
    ) -> pd.DataFrame:
        return fetch_all_macro(
            self._fred_key, start=start, lookback_years=self._lookback
        )

    def get_macro_series(self, name: str, start: str | None = None) -> pd.Series:
        return fetch_macro_series(
            name, api_key=self._fred_key, start=start, lookback_years=self._lookback
        )

    # ── Financial Statements (Deep Fundamentals) ────────────────────

    def get_statements(self, ticker: str) -> dict:
        """Return full financial statements for a single ticker."""
        return self._stmt_loader.fetch_all_statements(ticker)

    def get_all_statements(self) -> dict[str, dict]:
        """Return full financial statements for all universe tickers."""
        return self._stmt_loader.fetch_all_statements_bulk(self._tickers)

    # ── Alternative Data ────────────────────────────────────────────

    def get_insider_transactions(self, ticker: str) -> pd.DataFrame:
        return fetch_insider_transactions(ticker)

    def get_institutional_holders(self, ticker: str) -> pd.DataFrame:
        return fetch_institutional_holders(ticker)

    def get_all_alternative_data(self) -> dict[str, dict]:
        """Fetch insider, institutional, and short interest data for all tickers."""
        result = {}
        for t in self._tickers:
            try:
                result[t] = {
                    "insider_df": fetch_insider_transactions(t),
                    "holders_df": fetch_institutional_holders(t),
                    "short_df": fetch_short_interest_proxy(t),
                }
            except Exception as e:
                logger.debug("Alt data fetch failed for %s: %s", t, e)
                result[t] = {
                    "insider_df": pd.DataFrame(),
                    "holders_df": pd.DataFrame(),
                    "short_df": pd.DataFrame(),
                }
        return result

    # ── Universe ────────────────────────────────────────────────────

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)
