"""Unified data interface that coordinates all loaders."""

import pandas as pd

from config_loader import get_config
from data.price_loader import fetch_prices, fetch_prices_bulk
from data.fundamental_loader import (
    fetch_fundamentals,
    fetch_fundamentals_bulk,
)
from data.macro_loader import fetch_all_macro, fetch_macro_series


class DataManager:
    """Single entry point for all data access."""

    def __init__(self, config: dict | None = None):
        self.config = config or get_config()
        self._tickers = self.config["universe"]["tickers"]
        self._lookback = self.config["universe"]["lookback_years"]
        self._fred_key = self.config["fred"]["api_key"]

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

    # ── Universe ────────────────────────────────────────────────────

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)
