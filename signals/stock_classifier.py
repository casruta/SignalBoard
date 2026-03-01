"""Stock classification into growth, dividend, or value categories.

Classifies stocks based on dividend yield, payout ratio, and revenue growth
metrics. Thresholds are configurable via an optional config dict.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default classification thresholds
_DEFAULTS = {
    "dividend_yield_threshold": 0.015,
    "payout_ratio_threshold": 0.20,
    "revenue_growth_threshold": 0.10,
}


def _get_threshold(config: dict | None, key: str) -> float:
    """Retrieve a classification threshold from config or fall back to default.

    Config keys are nested under ``classification.*``, e.g.
    ``classification.dividend_yield_threshold``.
    """
    if config is not None:
        nested = config.get("classification", {})
        if isinstance(nested, dict) and key in nested:
            return float(nested[key])
        # Also accept flat keys for convenience
        flat_key = f"classification.{key}"
        if flat_key in config:
            return float(config[flat_key])
    return _DEFAULTS[key]


def classify_stock(info: dict, deep_fund: dict, config: dict | None = None) -> str:
    """Classify a stock as ``"growth"``, ``"dividend"``, or ``"value"``.

    Parameters
    ----------
    info : dict
        yfinance ``.info`` dict for the ticker.
    deep_fund : dict
        Deep fundamental data (e.g. from FundamentalLoader).
    config : dict, optional
        Override default thresholds via ``classification.*`` keys.

    Returns
    -------
    str
        One of ``"growth"``, ``"dividend"``, ``"value"``.
    """
    if info is None:
        info = {}
    if deep_fund is None:
        deep_fund = {}

    div_yield_thresh = _get_threshold(config, "dividend_yield_threshold")
    payout_thresh = _get_threshold(config, "payout_ratio_threshold")
    rev_growth_thresh = _get_threshold(config, "revenue_growth_threshold")

    # ---- Extract metrics (safe for None / missing) ----
    dividend_yield = _float_or(info.get("dividendYield"), 0.0)
    payout_ratio = _float_or(info.get("payoutRatio"), 0.0)
    deep_payout = _float_or(deep_fund.get("dividend_payout_ratio"), 0.0)
    trailing_div_rate = _float_or(info.get("trailingAnnualDividendRate"), 0.0)
    revenue_growth = _float_or(info.get("revenueGrowth"), 0.0)

    # ---- Dividend criteria ----
    is_dividend = (
        dividend_yield > div_yield_thresh
        and (payout_ratio > payout_thresh or deep_payout > payout_thresh)
        and trailing_div_rate > 0
    )

    # ---- Growth criteria ----
    is_growth = (
        revenue_growth > rev_growth_thresh
        or (revenue_growth > 0.05 and dividend_yield < 0.01)
    )

    # ---- Tiebreaker / fallback ----
    if is_dividend and is_growth:
        return "dividend" if dividend_yield > 0.025 else "growth"
    if is_dividend:
        return "dividend"
    if is_growth:
        return "growth"
    return "value"


def split_by_category(
    stocks_df: pd.DataFrame,
    info_map: dict,
    deep_fund_map: dict,
    config: dict | None = None,
    growth_n: int = 3,
    dividend_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split scored stocks into top-N growth and dividend DataFrames.

    Parameters
    ----------
    stocks_df : DataFrame
        Must contain ``ticker`` and ``composite_score`` columns.
    info_map : dict[str, dict]
        Mapping of ticker -> yfinance ``.info`` dict.
    deep_fund_map : dict[str, dict]
        Mapping of ticker -> deep fundamental data dict.
    config : dict, optional
        Classification threshold overrides.
    growth_n : int
        Number of growth stocks to return (default 3).
    dividend_n : int
        Number of dividend stocks to return (default 3).

    Returns
    -------
    tuple[DataFrame, DataFrame]
        ``(growth_df, dividend_df)`` — top N from each category, sorted by
        composite_score descending. Categories with fewer than N stocks are
        backfilled from "value" stocks by composite score.
    """
    if stocks_df is None or stocks_df.empty:
        empty = pd.DataFrame(columns=stocks_df.columns if stocks_df is not None else [])
        return empty.copy(), empty.copy()

    # Classify every ticker
    categories = {}
    for ticker in stocks_df["ticker"]:
        info = info_map.get(ticker, {})
        deep = deep_fund_map.get(ticker, {})
        categories[ticker] = classify_stock(info, deep, config)

    stocks_df = stocks_df.copy()
    stocks_df["_category"] = stocks_df["ticker"].map(categories)

    sorted_df = stocks_df.sort_values("composite_score", ascending=False)

    growth_df = sorted_df[sorted_df["_category"] == "growth"].head(growth_n)
    dividend_df = sorted_df[sorted_df["_category"] == "dividend"].head(dividend_n)
    value_df = sorted_df[sorted_df["_category"] == "value"]

    # Backfill from value stocks if a category is short
    if len(growth_df) < growth_n:
        shortfall = growth_n - len(growth_df)
        used_tickers = set(growth_df["ticker"]) | set(dividend_df["ticker"])
        available = value_df[~value_df["ticker"].isin(used_tickers)].head(shortfall)
        growth_df = pd.concat([growth_df, available], ignore_index=True)
        logger.info("Backfilled %d value stock(s) into growth category", len(available))

    if len(dividend_df) < dividend_n:
        shortfall = dividend_n - len(dividend_df)
        used_tickers = set(growth_df["ticker"]) | set(dividend_df["ticker"])
        available = value_df[~value_df["ticker"].isin(used_tickers)].head(shortfall)
        dividend_df = pd.concat([dividend_df, available], ignore_index=True)
        logger.info("Backfilled %d value stock(s) into dividend category", len(available))

    # Drop internal helper column
    growth_df = growth_df.drop(columns=["_category"], errors="ignore")
    dividend_df = dividend_df.drop(columns=["_category"], errors="ignore")

    return growth_df, dividend_df


def _float_or(val, default: float = 0.0) -> float:
    """Convert *val* to float, returning *default* on ``None`` or failure."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default
