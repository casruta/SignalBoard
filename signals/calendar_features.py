"""Calendar and event-based features: FOMC, options expiration, turn-of-month."""

import numpy as np
import pandas as pd


# FOMC meeting dates (approximate — 8 meetings per year)
# These are publicly available scheduled dates
_FOMC_MONTHS = [1, 3, 5, 6, 7, 9, 11, 12]  # typical months with meetings


def compute_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute calendar/event-based features.

    Parameters
    ----------
    dates : DatetimeIndex of trading days.

    Returns
    -------
    DataFrame indexed by date with calendar features.
    """
    df = pd.DataFrame(index=dates)

    # ── Day of week ─────────────────────────────────────────────
    # Monday=0, Friday=4 — known "Monday effect" and "Friday effect"
    df["day_of_week"] = dates.dayofweek
    df["is_monday"] = (dates.dayofweek == 0).astype(float)
    df["is_friday"] = (dates.dayofweek == 4).astype(float)

    # ── Turn of Month ───────────────────────────────────────────
    # Last 2 days of month and first 3 days of next month — documented anomaly
    day = dates.day
    days_in_month = dates.days_in_month
    df["is_turn_of_month"] = (
        (day <= 3) | (day >= days_in_month - 1)
    ).astype(float)

    # ── Month of year ───────────────────────────────────────────
    # "Sell in May", January effect, etc.
    df["month"] = dates.month
    df["is_january"] = (dates.month == 1).astype(float)
    df["is_december"] = (dates.month == 12).astype(float)

    # ── Quarter end ─────────────────────────────────────────────
    # Window dressing, rebalancing around quarter-end
    df["is_quarter_end"] = (
        (dates.month.isin([3, 6, 9, 12])) & (day >= days_in_month - 4)
    ).astype(float)

    # ── FOMC Proximity ──────────────────────────────────────────
    # Days until next approximate FOMC meeting (mid-month of FOMC months)
    df["fomc_proximity"] = _compute_fomc_proximity(dates)

    # ── Options Expiration (monthly OpEx) ───────────────────────
    # Third Friday of each month — known for increased volatility
    df["days_to_opex"] = _compute_opex_proximity(dates)
    df["is_opex_week"] = (df["days_to_opex"] <= 5).astype(float)

    # ── Year-end ────────────────────────────────────────────────
    # Tax-loss selling, window dressing in last 2 weeks
    df["is_year_end"] = (
        (dates.month == 12) & (day >= 15)
    ).astype(float)

    # ── Week of year (cyclical encoding) ────────────────────────
    week_num = dates.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_num / 52)

    return df


def _compute_fomc_proximity(dates: pd.DatetimeIndex) -> pd.Series:
    """Approximate days until the next FOMC meeting date."""
    results = []
    for dt in dates:
        year = dt.year
        # Generate approximate FOMC dates (typically Tue-Wed around mid-month)
        fomc_dates = []
        for m in _FOMC_MONTHS:
            fomc_dates.append(pd.Timestamp(year, m, 15))
        # Also add next year's January
        fomc_dates.append(pd.Timestamp(year + 1, 1, 15))

        # Find the next FOMC date
        future_dates = [f for f in fomc_dates if f >= dt]
        if future_dates:
            days_until = (future_dates[0] - dt).days
        else:
            days_until = 30  # fallback
        results.append(min(days_until, 60))  # cap at 60

    return pd.Series(results, index=dates, dtype=float)


def _compute_opex_proximity(dates: pd.DatetimeIndex) -> pd.Series:
    """Days until the next monthly options expiration (3rd Friday)."""
    results = []
    for dt in dates:
        # Find 3rd Friday of current month
        opex = _third_friday(dt.year, dt.month)
        if opex < dt:
            # Already passed — look at next month
            if dt.month == 12:
                opex = _third_friday(dt.year + 1, 1)
            else:
                opex = _third_friday(dt.year, dt.month + 1)
        results.append((opex - dt).days)

    return pd.Series(results, index=dates, dtype=float)


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the 3rd Friday of a given month."""
    # First day of month
    first = pd.Timestamp(year, month, 1)
    # Day of week: Monday=0, Friday=4
    first_friday_offset = (4 - first.dayofweek) % 7
    first_friday = first + pd.Timedelta(days=first_friday_offset)
    third_friday = first_friday + pd.Timedelta(weeks=2)
    return third_friday
