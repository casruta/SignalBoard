"""Fundamental analysis signals derived from financial data."""

import numpy as np
import pandas as pd


def compute_fundamental_signals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Compute value, quality, and growth scores from fundamentals DataFrame.

    Parameters
    ----------
    fundamentals : DataFrame indexed by ticker with columns from
                   fundamental_loader (pe_ratio, eps, roe, etc.)

    Returns
    -------
    DataFrame indexed by ticker with score columns.
    """
    df = pd.DataFrame(index=fundamentals.index)

    # ── Value Score ──────────────────────────────────────────────
    # Lower P/E and P/B are better; higher dividend yield is better.
    df["pe_rank"] = _rank_ascending(fundamentals.get("pe_ratio"))
    df["pb_rank"] = _rank_ascending(fundamentals.get("pb_ratio"))
    df["div_rank"] = _rank_descending(fundamentals.get("dividend_yield"))
    df["value_score"] = df[["pe_rank", "pb_rank", "div_rank"]].mean(axis=1)

    # ── Quality Score ────────────────────────────────────────────
    # Higher ROE is better; lower debt-to-equity is better.
    df["roe_rank"] = _rank_descending(fundamentals.get("roe"))
    df["debt_rank"] = _rank_ascending(fundamentals.get("debt_to_equity"))
    df["quality_score"] = df[["roe_rank", "debt_rank"]].mean(axis=1)

    # ── Growth Score ─────────────────────────────────────────────
    df["eps_growth_rank"] = _rank_descending(fundamentals.get("eps_growth"))
    df["rev_growth_rank"] = _rank_descending(fundamentals.get("revenue_growth"))
    df["growth_score"] = df[["eps_growth_rank", "rev_growth_rank"]].mean(axis=1)

    # ── Composite Fundamental Score ──────────────────────────────
    df["fundamental_score"] = (
        0.35 * df["value_score"]
        + 0.35 * df["quality_score"]
        + 0.30 * df["growth_score"]
    )

    # ── P/E vs sector average ────────────────────────────────────
    if "pe_ratio" in fundamentals.columns and "sector" in fundamentals.columns:
        sector_avg = fundamentals.groupby("sector")["pe_ratio"].transform("mean")
        pe = fundamentals["pe_ratio"]
        df["pe_vs_sector"] = np.where(
            sector_avg > 0, (pe - sector_avg) / sector_avg, np.nan
        )
    else:
        df["pe_vs_sector"] = np.nan

    # ── Quality grade (letter) ───────────────────────────────────
    df["quality_grade"] = pd.cut(
        df["quality_score"],
        bins=[0, 0.25, 0.5, 0.75, 1.01],
        labels=["D", "C", "B", "A"],
    )

    return df


def _rank_ascending(series: pd.Series | None) -> pd.Series:
    """Rank where lower raw values get higher scores (closer to 1.0)."""
    if series is None:
        return pd.Series(dtype=float)
    return 1.0 - series.rank(pct=True, na_option="keep")


def _rank_descending(series: pd.Series | None) -> pd.Series:
    """Rank where higher raw values get higher scores (closer to 1.0)."""
    if series is None:
        return pd.Series(dtype=float)
    return series.rank(pct=True, na_option="keep")
