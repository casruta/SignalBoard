"""Options-derived signals: implied volatility, put-call ratio, variance risk premium."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_options_chain(ticker: str) -> pd.DataFrame | None:
    """Fetch options chain from yfinance.

    Returns combined puts and calls for the nearest expiration.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        # Get the two nearest expirations
        chains = []
        for exp in expirations[:2]:
            opt = stock.option_chain(exp)
            calls = opt.calls.copy()
            calls["type"] = "call"
            calls["expiration"] = exp
            puts = opt.puts.copy()
            puts["type"] = "put"
            puts["expiration"] = exp
            chains.append(pd.concat([calls, puts], ignore_index=True))

        return pd.concat(chains, ignore_index=True) if chains else None
    except Exception as e:
        logger.debug("Failed to fetch options for %s: %s", ticker, e)
        return None


def compute_options_features(
    chain: pd.DataFrame | None,
    current_price: float,
) -> dict:
    """Compute options-derived features from an options chain.

    Parameters
    ----------
    chain : combined options chain DataFrame (from fetch_options_chain)
    current_price : current underlying price

    Returns
    -------
    Dict of feature name -> value.
    """
    features = {
        "opt_iv_atm": np.nan,
        "opt_iv_skew": np.nan,
        "opt_pcr_volume": np.nan,
        "opt_pcr_oi": np.nan,
        "opt_unusual_volume": np.nan,
        "opt_iv_term_slope": np.nan,
    }

    if chain is None or chain.empty or current_price <= 0:
        return features

    try:
        # Find ATM strikes (within 5% of current price)
        chain["moneyness"] = abs(chain["strike"] - current_price) / current_price
        atm_mask = chain["moneyness"] <= 0.05

        # ATM Implied Volatility
        if "impliedVolatility" in chain.columns:
            atm_iv = chain.loc[atm_mask, "impliedVolatility"]
            if not atm_iv.empty:
                features["opt_iv_atm"] = float(atm_iv.mean())

        # IV Skew: OTM puts IV - OTM calls IV (positive = put skew / fear)
        otm_puts = chain[(chain["type"] == "put") & (chain["strike"] < current_price * 0.95)]
        otm_calls = chain[(chain["type"] == "call") & (chain["strike"] > current_price * 1.05)]
        if "impliedVolatility" in chain.columns:
            put_iv = otm_puts["impliedVolatility"].mean()
            call_iv = otm_calls["impliedVolatility"].mean()
            if not (np.isnan(put_iv) or np.isnan(call_iv)) and call_iv > 0:
                features["opt_iv_skew"] = float(put_iv - call_iv)

        # Put-Call Ratio (volume-based)
        if "volume" in chain.columns:
            call_vol = chain.loc[chain["type"] == "call", "volume"].sum()
            put_vol = chain.loc[chain["type"] == "put", "volume"].sum()
            if call_vol > 0:
                features["opt_pcr_volume"] = float(put_vol / call_vol)

        # Put-Call Ratio (open interest)
        if "openInterest" in chain.columns:
            call_oi = chain.loc[chain["type"] == "call", "openInterest"].sum()
            put_oi = chain.loc[chain["type"] == "put", "openInterest"].sum()
            if call_oi > 0:
                features["opt_pcr_oi"] = float(put_oi / call_oi)

        # Unusual volume: total volume / total open interest
        if "volume" in chain.columns and "openInterest" in chain.columns:
            total_vol = chain["volume"].sum()
            total_oi = chain["openInterest"].sum()
            if total_oi > 0:
                features["opt_unusual_volume"] = float(total_vol / total_oi)

        # IV Term Structure Slope (front month vs back month)
        expirations = chain["expiration"].unique()
        if len(expirations) >= 2 and "impliedVolatility" in chain.columns:
            front_iv = chain.loc[
                (chain["expiration"] == expirations[0]) & atm_mask, "impliedVolatility"
            ].mean()
            back_iv = chain.loc[
                (chain["expiration"] == expirations[1]) & atm_mask, "impliedVolatility"
            ].mean()
            if not (np.isnan(front_iv) or np.isnan(back_iv)):
                features["opt_iv_term_slope"] = float(back_iv - front_iv)

    except Exception as e:
        logger.debug("Error computing options features: %s", e)

    return features


def compute_variance_risk_premium(
    implied_vol: float,
    realized_vol: float,
) -> float:
    """Compute Variance Risk Premium: IV - RV.

    Positive VRP = market is pricing in more volatility than realized.
    This is a well-documented risk premium (short vol is profitable on average).
    """
    if np.isnan(implied_vol) or np.isnan(realized_vol):
        return np.nan
    return implied_vol - realized_vol
