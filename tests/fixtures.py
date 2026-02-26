"""Synthetic data fixtures for offline testing. No API keys required."""

from datetime import datetime

import numpy as np
import pandas as pd


def make_config() -> dict:
    """Return a minimal config dict matching config.example.yaml structure."""
    return {
        "fred": {"api_key": "TEST_KEY"},
        "universe": {
            "tickers": ["AAPL", "MSFT", "JPM"],
            "lookback_years": 2,
        },
        "strategy": {
            "max_positions": 5,
            "max_position_size_pct": 20.0,
            "max_sector_exposure_pct": 40.0,
            "max_portfolio_drawdown_pct": 15.0,
            "daily_loss_limit_pct": 3.0,
            "take_profit_pct": 5.0,
            "stop_loss_pct": 3.0,
            "trailing_stop_trigger_pct": 3.0,
            "trailing_stop_distance_pct": 1.5,
            "time_stop_days": 15,
            "min_confidence_threshold": 0.60,
            "push_notification_threshold": 0.75,
        },
        "model": {
            "target_horizon_days": 5,
            "train_window_years": 1,
            "validation_window_months": 2,
            "retrain_frequency_days": 30,
            "calibrate_probabilities": False,
            "check_regime_shift": False,
        },
        "backtest": {
            "adaptive_slippage": False,
            "include_network_signals": True,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "database_path": ":memory:",
            "daily_run_hour": 16,
            "daily_run_minute": 30,
            "timezone": "US/Eastern",
        },
        "apns": {
            "key_path": "",
            "key_id": "",
            "team_id": "",
            "bundle_id": "",
            "use_sandbox": True,
        },
    }


def make_prices(
    tickers: list[str] | None = None, n_days: int = 600,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for multiple tickers."""
    tickers = tickers or ["AAPL", "MSFT", "JPM"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n_days)
    result = {}

    for i, ticker in enumerate(tickers):
        base = 100 + i * 50
        returns = rng.normal(0.0003, 0.015, n_days)
        close = base * np.cumprod(1 + returns)
        high = close * (1 + rng.uniform(0, 0.02, n_days))
        low = close * (1 - rng.uniform(0, 0.02, n_days))
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        volume = rng.integers(1_000_000, 10_000_000, n_days).astype(float)

        result[ticker] = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    return result


def make_fundamentals(tickers: list[str] | None = None) -> pd.DataFrame:
    """Generate synthetic fundamentals DataFrame."""
    tickers = tickers or ["AAPL", "MSFT", "JPM"]
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    rng = np.random.default_rng(42)

    rows = []
    for t in tickers:
        rows.append(
            {
                "pe_ratio": rng.uniform(10, 35),
                "forward_pe": rng.uniform(10, 30),
                "eps": rng.uniform(2, 15),
                "forward_eps": rng.uniform(2, 16),
                "eps_growth": rng.uniform(-0.1, 0.3),
                "revenue_growth": rng.uniform(-0.05, 0.2),
                "debt_to_equity": rng.uniform(0.2, 2.0),
                "roe": rng.uniform(0.05, 0.4),
                "dividend_yield": rng.uniform(0, 0.03),
                "pb_ratio": rng.uniform(1, 10),
                "sector": sectors.get(t, "Unknown"),
                "industry": "Software",
                "market_cap": rng.uniform(100e9, 3e12),
                "short_name": t,
            }
        )

    return pd.DataFrame(rows, index=tickers)


def make_macro(n_days: int = 600) -> pd.DataFrame:
    """Generate synthetic macro DataFrame matching macro_loader output."""
    dates = pd.bdate_range(end=datetime.now(), periods=n_days)
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "treasury_10y": 3.5 + rng.normal(0, 0.1, n_days).cumsum() * 0.01,
            "treasury_2y": 3.0 + rng.normal(0, 0.1, n_days).cumsum() * 0.01,
            "treasury_3m": 2.5 + rng.normal(0, 0.05, n_days).cumsum() * 0.01,
            "fed_funds": np.full(n_days, 5.25),
            "vix": np.clip(18 + rng.normal(0, 2, n_days).cumsum() * 0.1, 10, 60),
            "unemployment": np.full(n_days, 3.7),
            "cpi": 300 + np.arange(n_days) * 0.02,
            "oil_wti": 75 + rng.normal(0, 1, n_days).cumsum() * 0.1,
        },
        index=dates,
    )
