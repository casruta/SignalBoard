"""Backtest performance metrics."""

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """Compute comprehensive performance metrics.

    Parameters
    ----------
    equity_curve : DataFrame with columns: equity, daily_return
    trades : DataFrame with columns: pnl, pnl_pct, entry_date, exit_date
    initial_capital : starting capital
    """
    returns = equity_curve["daily_return"].dropna()

    # Basic return metrics
    total_return = (equity_curve["equity"].iloc[-1] / initial_capital) - 1
    trading_days = len(returns)
    annual_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

    # Risk metrics
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    max_dd, max_dd_duration = max_drawdown(equity_curve["equity"])
    volatility = returns.std() * np.sqrt(252)

    # Trade metrics
    trade_metrics = {}
    if not trades.empty:
        winners = trades[trades["pnl"] > 0]
        losers = trades[trades["pnl"] <= 0]
        trade_metrics = {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / max(len(trades), 1),
            "avg_win_pct": float(winners["pnl_pct"].mean()) if len(winners) > 0 else 0,
            "avg_loss_pct": float(losers["pnl_pct"].mean()) if len(losers) > 0 else 0,
            "largest_win": float(trades["pnl"].max()),
            "largest_loss": float(trades["pnl"].min()),
            "profit_factor": profit_factor(trades),
            "avg_holding_days": _avg_holding_days(trades),
        }

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": max_dd_duration,
        "annual_volatility": volatility,
        "trading_days": trading_days,
        **trade_metrics,
    }


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.04, periods: int = 252
) -> float:
    """Annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.04, periods: int = 252
) -> float:
    """Annualized Sortino ratio (penalizes only downside volatility)."""
    if returns.empty:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Return (max_drawdown_pct, duration_in_days)."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    # Duration: longest period below previous peak
    is_dd = equity < peak
    if not is_dd.any():
        return 0.0, 0
    groups = (~is_dd).cumsum()
    durations = is_dd.groupby(groups).sum()
    max_duration = int(durations.max()) if not durations.empty else 0

    return max_dd, max_duration


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profits / gross losses."""
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] <= 0, "pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def _avg_holding_days(trades: pd.DataFrame) -> float:
    if trades.empty or "entry_date" not in trades or "exit_date" not in trades:
        return 0.0
    entry = pd.to_datetime(trades["entry_date"])
    exit_ = pd.to_datetime(trades["exit_date"])
    return float((exit_ - entry).dt.days.mean())
