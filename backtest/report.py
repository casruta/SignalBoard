"""Generate backtest performance reports and charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent / "results"


def generate_report(
    results: dict,
    save_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """Generate a full backtest report with charts.

    Parameters
    ----------
    results : dict from BacktestEngine.run()
    save_dir : directory to save charts (defaults to backtest/results/)
    show : if True, display plots interactively
    """
    save_dir = save_dir or RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    eq = results["equity_curve"]
    trades = results["trades"]
    metrics = results["metrics"]

    # 1. Equity Curve
    _plot_equity_curve(eq, metrics, save_dir, show)

    # 2. Drawdown Chart
    _plot_drawdown(eq, save_dir, show)

    # 3. Monthly Returns Heatmap
    _plot_monthly_heatmap(eq, save_dir, show)

    # 4. Trade Distribution
    if not trades.empty:
        _plot_trade_distribution(trades, save_dir, show)

    # 5. Summary text
    _write_summary(metrics, save_dir)

    return save_dir


def _plot_equity_curve(eq, metrics, save_dir, show):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq.index, eq["equity"], linewidth=1.5, color="#2196F3")
    ax.fill_between(eq.index, eq["equity"], alpha=0.1, color="#2196F3")

    ax.set_title(
        f"Equity Curve | Sharpe: {metrics['sharpe_ratio']:.2f} | "
        f"Return: {metrics['total_return']:.1%} | "
        f"Max DD: {metrics['max_drawdown']:.1%}",
        fontsize=13,
    )
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "equity_curve.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_drawdown(eq, save_dir, show):
    peak = eq["equity"].expanding().max()
    drawdown = (eq["equity"] - peak) / peak

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown, 0, color="#F44336", alpha=0.4)
    ax.plot(drawdown.index, drawdown, color="#F44336", linewidth=0.8)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "drawdown.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_monthly_heatmap(eq, save_dir, show):
    monthly = eq["equity"].resample("ME").last().pct_change()
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Monthly Returns Heatmap")
    fig.tight_layout()
    fig.savefig(save_dir / "monthly_heatmap.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_trade_distribution(trades, save_dir, show):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PnL distribution
    axes[0].hist(trades["pnl_pct"], bins=40, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Trade Return Distribution")
    axes[0].set_xlabel("Return %")

    # PnL over time
    trade_dates = pd.to_datetime(trades["exit_date"])
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades["pnl"]]
    axes[1].bar(trade_dates, trades["pnl"], color=colors, alpha=0.7, width=2)
    axes[1].set_title("Trade P&L Over Time")
    axes[1].set_ylabel("P&L ($)")

    fig.tight_layout()
    fig.savefig(save_dir / "trade_distribution.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _write_summary(metrics, save_dir):
    lines = ["# Backtest Summary\n"]
    for key, val in metrics.items():
        if isinstance(val, float):
            if "pct" in key or "return" in key or "rate" in key or "drawdown" in key:
                lines.append(f"  {key}: {val:.2%}")
            else:
                lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")

    text = "\n".join(lines)
    (save_dir / "summary.txt").write_text(text)
