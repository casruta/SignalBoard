"""Event-driven backtesting engine."""

from datetime import timedelta

import numpy as np
import pandas as pd

from models.predict import Prediction, predict_batch
from strategy.portfolio import PortfolioState, Position, PortfolioConstructor
from strategy.entry_exit import EntryExitEngine
from strategy.risk_manager import RiskManager
from backtest.metrics import compute_metrics


class BacktestEngine:
    """Simulate the full trading strategy over historical data.

    Processes one day at a time to avoid lookahead bias.
    Accounts for slippage and commissions.
    """

    def __init__(
        self,
        config: dict,
        slippage_pct: float = 0.0005,
        commission_per_trade: float = 0.0,
    ):
        self.config = config
        self.slippage_pct = slippage_pct
        self.commission = commission_per_trade
        self.portfolio_constructor = PortfolioConstructor(config)
        self.entry_exit = EntryExitEngine(config)
        self.risk_manager = RiskManager(config)

    def run(
        self,
        model,
        feature_matrix: pd.DataFrame,
        prices: dict[str, pd.DataFrame],
        fundamentals: pd.DataFrame,
        initial_capital: float = 100_000.0,
    ) -> dict:
        """Run the full backtest.

        Parameters
        ----------
        model : trained LightGBM booster
        feature_matrix : MultiIndex (date, ticker) from SignalCombiner
        prices : {ticker: OHLCV DataFrame}
        fundamentals : DataFrame indexed by ticker
        initial_capital : starting cash

        Returns
        -------
        dict with keys: equity_curve, trades, metrics, daily_returns
        """
        portfolio = PortfolioState(
            cash=initial_capital,
            initial_capital=initial_capital,
        )

        dates = feature_matrix.index.get_level_values("date").unique().sort_values()

        # Drop dates where we don't have enough data for targets
        target_cols = ["target_return", "target_class"]
        non_feature = target_cols
        feature_cols = [
            c for c in feature_matrix.columns
            if c not in non_feature
            and feature_matrix[c].dtype in [np.float64, np.int64, float, int]
        ]

        equity_curve = []
        all_trades = []

        for i, date in enumerate(dates):
            # Get today's prices
            current_prices = {}
            for ticker in prices:
                if date in prices[ticker].index:
                    current_prices[ticker] = float(prices[ticker].loc[date, "Close"])

            if not current_prices:
                continue

            # Update risk manager for the day
            self.risk_manager.update_day_start(portfolio, current_prices)

            # ── Step 1: Evaluate exits on existing positions ─────
            today_predictions = self._get_predictions(
                model, feature_matrix, date, feature_cols
            )

            exit_signals = self.entry_exit.evaluate_exits(
                portfolio, current_prices, str(date.date()), today_predictions
            )

            for signal in exit_signals:
                if signal.ticker in portfolio.positions:
                    pos = portfolio.positions[signal.ticker]
                    exit_price = current_prices[signal.ticker]
                    # Apply slippage (worse price on exit)
                    exit_price *= (1 - self.slippage_pct)
                    proceeds = pos.shares * exit_price - self.commission
                    portfolio.cash += proceeds

                    pnl = (exit_price - pos.entry_price) * pos.shares
                    all_trades.append({
                        "ticker": signal.ticker,
                        "entry_date": pos.entry_date,
                        "exit_date": str(date.date()),
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "shares": pos.shares,
                        "pnl": pnl,
                        "pnl_pct": (exit_price - pos.entry_price) / pos.entry_price,
                        "reason": signal.reason,
                    })

                    del portfolio.positions[signal.ticker]

            # ── Step 2: Evaluate entries ─────────────────────────
            can_trade, reason = self.risk_manager.check_can_trade(
                portfolio, current_prices
            )

            if can_trade and today_predictions:
                entry_signals = self.entry_exit.evaluate_entries(
                    today_predictions, portfolio
                )

                for signal in entry_signals:
                    if signal.ticker not in current_prices:
                        continue

                    price = current_prices[signal.ticker]
                    # Apply slippage (worse price on entry)
                    entry_price = price * (1 + self.slippage_pct)

                    # Get ATR for position sizing
                    atr = self._get_atr(prices, signal.ticker, date)
                    sector = self._get_sector(fundamentals, signal.ticker)

                    shares = self.portfolio_constructor.size_position(
                        signal.prediction,
                        portfolio,
                        entry_price,
                        atr,
                        sector,
                        current_prices,
                    )

                    if shares <= 0:
                        continue

                    cost = shares * entry_price + self.commission
                    if cost > portfolio.cash:
                        continue

                    portfolio.cash -= cost

                    stop, target, trail = self.entry_exit.compute_stop_levels(
                        entry_price
                    )
                    time_stop = (date + timedelta(days=self.entry_exit.time_stop_days * 1.5))

                    portfolio.positions[signal.ticker] = Position(
                        ticker=signal.ticker,
                        entry_date=str(date.date()),
                        entry_price=entry_price,
                        shares=shares,
                        direction="long",
                        stop_loss=stop,
                        take_profit=target,
                        trailing_stop_trigger=trail,
                        time_stop_date=str(time_stop.date()),
                        sector=sector,
                    )

            # Record daily equity
            equity = portfolio.total_equity(current_prices)
            equity_curve.append({
                "date": date,
                "equity": equity,
                "cash": portfolio.cash,
                "num_positions": portfolio.num_positions,
            })

        # Build results
        eq_df = pd.DataFrame(equity_curve).set_index("date")
        eq_df["daily_return"] = eq_df["equity"].pct_change()

        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        metrics = compute_metrics(eq_df, trades_df, initial_capital)

        return {
            "equity_curve": eq_df,
            "trades": trades_df,
            "metrics": metrics,
        }

    def _get_predictions(
        self, model, feature_matrix, date, feature_cols
    ) -> list[Prediction]:
        """Get model predictions for a single date."""
        try:
            day_data = feature_matrix.loc[date]
            if isinstance(day_data, pd.Series):
                return []
            X = day_data[feature_cols].fillna(day_data[feature_cols].median())
            # Reconstruct MultiIndex for predict_batch
            X.index = pd.MultiIndex.from_tuples(
                [(date, t) for t in X.index], names=["date", "ticker"]
            )
            return predict_batch(model, X)
        except (KeyError, Exception):
            return []

    @staticmethod
    def _get_atr(
        prices: dict[str, pd.DataFrame], ticker: str, date
    ) -> float:
        """Get the 14-day ATR for a ticker at a given date."""
        if ticker not in prices:
            return 0.0
        df = prices[ticker]
        df_before = df[df.index <= date].tail(15)
        if len(df_before) < 2:
            return 0.0
        high = df_before["High"]
        low = df_before["Low"]
        close = df_before["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())

    @staticmethod
    def _get_sector(fundamentals: pd.DataFrame, ticker: str) -> str:
        if ticker in fundamentals.index and "sector" in fundamentals.columns:
            return str(fundamentals.loc[ticker, "sector"])
        return "Unknown"
