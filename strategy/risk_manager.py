"""Portfolio-level risk management with correlation-aware limits, CVaR, and stress testing."""

import numpy as np
import pandas as pd

from strategy.portfolio import PortfolioState


class RiskManager:
    """Enforce portfolio-level risk limits including correlation-aware position checks."""

    def __init__(self, config: dict):
        strat = config["strategy"]
        self.max_drawdown_pct = strat["max_portfolio_drawdown_pct"] / 100
        self.daily_loss_limit_pct = strat["daily_loss_limit_pct"] / 100
        self.max_sector_pct = strat["max_sector_exposure_pct"] / 100
        self.max_positions = strat["max_positions"]
        self.max_correlated = 5
        self.correlation_threshold = 0.70  # positions correlated above this count as "same bet"
        self.cvar_limit = 0.05  # max portfolio CVaR (5% of equity)

        self._peak_equity = 0.0
        self._day_start_equity = 0.0
        self._trading_paused = False
        self._return_history: dict[str, list[float]] = {}  # ticker -> recent returns

    def update_day_start(self, portfolio: PortfolioState, prices: dict[str, float]):
        """Call at the start of each trading day."""
        equity = portfolio.total_equity(prices)
        self._day_start_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def update_return_history(
        self,
        returns_df: pd.DataFrame | None = None,
        prices_history: dict[str, list[float]] | None = None,
    ):
        """Update the return history used for correlation and CVaR calculations.

        Parameters
        ----------
        returns_df : DataFrame with tickers as columns and daily returns as rows
        prices_history : {ticker: [price_t-N, ..., price_t]} — converted to returns internally
        """
        if returns_df is not None:
            for col in returns_df.columns:
                self._return_history[col] = returns_df[col].dropna().tolist()
        elif prices_history is not None:
            for ticker, prices_list in prices_history.items():
                if len(prices_list) >= 2:
                    arr = np.array(prices_list)
                    rets = np.diff(arr) / arr[:-1]
                    self._return_history[ticker] = rets.tolist()

    def check_can_trade(
        self, portfolio: PortfolioState, prices: dict[str, float]
    ) -> tuple[bool, str]:
        """Check if trading is allowed. Returns (allowed, reason)."""
        equity = portfolio.total_equity(prices)

        # Portfolio drawdown check
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                self._trading_paused = True
                return False, f"Max drawdown breached: {drawdown:.1%} (limit: {self.max_drawdown_pct:.1%})"

        # Daily loss check
        if self._day_start_equity > 0:
            daily_loss = (self._day_start_equity - equity) / self._day_start_equity
            if daily_loss >= self.daily_loss_limit_pct:
                return False, f"Daily loss limit hit: {daily_loss:.1%} (limit: {self.daily_loss_limit_pct:.1%})"

        if self._trading_paused:
            # Un-pause if drawdown recovers to within 50% of limit
            if self._peak_equity > 0:
                drawdown = (self._peak_equity - equity) / self._peak_equity
                if drawdown < self.max_drawdown_pct * 0.5:
                    self._trading_paused = False
                else:
                    return False, "Trading paused due to prior drawdown breach"

        return True, "OK"

    def check_sector_limit(
        self,
        portfolio: PortfolioState,
        sector: str,
        new_position_value: float,
        prices: dict[str, float],
    ) -> bool:
        """Check if adding a position in *sector* would breach sector limits."""
        equity = portfolio.total_equity(prices)
        if equity <= 0:
            return False

        current_sector = portfolio.sector_exposure(prices)
        current_pct = current_sector.get(sector, 0)
        additional_pct = new_position_value / equity

        return (current_pct + additional_pct) <= self.max_sector_pct

    def check_correlation_limit(
        self,
        new_ticker: str,
        portfolio: PortfolioState,
        lookback: int = 60,
    ) -> tuple[bool, str]:
        """Check if adding new_ticker would create too many correlated positions.

        Counts how many existing positions have rolling correlation > threshold
        with the new ticker. Blocks if count >= max_correlated.
        """
        if new_ticker not in self._return_history:
            return True, "OK (no return history for correlation check)"

        new_rets = np.array(self._return_history[new_ticker])
        if len(new_rets) < 20:
            return True, "OK (insufficient history for correlation)"

        correlated_count = 0
        high_corr_tickers = []

        for held_ticker in portfolio.positions:
            if held_ticker not in self._return_history:
                continue
            held_rets = np.array(self._return_history[held_ticker])

            # Align lengths
            min_len = min(len(new_rets), len(held_rets), lookback)
            if min_len < 20:
                continue

            a = new_rets[-min_len:]
            b = held_rets[-min_len:]

            corr = np.corrcoef(a, b)[0, 1]
            if not np.isnan(corr) and abs(corr) >= self.correlation_threshold:
                correlated_count += 1
                high_corr_tickers.append(f"{held_ticker}({corr:.2f})")

        if correlated_count >= self.max_correlated:
            return False, (
                f"Correlation limit: {new_ticker} correlated (>{self.correlation_threshold:.0%}) "
                f"with {correlated_count} positions: {', '.join(high_corr_tickers)}"
            )

        return True, "OK"

    # ── CVaR (Expected Shortfall) ──────────────────────────────

    def compute_portfolio_cvar(
        self,
        portfolio: PortfolioState,
        prices: dict[str, float],
        confidence: float = 0.95,
        lookback: int = 252,
    ) -> float:
        """Compute portfolio Conditional VaR (Expected Shortfall).

        Uses historical simulation: weight individual asset returns by
        current portfolio weights, then compute the mean of the worst
        (1-confidence)% of days.

        Returns CVaR as a positive fraction of portfolio equity.
        """
        equity = portfolio.total_equity(prices)
        if equity <= 0:
            return 0.0

        # Build portfolio return series
        weights = {}
        for ticker, pos in portfolio.positions.items():
            price = prices.get(ticker, pos.entry_price)
            weights[ticker] = (pos.shares * price) / equity

        tickers_with_data = [t for t in weights if t in self._return_history]
        if not tickers_with_data:
            return 0.0

        # Align to common length
        min_len = min(len(self._return_history[t]) for t in tickers_with_data)
        min_len = min(min_len, lookback)
        if min_len < 20:
            return 0.0

        portfolio_returns = np.zeros(min_len)
        for ticker in tickers_with_data:
            rets = np.array(self._return_history[ticker][-min_len:])
            portfolio_returns += weights[ticker] * rets

        # Sort returns ascending (worst first)
        sorted_returns = np.sort(portfolio_returns)
        cutoff_idx = max(1, int(len(sorted_returns) * (1 - confidence)))
        tail_returns = sorted_returns[:cutoff_idx]

        cvar = -np.mean(tail_returns)  # positive number = loss magnitude
        return float(cvar)

    def check_cvar_limit(
        self, portfolio: PortfolioState, prices: dict[str, float],
    ) -> tuple[bool, str]:
        """Check if portfolio CVaR exceeds the configured limit."""
        cvar = self.compute_portfolio_cvar(portfolio, prices)
        if cvar > self.cvar_limit:
            return False, f"CVaR limit breached: {cvar:.2%} > {self.cvar_limit:.2%}"
        return True, "OK"

    # ── Stress Testing ─────────────────────────────────────────

    def stress_test(
        self,
        portfolio: PortfolioState,
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Run standard stress scenarios on the current portfolio.

        Returns {scenario_name: portfolio_pnl_pct}.
        """
        equity = portfolio.total_equity(prices)
        if equity <= 0:
            return {}

        scenarios = {
            "market_crash_10pct": -0.10,
            "market_crash_20pct": -0.20,
            "sector_rotation_5pct": -0.05,
            "flash_crash_15pct": -0.15,
            "recovery_rally_10pct": 0.10,
        }

        results = {}
        for scenario_name, shock in scenarios.items():
            scenario_pnl = 0.0
            for ticker, pos in portfolio.positions.items():
                price = prices.get(ticker, pos.entry_price)
                position_value = pos.shares * price
                scenario_pnl += position_value * shock
            results[scenario_name] = scenario_pnl / equity

        # Correlation-amplified scenario
        if len(portfolio.positions) >= 2:
            avg_corr = self._avg_portfolio_correlation(portfolio)
            amplification = 1.0 + max(0, avg_corr - 0.3) * 0.5
            results["correlated_crash_10pct"] = -0.10 * amplification

        return results

    def _avg_portfolio_correlation(self, portfolio: PortfolioState) -> float:
        """Compute average pairwise correlation among held positions."""
        tickers = [t for t in portfolio.positions if t in self._return_history]
        if len(tickers) < 2:
            return 0.0

        correlations = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                a = np.array(self._return_history[tickers[i]])
                b = np.array(self._return_history[tickers[j]])
                min_len = min(len(a), len(b), 60)
                if min_len < 20:
                    continue
                corr = np.corrcoef(a[-min_len:], b[-min_len:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        return float(np.mean(correlations)) if correlations else 0.0

    # ── Market Impact Model ────────────────────────────────────

    @staticmethod
    def estimate_market_impact(
        shares: int,
        price: float,
        avg_daily_volume: float,
        spread_bps: float = 5.0,
    ) -> float:
        """Estimate total transaction cost including market impact.

        Uses a square-root market impact model:
            impact = spread/2 + sigma * sqrt(shares / ADV)

        Returns estimated cost as a fraction of trade notional.
        """
        if avg_daily_volume <= 0 or shares <= 0:
            return 0.0

        participation_rate = shares / avg_daily_volume
        half_spread = (spread_bps / 10_000) / 2

        # Square-root impact: typical sigma ~0.15 for large-cap US equities
        sigma = 0.15
        impact = sigma * np.sqrt(participation_rate)

        return float(half_spread + impact)

    def get_risk_summary(
        self, portfolio: PortfolioState, prices: dict[str, float]
    ) -> dict:
        """Return current risk metrics including CVaR and correlation info."""
        equity = portfolio.total_equity(prices)
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
        daily_pnl = 0.0
        if self._day_start_equity > 0:
            daily_pnl = (equity - self._day_start_equity) / self._day_start_equity

        cvar = self.compute_portfolio_cvar(portfolio, prices)
        avg_corr = self._avg_portfolio_correlation(portfolio)

        return {
            "equity": equity,
            "peak_equity": self._peak_equity,
            "drawdown_pct": drawdown,
            "daily_pnl_pct": daily_pnl,
            "num_positions": portfolio.num_positions,
            "max_positions": self.max_positions,
            "sector_exposure": portfolio.sector_exposure(prices),
            "trading_paused": self._trading_paused,
            "portfolio_cvar_95": cvar,
            "avg_position_correlation": avg_corr,
        }
