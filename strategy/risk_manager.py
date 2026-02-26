"""Portfolio-level risk management and circuit breakers."""

import pandas as pd

from strategy.portfolio import PortfolioState


class RiskManager:
    """Enforce portfolio-level risk limits."""

    def __init__(self, config: dict):
        strat = config["strategy"]
        self.max_drawdown_pct = strat["max_portfolio_drawdown_pct"] / 100
        self.daily_loss_limit_pct = strat["daily_loss_limit_pct"] / 100
        self.max_sector_pct = strat["max_sector_exposure_pct"] / 100
        self.max_positions = strat["max_positions"]
        self.max_correlated = 5

        self._peak_equity = 0.0
        self._day_start_equity = 0.0
        self._trading_paused = False

    def update_day_start(self, portfolio: PortfolioState, prices: dict[str, float]):
        """Call at the start of each trading day."""
        equity = portfolio.total_equity(prices)
        self._day_start_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

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

    def get_risk_summary(
        self, portfolio: PortfolioState, prices: dict[str, float]
    ) -> dict:
        """Return current risk metrics."""
        equity = portfolio.total_equity(prices)
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
        daily_pnl = 0.0
        if self._day_start_equity > 0:
            daily_pnl = (equity - self._day_start_equity) / self._day_start_equity

        return {
            "equity": equity,
            "peak_equity": self._peak_equity,
            "drawdown_pct": drawdown,
            "daily_pnl_pct": daily_pnl,
            "num_positions": portfolio.num_positions,
            "max_positions": self.max_positions,
            "sector_exposure": portfolio.sector_exposure(prices),
            "trading_paused": self._trading_paused,
        }
