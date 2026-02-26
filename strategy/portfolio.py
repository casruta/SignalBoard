"""Portfolio construction and position sizing."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.predict import Prediction


@dataclass
class Position:
    """An active portfolio position."""
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    direction: str  # "long" or "short" (short is future)
    stop_loss: float
    take_profit: float
    trailing_stop_trigger: float
    trailing_stop_price: float | None = None
    time_stop_date: str | None = None
    sector: str = "Unknown"
    peak_price: float | None = None

    @property
    def is_trailing(self) -> bool:
        return self.trailing_stop_price is not None


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    initial_capital: float = 100_000.0
    trade_history: list[dict] = field(default_factory=list)

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    def position_value(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos.entry_price)
            total += pos.shares * price
        return total

    def total_equity(self, current_prices: dict[str, float]) -> float:
        return self.cash + self.position_value(current_prices)

    def sector_exposure(self, current_prices: dict[str, float]) -> dict[str, float]:
        equity = self.total_equity(current_prices)
        if equity <= 0:
            return {}
        sectors = {}
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos.entry_price)
            val = pos.shares * price
            sectors[pos.sector] = sectors.get(pos.sector, 0) + val / equity
        return sectors


class PortfolioConstructor:
    """Determine position sizes for new trades."""

    def __init__(self, config: dict):
        strat = config["strategy"]
        self.max_positions = strat["max_positions"]
        self.max_position_pct = strat["max_position_size_pct"] / 100
        self.max_sector_pct = strat["max_sector_exposure_pct"] / 100

    def size_position(
        self,
        prediction: Prediction,
        portfolio: PortfolioState,
        current_price: float,
        atr: float,
        sector: str,
        current_prices: dict[str, float],
    ) -> int:
        """Calculate number of shares to buy for a given signal.

        Uses ATR-based volatility scaling: allocate a fixed dollar risk per trade,
        then scale position size inversely with volatility.

        Returns 0 if the position should be skipped (limits exceeded).
        """
        equity = portfolio.total_equity(current_prices)

        # Check position count limit
        if portfolio.num_positions >= self.max_positions:
            return 0

        # Check sector limit
        sector_exp = portfolio.sector_exposure(current_prices)
        if sector_exp.get(sector, 0) >= self.max_sector_pct:
            return 0

        # Max dollar allocation for this position
        max_dollars = equity * self.max_position_pct

        # Volatility-scaled sizing: risk 1% of equity per trade
        # Position size = risk_dollars / (ATR * 2)
        risk_dollars = equity * 0.01
        if atr > 0:
            vol_sized_dollars = (risk_dollars / (atr * 2)) * current_price
        else:
            vol_sized_dollars = max_dollars

        # Take the smaller of max allocation and vol-sized
        dollars = min(max_dollars, vol_sized_dollars, portfolio.cash)

        if dollars < current_price:
            return 0

        return int(dollars / current_price)
