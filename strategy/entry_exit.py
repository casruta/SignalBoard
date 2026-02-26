"""Entry and exit rule engine with ATR-adaptive stop levels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.predict import Prediction
from strategy.portfolio import Position, PortfolioState


@dataclass
class TradeSignal:
    """A decided trade action."""
    ticker: str
    action: str  # "OPEN_LONG", "CLOSE"
    reason: str
    prediction: Prediction | None = None


class EntryExitEngine:
    """Evaluate entry and exit conditions with ATR-adaptive stops.

    Stop levels scale with each asset's own volatility (ATR) instead of
    using fixed percentages, so a high-vol stock like NVDA gets wider stops
    while a low-vol stock like KO gets tighter ones.
    """

    # ATR multipliers for stop calculations
    STOP_ATR_MULT = 2.0       # stop loss = entry - ATR * 2
    TARGET_ATR_MULT = 3.0     # take profit = entry + ATR * 3
    TRAIL_TRIGGER_ATR = 2.0   # trailing activates after ATR * 2 gain
    TRAIL_DISTANCE_ATR = 1.5  # trail distance = ATR * 1.5

    def __init__(self, config: dict):
        strat = config["strategy"]
        self.min_confidence = strat["min_confidence_threshold"]
        # Fixed pct fallbacks (used only when ATR is unavailable)
        self.take_profit_pct = strat["take_profit_pct"] / 100
        self.stop_loss_pct = strat["stop_loss_pct"] / 100
        self.trailing_trigger_pct = strat["trailing_stop_trigger_pct"] / 100
        self.trailing_distance_pct = strat["trailing_stop_distance_pct"] / 100
        self.time_stop_days = strat["time_stop_days"]

    # ── Entry Logic ──────────────────────────────────────────────

    def evaluate_entries(
        self,
        predictions: list[Prediction],
        portfolio: PortfolioState,
        signal_agreement: dict[str, dict] | None = None,
    ) -> list[TradeSignal]:
        """Determine which predictions should trigger new positions.

        Parameters
        ----------
        predictions : list of model Predictions
        portfolio : current portfolio state
        signal_agreement : optional {ticker: {technical: bool, fundamental: bool, macro: bool}}
        """
        signals = []
        for pred in predictions:
            if pred.ticker in portfolio.positions:
                continue  # Already holding
            if pred.action == "HOLD":
                continue
            if pred.confidence < self.min_confidence:
                continue

            # Check signal agreement (at least 2 of 3 categories)
            if signal_agreement and pred.ticker in signal_agreement:
                agreement = signal_agreement[pred.ticker]
                agrees = sum(1 for v in agreement.values() if v)
                if agrees < 2:
                    continue

            if pred.action == "BUY":
                signals.append(
                    TradeSignal(
                        ticker=pred.ticker,
                        action="OPEN_LONG",
                        reason=f"ML confidence {pred.confidence:.0%}, action={pred.action}",
                        prediction=pred,
                    )
                )

        return signals

    # ── Exit Logic ───────────────────────────────────────────────

    def evaluate_exits(
        self,
        portfolio: PortfolioState,
        current_prices: dict[str, float],
        current_date: str,
        predictions: list[Prediction] | None = None,
    ) -> list[TradeSignal]:
        """Determine which positions should be closed.

        Uses the per-position stop/target levels set at entry time (ATR-adaptive).
        Also checks: trailing stop, time stop, signal reversal.
        """
        signals = []
        pred_map = {}
        if predictions:
            pred_map = {p.ticker: p for p in predictions}

        for ticker, pos in portfolio.positions.items():
            price = current_prices.get(ticker)
            if price is None:
                continue

            # Take profit — use the stored ATR-adaptive target
            if price >= pos.take_profit:
                pnl_pct = (price - pos.entry_price) / pos.entry_price
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="CLOSE",
                    reason=f"Take profit hit: {pnl_pct:+.1%} (target ${pos.take_profit:.2f})",
                ))
                continue

            # Stop loss — use the stored ATR-adaptive stop
            if price <= pos.stop_loss:
                pnl_pct = (price - pos.entry_price) / pos.entry_price
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="CLOSE",
                    reason=f"Stop loss hit: {pnl_pct:+.1%} (stop ${pos.stop_loss:.2f})",
                ))
                continue

            # Trailing stop
            if pos.peak_price is None:
                pos.peak_price = pos.entry_price
            pos.peak_price = max(pos.peak_price, price)

            if price >= pos.trailing_stop_trigger:
                trail_price = pos.peak_price - (pos.trailing_stop_trigger - pos.entry_price) * (
                    self.TRAIL_DISTANCE_ATR / self.TRAIL_TRIGGER_ATR
                )
                # Only ratchet trailing stop upward
                if pos.trailing_stop_price is not None:
                    trail_price = max(trail_price, pos.trailing_stop_price)
                pos.trailing_stop_price = trail_price

                if price <= trail_price:
                    signals.append(TradeSignal(
                        ticker=ticker,
                        action="CLOSE",
                        reason=f"Trailing stop hit: price {price:.2f} < trail {trail_price:.2f}",
                    ))
                    continue

            # Time stop
            if pos.time_stop_date and current_date >= pos.time_stop_date:
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="CLOSE",
                    reason=f"Time stop: held {self.time_stop_days} days",
                ))
                continue

            # Signal reversal
            if ticker in pred_map and pred_map[ticker].action == "SELL":
                if pred_map[ticker].confidence >= self.min_confidence:
                    signals.append(TradeSignal(
                        ticker=ticker,
                        action="CLOSE",
                        reason=f"Signal reversal to SELL ({pred_map[ticker].confidence:.0%})",
                    ))
                    continue

        return signals

    def compute_stop_levels(
        self, entry_price: float, atr: float | None = None,
    ) -> tuple[float, float, float]:
        """Return (stop_loss, take_profit, trailing_trigger) prices.

        When ATR is available, levels are adaptive to the asset's volatility:
            stop = entry - ATR * 2.0
            target = entry + ATR * 3.0  (1.5:1 reward/risk)
            trail_trigger = entry + ATR * 2.0

        Falls back to fixed-percentage levels when ATR is unavailable.
        """
        if atr is not None and atr > 0:
            stop = entry_price - atr * self.STOP_ATR_MULT
            target = entry_price + atr * self.TARGET_ATR_MULT
            trail_trigger = entry_price + atr * self.TRAIL_TRIGGER_ATR
        else:
            stop = entry_price * (1 - self.stop_loss_pct)
            target = entry_price * (1 + self.take_profit_pct)
            trail_trigger = entry_price * (1 + self.trailing_trigger_pct)
        return stop, target, trail_trigger
