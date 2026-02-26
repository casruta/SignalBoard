"""Execution optimization: adaptive slippage, VWAP/TWAP models, turnover-aware construction."""

import numpy as np
import pandas as pd


class AdaptiveSlippageModel:
    """Regime-dependent slippage model.

    Replaces fixed 5bps with a model that accounts for:
    - Market volatility regime (higher VIX = wider spreads)
    - Asset liquidity (volume, market cap)
    - Time of day (opening/closing higher impact)
    - Order size relative to ADV
    """

    # Base spread assumptions by market cap tier (bps)
    SPREAD_BY_TIER = {
        "large": 3.0,    # >$50B
        "mid": 8.0,      # $5B-$50B
        "small": 15.0,   # <$5B
    }

    def __init__(self):
        self.vix_regime_mult = {0: 0.8, 1: 1.0, 2: 1.5}  # low/normal/high

    def estimate_slippage(
        self,
        price: float,
        shares: int,
        avg_daily_volume: float,
        market_cap: float | None = None,
        vix_level: float | None = None,
    ) -> float:
        """Estimate execution slippage as a fraction of trade value.

        Returns slippage as a decimal (e.g., 0.001 = 10bps).
        """
        if shares <= 0 or price <= 0:
            return 0.0

        # Base spread from market cap tier
        tier = self._market_cap_tier(market_cap)
        base_spread_bps = self.SPREAD_BY_TIER.get(tier, 8.0)

        # VIX regime multiplier
        vix_mult = 1.0
        if vix_level is not None:
            if vix_level < 15:
                vix_mult = 0.8
            elif vix_level > 25:
                vix_mult = 1.5
            elif vix_level > 35:
                vix_mult = 2.0

        # Participation rate impact (square-root model)
        participation = shares / max(avg_daily_volume, 1)
        impact_bps = 15.0 * np.sqrt(participation)  # typical for large-cap

        # Total slippage in bps
        total_bps = (base_spread_bps / 2 + impact_bps) * vix_mult

        return total_bps / 10_000

    @staticmethod
    def _market_cap_tier(market_cap: float | None) -> str:
        if market_cap is None:
            return "mid"
        if market_cap >= 50e9:
            return "large"
        elif market_cap >= 5e9:
            return "mid"
        return "small"


class VWAPExecutor:
    """Simulate VWAP (Volume-Weighted Average Price) execution.

    Splits a large order across the day proportional to historical volume profile.
    """

    # Typical U-shaped intraday volume profile (30-minute buckets, 13 buckets)
    VOLUME_PROFILE = np.array([
        0.12, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05,
        0.05, 0.06, 0.06, 0.07, 0.10, 0.20,
    ])

    def simulate_vwap_execution(
        self,
        total_shares: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        daily_volume: int,
    ) -> dict:
        """Simulate VWAP execution for a single day.

        Returns dict with 'avg_fill_price', 'slippage_bps', 'execution_pct'.
        """
        if total_shares <= 0:
            return {"avg_fill_price": open_price, "slippage_bps": 0, "execution_pct": 0}

        profile = self.VOLUME_PROFILE / self.VOLUME_PROFILE.sum()
        n_buckets = len(profile)

        # Simulate prices at each bucket (linear interpolation within day range)
        bucket_prices = np.linspace(open_price, close, n_buckets)
        # Add some high/low noise
        noise = np.random.default_rng(42).uniform(-1, 1, n_buckets)
        price_range = high - low
        bucket_prices += noise * price_range * 0.1

        # Shares per bucket
        shares_per_bucket = (profile * total_shares).astype(int)
        shares_per_bucket[-1] += total_shares - shares_per_bucket.sum()  # remainder

        # Volume-weighted average fill
        dollar_total = (shares_per_bucket * bucket_prices).sum()
        shares_filled = shares_per_bucket.sum()
        avg_fill = dollar_total / shares_filled if shares_filled > 0 else open_price

        # VWAP of the day
        vwap = (bucket_prices * profile).sum()

        # Slippage vs VWAP
        slippage_bps = abs(avg_fill - vwap) / vwap * 10_000

        return {
            "avg_fill_price": float(avg_fill),
            "vwap": float(vwap),
            "slippage_vs_vwap_bps": float(slippage_bps),
            "execution_pct": float(shares_filled / max(daily_volume, 1)),
        }


class TurnoverOptimizer:
    """Penalize portfolio turnover to reduce transaction costs.

    Only trade when the expected marginal alpha exceeds estimated costs.
    """

    def __init__(self, min_alpha_to_trade: float = 0.003, cost_per_trade_bps: float = 10.0):
        self.min_alpha = min_alpha_to_trade
        self.cost_bps = cost_per_trade_bps / 10_000

    def filter_trades(
        self,
        proposed_trades: list[dict],
        current_positions: dict[str, float],
    ) -> list[dict]:
        """Filter proposed trades to only include those exceeding cost threshold.

        Parameters
        ----------
        proposed_trades : list of {'ticker', 'action', 'expected_return', ...}
        current_positions : {ticker: current_value}

        Returns
        -------
        Filtered list of trades worth executing.
        """
        filtered = []
        for trade in proposed_trades:
            er = trade.get("expected_return", 0)

            # New position: alpha must exceed round-trip cost
            if trade.get("action") == "OPEN_LONG":
                if abs(er) > self.min_alpha + 2 * self.cost_bps:
                    filtered.append(trade)

            # Close position: always allow stops and signal reversals
            elif trade.get("action") == "CLOSE":
                reason = trade.get("reason", "")
                # Always honor stop losses and signal reversals
                if "stop" in reason.lower() or "reversal" in reason.lower():
                    filtered.append(trade)
                # For discretionary closes, check if cost is worth it
                elif abs(er) > self.cost_bps:
                    filtered.append(trade)

        return filtered

    def compute_optimal_rebalance_frequency(
        self,
        signal_autocorrelation: float,
        avg_trade_cost_bps: float,
        signal_ic: float,
    ) -> str:
        """Determine optimal rebalancing frequency from signal properties.

        Parameters
        ----------
        signal_autocorrelation : lag-1 autocorrelation of the trading signal
        avg_trade_cost_bps : average round-trip cost
        signal_ic : information coefficient of the signal

        Returns
        -------
        Recommended frequency: 'daily', 'weekly', 'biweekly', 'monthly'
        """
        # Alpha decay: signals with high autocorrelation decay slowly
        # Higher IC * lower turnover cost = can afford to trade more often

        if signal_ic <= 0:
            return "monthly"

        # Rough heuristic: IC * sqrt(252/freq) - cost * freq
        # Higher autocorrelation = slower decay = less frequent rebalancing
        decay_rate = 1 - signal_autocorrelation

        if decay_rate > 0.5 and signal_ic > 0.05:
            return "daily"
        elif decay_rate > 0.3 and signal_ic > 0.03:
            return "weekly"
        elif decay_rate > 0.1:
            return "biweekly"
        return "monthly"
