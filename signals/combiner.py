"""Combine technical, fundamental, macro, and advanced signals into a unified feature matrix."""

import numpy as np
import pandas as pd

from signals.technical import compute_all_technical
from signals.fundamental import compute_fundamental_signals
from signals.macro import compute_macro_signals
from signals.microstructure import compute_microstructure_features
from signals.statistical import compute_statistical_features
from signals.calendar_features import compute_calendar_features
from signals.cross_sectional import compute_cross_sectional_features
from signals.interactions import compute_interaction_features
from signals.network_analysis import compute_network_features


class SignalCombiner:
    """Merge all signal categories into a single feature matrix."""

    def build_feature_matrix(
        self,
        prices: dict[str, pd.DataFrame],
        fundamentals: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the combined feature matrix.

        Parameters
        ----------
        prices : {ticker: OHLCV DataFrame}
        fundamentals : DataFrame indexed by ticker (from fundamental_loader)
        macro_df : DataFrame from macro_loader.fetch_all_macro()

        Returns
        -------
        DataFrame with MultiIndex (date, ticker) and all signal columns.
        """
        # Compute macro signals once (shared across all tickers)
        macro_signals = compute_macro_signals(macro_df)

        # Compute fundamental signals once
        fund_signals = compute_fundamental_signals(fundamentals)

        # Compute calendar features once (shared across all tickers)
        all_dates = set()
        for price_df in prices.values():
            all_dates.update(price_df.index)
        if all_dates:
            calendar_idx = pd.DatetimeIndex(sorted(all_dates))
            calendar_features = compute_calendar_features(calendar_idx)
        else:
            calendar_features = pd.DataFrame()

        # Compute network features once (cross-asset correlations)
        try:
            network_features = compute_network_features(prices, window=60)
        except Exception:
            network_features = pd.DataFrame()

        rows = []
        for ticker, price_df in prices.items():
            # Technical signals for this ticker
            tech = compute_all_technical(price_df)

            # Microstructure signals
            micro = compute_microstructure_features(price_df)

            # Statistical regime signals
            stat = compute_statistical_features(price_df)

            # Merge macro signals onto the same dates (forward-fill to daily)
            macro_aligned = macro_signals.reindex(tech.index, method="ffill")

            # Calendar features aligned
            cal_aligned = (
                calendar_features.reindex(tech.index, method="nearest")
                if len(calendar_features) > 0
                else pd.DataFrame(index=tech.index)
            )

            # Fundamental signals are static per ticker — broadcast
            fund_row = {}
            if ticker in fund_signals.index:
                fund_row = fund_signals.loc[ticker].to_dict()

            # Combine into one row per date
            combined = tech.copy()

            # Add microstructure features
            for col in micro.columns:
                combined[f"micro_{col}"] = micro[col]

            # Add statistical features
            for col in stat.columns:
                combined[f"stat_{col}"] = stat[col]

            # Add macro features
            for col in macro_aligned.columns:
                combined[f"macro_{col}"] = macro_aligned[col]

            # Add calendar features
            for col in cal_aligned.columns:
                combined[f"cal_{col}"] = cal_aligned[col]

            # Add fundamental features
            for key, val in fund_row.items():
                combined[f"fund_{key}"] = val

            # Add network features (static per ticker)
            if len(network_features) > 0 and ticker in network_features.index:
                for col in network_features.columns:
                    combined[col] = float(network_features.loc[ticker, col])

            combined["ticker"] = ticker
            rows.append(combined)

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows)
        result = result.reset_index().rename(columns={"index": "date"})
        result = result.set_index(["date", "ticker"]).sort_index()

        # Add cross-sectional features (need full universe at once)
        result = compute_cross_sectional_features(result)

        # Add interaction features
        result = compute_interaction_features(result)

        return result

    @staticmethod
    def normalize_features(
        df: pd.DataFrame, method: str = "zscore"
    ) -> pd.DataFrame:
        """Normalize numeric columns to comparable scales.

        Parameters
        ----------
        method : 'zscore' or 'minmax'
        """
        numeric = df.select_dtypes(include=[np.number])
        non_numeric = df.select_dtypes(exclude=[np.number])

        if method == "zscore":
            normalized = (numeric - numeric.mean()) / numeric.std().replace(0, 1)
        elif method == "minmax":
            min_vals = numeric.min()
            range_vals = (numeric.max() - min_vals).replace(0, 1)
            normalized = (numeric - min_vals) / range_vals
        else:
            raise ValueError(f"Unknown method: {method}")

        return pd.concat([normalized, non_numeric], axis=1)
