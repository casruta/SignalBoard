"""Combine technical, fundamental, and macro signals into a unified feature matrix."""

import numpy as np
import pandas as pd

from signals.technical import compute_all_technical
from signals.fundamental import compute_fundamental_signals
from signals.macro import compute_macro_signals


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

        rows = []
        for ticker, price_df in prices.items():
            # Technical signals for this ticker
            tech = compute_all_technical(price_df)

            # Merge macro signals onto the same dates (forward-fill to daily)
            macro_aligned = macro_signals.reindex(tech.index, method="ffill")

            # Fundamental signals are static per ticker — broadcast
            fund_row = {}
            if ticker in fund_signals.index:
                fund_row = fund_signals.loc[ticker].to_dict()

            # Combine into one row per date
            combined = tech.copy()
            for col in macro_aligned.columns:
                combined[f"macro_{col}"] = macro_aligned[col]
            for key, val in fund_row.items():
                combined[f"fund_{key}"] = val

            combined["ticker"] = ticker
            rows.append(combined)

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows)
        result = result.reset_index().rename(columns={"index": "date"})
        result = result.set_index(["date", "ticker"]).sort_index()

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
