"""Combine technical, fundamental, macro, and advanced signals into a unified feature matrix."""

import logging

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
from signals.fundamental_deep import (
    compute_deep_fundamentals,
    compute_industry_relative_metrics,
    compute_institutional_blindspot,
)
from signals.dcf_valuation import compute_dcf_valuation

logger = logging.getLogger(__name__)


class SignalCombiner:
    """Merge all signal categories into a single feature matrix."""

    def build_feature_matrix(
        self,
        prices: dict[str, pd.DataFrame],
        fundamentals: pd.DataFrame,
        macro_df: pd.DataFrame,
        statements: dict[str, dict] | None = None,
        alt_data: dict[str, dict] | None = None,
        risk_free_rate: float = 0.04,
    ) -> pd.DataFrame:
        """Build the combined feature matrix.

        Parameters
        ----------
        prices : {ticker: OHLCV DataFrame}
        fundamentals : DataFrame indexed by ticker (from fundamental_loader)
        macro_df : DataFrame from macro_loader.fetch_all_macro()
        statements : {ticker: dict of financial statements} (optional, for deep fundamentals)
        alt_data : {ticker: {insider_df, holders_df, short_df}} (optional)
        risk_free_rate : 10Y Treasury rate for DCF calculations

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

        # ── Deep Fundamentals: DCF + quarterly trends + institutional blindspot ──
        deep_fund_map: dict[str, dict] = {}
        dcf_map: dict[str, dict] = {}
        blindspot_map: dict[str, dict] = {}

        if statements:
            logger.info("Computing deep fundamentals for %d tickers...", len(statements))
            for ticker, stmts in statements.items():
                info = stmts.get("info", {})
                try:
                    deep_fund_map[ticker] = compute_deep_fundamentals(stmts, info)
                except Exception as e:
                    logger.debug("Deep fundamentals failed for %s: %s", ticker, e)

                try:
                    dcf_map[ticker] = compute_dcf_valuation(
                        stmts, info, risk_free_rate=risk_free_rate
                    )
                except Exception as e:
                    logger.debug("DCF failed for %s: %s", ticker, e)

            # Industry-relative metrics (need all tickers computed first)
            industry_map = {}
            if not fundamentals.empty and "industry" in fundamentals.columns:
                industry_map = fundamentals["industry"].to_dict()
            elif not fundamentals.empty and "sector" in fundamentals.columns:
                industry_map = fundamentals["sector"].to_dict()

            if deep_fund_map and industry_map:
                for ticker in deep_fund_map:
                    try:
                        industry_rel = compute_industry_relative_metrics(
                            ticker, deep_fund_map[ticker], deep_fund_map, industry_map,
                        )
                        deep_fund_map[ticker].update(industry_rel)
                    except Exception as e:
                        logger.debug("Industry-relative failed for %s: %s", ticker, e)

        # Institutional blindspot signals
        if alt_data:
            for ticker, adata in alt_data.items():
                info = {}
                if statements and ticker in statements:
                    info = statements[ticker].get("info", {})
                try:
                    blindspot_map[ticker] = compute_institutional_blindspot(
                        info,
                        adata.get("holders_df", pd.DataFrame()),
                        adata.get("insider_df", pd.DataFrame()),
                    )
                except Exception as e:
                    logger.debug("Blindspot signals failed for %s: %s", ticker, e)

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

            # Add fundamental features (original simple)
            for key, val in fund_row.items():
                combined[f"fund_{key}"] = val

            # Add deep fundamental features (quarterly trends, balance sheet, etc.)
            if ticker in deep_fund_map:
                for key, val in deep_fund_map[ticker].items():
                    combined[f"fund_{key}"] = val

            # Add DCF valuation features
            if ticker in dcf_map:
                for key, val in dcf_map[ticker].items():
                    combined[f"fund_dcf_{key}"] = val

            # Add institutional blindspot features
            if ticker in blindspot_map:
                for key, val in blindspot_map[ticker].items():
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
