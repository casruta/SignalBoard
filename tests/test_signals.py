"""Unit tests for signal computation modules."""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import make_prices, make_fundamentals, make_macro


class TestTechnicalSignals:
    def test_compute_all_technical(self):
        prices = make_prices(["AAPL"], n_days=100)
        from signals.technical import compute_all_technical

        result = compute_all_technical(prices["AAPL"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Check key columns exist
        expected = ["rsi_14", "macd_histogram", "atr_14", "momentum_5"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_all_nan_columns_after_warmup(self):
        prices = make_prices(["AAPL"], n_days=200)
        from signals.technical import compute_all_technical

        result = compute_all_technical(prices["AAPL"])
        # After 50+ days of warmup, no column should be all NaN
        after_warmup = result.iloc[60:]
        for col in after_warmup.columns:
            assert not after_warmup[col].isna().all(), f"{col} is all NaN after warmup"


class TestFundamentalSignals:
    def test_compute_fundamental_signals(self):
        fundamentals = make_fundamentals()
        from signals.fundamental import compute_fundamental_signals

        result = compute_fundamental_signals(fundamentals)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(fundamentals)


class TestMacroSignals:
    def test_compute_macro_signals(self):
        macro = make_macro(n_days=200)
        from signals.macro import compute_macro_signals

        result = compute_macro_signals(macro)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "yield_curve_slope" in result.columns


class TestMicrostructureSignals:
    def test_compute_microstructure_features(self):
        prices = make_prices(["AAPL"], n_days=100)
        from signals.microstructure import compute_microstructure_features

        result = compute_microstructure_features(prices["AAPL"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestStatisticalSignals:
    def test_compute_statistical_features(self):
        prices = make_prices(["AAPL"], n_days=100)
        from signals.statistical import compute_statistical_features

        result = compute_statistical_features(prices["AAPL"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestNetworkFeatures:
    def test_compute_network_features(self):
        prices = make_prices(["AAPL", "MSFT", "JPM"], n_days=200)
        from signals.network_analysis import compute_network_features

        result = compute_network_features(prices, window=60)
        assert isinstance(result, pd.DataFrame)
        assert "net_degree" in result.columns
        assert "net_avg_corr" in result.columns
        assert "net_lead_lag_score" in result.columns
        # Should have one row per ticker
        assert len(result) == 3


class TestSignalCombiner:
    def test_build_feature_matrix(self):
        prices = make_prices(n_days=200)
        fundamentals = make_fundamentals()
        macro = make_macro(n_days=200)

        from signals.combiner import SignalCombiner

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)

        assert isinstance(fm, pd.DataFrame)
        assert isinstance(fm.index, pd.MultiIndex)
        assert fm.index.names == ["date", "ticker"]
        assert len(fm.columns) > 30
        assert len(fm) > 100

    def test_normalize_features(self):
        prices = make_prices(n_days=100)
        fundamentals = make_fundamentals()
        macro = make_macro(n_days=100)

        from signals.combiner import SignalCombiner

        combiner = SignalCombiner()
        fm = combiner.build_feature_matrix(prices, fundamentals, macro)
        normalized = combiner.normalize_features(fm, method="zscore")
        assert len(normalized) == len(fm)
