"""
Tests for technical indicators module.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.technical_indicators import TechnicalIndicators, calculate_features


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 300  # Need enough data for 252-day calculations

        # Generate random walk for close price
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))

        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n)) * 0.01),
            "low": close * (1 - np.abs(np.random.randn(n)) * 0.01),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n),
        }, index=dates)

        return df

    @pytest.fixture
    def indicators(self):
        """Create indicators instance."""
        return TechnicalIndicators()

    def test_add_returns(self, indicators, sample_data):
        """Test return calculations."""
        result = indicators.add_returns(sample_data)

        assert "returns_1d" in result.columns
        assert "returns_5d" in result.columns
        assert "returns_20d" in result.columns
        assert "log_returns_1d" in result.columns

        # Check that returns are reasonable
        assert result["returns_1d"].dropna().abs().max() < 0.5  # No 50%+ daily moves

    def test_add_rsi(self, indicators, sample_data):
        """Test RSI calculation."""
        result = indicators.add_rsi(sample_data)

        assert "rsi_14" in result.columns
        assert "rsi_28" in result.columns

        # RSI should be between 0 and 100
        rsi_14 = result["rsi_14"].dropna()
        assert rsi_14.min() >= 0
        assert rsi_14.max() <= 100

    def test_add_macd(self, indicators, sample_data):
        """Test MACD calculation."""
        result = indicators.add_macd(sample_data)

        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert "macd_pct" in result.columns

        # Histogram should equal macd - signal
        hist_calc = result["macd"] - result["macd_signal"]
        np.testing.assert_array_almost_equal(
            result["macd_histogram"].dropna(),
            hist_calc.dropna()
        )

    def test_add_bollinger_bands(self, indicators, sample_data):
        """Test Bollinger Bands calculation."""
        result = indicators.add_bollinger_bands(sample_data)

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        assert "bb_position" in result.columns

        # Upper should be above lower
        valid = result[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_add_moving_averages(self, indicators, sample_data):
        """Test moving average calculations."""
        result = indicators.add_moving_averages(sample_data)

        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "sma_200" in result.columns
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns
        assert "sma_cross_20_50" in result.columns

        # Cross signals should be binary
        assert set(result["sma_cross_20_50"].dropna().unique()).issubset({0, 1})

    def test_add_volatility(self, indicators, sample_data):
        """Test volatility calculations."""
        result = indicators.add_volatility(sample_data)

        assert "hv_5" in result.columns
        assert "hv_20" in result.columns
        assert "hv_60" in result.columns
        assert "hv_ratio_5_20" in result.columns
        assert "parkinson_hv" in result.columns

        # HV should be positive
        hv_20 = result["hv_20"].dropna()
        assert hv_20.min() >= 0

    def test_add_momentum(self, indicators, sample_data):
        """Test momentum indicators."""
        result = indicators.add_momentum(sample_data)

        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
        assert "roc_10" in result.columns
        assert "mfi" in result.columns

        # Stochastic should be between 0 and 100
        stoch_k = result["stoch_k"].dropna()
        assert stoch_k.min() >= 0
        assert stoch_k.max() <= 100

    def test_add_volume_indicators(self, indicators, sample_data):
        """Test volume indicators."""
        result = indicators.add_volume_indicators(sample_data)

        assert "volume_sma_20" in result.columns
        assert "volume_ratio" in result.columns
        assert "volume_surge" in result.columns
        assert "obv" in result.columns

        # Volume surge should be binary
        assert set(result["volume_surge"].dropna().unique()).issubset({0, 1})

    def test_add_price_levels(self, indicators, sample_data):
        """Test price level indicators."""
        result = indicators.add_price_levels(sample_data)

        assert "high_52w" in result.columns
        assert "low_52w" in result.columns
        assert "atr_14" in result.columns
        assert "atr_pct" in result.columns

        # ATR should be positive
        atr = result["atr_14"].dropna()
        assert atr.min() >= 0

    def test_calculate_all(self, indicators, sample_data):
        """Test calculating all indicators."""
        result = indicators.calculate_all(sample_data)

        # Should have many more columns
        assert len(result.columns) > len(sample_data.columns) + 50

        # Original columns should still exist
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_calculate_features_convenience_function(self, sample_data):
        """Test the convenience function."""
        result = calculate_features(sample_data)

        assert len(result.columns) > len(sample_data.columns) + 50

    def test_missing_column_raises_error(self, indicators):
        """Test that missing required columns raise errors."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required column"):
            indicators.calculate_all(df)
