"""
Tests for YFinanceLoader data fetching module.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.yfinance_loader import (
    DataFetchError,
    RateLimiter,
    YFinanceLoader,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_allows_calls_under_limit(self):
        """Test that calls under the limit are allowed immediately."""
        limiter = RateLimiter(max_calls=10, period=60)

        # Should not sleep for first few calls
        for _ in range(5):
            with limiter:
                pass

        assert len(limiter.calls) == 5

    def test_rate_limiter_tracks_calls(self):
        """Test that rate limiter tracks call timestamps."""
        limiter = RateLimiter(max_calls=100, period=60)

        with limiter:
            pass

        assert len(limiter.calls) == 1


class TestYFinanceLoader:
    """Tests for YFinanceLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        return YFinanceLoader(tickers=["SPY"], lookback_years=1)

    def test_initialization(self, loader):
        """Test loader initialization."""
        assert "SPY" in loader.tickers
        assert loader.lookback_years == 1
        assert loader.rate_limiter is not None

    def test_calculate_date_range(self, loader):
        """Test date range calculation."""
        start, end = loader._calculate_date_range()

        assert isinstance(start, date)
        assert isinstance(end, date)
        assert start < end
        assert (end - start).days >= 365  # At least 1 year

    @patch("yfinance.Ticker")
    def test_fetch_ticker_data_success(self, mock_ticker, loader):
        """Test successful data fetch."""
        # Create mock data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        mock_df = pd.DataFrame({
            "Open": [100] * 10,
            "High": [105] * 10,
            "Low": [95] * 10,
            "Close": [102] * 10,
            "Volume": [1000000] * 10,
        }, index=dates)

        mock_ticker.return_value.history.return_value = mock_df

        result = loader.fetch_ticker_data("SPY")

        assert len(result) == 10
        assert "close" in result.columns
        assert "returns" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "SPY"

    @patch("yfinance.Ticker")
    def test_fetch_ticker_data_empty_raises_error(self, mock_ticker, loader):
        """Test that empty data raises DataFetchError."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        with pytest.raises(DataFetchError):
            loader.fetch_ticker_data("INVALID")

    @patch("yfinance.Ticker")
    def test_fetch_vix_data_success(self, mock_ticker, loader):
        """Test successful VIX data fetch."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        mock_df = pd.DataFrame({
            "Close": [20] * 10,
        }, index=dates)

        mock_ticker.return_value.history.return_value = mock_df

        result = loader.fetch_vix_data()

        assert len(result) == 10
        assert "vix" in result.columns

    def test_merge_with_vix(self, loader):
        """Test merging ticker data with VIX."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        ticker_df = pd.DataFrame({
            "close": [100 + i for i in range(10)],
            "volume": [1000000] * 10,
        }, index=dates)

        vix_df = pd.DataFrame({
            "vix": [20 + i * 0.5 for i in range(10)],
        }, index=dates)

        result = loader.merge_with_vix(ticker_df, vix_df)

        assert "vix" in result.columns
        assert "vix_rank" in result.columns
        assert len(result) == 10


class TestIntegration:
    """Integration tests (require network)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_real_data(self):
        """Test fetching real data from yfinance (requires network)."""
        loader = YFinanceLoader(tickers=["SPY"], lookback_years=1)

        # This will make real API calls
        df = loader.fetch_ticker_data("SPY")

        assert len(df) > 200  # Should have ~252 trading days
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["close"].min() > 0
