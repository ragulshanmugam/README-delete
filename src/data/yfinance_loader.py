"""
Data fetching module using yfinance.

Fetches historical OHLCV data and VIX data for options trading analysis.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class RateLimiter:
    """
    Simple rate limiter for API calls.

    Tracks call timestamps and sleeps if rate limit is exceeded.
    """

    def __init__(self, max_calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls per period
            period: Period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []

    def __enter__(self):
        """Check rate limit and wait if necessary."""
        now = time.time()

        # Remove calls outside the window
        self.calls = [t for t in self.calls if now - t < self.period]

        # Wait if limit exceeded
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.calls = []

        self.calls.append(time.time())
        return self

    def __exit__(self, *args):
        """Exit context."""
        pass


class DataFetchError(Exception):
    """Raised when data fetching fails."""

    pass


class YFinanceLoader:
    """
    Data loader using yfinance.

    Fetches historical OHLCV and VIX data with rate limiting
    and error handling.

    Attributes:
        tickers: List of ticker symbols to fetch
        lookback_years: Number of years of historical data
        rate_limiter: Rate limiter instance
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        lookback_years: Optional[int] = None,
    ):
        """
        Initialize the loader.

        Args:
            tickers: List of ticker symbols (default from settings)
            lookback_years: Years of historical data (default from settings)
        """
        settings = get_settings()
        self.tickers = tickers or settings.tickers
        self.lookback_years = lookback_years or settings.lookback_years
        self.vix_symbol = settings.vix_symbol
        self.rate_limiter = RateLimiter(
            max_calls=settings.rate_limit_calls,
            period=settings.rate_limit_period
        )
        logger.info(
            f"Initialized YFinanceLoader with tickers={self.tickers}, "
            f"lookback={self.lookback_years} years"
        )

    def _calculate_date_range(self) -> Tuple[date, date]:
        """
        Calculate start and end dates for data fetching.

        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_years * 365)
        return start_date, end_date

    @log_execution_time(logger)
    def fetch_ticker_data(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Ticker symbol (e.g., 'SPY')
            start: Start date (default: lookback_years ago)
            end: End date (default: today)

        Returns:
            DataFrame with OHLCV data, indexed by date

        Raises:
            DataFetchError: If data fetching fails
        """
        if start is None or end is None:
            start, end = self._calculate_date_range()

        logger.info(f"Fetching {ticker} data from {start} to {end}")

        try:
            with self.rate_limiter:
                yf_ticker = yf.Ticker(ticker)
                df = yf_ticker.history(
                    start=start,
                    end=end,
                    auto_adjust=True,  # Adjust for splits/dividends
                )

            if df.empty:
                raise DataFetchError(f"No data returned for {ticker}")

            # Clean up index
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"

            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Add ticker column
            df["ticker"] = ticker

            # Calculate returns
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            raise DataFetchError(f"Failed to fetch {ticker}: {e}") from e

    @log_execution_time(logger)
    def fetch_vix_data(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch VIX index data.

        Args:
            start: Start date (default: lookback_years ago)
            end: End date (default: today)

        Returns:
            DataFrame with VIX data, indexed by date

        Raises:
            DataFetchError: If data fetching fails
        """
        if start is None or end is None:
            start, end = self._calculate_date_range()

        logger.info(f"Fetching VIX data from {start} to {end}")

        try:
            with self.rate_limiter:
                vix = yf.Ticker(self.vix_symbol)
                df = vix.history(start=start, end=end)

            if df.empty:
                raise DataFetchError("No VIX data returned")

            # Clean up
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"

            # Keep only close price and rename
            df = df[["Close"]].rename(columns={"Close": "vix"})

            logger.info(f"Fetched {len(df)} VIX data points")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch VIX: {e}")
            raise DataFetchError(f"Failed to fetch VIX: {e}") from e

    @log_execution_time(logger)
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all configured tickers and VIX.

        Returns:
            Dictionary mapping ticker symbols to DataFrames

        Raises:
            DataFetchError: If any fetch fails
        """
        data = {}
        start, end = self._calculate_date_range()

        # Fetch ticker data
        for ticker in self.tickers:
            try:
                data[ticker] = self.fetch_ticker_data(ticker, start, end)
            except DataFetchError as e:
                logger.error(f"Skipping {ticker}: {e}")
                continue

        # Fetch VIX
        try:
            data["VIX"] = self.fetch_vix_data(start, end)
        except DataFetchError as e:
            logger.error(f"Failed to fetch VIX: {e}")

        if not data:
            raise DataFetchError("No data fetched for any ticker")

        return data

    @log_execution_time(logger)
    def merge_with_vix(
        self,
        ticker_df: pd.DataFrame,
        vix_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge ticker data with VIX data.

        Args:
            ticker_df: DataFrame with ticker OHLCV data
            vix_df: DataFrame with VIX data

        Returns:
            Merged DataFrame with VIX column added
        """
        # Ensure both have date index
        ticker_df = ticker_df.copy()

        # Merge on date index
        merged = ticker_df.join(vix_df, how="left")

        # Forward fill VIX for weekends/holidays
        merged["vix"] = merged["vix"].ffill()

        # Calculate VIX rank (252-day percentile)
        merged["vix_rank"] = merged["vix"].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
            if x.max() > x.min() else 0.5,
            raw=False
        )

        logger.info(f"Merged data: {len(merged)} rows")
        return merged

    def create_combined_dataset(self) -> pd.DataFrame:
        """
        Create a combined dataset with all tickers and VIX.

        Returns:
            Combined DataFrame with all ticker data and VIX

        Raises:
            DataFetchError: If data fetching fails
        """
        all_data = self.fetch_all_data()

        if "VIX" not in all_data:
            raise DataFetchError("VIX data required for combined dataset")

        vix_df = all_data.pop("VIX")
        combined = []

        for ticker, ticker_df in all_data.items():
            merged = self.merge_with_vix(ticker_df, vix_df)
            combined.append(merged)

        if not combined:
            raise DataFetchError("No ticker data to combine")

        result = pd.concat(combined, ignore_index=False)
        result = result.sort_index()

        logger.info(f"Created combined dataset: {len(result)} rows")
        return result


def fetch_and_save_data(
    output_dir: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Convenience function to fetch and save all data.

    Args:
        output_dir: Directory to save data (default from settings)
        tickers: List of tickers (default from settings)

    Returns:
        Dictionary mapping ticker to saved file path
    """
    settings = get_settings()
    output_dir = output_dir or settings.get_data_path("raw")

    loader = YFinanceLoader(tickers=tickers)
    data = loader.fetch_all_data()

    saved_files = {}
    for ticker, df in data.items():
        file_path = output_dir / f"{ticker.replace('^', '')}_ohlcv.parquet"
        df.to_parquet(file_path)
        saved_files[ticker] = file_path
        logger.info(f"Saved {ticker} data to {file_path}")

    return saved_files


if __name__ == "__main__":
    # Quick test
    loader = YFinanceLoader(tickers=["SPY"])
    data = loader.fetch_all_data()

    for ticker, df in data.items():
        print(f"\n{ticker}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
