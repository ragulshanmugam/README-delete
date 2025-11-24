"""
Technical indicators for feature engineering.

Calculates RSI, MACD, Bollinger Bands, Moving Averages, and other
technical indicators used for ML model training.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for trading features.

    Provides methods to calculate various technical indicators
    commonly used in quantitative trading.

    Attributes:
        windows: List of window sizes for rolling calculations
    """

    def __init__(self, windows: Optional[List[int]] = None):
        """
        Initialize technical indicators calculator.

        Args:
            windows: List of window sizes (default from settings)
        """
        settings = get_settings()
        self.windows = windows or settings.feature_windows
        logger.info(f"Initialized TechnicalIndicators with windows={self.windows}")

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with all technical indicators added
        """
        result = df.copy()

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in result.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate indicators
        logger.info("Calculating technical indicators...")

        result = self.add_returns(result)
        result = self.add_rsi(result)
        result = self.add_macd(result)
        result = self.add_bollinger_bands(result)
        result = self.add_moving_averages(result)
        result = self.add_volatility(result)
        result = self.add_momentum(result)
        result = self.add_volume_indicators(result)
        result = self.add_adx(result)
        result = self.add_price_levels(result)

        # Count features added
        new_features = len(result.columns) - len(df.columns)
        logger.info(f"Added {new_features} technical indicator features")

        return result

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return calculations.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with return features
        """
        result = df.copy()
        close = result["close"]

        # Simple returns for multiple windows
        for window in [1, 5, 10, 20, 60]:
            result[f"returns_{window}d"] = close.pct_change(window)

        # Log returns
        result["log_returns_1d"] = np.log(close / close.shift(1))

        # Cumulative returns (YTD approximation using 252 trading days)
        result["returns_252d"] = close.pct_change(252)

        return result

    def add_rsi(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = average gain / average loss

        Args:
            df: DataFrame with close prices
            periods: RSI periods (default: [14, 28])

        Returns:
            DataFrame with RSI features
        """
        result = df.copy()
        periods = periods or [14, 28]

        for period in periods:
            delta = result["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return result

    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD).

        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal

        Args:
            df: DataFrame with close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD features
        """
        result = df.copy()
        close = result["close"]

        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # MACD line
        result["macd"] = ema_fast - ema_slow

        # Signal line
        result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()

        # Histogram
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        # Normalized MACD (as percentage of price)
        result["macd_pct"] = result["macd"] / close * 100

        return result

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Middle = SMA(window)
        Upper = Middle + num_std * std(window)
        Lower = Middle - num_std * std(window)

        Args:
            df: DataFrame with close prices
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band features
        """
        result = df.copy()
        close = result["close"]

        # Calculate bands
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()

        result["bb_middle"] = sma
        result["bb_upper"] = sma + (num_std * std)
        result["bb_lower"] = sma - (num_std * std)

        # Band width (volatility measure)
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]

        # Position within bands (0 = lower, 1 = upper)
        result["bb_position"] = (close - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        )

        # Distance from middle band (normalized)
        result["bb_distance"] = (close - result["bb_middle"]) / std

        return result

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add various moving averages.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with moving average features
        """
        result = df.copy()
        close = result["close"]

        # Simple Moving Averages
        for window in [20, 50, 200]:
            result[f"sma_{window}"] = close.rolling(window=window).mean()
            result[f"price_vs_sma{window}"] = (close - result[f"sma_{window}"]) / result[f"sma_{window}"]

        # Exponential Moving Averages
        for span in [12, 26]:
            result[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

        # Moving average crossovers (binary)
        result["sma_cross_20_50"] = (result["sma_20"] > result["sma_50"]).astype(int)
        result["sma_cross_50_200"] = (result["sma_50"] > result["sma_200"]).astype(int)

        # Price vs MA signals
        result["price_above_sma20"] = (close > result["sma_20"]).astype(int)
        result["price_above_sma50"] = (close > result["sma_50"]).astype(int)
        result["price_above_sma200"] = (close > result["sma_200"]).astype(int)

        return result

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical volatility features.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with volatility features
        """
        result = df.copy()

        # Log returns for volatility calculation
        log_returns = np.log(result["close"] / result["close"].shift(1))

        # Historical volatility (annualized) for multiple windows
        for window in [5, 10, 20, 60, 120]:
            result[f"hv_{window}"] = log_returns.rolling(window=window).std() * np.sqrt(252)

        # Volatility ratios
        result["hv_ratio_5_20"] = result["hv_5"] / result["hv_20"]
        result["hv_ratio_20_60"] = result["hv_20"] / result["hv_60"]

        # Parkinson volatility (high-low based)
        hl_ratio = np.log(result["high"] / result["low"])
        result["parkinson_hv"] = hl_ratio.rolling(window=20).apply(
            lambda x: np.sqrt((1 / (4 * np.log(2))) * (x ** 2).sum() / len(x)) * np.sqrt(252)
        )

        # Volatility trend
        result["hv_trend"] = result["hv_20"] - result["hv_60"]

        return result

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with momentum features
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # Stochastic oscillator
        window = 14
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()

        result["stoch_k"] = (
            (close - lowest_low) / (highest_high - lowest_low) * 100
        )
        result["stoch_d"] = result["stoch_k"].rolling(window=3).mean()

        # Rate of change
        for period in [5, 10, 20]:
            result[f"roc_{period}"] = (close - close.shift(period)) / close.shift(period) * 100

        # Money Flow Index (simplified - volume-weighted RSI)
        typical_price = (high + low + close) / 3
        raw_mf = typical_price * result["volume"]
        mf_direction = np.where(typical_price > typical_price.shift(1), raw_mf, -raw_mf)
        mf_series = pd.Series(mf_direction, index=result.index)

        positive_mf = mf_series.where(mf_series > 0, 0).rolling(14).sum()
        negative_mf = (-mf_series.where(mf_series < 0, 0)).rolling(14).sum()

        mf_ratio = positive_mf / negative_mf
        result["mfi"] = 100 - (100 / (1 + mf_ratio))

        return result

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with volume features
        """
        result = df.copy()
        volume = result["volume"]

        # Volume moving averages
        for window in [10, 20]:
            result[f"volume_sma_{window}"] = volume.rolling(window=window).mean()

        # Volume ratio (current vs average)
        result["volume_ratio"] = volume / result["volume_sma_20"]

        # Volume surge flag
        result["volume_surge"] = (result["volume_ratio"] > 2.0).astype(int)

        # Log volume (for normalization)
        result["log_volume"] = np.log(volume + 1)

        # On-Balance Volume (OBV)
        obv_direction = np.where(
            result["close"] > result["close"].shift(1), volume,
            np.where(result["close"] < result["close"].shift(1), -volume, 0)
        )
        result["obv"] = pd.Series(obv_direction, index=result.index).cumsum()
        result["obv_sma"] = result["obv"].rolling(window=20).mean()

        return result

    def add_adx(
        self,
        df: pd.DataFrame,
        window: int = 14,
    ) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) for trend strength.

        ADX > 25: Strong trend (good for momentum strategies)
        ADX < 20: Weak trend / ranging market (good for mean reversion)

        Also calculates +DI and -DI for trend direction.

        Args:
            df: DataFrame with OHLCV data
            window: Lookback period (default 14)

        Returns:
            DataFrame with ADX features
        """
        result = df.copy()
        high = result["high"]
        low = result["low"]
        close = result["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        # +DM and -DM
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=result.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=result.index
        )

        # Smoothed Directional Indicators
        plus_di = 100 * plus_dm.rolling(window=window).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(window=window).mean() / (atr + 1e-10)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=window).mean()

        # Add features
        result[f"adx_{window}"] = adx
        result[f"plus_di_{window}"] = plus_di
        result[f"minus_di_{window}"] = minus_di

        # Trend strength categories
        result["adx_strong_trend"] = (adx > 25).astype(int)
        result["adx_weak_trend"] = (adx < 20).astype(int)

        # DI difference (positive = bullish, negative = bearish)
        result[f"di_diff_{window}"] = plus_di - minus_di

        # Trend direction signal
        result["trend_bullish"] = ((plus_di > minus_di) & (adx > 20)).astype(int)
        result["trend_bearish"] = ((minus_di > plus_di) & (adx > 20)).astype(int)

        logger.debug(f"Added ADX features with window={window}")

        return result

    def add_price_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price level indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price level features
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # 52-week high/low
        result["high_52w"] = high.rolling(window=252).max()
        result["low_52w"] = low.rolling(window=252).min()

        # Distance from 52-week levels
        result["dist_from_high_52w"] = (result["high_52w"] - close) / result["high_52w"]
        result["dist_from_low_52w"] = (close - result["low_52w"]) / result["low_52w"]

        # Near 52-week high/low flags
        result["near_52w_high"] = (result["dist_from_high_52w"] < 0.05).astype(int)
        result["near_52w_low"] = (result["dist_from_low_52w"] < 0.1).astype(int)

        # Average True Range (ATR)
        tr = pd.DataFrame({
            "hl": high - low,
            "hpc": abs(high - close.shift(1)),
            "lpc": abs(low - close.shift(1))
        }).max(axis=1)
        result["atr_14"] = tr.rolling(window=14).mean()
        result["atr_pct"] = result["atr_14"] / close * 100

        return result


def calculate_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators.

    Args:
        df: DataFrame with OHLCV data
        windows: Optional window sizes

    Returns:
        DataFrame with all features
    """
    calculator = TechnicalIndicators(windows=windows)
    return calculator.calculate_all(df)


if __name__ == "__main__":
    # Quick test with sample data
    import yfinance as yf

    # Fetch sample data
    spy = yf.Ticker("SPY")
    df = spy.history(period="1y")

    # Rename columns to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Calculate indicators
    result = calculate_features(df)

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {result.shape}")
    print(f"New features: {len(result.columns) - len(df.columns)}")
    print(f"\nFeature columns:\n{list(result.columns)}")
