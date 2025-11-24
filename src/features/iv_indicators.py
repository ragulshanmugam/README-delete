"""
Implied Volatility indicators using VIX as proxy.

For broad market ETFs (SPY, QQQ, IWM), VIX-based proxies provide
sufficient IV signal for regime classification and strategy selection.

Features:
- IV Rank: Percentile of current IV over 252-day lookback
- IV Percentile: % of days IV was lower than current
- IV-HV Spread: Implied vs Historical volatility gap
- VIX Term Structure: Contango/Backwardation signal
- IV Regime: Categorical classification (low/normal/elevated/high)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


# VIX proxy mapping for each ticker
VIX_PROXY_MAP = {
    "SPY": "^VIX",      # S&P 500 Volatility Index
    "QQQ": "^VXN",      # Nasdaq 100 Volatility Index
    "IWM": "^RVX",      # Russell 2000 Volatility Index
    "DIA": "^VXD",      # Dow Jones Volatility Index
    "XLV": "^VIX",      # Use VIX as fallback for sectors
    "XLK": "^VIX",      # Use VIX as fallback for sectors
    "XLF": "^VIX",      # Use VIX as fallback for sectors
}

# VIX term structure symbols
VIX_TERM_STRUCTURE = {
    "VIX": "^VIX",          # 30-day
    "VIX9D": "^VIX9D",      # 9-day
    "VIX3M": "^VIX3M",      # 3-month
    "VIX6M": "^VIX6M",      # 6-month
}


class IVIndicators:
    """
    Calculate Implied Volatility features using VIX as proxy.

    For SPY/QQQ/IWM, uses the corresponding volatility index.
    For other tickers, falls back to VIX.
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize IV indicator calculator.

        Args:
            lookback_days: Days for IV rank/percentile calculation (default: 252 = 1 year)
        """
        self.lookback_days = lookback_days
        self._vix_cache: Dict[str, pd.DataFrame] = {}
        logger.info(f"Initialized IVIndicators with lookback={lookback_days} days")

    def fetch_vix_data(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch VIX proxy data for a ticker.

        Args:
            ticker: Stock ticker (SPY, QQQ, IWM, etc.)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (optional)

        Returns:
            DataFrame with VIX proxy data
        """
        vix_symbol = VIX_PROXY_MAP.get(ticker, "^VIX")

        # Check cache
        cache_key = f"{vix_symbol}_{start_date}_{end_date}"
        if cache_key in self._vix_cache:
            return self._vix_cache[cache_key]

        try:
            logger.info(f"Fetching {vix_symbol} data for {ticker}")
            vix = yf.Ticker(vix_symbol)

            if end_date:
                data = vix.history(start=start_date, end=end_date)
            else:
                data = vix.history(start=start_date)

            if data.empty:
                logger.warning(f"No data returned for {vix_symbol}, falling back to ^VIX")
                vix = yf.Ticker("^VIX")
                data = vix.history(start=start_date, end=end_date) if end_date else vix.history(start=start_date)

            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]

            result = pd.DataFrame({
                "vix_close": data["close"],
                "vix_high": data["high"],
                "vix_low": data["low"],
            }, index=data.index)

            # Remove timezone info for compatibility
            if result.index.tz is not None:
                result.index = result.index.tz_localize(None)

            self._vix_cache[cache_key] = result
            logger.info(f"Fetched {len(result)} VIX data points")

            return result

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return pd.DataFrame()

    def fetch_term_structure(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch VIX term structure data.

        Returns DataFrame with VIX, VIX9D, VIX3M columns for term structure analysis.
        """
        term_data = {}

        for name, symbol in VIX_TERM_STRUCTURE.items():
            try:
                ticker = yf.Ticker(symbol)
                if end_date:
                    data = ticker.history(start=start_date, end=end_date)
                else:
                    data = ticker.history(start=start_date)

                if not data.empty:
                    series = data["Close"]
                    if series.index.tz is not None:
                        series.index = series.index.tz_localize(None)
                    term_data[name.lower()] = series

            except Exception as e:
                logger.warning(f"Could not fetch {symbol}: {e}")

        if not term_data:
            return pd.DataFrame()

        result = pd.DataFrame(term_data)
        logger.info(f"Fetched term structure data: {list(result.columns)}")

        return result

    def calculate_iv_rank(
        self,
        vix_series: pd.Series,
        lookback: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate IV Rank (percentile based on high/low range).

        IV Rank = (Current - 52wk Low) / (52wk High - 52wk Low) * 100

        Values:
        - 0-20: Low IV (options cheap, favor buying premium)
        - 20-50: Normal IV
        - 50-80: Elevated IV (favor selling premium)
        - 80-100: High IV (options expensive, strongly favor selling)

        Returns:
            Series with IV rank (0-100)
        """
        lookback = lookback or self.lookback_days

        rolling_min = vix_series.rolling(window=lookback, min_periods=20).min()
        rolling_max = vix_series.rolling(window=lookback, min_periods=20).max()

        # Avoid division by zero
        range_val = rolling_max - rolling_min
        range_val = range_val.replace(0, np.nan)

        iv_rank = ((vix_series - rolling_min) / range_val * 100).fillna(50)

        return iv_rank.clip(0, 100)

    def calculate_iv_percentile(
        self,
        vix_series: pd.Series,
        lookback: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate IV Percentile (% of days IV was lower).

        Different from IV Rank - considers full distribution, not just extremes.
        More robust to outliers.

        Returns:
            Series with IV percentile (0-100)
        """
        lookback = lookback or self.lookback_days

        def percentile_rank(window):
            if len(window) < 10:
                return 50.0
            current = window.iloc[-1]
            return (window.iloc[:-1] < current).sum() / (len(window) - 1) * 100

        return vix_series.rolling(window=lookback, min_periods=20).apply(
            percentile_rank, raw=False
        ).fillna(50)

    def calculate_iv_hv_spread(
        self,
        vix_series: pd.Series,
        hv_series: pd.Series,
    ) -> pd.Series:
        """
        Calculate IV - HV spread (implied vs realized volatility).

        Positive spread: IV > HV (options expensive relative to actual moves)
        Negative spread: IV < HV (options cheap relative to actual moves)

        Args:
            vix_series: VIX values (already in % terms)
            hv_series: Historical volatility (annualized, in decimal form like 0.20)

        Returns:
            Series with IV-HV spread in percentage points
        """
        # Convert HV to percentage if needed (0.20 -> 20)
        hv_pct = hv_series * 100 if hv_series.max() < 5 else hv_series

        return vix_series - hv_pct

    def calculate_term_structure_slope(
        self,
        term_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate VIX term structure slope.

        Contango (positive): VIX3M > VIX (normal market, calm)
        Backwardation (negative): VIX > VIX3M (stressed market, fear)

        Returns:
            Series with slope (VIX3M - VIX)
        """
        if "vix" not in term_df.columns or "vix3m" not in term_df.columns:
            logger.warning("Missing VIX or VIX3M for term structure calculation")
            return pd.Series(index=term_df.index, data=0.0)

        slope = term_df["vix3m"] - term_df["vix"]

        return slope

    def classify_iv_regime(
        self,
        iv_rank: pd.Series,
        iv_hv_spread: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Classify IV regime based on IV rank and IV-HV spread.

        Regimes:
        - low: IV rank < 25 (buy premium)
        - normal: IV rank 25-50
        - elevated: IV rank 50-75
        - high: IV rank > 75 (sell premium)

        Returns:
            Series with regime labels
        """
        conditions = [
            iv_rank < 25,
            (iv_rank >= 25) & (iv_rank < 50),
            (iv_rank >= 50) & (iv_rank < 75),
            iv_rank >= 75,
        ]
        choices = ["low", "normal", "elevated", "high"]

        regime = pd.Series(
            np.select(conditions, choices, default="normal"),
            index=iv_rank.index
        )

        return regime

    def calculate_all(
        self,
        df: pd.DataFrame,
        ticker: str,
        hv_column: str = "hv_20",
    ) -> pd.DataFrame:
        """
        Calculate all IV indicators and add to DataFrame.

        Args:
            df: DataFrame with price data (must have DatetimeIndex)
            ticker: Stock ticker for VIX proxy selection
            hv_column: Column name for historical volatility

        Returns:
            DataFrame with IV features added
        """
        result = df.copy()

        # Ensure index is datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, skipping IV features")
            return result

        # Get date range
        start_date = (result.index.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = (result.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Fetch VIX data
        vix_data = self.fetch_vix_data(ticker, start_date, end_date)

        if vix_data.empty:
            logger.warning(f"No VIX data available for {ticker}, skipping IV features")
            return result

        # Merge VIX data
        result = result.join(vix_data, how="left")
        result["vix_close"] = result["vix_close"].ffill()

        # Calculate IV features
        vix = result["vix_close"]

        # IV Rank and Percentile
        result["iv_rank"] = self.calculate_iv_rank(vix)
        result["iv_percentile"] = self.calculate_iv_percentile(vix)

        # IV-HV Spread
        if hv_column in result.columns:
            result["iv_hv_spread"] = self.calculate_iv_hv_spread(vix, result[hv_column])
            result["iv_premium"] = (result["iv_hv_spread"] > 0).astype(int)

        # VIX momentum features
        result["vix_sma_10"] = vix.rolling(10).mean()
        result["vix_sma_20"] = vix.rolling(20).mean()
        result["vix_change_1d"] = vix.pct_change(1) * 100
        result["vix_change_5d"] = vix.pct_change(5) * 100
        result["vix_zscore"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
        result["vix_trend"] = result["vix_sma_10"] - result["vix_sma_20"]

        # IV Regime
        result["iv_regime"] = self.classify_iv_regime(result["iv_rank"])

        # Regime binary flags for ML
        result["iv_regime_low"] = (result["iv_regime"] == "low").astype(int)
        result["iv_regime_high"] = (result["iv_regime"] == "high").astype(int)

        # Fetch term structure
        term_data = self.fetch_term_structure(start_date, end_date)
        if not term_data.empty:
            term_data = term_data.reindex(result.index, method="ffill")
            result["vix_term_slope"] = self.calculate_term_structure_slope(term_data)
            result["vix_contango"] = (result["vix_term_slope"] > 0).astype(int)

        # Count features added
        new_features = [c for c in result.columns if c not in df.columns]
        logger.info(f"Added {len(new_features)} IV indicator features")

        return result


def calculate_iv_features(
    df: pd.DataFrame,
    ticker: str,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Convenience function to calculate IV features.

    Args:
        df: DataFrame with price data
        ticker: Stock ticker
        lookback_days: Days for IV rank calculation

    Returns:
        DataFrame with IV features added
    """
    calculator = IVIndicators(lookback_days=lookback_days)
    return calculator.calculate_all(df, ticker)


if __name__ == "__main__":
    # Quick test
    import yfinance as yf

    # Fetch sample data
    spy = yf.Ticker("SPY")
    df = spy.history(period="2y")
    df.columns = [col.lower() for col in df.columns]

    # Remove timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Add HV for IV-HV spread calculation
    log_returns = np.log(df["close"] / df["close"].shift(1))
    df["hv_20"] = log_returns.rolling(20).std() * np.sqrt(252)

    # Calculate IV features
    result = calculate_iv_features(df, "SPY")

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {result.shape}")
    print(f"\nIV Features added:")
    iv_cols = [c for c in result.columns if c not in df.columns]
    for col in iv_cols:
        print(f"  - {col}")

    print(f"\nLatest IV values:")
    print(f"  VIX: {result['vix_close'].iloc[-1]:.2f}")
    print(f"  IV Rank: {result['iv_rank'].iloc[-1]:.1f}")
    print(f"  IV Percentile: {result['iv_percentile'].iloc[-1]:.1f}")
    print(f"  IV Regime: {result['iv_regime'].iloc[-1]}")
    if "iv_hv_spread" in result.columns:
        print(f"  IV-HV Spread: {result['iv_hv_spread'].iloc[-1]:.2f}")
