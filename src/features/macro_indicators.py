"""
Macro/Fed Feature Engineering for Options Trading.

Free data from FRED (Federal Reserve Economic Data).
Get API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Key insight: For 3-45 DTE options, EVENT TIMING matters more than
the actual macro numbers. Markets price in expectations - the
surprise and volatility around events is what moves options.
"""

import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy import fredapi to avoid import errors if not installed
_fred = None


def _get_fred():
    """Lazy load FRED client."""
    global _fred
    if _fred is None:
        try:
            from fredapi import Fred
            # Get API key from environment directly to avoid settings parsing issues
            api_key = os.environ.get("FRED_API_KEY")
            if not api_key:
                # Try settings as fallback
                try:
                    settings = get_settings()
                    api_key = settings.fred_api_key
                except Exception:
                    pass
            if not api_key:
                raise ValueError(
                    "FRED API key required. Set FRED_API_KEY env var or get free key at: "
                    "https://fred.stlouisfed.org/docs/api/api_key.html"
                )
            _fred = Fred(api_key=api_key)
            logger.info("Initialized FRED client")
        except ImportError:
            raise ImportError("fredapi not installed. Run: pip install fredapi")
    return _fred


class MacroIndicators:
    """
    Fetches and engineers Fed/macro features for options trading.

    Features are grouped into:
    1. Event Timing - Days to FOMC, NFP, CPI
    2. Rate Environment - Yields, curve shape
    3. Risk Sentiment - Credit spreads, inflation expectations
    4. Policy Regime - Easing/holding/tightening
    """

    # FRED Series IDs for key macro data
    SERIES = {
        # Treasury Yields (Daily)
        "yield_10y": "DGS10",
        "yield_2y": "DGS2",
        "yield_3m": "DGS3MO",
        "yield_spread_10y_2y": "T10Y2Y",

        # Fed Funds (Daily)
        "fed_funds_effective": "DFF",

        # Inflation Expectations (Daily)
        "breakeven_inflation_10y": "T10YIE",
        "breakeven_inflation_5y": "T5YIE",

        # Risk Sentiment (Daily)
        "high_yield_spread": "BAMLH0A0HYM2",

        # Economic Indicators (Monthly - will be forward-filled)
        "cpi_yoy": "CPIAUCSL",
        "unemployment_rate": "UNRATE",
    }

    # FOMC Meeting Dates (End dates of 2-day meetings)
    # Update annually from: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    FOMC_DATES = {
        2024: [
            date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
            date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
            date(2024, 11, 7), date(2024, 12, 18),
        ],
        2025: [
            date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
            date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
            date(2025, 11, 5), date(2025, 12, 17),
        ],
        2026: [
            date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
            date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
            date(2026, 11, 4), date(2026, 12, 16),
        ],
    }

    def __init__(self):
        """Initialize macro indicators calculator."""
        self._cache: Dict[str, pd.Series] = {}
        self._fred = None
        logger.info("Initialized MacroIndicators")

    @property
    def fred(self):
        """Lazy-load FRED client."""
        if self._fred is None:
            self._fred = _get_fred()
        return self._fred

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "2019-01-01"
    ) -> pd.Series:
        """
        Fetch a single FRED series with caching.

        Args:
            series_id: FRED series ID (e.g., 'DGS10')
            start_date: Start date for data fetch

        Returns:
            pd.Series with the data
        """
        cache_key = f"{series_id}_{start_date}"

        if cache_key not in self._cache:
            logger.debug(f"Fetching FRED series: {series_id}")
            self._cache[cache_key] = self.fred.get_series(
                series_id,
                observation_start=start_date
            )
        return self._cache[cache_key]

    def fetch_all_series(self, start_date: str = "2019-01-01") -> pd.DataFrame:
        """
        Fetch all macro series into a single DataFrame.

        Returns daily DataFrame with forward-filled values for
        monthly/quarterly series.
        """
        dfs = {}

        for name, series_id in self.SERIES.items():
            try:
                series = self.fetch_series(series_id, start_date)
                dfs[name] = series
                logger.debug(f"Fetched {name}: {len(series)} observations")
            except Exception as e:
                logger.warning(f"Could not fetch {name} ({series_id}): {e}")

        if not dfs:
            raise ValueError("No FRED data could be fetched")

        # Combine into single DataFrame
        df = pd.DataFrame(dfs)

        # Resample to daily and forward-fill gaps
        df = df.resample("D").last().ffill()

        # Clean up index
        df.index.name = "date"

        logger.info(f"Fetched {len(dfs)} FRED series, {len(df)} days")
        return df

    def _get_all_fomc_dates(self) -> List[date]:
        """Get all FOMC dates from the calendar."""
        all_dates = []
        for year_dates in self.FOMC_DATES.values():
            all_dates.extend(year_dates)
        return sorted(all_dates)

    def _calc_fomc_features(self, current_date: date) -> Dict:
        """Calculate FOMC timing features for a single date."""
        if isinstance(current_date, (pd.Timestamp, datetime)):
            current_date = current_date.date() if hasattr(current_date, 'date') else current_date

        fomc_dates = self._get_all_fomc_dates()

        # Find next and previous FOMC
        future_fomc = [d for d in fomc_dates if d > current_date]
        past_fomc = [d for d in fomc_dates if d <= current_date]

        days_to_fomc = (future_fomc[0] - current_date).days if future_fomc else 60
        days_since_fomc = (current_date - past_fomc[-1]).days if past_fomc else 60

        return {
            "days_to_fomc": min(days_to_fomc, 60),  # Cap at 60
            "days_since_fomc": min(days_since_fomc, 60),
            "is_fomc_week": int(days_to_fomc <= 5),
            "is_fomc_day": int(days_to_fomc == 0),
        }

    def _calc_nfp_features(self, current_date: date) -> Dict:
        """
        Calculate NFP (Non-Farm Payrolls) timing features.
        NFP is released first Friday of each month at 8:30 AM ET.
        """
        if isinstance(current_date, (pd.Timestamp, datetime)):
            current_date = current_date.date() if hasattr(current_date, 'date') else current_date

        # Find first Friday of current month
        first_of_month = current_date.replace(day=1)
        days_until_friday = (4 - first_of_month.weekday()) % 7
        nfp_this_month = first_of_month + timedelta(days=days_until_friday)

        # If NFP this month has passed, find next month's
        if current_date >= nfp_this_month:
            next_month = (first_of_month + timedelta(days=32)).replace(day=1)
            days_until_friday = (4 - next_month.weekday()) % 7
            next_nfp = next_month + timedelta(days=days_until_friday)
        else:
            next_nfp = nfp_this_month

        days_to_nfp = (next_nfp - current_date).days

        return {
            "days_to_nfp": min(days_to_nfp, 35),  # Cap at ~1 month
            "is_nfp_week": int(days_to_nfp <= 5),
        }

    def _calc_cpi_features(self, current_date: date) -> Dict:
        """
        Calculate CPI timing features.
        CPI is typically released around 10th-15th of each month.
        """
        if isinstance(current_date, (pd.Timestamp, datetime)):
            current_date = current_date.date() if hasattr(current_date, 'date') else current_date

        # Approximate CPI as 12th of each month
        cpi_this_month = current_date.replace(day=12)

        if current_date >= cpi_this_month:
            next_month = (current_date.replace(day=1) + timedelta(days=32))
            next_cpi = next_month.replace(day=12)
        else:
            next_cpi = cpi_this_month

        days_to_cpi = (next_cpi - current_date).days

        return {
            "days_to_cpi": min(days_to_cpi, 35),
            "is_cpi_week": int(days_to_cpi <= 5),
        }

    def add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add event timing features - CRITICAL for options.

        These features capture:
        1. Days until next FOMC/NFP/CPI
        2. Days since last event
        3. Whether we're in an "event week"
        4. Combined event density
        """
        result = df.copy()

        # Get dates from index
        if isinstance(result.index, pd.DatetimeIndex):
            dates = result.index.date
        else:
            dates = pd.to_datetime(result.index).date

        # Calculate event features for each date
        fomc_features = [self._calc_fomc_features(d) for d in dates]
        nfp_features = [self._calc_nfp_features(d) for d in dates]
        cpi_features = [self._calc_cpi_features(d) for d in dates]

        # Add FOMC features
        result["days_to_fomc"] = [f["days_to_fomc"] for f in fomc_features]
        result["days_since_fomc"] = [f["days_since_fomc"] for f in fomc_features]
        result["is_fomc_week"] = [f["is_fomc_week"] for f in fomc_features]

        # Add NFP features
        result["days_to_nfp"] = [f["days_to_nfp"] for f in nfp_features]
        result["is_nfp_week"] = [f["is_nfp_week"] for f in nfp_features]

        # Add CPI features
        result["days_to_cpi"] = [f["days_to_cpi"] for f in cpi_features]
        result["is_cpi_week"] = [f["is_cpi_week"] for f in cpi_features]

        # Combined event density
        result["major_events_next_7d"] = (
            (result["days_to_fomc"] <= 7).astype(int) +
            (result["days_to_nfp"] <= 7).astype(int) +
            (result["days_to_cpi"] <= 7).astype(int)
        )

        result["is_event_week"] = (
            result["is_fomc_week"] |
            result["is_nfp_week"] |
            result["is_cpi_week"]
        ).astype(int)

        logger.debug("Added event timing features")
        return result

    def add_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate environment features."""
        result = df.copy()

        # Yield curve status
        if "yield_spread_10y_2y" in result.columns:
            result["yield_curve_inverted"] = (
                result["yield_spread_10y_2y"] < 0
            ).astype(int)

        # Rate momentum (changes over time)
        for col in ["yield_10y", "yield_2y", "fed_funds_effective"]:
            if col in result.columns:
                result[f"{col}_change_5d"] = result[col].diff(5)
                result[f"{col}_change_20d"] = result[col].diff(20)

        # Yield curve dynamics
        if "yield_spread_10y_2y" in result.columns:
            result["yield_curve_steepening"] = (
                result["yield_spread_10y_2y"].diff(5) > 0
            ).astype(int)
            result["yield_spread_change_5d"] = result["yield_spread_10y_2y"].diff(5)

        # Real rates proxy (10Y - breakeven inflation)
        if "yield_10y" in result.columns and "breakeven_inflation_10y" in result.columns:
            result["real_rate_10y"] = (
                result["yield_10y"] - result["breakeven_inflation_10y"]
            )

        logger.debug("Added rate environment features")
        return result

    def add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk sentiment features."""
        result = df.copy()

        # High yield spread momentum (widening = risk-off)
        if "high_yield_spread" in result.columns:
            result["hy_spread_change_5d"] = result["high_yield_spread"].diff(5)
            result["hy_spread_change_20d"] = result["high_yield_spread"].diff(20)

            # Elevated spread (risk-off signal)
            hy_ma_60 = result["high_yield_spread"].rolling(60).mean()
            result["hy_spread_elevated"] = (
                result["high_yield_spread"] > hy_ma_60
            ).astype(int)

        # Inflation expectations momentum
        if "breakeven_inflation_10y" in result.columns:
            result["inflation_expectations_rising"] = (
                result["breakeven_inflation_10y"].diff(20) > 0
            ).astype(int)

        logger.debug("Added risk sentiment features")
        return result

    def add_policy_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify monetary policy regime.

        0 = Easing (rates falling)
        1 = Holding (rates stable)
        2 = Tightening (rates rising)
        """
        result = df.copy()

        if "fed_funds_effective" in result.columns:
            ff_change_60d = result["fed_funds_effective"].diff(60)

            result["monetary_policy_regime"] = np.select(
                [ff_change_60d < -0.1, ff_change_60d > 0.1],
                [0, 2],
                default=1
            )

        logger.debug("Added policy regime feature")
        return result

    def calculate_all(
        self,
        df: Optional[pd.DataFrame] = None,
        start_date: str = "2019-01-01",
    ) -> pd.DataFrame:
        """
        Calculate all macro features.

        Args:
            df: Optional DataFrame to merge with (uses index dates)
            start_date: Start date for FRED data if df not provided

        Returns:
            DataFrame with all macro features
        """
        logger.info("Calculating macro features...")

        # Fetch raw FRED data
        macro_df = self.fetch_all_series(start_date)

        # Add engineered features
        macro_df = self.add_event_features(macro_df)
        macro_df = self.add_rate_features(macro_df)
        macro_df = self.add_risk_features(macro_df)
        macro_df = self.add_policy_regime(macro_df)

        # If input df provided, merge on date
        if df is not None:
            # Ensure both have datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            if not isinstance(macro_df.index, pd.DatetimeIndex):
                macro_df.index = pd.to_datetime(macro_df.index)

            # Merge
            result = df.join(macro_df, how="left")
            result = result.ffill()  # Forward-fill any gaps

            logger.info(f"Merged macro features: {macro_df.shape[1]} new columns")
            return result

        logger.info(f"Calculated {macro_df.shape[1]} macro features")
        return macro_df

    def get_feature_names(self) -> List[str]:
        """Return list of all macro feature names."""
        return [
            # Raw FRED series
            "yield_10y", "yield_2y", "yield_3m", "yield_spread_10y_2y",
            "fed_funds_effective",
            "breakeven_inflation_10y", "breakeven_inflation_5y",
            "high_yield_spread",
            "cpi_yoy", "unemployment_rate",
            # Event timing
            "days_to_fomc", "days_since_fomc", "is_fomc_week",
            "days_to_nfp", "is_nfp_week",
            "days_to_cpi", "is_cpi_week",
            "major_events_next_7d", "is_event_week",
            # Rate environment
            "yield_curve_inverted", "yield_curve_steepening",
            "yield_10y_change_5d", "yield_10y_change_20d",
            "yield_2y_change_5d", "yield_2y_change_20d",
            "yield_spread_change_5d", "real_rate_10y",
            # Risk sentiment
            "hy_spread_change_5d", "hy_spread_change_20d",
            "hy_spread_elevated", "inflation_expectations_rising",
            # Policy regime
            "monetary_policy_regime",
        ]


def calculate_macro_features(
    df: Optional[pd.DataFrame] = None,
    start_date: str = "2019-01-01",
) -> pd.DataFrame:
    """
    Convenience function to calculate all macro features.

    Args:
        df: Optional DataFrame to merge with
        start_date: Start date for data

    Returns:
        DataFrame with macro features
    """
    calculator = MacroIndicators()
    return calculator.calculate_all(df=df, start_date=start_date)


if __name__ == "__main__":
    # Quick test
    print("Testing MacroIndicators...")

    macro = MacroIndicators()
    features = macro.calculate_all(start_date="2024-01-01")

    print(f"\nShape: {features.shape}")
    print(f"\nFeatures ({len(features.columns)}):")
    for col in sorted(features.columns):
        print(f"  - {col}")

    print(f"\nSample (last 5 days):")
    print(features.tail())

    # Check event features for today
    today = date.today()
    fomc = macro._calc_fomc_features(today)
    nfp = macro._calc_nfp_features(today)
    cpi = macro._calc_cpi_features(today)

    print(f"\n--- Event Features for {today} ---")
    print(f"Days to FOMC: {fomc['days_to_fomc']}")
    print(f"Days to NFP: {nfp['days_to_nfp']}")
    print(f"Days to CPI: {cpi['days_to_cpi']}")
