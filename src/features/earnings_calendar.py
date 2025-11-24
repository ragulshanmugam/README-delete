"""
Earnings Calendar Features for ML Options Trading.

Integrates with Finnhub free tier to get upcoming earnings dates
and adds earnings-related features for the model.

Key Features:
- Days to next earnings announcement
- Days since last earnings
- Earnings surprise history
- Earnings week indicator (binary)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EarningsInfo:
    """Earnings information for a symbol."""
    symbol: str
    next_earnings_date: Optional[datetime] = None
    last_earnings_date: Optional[datetime] = None
    days_to_earnings: Optional[int] = None
    days_since_earnings: Optional[int] = None
    earnings_week: bool = False
    last_surprise_pct: Optional[float] = None


class EarningsCalendarFetcher:
    """
    Fetch earnings calendar data from Finnhub.

    Uses the free tier earnings calendar endpoint.
    """

    FINNHUB_BASE = "https://finnhub.io/api/v1"

    # ETFs don't have earnings - use their top holdings
    ETF_PROXIES = {
        "SPY": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "QQQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
        "IWM": [],  # Small caps - skip for now
    }

    def __init__(self, finnhub_api_key: Optional[str] = None):
        """Initialize with Finnhub API key."""
        self.api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY", "")
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = timedelta(hours=6)
        self._cache_times: Dict[str, datetime] = {}

        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not set - earnings features will use defaults")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_times:
            return False
        return datetime.now() - self._cache_times[key] < self._cache_ttl

    def fetch_earnings_calendar(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch earnings calendar for a symbol.

        Args:
            symbol: Stock ticker
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of earnings events
        """
        if not self.api_key:
            return []

        # Default date range: 60 days back, 60 days forward
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")

        cache_key = f"earnings_{symbol}_{from_date}_{to_date}"
        if cache_key in self._cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            url = f"{self.FINNHUB_BASE}/calendar/earnings"
            params = {
                "symbol": symbol.upper(),
                "from": from_date,
                "to": to_date,
                "token": self.api_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            earnings_list = data.get("earningsCalendar", [])

            self._cache[cache_key] = earnings_list
            self._cache_times[cache_key] = datetime.now()

            logger.debug(f"Fetched {len(earnings_list)} earnings events for {symbol}")

            return earnings_list

        except Exception as e:
            logger.warning(f"Error fetching earnings calendar for {symbol}: {e}")
            return []

    def get_earnings_info(
        self,
        symbol: str,
        reference_date: Optional[datetime] = None,
    ) -> EarningsInfo:
        """
        Get earnings info relative to a reference date.

        Args:
            symbol: Stock ticker
            reference_date: Date to calculate days from (default: today)

        Returns:
            EarningsInfo with calculated metrics
        """
        if reference_date is None:
            reference_date = datetime.now()

        ref_date = reference_date.date() if hasattr(reference_date, 'date') else reference_date

        # Handle ETFs
        if symbol.upper() in self.ETF_PROXIES:
            return self._get_etf_earnings_info(symbol, reference_date)

        # Fetch earnings data
        earnings = self.fetch_earnings_calendar(symbol)

        if not earnings:
            return EarningsInfo(symbol=symbol)

        # Parse dates and find next/last earnings
        next_earnings = None
        last_earnings = None
        last_surprise = None

        for event in earnings:
            event_date_str = event.get("date")
            if not event_date_str:
                continue

            try:
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()

                if event_date >= ref_date:
                    if next_earnings is None or event_date < next_earnings:
                        next_earnings = event_date
                else:
                    if last_earnings is None or event_date > last_earnings:
                        last_earnings = event_date
                        # Get surprise data if available
                        actual = event.get("epsActual")
                        estimate = event.get("epsEstimate")
                        if actual is not None and estimate is not None and estimate != 0:
                            last_surprise = ((actual - estimate) / abs(estimate)) * 100

            except ValueError:
                continue

        # Calculate days
        days_to_earnings = None
        days_since_earnings = None
        earnings_week = False

        if next_earnings:
            days_to_earnings = (next_earnings - ref_date).days
            earnings_week = days_to_earnings <= 7

        if last_earnings:
            days_since_earnings = (ref_date - last_earnings).days

        return EarningsInfo(
            symbol=symbol,
            next_earnings_date=datetime.combine(next_earnings, datetime.min.time()) if next_earnings else None,
            last_earnings_date=datetime.combine(last_earnings, datetime.min.time()) if last_earnings else None,
            days_to_earnings=days_to_earnings,
            days_since_earnings=days_since_earnings,
            earnings_week=earnings_week,
            last_surprise_pct=last_surprise,
        )

    def _get_etf_earnings_info(
        self,
        etf_symbol: str,
        reference_date: datetime,
    ) -> EarningsInfo:
        """
        Get aggregated earnings info for an ETF based on top holdings.

        For ETFs, we check if any major holdings have earnings this week.
        """
        proxies = self.ETF_PROXIES.get(etf_symbol.upper(), [])

        if not proxies:
            return EarningsInfo(symbol=etf_symbol)

        # Check each proxy stock
        min_days_to = float('inf')
        any_earnings_week = False
        avg_surprise = []

        for proxy in proxies[:5]:  # Limit to 5 to avoid rate limits
            info = self.get_earnings_info(proxy, reference_date)

            if info.days_to_earnings is not None:
                min_days_to = min(min_days_to, info.days_to_earnings)

            if info.earnings_week:
                any_earnings_week = True

            if info.last_surprise_pct is not None:
                avg_surprise.append(info.last_surprise_pct)

        return EarningsInfo(
            symbol=etf_symbol,
            days_to_earnings=int(min_days_to) if min_days_to != float('inf') else None,
            earnings_week=any_earnings_week,
            last_surprise_pct=sum(avg_surprise) / len(avg_surprise) if avg_surprise else None,
        )


def calculate_earnings_features(
    df: pd.DataFrame,
    symbol: str,
    finnhub_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add earnings-related features to a DataFrame.

    For backtesting: Uses a simplified approach since historical earnings
    calendar requires premium API access.

    For live trading: Uses real upcoming earnings data.

    Args:
        df: DataFrame with price/feature data (DatetimeIndex)
        symbol: Stock/ETF ticker
        finnhub_api_key: Optional API key override

    Returns:
        DataFrame with earnings features added
    """
    fetcher = EarningsCalendarFetcher(finnhub_api_key)
    result = df.copy()

    # Get current earnings info (for the most recent date)
    current_info = fetcher.get_earnings_info(symbol)

    # For historical data, we'll use static values from current info
    # (Historical earnings calendar requires premium access)

    # =====================
    # Core Earnings Features
    # =====================

    # Days to next earnings (capped at 90 for normalization)
    if current_info.days_to_earnings is not None:
        result["days_to_earnings"] = min(current_info.days_to_earnings, 90)
    else:
        result["days_to_earnings"] = 45  # Assume mid-quarter

    # Normalize to 0-1 scale
    result["earnings_proximity"] = 1 - (result["days_to_earnings"] / 90)

    # Binary: Is earnings this week?
    result["earnings_week"] = int(current_info.earnings_week)

    # Days since last earnings (capped at 90)
    if current_info.days_since_earnings is not None:
        result["days_since_earnings"] = min(current_info.days_since_earnings, 90)
    else:
        result["days_since_earnings"] = 45

    # =====================
    # Earnings Risk Categories
    # =====================

    # Approaching earnings (7-14 days out) - high IV environment
    result["earnings_approaching"] = (
        (result["days_to_earnings"] >= 7) & (result["days_to_earnings"] <= 14)
    ).astype(int)

    # Just after earnings (0-7 days) - IV crush expected
    result["post_earnings"] = (result["days_since_earnings"] <= 7).astype(int)

    # Safe from earnings (> 30 days)
    result["earnings_safe"] = (result["days_to_earnings"] > 30).astype(int)

    # =====================
    # Earnings Surprise
    # =====================

    # Last earnings surprise (as percentage)
    if current_info.last_surprise_pct is not None:
        result["last_earnings_surprise"] = current_info.last_surprise_pct
    else:
        result["last_earnings_surprise"] = 0.0

    # Surprise categories
    result["earnings_beat"] = (result["last_earnings_surprise"] > 5).astype(int)
    result["earnings_miss"] = (result["last_earnings_surprise"] < -5).astype(int)

    # =====================
    # IV + Earnings Interactions
    # =====================

    # If IV features exist, create interactions
    if "iv_rank" in result.columns:
        # High IV + Earnings approaching = classic IV crush setup
        result["earnings_iv_crush_setup"] = (
            (result["earnings_approaching"] == 1) & (result["iv_rank"] > 60)
        ).astype(int)

        # Low IV + Earnings week = potential straddle opportunity
        result["earnings_straddle_setup"] = (
            (result["earnings_week"] == 1) & (result["iv_rank"] < 40)
        ).astype(int)

    feature_count = len([c for c in result.columns if 'earnings' in c])
    logger.info(f"Added {feature_count} earnings features for {symbol}")

    return result


def get_upcoming_earnings_summary(
    symbols: List[str],
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get a summary of upcoming earnings for multiple symbols.

    Args:
        symbols: List of stock tickers
        api_key: Optional Finnhub API key

    Returns:
        DataFrame with earnings summary
    """
    fetcher = EarningsCalendarFetcher(api_key)
    results = []

    for symbol in symbols:
        info = fetcher.get_earnings_info(symbol)
        results.append({
            "Symbol": symbol,
            "Next Earnings": info.next_earnings_date.strftime("%Y-%m-%d") if info.next_earnings_date else "N/A",
            "Days To": info.days_to_earnings if info.days_to_earnings else "N/A",
            "Earnings Week": "Yes" if info.earnings_week else "No",
            "Last Surprise": f"{info.last_surprise_pct:.1f}%" if info.last_surprise_pct else "N/A",
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test earnings features
    print("Testing EarningsCalendarFetcher...\n")

    # Test with individual stocks
    for symbol in ["AAPL", "SPY", "QQQ"]:
        fetcher = EarningsCalendarFetcher()
        info = fetcher.get_earnings_info(symbol)

        print(f"{symbol}:")
        print(f"  Next earnings: {info.next_earnings_date}")
        print(f"  Days to earnings: {info.days_to_earnings}")
        print(f"  Earnings week: {info.earnings_week}")
        print(f"  Last surprise: {info.last_surprise_pct}")
        print()
