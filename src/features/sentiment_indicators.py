"""
Sentiment indicators from financial news and social media.

Integrates with Finnhub (news sentiment) and ApeWisdom (Reddit mentions)
to provide sentiment features for the ML options trading pipeline.
"""

import os
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentData:
    """Container for sentiment data from various sources."""
    # News sentiment (Finnhub)
    news_score: float = 0.5  # Company news score (0-1)
    bullish_pct: float = 0.5  # % of bullish articles
    bearish_pct: float = 0.5  # % of bearish articles
    buzz_score: float = 0.0  # News buzz/attention
    articles_week: int = 0  # Articles in last week
    sector_bullish: float = 0.5  # Sector average bullish %

    # Social sentiment (Reddit)
    reddit_mentions: int = 0
    reddit_rank: int = 999
    reddit_mentions_24h_ago: int = 0
    reddit_upvotes: int = 0


class SentimentFetcher:
    """
    Fetch and process sentiment data from multiple sources.

    Primary source: Finnhub (news sentiment)
    Secondary source: ApeWisdom (Reddit mentions)
    """

    FINNHUB_BASE = "https://finnhub.io/api/v1"
    APEWISDOM_BASE = "https://apewisdom.io/api/v1.0"

    # Keywords for simple sentiment analysis from news headlines
    BULLISH_KEYWORDS = [
        'surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'bull', 'buy',
        'upgrade', 'beat', 'record', 'high', 'growth', 'profit', 'positive',
        'optimistic', 'strong', 'outperform', 'breakout', 'momentum'
    ]
    BEARISH_KEYWORDS = [
        'drop', 'fall', 'decline', 'plunge', 'crash', 'bear', 'sell',
        'downgrade', 'miss', 'loss', 'low', 'weak', 'negative', 'concern',
        'risk', 'warning', 'fear', 'recession', 'cut', 'layoff'
    ]

    # ETF to stock mapping for sentiment (ETFs don't have direct sentiment)
    ETF_SENTIMENT_PROXIES = {
        "SPY": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],  # Top S&P 500 holdings
        "QQQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],  # Top Nasdaq holdings
        "IWM": ["AMC", "GME", "SOFI", "PLTR"],  # Popular small caps
    }

    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY", "")
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = timedelta(hours=1)
        self._cache_times: Dict[str, datetime] = {}

        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY not set - sentiment features will use defaults")
        else:
            logger.info("Initialized SentimentFetcher with Finnhub API")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_times:
            return False
        return datetime.now() - self._cache_times[key] < self._cache_ttl

    def _analyze_headline_sentiment(self, headline: str) -> float:
        """
        Analyze sentiment from a news headline using keyword matching.

        Args:
            headline: News headline text

        Returns:
            Sentiment score from -1 (bearish) to +1 (bullish)
        """
        headline_lower = headline.lower()
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in headline_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in headline_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total

    def fetch_market_news_sentiment(self) -> Dict:
        """
        Fetch market news from Finnhub (free tier) and analyze sentiment.

        Returns:
            Dict with aggregated market sentiment
        """
        if not self.finnhub_key:
            return self._empty_sentiment()

        cache_key = "market_news"
        if cache_key in self._cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            url = f"{self.FINNHUB_BASE}/news"
            params = {"category": "general", "token": self.finnhub_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            news_items = response.json()

            if not news_items:
                return self._empty_sentiment()

            # Analyze sentiment from headlines
            sentiments = []
            for item in news_items[:50]:  # Analyze top 50 news
                headline = item.get("headline", "")
                if headline:
                    sentiments.append(self._analyze_headline_sentiment(headline))

            if not sentiments:
                return self._empty_sentiment()

            avg_sentiment = sum(sentiments) / len(sentiments)

            # Convert to bullish/bearish percentages
            bullish_pct = 0.5 + (avg_sentiment / 2)  # Map -1,1 to 0,1
            bearish_pct = 1 - bullish_pct

            result = {
                "news_score": bullish_pct,
                "bullish_pct": bullish_pct,
                "bearish_pct": bearish_pct,
                "buzz_score": len(news_items) / 100,  # Normalize
                "articles_week": len(news_items),
                "sector_bullish": 0.5,  # Not available in free tier
            }

            self._cache[cache_key] = result
            self._cache_times[cache_key] = datetime.now()

            logger.info(
                f"Market news sentiment: bullish={bullish_pct:.1%}, "
                f"from {len(news_items)} articles"
            )

            return result

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return self._empty_sentiment()

    def fetch_finnhub_sentiment(self, symbol: str) -> Dict:
        """
        Fetch news sentiment from Finnhub.

        Uses Market News (free tier) since news-sentiment requires premium.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with sentiment metrics
        """
        if not self.finnhub_key:
            return self._empty_sentiment()

        # Use market news sentiment as baseline for all symbols
        # (Symbol-specific sentiment requires premium)
        return self.fetch_market_news_sentiment()

    def fetch_etf_sentiment(self, etf_symbol: str) -> Dict:
        """
        Fetch sentiment for an ETF.

        Uses market-wide news sentiment since ETF-specific sentiment
        requires premium API access.

        Args:
            etf_symbol: ETF ticker (SPY, QQQ, IWM)

        Returns:
            Market sentiment (applies to all ETFs)
        """
        # Use market news sentiment for ETFs (free tier limitation)
        return self.fetch_market_news_sentiment()

    def fetch_reddit_mentions(self, symbol: str) -> Dict:
        """
        Fetch Reddit mention data from ApeWisdom.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with Reddit mention metrics
        """
        cache_key = f"reddit_{symbol}"
        if cache_key in self._cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            url = f"{self.APEWISDOM_BASE}/filter/all-stocks"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Find our symbol in the results
            for stock in data.get("results", []):
                if stock.get("ticker", "").upper() == symbol.upper():
                    result = {
                        "reddit_mentions": stock.get("mentions", 0),
                        "reddit_rank": stock.get("rank", 999),
                        "reddit_mentions_24h_ago": stock.get("mentions_24h_ago", 0),
                        "reddit_upvotes": stock.get("upvotes", 0),
                    }
                    self._cache[cache_key] = result
                    self._cache_times[cache_key] = datetime.now()

                    logger.info(
                        f"Reddit mentions for {symbol}: "
                        f"{result['reddit_mentions']} (rank #{result['reddit_rank']})"
                    )
                    return result

            # Symbol not found in trending
            result = {
                "reddit_mentions": 0,
                "reddit_rank": 999,
                "reddit_mentions_24h_ago": 0,
                "reddit_upvotes": 0,
            }
            self._cache[cache_key] = result
            self._cache_times[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.warning(f"Error fetching Reddit data for {symbol}: {e}")
            return {
                "reddit_mentions": 0,
                "reddit_rank": 999,
                "reddit_mentions_24h_ago": 0,
                "reddit_upvotes": 0,
            }

    def _empty_sentiment(self) -> Dict:
        """Return neutral sentiment when data unavailable."""
        return {
            "news_score": 0.5,
            "bullish_pct": 0.5,
            "bearish_pct": 0.5,
            "buzz_score": 0,
            "articles_week": 0,
            "sector_bullish": 0.5,
        }

    def get_all_sentiment(self, symbol: str) -> SentimentData:
        """
        Fetch all sentiment data for a symbol.

        Args:
            symbol: Stock or ETF ticker

        Returns:
            SentimentData with all metrics
        """
        # Use ETF aggregation for known ETFs
        if symbol.upper() in self.ETF_SENTIMENT_PROXIES:
            news = self.fetch_etf_sentiment(symbol)
        else:
            news = self.fetch_finnhub_sentiment(symbol)

        reddit = self.fetch_reddit_mentions(symbol)

        return SentimentData(
            news_score=news["news_score"],
            bullish_pct=news["bullish_pct"],
            bearish_pct=news["bearish_pct"],
            buzz_score=news["buzz_score"],
            articles_week=news["articles_week"],
            sector_bullish=news["sector_bullish"],
            reddit_mentions=reddit["reddit_mentions"],
            reddit_rank=reddit["reddit_rank"],
            reddit_mentions_24h_ago=reddit["reddit_mentions_24h_ago"],
            reddit_upvotes=reddit["reddit_upvotes"],
        )


def calculate_sentiment_features(
    df: pd.DataFrame,
    symbol: str,
    finnhub_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add sentiment features to a DataFrame.

    For backtesting: Uses current sentiment as static proxy
    (sentiment history is not available in free tier).

    For live trading: Call daily for fresh sentiment data.

    Args:
        df: DataFrame with price/feature data (DatetimeIndex)
        symbol: Stock/ETF ticker
        finnhub_api_key: Optional API key override

    Returns:
        DataFrame with sentiment features added
    """
    fetcher = SentimentFetcher(finnhub_api_key)
    result = df.copy()

    # Fetch current sentiment
    sentiment = fetcher.get_all_sentiment(symbol)

    # ===================
    # Core Sentiment Score
    # ===================
    # Normalized to -1 (bearish) to +1 (bullish)
    result["sentiment_score"] = sentiment.bullish_pct - sentiment.bearish_pct

    # Relative to sector (positive = outperforming sector sentiment)
    result["sentiment_vs_sector"] = sentiment.bullish_pct - sentiment.sector_bullish

    # News intensity/attention
    result["news_buzz_score"] = sentiment.buzz_score
    result["news_article_count"] = sentiment.articles_week

    # ===================
    # Social Media Features
    # ===================
    result["reddit_mentions"] = sentiment.reddit_mentions

    # Reddit momentum (change from 24h ago)
    if sentiment.reddit_mentions_24h_ago > 0:
        result["reddit_momentum"] = (
            (sentiment.reddit_mentions - sentiment.reddit_mentions_24h_ago)
            / sentiment.reddit_mentions_24h_ago
        )
    else:
        result["reddit_momentum"] = 0.0

    # ===================
    # Binary Classification
    # ===================
    # Sentiment regime flags
    result["sentiment_bullish"] = (result["sentiment_score"] > 0.15).astype(int)
    result["sentiment_bearish"] = (result["sentiment_score"] < -0.15).astype(int)
    result["sentiment_neutral"] = (
        (result["sentiment_score"] >= -0.15) & (result["sentiment_score"] <= 0.15)
    ).astype(int)

    # Extreme sentiment (contrarian signals)
    result["sentiment_extreme_bullish"] = (result["sentiment_score"] > 0.35).astype(int)
    result["sentiment_extreme_bearish"] = (result["sentiment_score"] < -0.35).astype(int)

    # High social attention flag
    result["high_social_attention"] = (sentiment.reddit_mentions > 50).astype(int)

    # ===================
    # Cross-Feature Interactions
    # ===================
    # Sentiment + IV interactions (if IV features exist)
    if "iv_rank" in result.columns:
        # Bullish sentiment + High IV = potential IV crush opportunity (sell premium)
        result["sentiment_bullish_high_iv"] = (
            (result["sentiment_score"] > 0.1) & (result["iv_rank"] > 60)
        ).astype(int)

        # Bearish sentiment + Low IV = cheap puts
        result["sentiment_bearish_low_iv"] = (
            (result["sentiment_score"] < -0.1) & (result["iv_rank"] < 30)
        ).astype(int)

        # Sentiment-IV divergence (contrarian signal)
        # High IV (fear) but bullish sentiment = market may be oversold
        result["sentiment_iv_divergence"] = (
            (result["sentiment_score"] > 0.1) & (result["iv_rank"] > 70)
        ).astype(int)

    logger.info(
        f"Added {len([c for c in result.columns if 'sentiment' in c or 'reddit' in c or 'news' in c])} "
        f"sentiment features for {symbol}"
    )

    return result


# Convenience function for standalone usage
def get_sentiment_summary(symbol: str, api_key: Optional[str] = None) -> Dict:
    """
    Get a summary of current sentiment for a symbol.

    Args:
        symbol: Stock/ETF ticker
        api_key: Optional Finnhub API key

    Returns:
        Dict with sentiment summary
    """
    fetcher = SentimentFetcher(api_key)
    sentiment = fetcher.get_all_sentiment(symbol)

    score = sentiment.bullish_pct - sentiment.bearish_pct

    # Classify sentiment
    if score > 0.35:
        regime = "VERY_BULLISH"
    elif score > 0.15:
        regime = "BULLISH"
    elif score < -0.35:
        regime = "VERY_BEARISH"
    elif score < -0.15:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    return {
        "symbol": symbol,
        "sentiment_score": score,
        "sentiment_regime": regime,
        "bullish_pct": sentiment.bullish_pct,
        "bearish_pct": sentiment.bearish_pct,
        "news_buzz": sentiment.buzz_score,
        "articles_this_week": sentiment.articles_week,
        "reddit_mentions": sentiment.reddit_mentions,
        "reddit_rank": sentiment.reddit_rank if sentiment.reddit_rank < 999 else "Not trending",
    }
