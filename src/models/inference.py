"""
Model Inference Module.

Provides a unified interface for loading trained models
and generating predictions for the dashboard.
"""

import joblib
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default model directory
MODEL_DIR = Path(__file__).parent.parent.parent / "models"


class ModelInference:
    """
    Unified inference interface for ML options trading models.

    Loads direction classifier and volatility forecaster, generates
    predictions with strategy recommendations.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize inference engine.

        Args:
            model_dir: Path to models directory (defaults to project models/)
        """
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._models: Dict[str, Dict] = {}

        logger.info(f"Initialized ModelInference with model_dir={self.model_dir}")

    def _load_model(self, ticker: str, model_type: str) -> Optional[Dict]:
        """
        Load a model from disk.

        Args:
            ticker: Stock ticker (spy, qqq, iwm)
            model_type: 'direction' or 'volatility'

        Returns:
            Model dictionary with 'model', 'feature_names', 'metadata'
        """
        cache_key = f"{ticker}_{model_type}"
        if cache_key in self._models:
            return self._models[cache_key]

        model_filename = f"{ticker.lower()}_{model_type}_model.joblib"
        model_path = self.model_dir / model_filename

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None

        try:
            model_data = joblib.load(model_path)
            self._models[cache_key] = model_data
            logger.info(f"Loaded model: {model_path}")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None

    def _prepare_features(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        include_sentiment: bool = False,
    ) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            ticker: Stock ticker
            price_data: Raw price data from data fetcher
            include_sentiment: Whether to include sentiment features

        Returns:
            DataFrame with features ready for prediction
        """
        from src.models.feature_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            ticker=ticker,
            include_macro=True,
            include_sentiment=include_sentiment,
        )

        # Get technical features from price data
        features_df, _ = pipeline.prepare_training_data(
            price_df=price_data,
            target_col="direction",
            forward_days=5,
        )

        return features_df

    def predict(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        include_sentiment: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for a ticker.

        Args:
            ticker: Stock ticker (SPY, QQQ, IWM)
            price_data: Price data DataFrame
            include_sentiment: Include sentiment features

        Returns:
            Dictionary with prediction details
        """
        ticker_lower = ticker.lower()

        # Load models
        direction_model = self._load_model(ticker_lower, "direction")
        volatility_model = self._load_model(ticker_lower, "volatility")

        if not direction_model:
            logger.error(f"Direction model not found for {ticker}")
            return None

        # Prepare features
        try:
            features_df = self._prepare_features(ticker, price_data, include_sentiment)
            if features_df.empty:
                logger.error(f"No features generated for {ticker}")
                return None

            latest_features = features_df.iloc[[-1]]
            latest_date = latest_features.index[0] if hasattr(latest_features.index, '__getitem__') else datetime.now()

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None

        # Direction prediction
        try:
            dir_model = direction_model["model"]
            dir_feature_names = direction_model.get("feature_names", [])

            # Select features that model was trained on
            available_features = [f for f in dir_feature_names if f in latest_features.columns]
            if len(available_features) < len(dir_feature_names) * 0.5:
                logger.warning(f"Only {len(available_features)}/{len(dir_feature_names)} features available")

            X_dir = latest_features[available_features].fillna(0)

            # Get prediction
            dir_pred = dir_model.predict(X_dir)[0]
            dir_proba = dir_model.predict_proba(X_dir)[0]

            if dir_pred == 1:
                direction = "up"
                direction_prob = dir_proba[1] if len(dir_proba) > 1 else 0.5
            else:
                direction = "down"
                direction_prob = dir_proba[0] if len(dir_proba) > 0 else 0.5

        except Exception as e:
            logger.error(f"Direction prediction failed: {e}")
            direction = "neutral"
            direction_prob = 0.5

        # Volatility prediction (if model exists)
        volatility_regime = "medium"
        iv_rank = None

        if volatility_model:
            try:
                vol_model = volatility_model["model"]
                vol_feature_names = volatility_model.get("feature_names", [])

                available_vol_features = [f for f in vol_feature_names if f in latest_features.columns]
                X_vol = latest_features[available_vol_features].fillna(0)

                vol_pred = vol_model.predict(X_vol)[0]

                # Map prediction to regime
                vol_mapping = {0: "low", 1: "medium", 2: "high"}
                volatility_regime = vol_mapping.get(vol_pred, "medium")

                # Get IV rank if available
                if "iv_rank" in latest_features.columns:
                    iv_rank = latest_features["iv_rank"].iloc[0]

            except Exception as e:
                logger.warning(f"Volatility prediction failed: {e}")

        # Strategy recommendation
        strategy = self._select_strategy(direction, direction_prob, volatility_regime, iv_rank)

        # Current price
        current_price = None
        if "close" in latest_features.columns:
            current_price = latest_features["close"].iloc[0]
        elif hasattr(price_data, "columns") and "Close" in price_data.columns:
            current_price = price_data["Close"].iloc[-1]

        result = {
            "ticker": ticker.upper(),
            "date": str(latest_date)[:10],
            "direction": direction,
            "direction_confidence": direction_prob,
            "volatility_regime": volatility_regime,
            "iv_rank": iv_rank,
            "strategy": strategy,
            "strategy_confidence": direction_prob * 0.8,  # Slightly discount
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Prediction for {ticker}: {direction} ({direction_prob:.1%}), strategy={strategy}")

        return result

    def _select_strategy(
        self,
        direction: str,
        confidence: float,
        volatility_regime: str,
        iv_rank: Optional[float],
    ) -> str:
        """
        Select options strategy based on direction and volatility.

        Args:
            direction: 'up', 'down', or 'neutral'
            confidence: Prediction confidence
            volatility_regime: 'low', 'medium', 'high'
            iv_rank: IV rank (0-100) if available

        Returns:
            Strategy name
        """
        # Use IV rank to determine regime if available
        if iv_rank is not None:
            if iv_rank < 30:
                vol_regime = "low"
            elif iv_rank > 60:
                vol_regime = "high"
            else:
                vol_regime = "medium"
        else:
            vol_regime = volatility_regime

        # Low confidence = no trade
        if confidence < 0.52:
            return "no_trade"

        # Strategy matrix
        if direction == "up":
            if vol_regime == "low":
                return "long_call"
            elif vol_regime == "medium":
                return "bull_call_spread"
            else:  # high
                return "bull_put_spread"

        elif direction == "down":
            if vol_regime == "low":
                return "long_put"
            elif vol_regime == "medium":
                return "bear_put_spread"
            else:  # high
                return "bear_call_spread"

        else:  # neutral
            if vol_regime == "high":
                return "iron_condor"
            return "no_trade"

    def batch_predict(
        self,
        tickers: list,
        data_fetcher: Any,
        include_sentiment: bool = False,
    ) -> Dict[str, Dict]:
        """
        Generate predictions for multiple tickers.

        Args:
            tickers: List of ticker symbols
            data_fetcher: DataFetcher instance
            include_sentiment: Include sentiment features

        Returns:
            Dictionary of ticker -> prediction
        """
        results = {}

        for ticker in tickers:
            try:
                # Fetch data
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

                price_data = data_fetcher.fetch_ticker_data(ticker, start_date, end_date)

                if price_data is None or price_data.empty:
                    logger.warning(f"No price data for {ticker}")
                    continue

                # Generate prediction
                prediction = self.predict(ticker, price_data, include_sentiment)

                if prediction:
                    results[ticker] = prediction

            except Exception as e:
                logger.error(f"Failed to predict {ticker}: {e}")

        return results


# Convenience function for CLI usage
def generate_signals(
    tickers: list = ["SPY", "QQQ", "IWM"],
    include_sentiment: bool = False,
) -> Dict[str, Dict]:
    """
    Generate trading signals for tickers.

    Args:
        tickers: List of ticker symbols
        include_sentiment: Include sentiment features

    Returns:
        Dictionary of predictions
    """
    from src.data.data_fetcher import DataFetcher

    inference = ModelInference()
    data_fetcher = DataFetcher()

    return inference.batch_predict(tickers, data_fetcher, include_sentiment)


if __name__ == "__main__":
    # Test inference
    print("Testing ModelInference...\n")

    signals = generate_signals()

    for ticker, signal in signals.items():
        print(f"\n{ticker}:")
        print(f"  Direction: {signal['direction']} ({signal['direction_confidence']:.1%})")
        print(f"  Strategy: {signal['strategy']}")
        print(f"  IV Rank: {signal['iv_rank']}")
