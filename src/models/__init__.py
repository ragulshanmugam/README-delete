"""
ML models for direction prediction, volatility forecasting, and regime classification.

This module provides:
- RobustDirectionClassifier: Robust binary classifier with regularization (RECOMMENDED)
- DirectionClassifier: XGBoost classifier for market direction (higher variance)
- FeaturePipeline: Data loading and preprocessing pipeline for ML training
- VolatilityForecaster: Forecasts future volatility levels
- RegimeClassifier: Classifies market regimes

Example usage:
    from src.models import RobustDirectionClassifier, FeaturePipeline

    # Prepare training data
    pipeline = FeaturePipeline(ticker="SPY", include_macro=True, classification_mode="binary")
    X, y, metadata = pipeline.prepare_training_data()

    # Train robust classifier (recommended)
    classifier = RobustDirectionClassifier(
        model_type="logistic",
        regularization="medium",
        use_safe_features=True,
    )
    metrics = classifier.train(X, y, n_splits=5)

    # Make predictions
    predictions = classifier.predict_with_confidence(X.tail(5))
"""

from src.models.direction_classifier import (
    DirectionClassifier,
    ClassificationResult,
    EvaluationMetrics,
    WalkForwardSplit,
)
from src.models.robust_classifier import (
    RobustDirectionClassifier,
    RobustEvaluationResult,
    get_safe_features,
    LEVEL_FEATURES,
    MOMENTUM_FEATURES,
)
from src.models.feature_pipeline import (
    FeaturePipeline,
    load_training_data,
)
from src.models.volatility_forecaster import VolatilityForecaster
from src.models.regime_classifier import RegimeClassifier, MarketRegime

__all__ = [
    # Robust Direction Classifier (recommended)
    "RobustDirectionClassifier",
    "RobustEvaluationResult",
    "get_safe_features",
    "LEVEL_FEATURES",
    "MOMENTUM_FEATURES",
    # Direction Classifier (XGBoost)
    "DirectionClassifier",
    "ClassificationResult",
    "EvaluationMetrics",
    "WalkForwardSplit",
    # Feature Pipeline
    "FeaturePipeline",
    "load_training_data",
    # Volatility Forecaster
    "VolatilityForecaster",
    # Regime Classifier
    "RegimeClassifier",
    "MarketRegime",
]
