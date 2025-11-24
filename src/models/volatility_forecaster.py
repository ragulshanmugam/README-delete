"""
Volatility Forecaster Model.

Predicts IV regime (LOW/NORMAL/HIGH) 5 days ahead to help with
options strategy selection and timing.

Key insight: Knowing if IV will expand or contract helps decide:
- Sell premium when IV is HIGH and expected to drop (IV crush)
- Buy premium when IV is LOW and expected to rise

Two modes:
1. Regression: Predict actual IV rank value (0-100)
2. Classification: Predict IV regime category (LOW/NORMAL/HIGH)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# IV Regime thresholds (based on IV Rank)
IV_REGIME_THRESHOLDS = {
    "LOW": (0, 30),      # IV Rank 0-30: Options cheap
    "NORMAL": (30, 60),  # IV Rank 30-60: Fair value
    "HIGH": (60, 100),   # IV Rank 60-100: Options expensive
}


@dataclass
class VolatilityForecastResult:
    """Results from volatility forecast."""
    predicted_regime: str  # LOW, NORMAL, HIGH
    regime_probabilities: Dict[str, float]  # Probability for each regime
    current_iv_rank: float
    predicted_iv_rank: Optional[float]  # If regression mode
    expected_iv_direction: str  # EXPANDING, CONTRACTING, STABLE
    confidence: float


class VolatilityForecaster:
    """
    Forecasts IV regime 5 days ahead.

    Uses current IV features, lagged IV values, and macro indicators
    to predict whether IV will be LOW, NORMAL, or HIGH.

    This helps with strategy selection:
    - HIGH IV predicted: Consider selling premium (credit spreads)
    - LOW IV predicted: Consider buying premium (directional plays)
    """

    # Features specifically useful for volatility prediction
    VOLATILITY_FEATURES = [
        # Current IV state
        "iv_rank",
        "iv_percentile",
        "iv_hv_spread",
        "vix_close",
        "vix_zscore",
        "vix_trend",
        "vix_change_1d",
        "vix_change_5d",

        # Term structure (predicts IV movement)
        "vix_term_slope",
        "vix_contango",

        # Historical volatility (HV often leads IV)
        "hv_5",
        "hv_10",
        "hv_20",
        "hv_60",
        "hv_ratio_5_20",
        "hv_ratio_20_60",
        "hv_trend",

        # Volatility of volatility
        "bb_width",
        "atr_pct",
        "parkinson_hv",

        # Market stress indicators
        "returns_5d",
        "returns_10d",

        # Macro rates (affect vol regime)
        "yield_spread_10y_2y",
        "yield_curve_inverted",
        "vix_regime_elevated",
    ]

    def __init__(
        self,
        prediction_horizon: Optional[int] = None,
        mode: str = "classification",
        regularization: str = "medium",
    ):
        """
        Initialize volatility forecaster.

        Args:
            prediction_horizon: Days ahead to predict (default from settings)
            mode: "classification" (regime) or "regression" (IV rank value)
            regularization: Regularization strength ("light", "medium", "strong")
        """
        settings = get_settings()
        self.prediction_horizon = prediction_horizon or settings.prediction_horizon
        self.mode = mode
        self.regularization = regularization

        # Regularization mapping
        reg_map = {"light": 1.0, "medium": 0.1, "strong": 0.01}
        C = reg_map.get(regularization, 0.1)

        if mode == "classification":
            # Multinomial logistic regression for 3-class prediction
            self.model = LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=1000,
                multi_class="multinomial",
                random_state=42,
            )
        else:
            # For regression, we'll use XGBoost if available, else Ridge
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    max_depth=5,
                    learning_rate=0.05,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    random_state=42,
                    n_jobs=-1,
                )
            except ImportError:
                from sklearn.linear_model import Ridge
                self.model = Ridge(alpha=1.0/C)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if mode == "classification" else None
        self.feature_names: List[str] = []
        self.is_fitted = False

        # Training metrics
        self.train_metrics: Dict = {}

        logger.info(
            f"Initialized VolatilityForecaster: horizon={self.prediction_horizon}d, "
            f"mode={mode}, regularization={regularization}"
        )

    def _create_iv_regime_labels(self, iv_rank: pd.Series) -> pd.Series:
        """
        Create IV regime labels based on IV rank.

        Args:
            iv_rank: Series with IV rank values

        Returns:
            Series with regime labels (LOW, NORMAL, HIGH)
        """
        conditions = [
            iv_rank < 30,
            (iv_rank >= 30) & (iv_rank < 60),
            iv_rank >= 60,
        ]
        choices = ["LOW", "NORMAL", "HIGH"]

        return pd.Series(
            np.select(conditions, choices, default="NORMAL"),
            index=iv_rank.index
        )

    def _create_forward_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Create forward-looking labels.

        Args:
            df: DataFrame with iv_rank column
            horizon: Days ahead to predict

        Returns:
            Tuple of (regime labels, iv_rank values) for future
        """
        # Get IV rank `horizon` days in the future
        future_iv_rank = df["iv_rank"].shift(-horizon)

        # Create regime labels for future IV
        future_regime = self._create_iv_regime_labels(future_iv_rank)

        return future_regime, future_iv_rank

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select available volatility-related features.

        Args:
            df: DataFrame with features

        Returns:
            List of available feature names
        """
        available = [f for f in self.VOLATILITY_FEATURES if f in df.columns]

        # Add lagged IV features if present
        for lag in [5, 10, 20]:
            lag_col = f"iv_rank_lag_{lag}"
            if lag_col in df.columns:
                available.append(lag_col)

        # Add momentum features
        for col in ["iv_rank_change_5d", "iv_rank_change_10d"]:
            if col in df.columns:
                available.append(col)

        logger.info(f"Selected {len(available)} volatility features")
        return available

    def _add_lagged_iv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged IV rank features for better prediction.

        Args:
            df: DataFrame with iv_rank column

        Returns:
            DataFrame with lagged features added
        """
        result = df.copy()

        if "iv_rank" in result.columns:
            for lag in [5, 10, 20]:
                result[f"iv_rank_lag_{lag}"] = result["iv_rank"].shift(lag)

            # IV rank momentum
            result["iv_rank_change_5d"] = result["iv_rank"] - result["iv_rank"].shift(5)
            result["iv_rank_change_10d"] = result["iv_rank"] - result["iv_rank"].shift(10)

            logger.info("Added lagged IV features")

        return result

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training.

        Args:
            df: DataFrame with all features including iv_rank

        Returns:
            Tuple of (X features, y labels)
        """
        # Add lagged features
        data = self._add_lagged_iv_features(df)

        # Create forward labels
        future_regime, future_iv_rank = self._create_forward_labels(
            data, self.prediction_horizon
        )

        if self.mode == "classification":
            data["target"] = future_regime
        else:
            data["target"] = future_iv_rank

        # Select features
        self.feature_names = self._select_features(data)

        # Drop rows with NaN (from lagging and forward shift)
        data = data.dropna(subset=self.feature_names + ["target"])

        X = data[self.feature_names]
        y = data["target"]

        logger.info(f"Prepared data: {len(X)} samples, {len(self.feature_names)} features")

        if self.mode == "classification":
            logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict:
        """
        Train the volatility forecaster with walk-forward validation.

        Args:
            X: Feature DataFrame
            y: Target labels
            n_splits: Number of validation splits

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training VolatilityForecaster ({self.mode}) with {len(X)} samples")

        if self.mode == "classification":
            return self._train_classifier(X, y, n_splits)
        else:
            return self._train_regressor(X, y, n_splits)

    def _train_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
    ) -> Dict:
        """Train classification model."""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            fold_metrics.append({
                "fold": fold,
                "accuracy": acc,
                "f1_macro": f1,
                "test_size": len(y_test),
            })

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            logger.info(f"Fold {fold}: Accuracy={acc:.3f}, F1={f1:.3f}")

        # Final fit on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True

        # Overall metrics
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        overall_f1 = f1_score(all_y_true, all_y_pred, average="macro")

        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)

        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            all_y_true, all_y_pred,
            target_names=class_names,
            output_dict=True
        )

        self.train_metrics = {
            "mode": "classification",
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "fold_metrics": fold_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "class_names": list(class_names),
        }

        logger.info(
            f"Training complete: Accuracy={overall_acc:.3f}, F1={overall_f1:.3f}"
        )

        return self.train_metrics

    def _train_regressor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
    ) -> Dict:
        """Train regression model."""
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Direction accuracy
            if "iv_rank" in X_test.columns:
                current_iv = X_test["iv_rank"].values
                actual_direction = y_test > current_iv
                pred_direction = y_pred > current_iv
                dir_acc = (actual_direction == pred_direction).mean()
            else:
                dir_acc = None

            fold_metrics.append({
                "fold": fold,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "direction_accuracy": dir_acc,
            })

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            logger.info(f"Fold {fold}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

        # Final fit on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.values)
        self.is_fitted = True

        # Overall metrics
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        overall_mae = mean_absolute_error(all_y_true, all_y_pred)
        overall_r2 = r2_score(all_y_true, all_y_pred)

        self.train_metrics = {
            "mode": "regression",
            "rmse": overall_rmse,
            "mae": overall_mae,
            "r2": overall_r2,
            "fold_metrics": fold_metrics,
            "n_samples": len(X),
            "n_features": len(self.feature_names),
        }

        logger.info(
            f"Training complete: RMSE={overall_rmse:.3f}, MAE={overall_mae:.3f}, R2={overall_r2:.3f}"
        )

        return self.train_metrics

    def predict(self, X: pd.DataFrame) -> List[VolatilityForecastResult]:
        """
        Predict IV regime for new data.

        Args:
            X: Feature DataFrame

        Returns:
            List of VolatilityForecastResult objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        # Ensure we have required features
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Scale and predict
        X_subset = X[self.feature_names]
        X_scaled = self.scaler.transform(X_subset)

        if self.mode == "classification":
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)
            regime_names = self.label_encoder.classes_
        else:
            y_pred_values = self.model.predict(X_scaled)
            y_pred_values = np.clip(y_pred_values, 0, 100)

        results = []
        for i in range(len(X)):
            # Get current IV rank
            current_iv = X.iloc[i].get("iv_rank", 50.0)
            current_regime = self._get_regime_from_rank(current_iv)

            if self.mode == "classification":
                pred_regime = self.label_encoder.inverse_transform([y_pred[i]])[0]
                proba_dict = {
                    regime: float(y_proba[i, j])
                    for j, regime in enumerate(regime_names)
                }
                predicted_iv_rank = None
                confidence = max(proba_dict.values())
            else:
                predicted_iv_rank = float(y_pred_values[i])
                pred_regime = self._get_regime_from_rank(predicted_iv_rank)
                proba_dict = {pred_regime: 1.0}  # No probabilities in regression
                confidence = 1.0 - abs(predicted_iv_rank - 50) / 50  # Higher near extremes

            # Determine expected direction
            if pred_regime == "HIGH" and current_regime == "LOW":
                direction = "EXPANDING"
            elif pred_regime == "LOW" and current_regime == "HIGH":
                direction = "CONTRACTING"
            elif pred_regime == current_regime:
                direction = "STABLE"
            elif pred_regime == "HIGH" or (pred_regime == "NORMAL" and current_regime == "LOW"):
                direction = "EXPANDING"
            else:
                direction = "CONTRACTING"

            results.append(VolatilityForecastResult(
                predicted_regime=pred_regime,
                regime_probabilities=proba_dict,
                current_iv_rank=current_iv,
                predicted_iv_rank=predicted_iv_rank,
                expected_iv_direction=direction,
                confidence=confidence,
            ))

        return results

    def _get_regime_from_rank(self, iv_rank: float) -> str:
        """Get IV regime from IV rank value."""
        if iv_rank < 30:
            return "LOW"
        elif iv_rank < 60:
            return "NORMAL"
        else:
            return "HIGH"

    def get_strategy_recommendation(
        self,
        direction_prediction: str,
        vol_forecast: VolatilityForecastResult,
        direction_confidence: float = 0.5,
    ) -> Dict:
        """
        Get options strategy recommendation based on direction and IV forecast.

        Args:
            direction_prediction: "UP" or "DOWN" from direction classifier
            vol_forecast: Volatility forecast result
            direction_confidence: Confidence from direction classifier

        Returns:
            Dictionary with strategy recommendation
        """
        iv_regime = vol_forecast.predicted_regime
        iv_direction = vol_forecast.expected_iv_direction
        iv_confidence = vol_forecast.confidence

        # Low confidence = no trade
        if direction_confidence < 0.55 and iv_confidence < 0.4:
            return {
                "strategy": "NO TRADE",
                "reasoning": "Low confidence on both direction and IV",
                "position": "Wait for clearer signal",
                "iv_context": {
                    "current_iv_rank": vol_forecast.current_iv_rank,
                    "predicted_regime": iv_regime,
                    "iv_direction": iv_direction,
                },
            }

        # Strategy selection matrix
        strategy_matrix = {
            ("UP", "LOW"): {
                "strategy": "Buy Calls",
                "reasoning": "Bullish + cheap options = buy directional",
                "position": "Long ATM/OTM calls, 2-4 weeks expiry",
                "risk_level": "Medium",
            },
            ("UP", "NORMAL"): {
                "strategy": "Bull Call Spread",
                "reasoning": "Bullish + fair IV = defined risk spread",
                "position": "Buy ATM call, sell OTM call, 2-3 weeks",
                "risk_level": "Low-Medium",
            },
            ("UP", "HIGH"): {
                "strategy": "Sell Put Spread (Bull Put)",
                "reasoning": "Bullish + expensive options = sell premium, expect IV crush",
                "position": "Sell ATM put, buy OTM put for protection, 2-4 weeks",
                "risk_level": "Medium",
            },
            ("DOWN", "LOW"): {
                "strategy": "Buy Puts",
                "reasoning": "Bearish + cheap options = buy directional protection",
                "position": "Long ATM/OTM puts, 2-4 weeks expiry",
                "risk_level": "Medium",
            },
            ("DOWN", "NORMAL"): {
                "strategy": "Bear Put Spread",
                "reasoning": "Bearish + fair IV = defined risk spread",
                "position": "Buy ATM put, sell OTM put, 2-3 weeks",
                "risk_level": "Low-Medium",
            },
            ("DOWN", "HIGH"): {
                "strategy": "Sell Call Spread (Bear Call)",
                "reasoning": "Bearish + expensive options = sell premium",
                "position": "Sell ATM call, buy OTM call for protection, 2-4 weeks",
                "risk_level": "Medium",
            },
        }

        key = (direction_prediction, iv_regime)
        recommendation = strategy_matrix.get(key, {
            "strategy": "No Trade",
            "reasoning": "Unclear signal combination",
            "position": "Wait for better setup",
            "risk_level": "N/A",
        })

        # Add IV context
        recommendation["iv_context"] = {
            "current_iv_rank": vol_forecast.current_iv_rank,
            "predicted_regime": iv_regime,
            "predicted_iv_rank": vol_forecast.predicted_iv_rank,
            "iv_direction": iv_direction,
            "iv_confidence": iv_confidence,
        }

        recommendation["direction_context"] = {
            "prediction": direction_prediction,
            "confidence": direction_confidence,
        }

        # Add cautions
        cautions = []
        if iv_confidence < 0.4:
            cautions.append("Low IV forecast confidence - consider smaller position")
        if direction_confidence < 0.55:
            cautions.append("Low direction confidence - tighter stops recommended")
        if iv_direction == "EXPANDING" and "Sell" in recommendation["strategy"]:
            cautions.append("IV expected to expand - selling premium is contrarian")

        if cautions:
            recommendation["cautions"] = cautions

        return recommendation

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not trained.")

        if self.mode == "classification":
            # For logistic regression, use absolute coefficients
            importance = np.abs(self.model.coef_).mean(axis=0)
        else:
            # For XGBoost, use built-in importance
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            else:
                # For Ridge, use absolute coefficients
                importance = np.abs(self.model.coef_)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        return df

    def save(self, path: str) -> None:
        """Save model to file."""
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "train_metrics": self.train_metrics,
            "prediction_horizon": self.prediction_horizon,
            "mode": self.mode,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VolatilityForecaster":
        """Load model from file."""
        data = joblib.load(path)

        forecaster = cls(
            prediction_horizon=data["prediction_horizon"],
            mode=data["mode"],
        )
        forecaster.model = data["model"]
        forecaster.scaler = data["scaler"]
        forecaster.label_encoder = data["label_encoder"]
        forecaster.feature_names = data["feature_names"]
        forecaster.train_metrics = data["train_metrics"]
        forecaster.is_fitted = True

        logger.info(f"Model loaded from {path}")
        return forecaster


def train_volatility_forecaster(
    df: pd.DataFrame,
    ticker: str,
    mode: str = "classification",
    save_path: Optional[str] = None,
) -> Tuple[VolatilityForecaster, Dict]:
    """
    Convenience function to train a volatility forecaster.

    Args:
        df: DataFrame with all features including IV features
        ticker: Stock ticker (for logging)
        mode: "classification" or "regression"
        save_path: Optional path to save model

    Returns:
        Tuple of (trained forecaster, metrics)
    """
    logger.info(f"Training volatility forecaster for {ticker} (mode={mode})")

    forecaster = VolatilityForecaster(mode=mode)
    X, y = forecaster.prepare_data(df)
    metrics = forecaster.train(X, y)

    if save_path:
        forecaster.save(save_path)

    return forecaster, metrics


if __name__ == "__main__":
    # Quick test with sample data
    print("Testing VolatilityForecaster...")

    # Create synthetic data for testing
    np.random.seed(42)
    n = 500

    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Simulate IV rank with mean reversion
    iv_rank = np.zeros(n)
    iv_rank[0] = 50
    for i in range(1, n):
        iv_rank[i] = iv_rank[i-1] + np.random.randn() * 5 - 0.1 * (iv_rank[i-1] - 50)
    iv_rank = np.clip(iv_rank, 0, 100)

    df = pd.DataFrame({
        "iv_rank": iv_rank,
        "iv_percentile": iv_rank + np.random.randn(n) * 5,
        "iv_hv_spread": np.random.randn(n) * 3,
        "vix_close": 15 + iv_rank * 0.2 + np.random.randn(n) * 2,
        "vix_zscore": (iv_rank - 50) / 20,
        "vix_trend": np.random.randn(n) * 0.5,
        "vix_change_1d": np.random.randn(n) * 2,
        "vix_change_5d": np.random.randn(n) * 5,
        "vix_term_slope": np.random.randn(n) * 1,
        "vix_contango": (np.random.rand(n) > 0.3).astype(int),
        "hv_5": 0.15 + np.random.rand(n) * 0.1,
        "hv_10": 0.15 + np.random.rand(n) * 0.08,
        "hv_20": 0.15 + np.random.rand(n) * 0.06,
        "hv_60": 0.15 + np.random.rand(n) * 0.04,
        "hv_ratio_5_20": 1.0 + np.random.randn(n) * 0.2,
        "hv_ratio_20_60": 1.0 + np.random.randn(n) * 0.1,
        "hv_trend": np.random.randn(n) * 0.02,
        "bb_width": 0.05 + np.random.rand(n) * 0.03,
        "atr_pct": 1.0 + np.random.rand(n) * 0.5,
        "returns_5d": np.random.randn(n) * 0.02,
        "returns_10d": np.random.randn(n) * 0.03,
    }, index=dates)

    # Test classification mode
    print("\n=== Classification Mode ===")
    forecaster = VolatilityForecaster(mode="classification")
    X, y = forecaster.prepare_data(df)

    print(f"Data shape: X={X.shape}, y={len(y)}")
    print(f"Label distribution:\n{y.value_counts()}")

    metrics = forecaster.train(X, y, n_splits=3)

    print(f"\nTraining Results:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.3f}")

    # Test prediction
    sample = X.iloc[-1:].copy()
    results = forecaster.predict(sample)

    print(f"\nSample Prediction:")
    print(f"  Current IV Rank: {results[0].current_iv_rank:.1f}")
    print(f"  Predicted Regime: {results[0].predicted_regime}")
    print(f"  Probabilities: {results[0].regime_probabilities}")
    print(f"  Expected Direction: {results[0].expected_iv_direction}")

    # Test strategy recommendation
    recommendation = forecaster.get_strategy_recommendation("UP", results[0], 0.6)
    print(f"\nStrategy Recommendation (Direction=UP, conf=0.6):")
    print(f"  Strategy: {recommendation['strategy']}")
    print(f"  Reasoning: {recommendation['reasoning']}")
    print(f"  Position: {recommendation['position']}")

    # Test regression mode
    print("\n=== Regression Mode ===")
    forecaster_reg = VolatilityForecaster(mode="regression")
    X_reg, y_reg = forecaster_reg.prepare_data(df)
    metrics_reg = forecaster_reg.train(X_reg, y_reg, n_splits=3)

    print(f"\nRegression Results:")
    print(f"  RMSE: {metrics_reg['rmse']:.3f}")
    print(f"  MAE: {metrics_reg['mae']:.3f}")
    print(f"  R2: {metrics_reg['r2']:.3f}")
