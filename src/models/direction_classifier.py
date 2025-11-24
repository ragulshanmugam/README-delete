"""
Direction classifier for predicting 5-day market direction.

Uses XGBoost to predict 3 classes:
- Class 0: BEARISH (5-day return < -1%)
- Class 1: NEUTRAL (5-day return between -1% and +1%)
- Class 2: BULLISH (5-day return > +1%)

Features:
- Walk-forward validation (time series aware, no data leakage)
- MLflow experiment tracking
- Feature importance tracking
- Class imbalance handling
- Confidence scores for predictions
- Model serialization
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardSplit:
    """Represents a single walk-forward validation split."""

    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int


@dataclass
class ClassificationResult:
    """Results from a single prediction."""

    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for the classifier."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc_ovr: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    fold_metrics: Optional[List[Dict[str, float]]] = None


class DirectionClassifier:
    """
    XGBoost-based 3-class direction classifier.

    Predicts whether the market will be BEARISH, NEUTRAL, or BULLISH
    over a specified prediction horizon (default: 5 days).

    Attributes:
        model: Trained XGBoost model
        feature_names: List of feature names used for training
        prediction_horizon: Days ahead to predict
        direction_threshold: Return threshold for classification
        class_names: Names for the 3 classes
        feature_importance: DataFrame with feature importance scores
        training_metadata: Metadata from training run
    """

    # Class name mappings for different modes
    CLASS_NAMES_BINARY = {0: "DOWN", 1: "UP"}
    CLASS_NAMES_TERNARY = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
    CLASS_LABELS_BINARY = {"DOWN": 0, "UP": 1}
    CLASS_LABELS_TERNARY = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}

    def __init__(
        self,
        prediction_horizon: Optional[int] = None,
        direction_threshold: Optional[float] = None,
        random_state: int = 42,
        num_classes: int = 2,
    ):
        """
        Initialize the direction classifier.

        Args:
            prediction_horizon: Days ahead to predict (default from settings: 5)
            direction_threshold: Return threshold (default from settings: 0.01)
            random_state: Random seed for reproducibility
            num_classes: Number of classes (2 for binary, 3 for ternary)
        """
        settings = get_settings()
        self.prediction_horizon = prediction_horizon or settings.prediction_horizon
        self.direction_threshold = direction_threshold or settings.direction_threshold
        self.random_state = random_state
        self.num_classes = num_classes

        # Set class names based on mode
        if num_classes == 2:
            self.CLASS_NAMES = self.CLASS_NAMES_BINARY
            self.CLASS_LABELS = self.CLASS_LABELS_BINARY
        else:
            self.CLASS_NAMES = self.CLASS_NAMES_TERNARY
            self.CLASS_LABELS = self.CLASS_LABELS_TERNARY

        # Model state
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.training_metadata: Dict[str, Any] = {}

        # XGBoost parameters
        self._params = self._get_default_params()

        logger.info(
            f"Initialized DirectionClassifier: "
            f"horizon={self.prediction_horizon}d, "
            f"threshold={self.direction_threshold:.1%}, "
            f"classes={num_classes}"
        )

    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default XGBoost parameters optimized for classification.

        Returns:
            Dictionary of XGBoost parameters
        """
        if self.num_classes == 2:
            # Binary classification
            return {
                "objective": "binary:logistic",
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": 1,
                "random_state": self.random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
                "early_stopping_rounds": 20,
            }
        else:
            # Multi-class classification
            return {
                "objective": "multi:softprob",
                "num_class": self.num_classes,
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": 1,
                "random_state": self.random_state,
                "n_jobs": -1,
                "eval_metric": "mlogloss",
                "early_stopping_rounds": 20,
            }

    def set_params(self, **params: Any) -> "DirectionClassifier":
        """
        Set XGBoost parameters.

        Args:
            **params: XGBoost parameters to set

        Returns:
            Self for method chaining
        """
        self._params.update(params)
        logger.info(f"Updated parameters: {params}")
        return self

    def _compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Args:
            y: Labels series

        Returns:
            Dictionary mapping class to weight
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y.values
        )
        return dict(zip(classes, weights))

    def _create_walk_forward_splits(
        self,
        X: pd.DataFrame,
        n_splits: int = 5,
        test_size: float = 0.1,
        min_train_size: float = 0.3,
    ) -> List[Tuple[np.ndarray, np.ndarray, WalkForwardSplit]]:
        """
        Create walk-forward validation splits (time series aware).

        This ensures no data leakage by always training on past data
        and testing on future data.

        Args:
            X: Feature DataFrame with datetime index
            n_splits: Number of validation splits
            test_size: Fraction of data for each test set
            min_train_size: Minimum fraction of data for training

        Returns:
            List of (train_indices, test_indices, split_info) tuples
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * test_size)
        min_train_samples = int(n_samples * min_train_size)

        splits = []

        for i in range(n_splits):
            # Calculate split points
            # Each fold moves forward in time
            train_end_idx = min_train_samples + i * test_size_samples
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size_samples, n_samples)

            # Ensure we have enough data
            if train_end_idx >= n_samples or test_end_idx > n_samples:
                logger.warning(f"Skipping fold {i+1}: insufficient data")
                continue

            train_idx = np.arange(0, train_end_idx)
            test_idx = np.arange(test_start_idx, test_end_idx)

            # Create split info
            split_info = WalkForwardSplit(
                fold=i + 1,
                train_start=X.index[0],
                train_end=X.index[train_end_idx - 1],
                test_start=X.index[test_start_idx],
                test_end=X.index[test_end_idx - 1],
                train_size=len(train_idx),
                test_size=len(test_idx),
            )

            splits.append((train_idx, test_idx, split_info))

            logger.debug(
                f"Fold {i+1}: train={len(train_idx)} samples "
                f"({split_info.train_start.date()} to {split_info.train_end.date()}), "
                f"test={len(test_idx)} samples "
                f"({split_info.test_start.date()} to {split_info.test_end.date()})"
            )

        return splits

    def _train_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train model on a single fold.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Optional sample weights

        Returns:
            Tuple of (trained model, metrics dict)
        """
        import xgboost as xgb
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Extract early stopping rounds from params
        params = self._params.copy()
        early_stopping = params.pop("early_stopping_rounds", 20)

        # Create model
        model = xgb.XGBClassifier(**params)

        # Fit with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=False,
        )

        # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
        }

        # AUC calculation (different for binary vs multi-class)
        try:
            if self.num_classes == 2:
                # Binary: use probability of positive class
                metrics["auc_ovr"] = roc_auc_score(y_val, y_pred_proba[:, 1])
            else:
                # Multi-class: one-vs-rest
                metrics["auc_ovr"] = roc_auc_score(
                    y_val, y_pred_proba, multi_class="ovr", average="macro"
                )
        except ValueError:
            # Can happen if only one class in validation set
            metrics["auc_ovr"] = 0.5

        return model, metrics

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        handle_imbalance: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> EvaluationMetrics:
        """
        Train the classifier using walk-forward validation.

        Args:
            X: Feature DataFrame with datetime index
            y: Labels Series (0, 1, 2)
            n_splits: Number of walk-forward splits
            handle_imbalance: Whether to use class weights for imbalanced data
            params: Optional XGBoost parameters to override defaults

        Returns:
            EvaluationMetrics with overall and per-fold metrics
        """
        import xgboost as xgb
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        logger.info(f"Training with {n_splits}-fold walk-forward validation")

        # Update params if provided
        if params:
            self._params.update(params)

        # Store feature names
        self.feature_names = list(X.columns)

        # Compute class weights if needed
        class_weights = None
        if handle_imbalance:
            class_weights = self._compute_class_weights(y)
            logger.info(f"Class weights: {class_weights}")

        # Create walk-forward splits
        splits = self._create_walk_forward_splits(X, n_splits=n_splits)

        if not splits:
            raise ValueError("Could not create any valid splits. Check data size.")

        # Train on each fold
        fold_metrics: List[Dict[str, float]] = []
        all_test_preds = []
        all_test_true = []
        all_test_proba = []

        for train_idx, test_idx, split_info in splits:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Compute sample weights if using class weights
            sample_weights = None
            if class_weights:
                sample_weights = np.array([class_weights[int(c)] for c in y_train])

            # Train fold
            model, metrics = self._train_single_fold(
                X_train, y_train, X_test, y_test, sample_weights
            )

            fold_metrics.append(metrics)
            all_test_preds.extend(model.predict(X_test))
            all_test_true.extend(y_test.values)
            all_test_proba.extend(model.predict_proba(X_test))

            logger.info(
                f"Fold {split_info.fold}: accuracy={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1_macro']:.3f}, auc={metrics['auc_ovr']:.3f}"
            )

        # Train final model on all data
        logger.info("Training final model on full dataset")

        sample_weights = None
        if class_weights:
            sample_weights = np.array([class_weights[int(c)] for c in y])

        # Use last 20% as validation for early stopping
        val_split = int(len(X) * 0.8)
        X_train_final = X.iloc[:val_split]
        y_train_final = y.iloc[:val_split]
        X_val_final = X.iloc[val_split:]
        y_val_final = y.iloc[val_split:]
        weights_train = sample_weights[:val_split] if sample_weights is not None else None

        self.model, _ = self._train_single_fold(
            X_train_final, y_train_final, X_val_final, y_val_final, weights_train
        )

        # Store feature importance
        self.feature_importance = self._compute_feature_importance()

        # Calculate overall metrics
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)
        all_test_proba = np.array(all_test_proba)

        # Per-class metrics
        per_class_metrics = {}
        for class_idx, class_name in self.CLASS_NAMES.items():
            binary_true = (all_test_true == class_idx).astype(int)
            binary_pred = (all_test_preds == class_idx).astype(int)

            per_class_metrics[class_name] = {
                "precision": precision_score(binary_true, binary_pred, zero_division=0),
                "recall": recall_score(binary_true, binary_pred, zero_division=0),
                "f1": f1_score(binary_true, binary_pred, zero_division=0),
                "support": int(binary_true.sum()),
            }

        # Calculate AUC (different for binary vs multi-class)
        try:
            if self.num_classes == 2:
                # Binary: use probability of positive class
                auc_score = roc_auc_score(all_test_true, all_test_proba[:, 1])
            else:
                # Multi-class: one-vs-rest
                auc_score = roc_auc_score(all_test_true, all_test_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc_score = 0.5

        # Overall metrics
        metrics = EvaluationMetrics(
            accuracy=accuracy_score(all_test_true, all_test_preds),
            precision_macro=precision_score(all_test_true, all_test_preds, average="macro", zero_division=0),
            recall_macro=recall_score(all_test_true, all_test_preds, average="macro", zero_division=0),
            f1_macro=f1_score(all_test_true, all_test_preds, average="macro", zero_division=0),
            auc_ovr=auc_score,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix(all_test_true, all_test_preds),
            fold_metrics=fold_metrics,
        )

        # Store training metadata
        self.training_metadata = {
            "n_splits": n_splits,
            "handle_imbalance": handle_imbalance,
            "class_weights": class_weights,
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "date_range": {
                "start": str(X.index.min()),
                "end": str(X.index.max()),
            },
            "label_distribution": y.value_counts().to_dict(),
            "params": self._params,
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Training complete: accuracy={metrics.accuracy:.3f}, "
            f"f1={metrics.f1_macro:.3f}, auc={metrics.auc_ovr:.3f}"
        )

        return metrics

    def train_with_mlflow(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        experiment_name: str = "direction_classifier",
        run_name: Optional[str] = None,
        n_splits: int = 5,
        handle_imbalance: bool = True,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Tuple[EvaluationMetrics, str]:
        """
        Train the classifier with MLflow experiment tracking.

        Args:
            X: Feature DataFrame
            y: Labels Series
            experiment_name: MLflow experiment name
            run_name: Optional run name (auto-generated if None)
            n_splits: Number of walk-forward splits
            handle_imbalance: Whether to use class weights
            params: Optional XGBoost parameters
            tags: Optional tags for the run

        Returns:
            Tuple of (EvaluationMetrics, run_id)
        """
        import mlflow
        import mlflow.xgboost

        settings = get_settings()

        # Set tracking URI
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Generate run name if not provided
        if run_name is None:
            ticker = tags.get("ticker", "unknown") if tags else "unknown"
            run_name = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log parameters
            mlflow.log_params({
                "prediction_horizon": self.prediction_horizon,
                "direction_threshold": self.direction_threshold,
                "n_splits": n_splits,
                "handle_imbalance": handle_imbalance,
                "n_samples": len(X),
                "n_features": len(X.columns),
            })

            # Log XGBoost params
            log_params = self._params.copy()
            if params:
                log_params.update(params)
            mlflow.log_params({f"xgb_{k}": v for k, v in log_params.items()})

            # Train model
            metrics = self.train(
                X, y,
                n_splits=n_splits,
                handle_imbalance=handle_imbalance,
                params=params,
            )

            # Log metrics
            mlflow.log_metrics({
                "accuracy": metrics.accuracy,
                "precision_macro": metrics.precision_macro,
                "recall_macro": metrics.recall_macro,
                "f1_macro": metrics.f1_macro,
                "auc_ovr": metrics.auc_ovr,
            })

            # Log per-class metrics
            for class_name, class_metrics in metrics.per_class_metrics.items():
                for metric_name, value in class_metrics.items():
                    mlflow.log_metric(f"{class_name.lower()}_{metric_name}", value)

            # Log fold metrics
            if metrics.fold_metrics:
                for i, fold_metric in enumerate(metrics.fold_metrics):
                    for metric_name, value in fold_metric.items():
                        mlflow.log_metric(f"fold{i+1}_{metric_name}", value)

            # Log feature importance
            if self.feature_importance is not None:
                # Save as artifact
                importance_path = f"/tmp/feature_importance_{run_id}.csv"
                self.feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)

                # Log top features
                top_features = self.feature_importance.head(10)
                for _, row in top_features.iterrows():
                    mlflow.log_metric(f"importance_{row['feature']}", row["importance"])

            # Log model
            mlflow.xgboost.log_model(self.model, "model")

            # Log confusion matrix as artifact
            cm_path = f"/tmp/confusion_matrix_{run_id}.json"
            with open(cm_path, "w") as f:
                json.dump({
                    "matrix": metrics.confusion_matrix.tolist(),
                    "labels": list(self.CLASS_NAMES.values()),
                }, f)
            mlflow.log_artifact(cm_path)

            logger.info(f"MLflow run completed: {run_id}")

        return metrics, run_id

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict direction class.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predicted classes (0=BEARISH, 1=NEUTRAL, 2=BULLISH)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure feature order matches training
        X_aligned = X[self.feature_names]

        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 3) with probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_aligned = X[self.feature_names]

        return self.model.predict_proba(X_aligned)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.5,
    ) -> List[ClassificationResult]:
        """
        Predict with confidence scores and class names.

        Args:
            X: Feature DataFrame
            confidence_threshold: Minimum confidence for a prediction

        Returns:
            List of ClassificationResult objects
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_aligned = X[self.feature_names]

        predictions = self.model.predict(X_aligned)
        probabilities = self.model.predict_proba(X_aligned)

        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = probs[pred]
            result = ClassificationResult(
                predicted_class=int(pred),
                class_name=self.CLASS_NAMES[pred],
                confidence=float(confidence),
                probabilities={
                    self.CLASS_NAMES[j]: float(p) for j, p in enumerate(probs)
                }
            )
            results.append(result)

        return results

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> EvaluationMetrics:
        """
        Evaluate the model on a test set.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            EvaluationMetrics object
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Predict
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Per-class metrics
        per_class_metrics = {}
        for class_idx, class_name in self.CLASS_NAMES.items():
            binary_true = (y == class_idx).astype(int)
            binary_pred = (y_pred == class_idx).astype(int)

            per_class_metrics[class_name] = {
                "precision": precision_score(binary_true, binary_pred, zero_division=0),
                "recall": recall_score(binary_true, binary_pred, zero_division=0),
                "f1": f1_score(binary_true, binary_pred, zero_division=0),
                "support": int(binary_true.sum()),
            }

        # Calculate AUC (different for binary vs multi-class)
        try:
            if self.num_classes == 2:
                auc_score = roc_auc_score(y, y_proba[:, 1])
            else:
                auc_score = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc_score = 0.5

        # Overall metrics
        return EvaluationMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision_macro=precision_score(y, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y, y_pred, average="macro", zero_division=0),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            auc_ovr=auc_score,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix(y, y_pred),
        )

    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute feature importance from the trained model.

        Returns:
            DataFrame with features sorted by importance
        """
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importances_

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # Add rank
        df["rank"] = range(1, len(df) + 1)

        return df

    def get_feature_importance(
        self,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Return only top N features (optional)

        Returns:
            DataFrame with features sorted by importance
        """
        if self.feature_importance is None:
            if self.model is not None:
                self.feature_importance = self._compute_feature_importance()
            else:
                raise ValueError("Model not trained.")

        if top_n:
            return self.feature_importance.head(top_n)
        return self.feature_importance

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save the model and metadata to disk.

        Args:
            path: Path to save the model (directory or file)

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        path = Path(path)

        # If path is a directory, create filename
        if path.is_dir():
            filename = f"direction_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            path = path / filename

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        save_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "prediction_horizon": self.prediction_horizon,
            "direction_threshold": self.direction_threshold,
            "training_metadata": self.training_metadata,
            "params": self._params,
            "class_names": self.CLASS_NAMES,
            "version": "2.0",
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(save_data, path)
        logger.info(f"Model saved to {path}")

        return path

    def load(self, path: Union[str, Path]) -> "DirectionClassifier":
        """
        Load a model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Self for method chaining
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)

        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importance = data.get("feature_importance")
        self.prediction_horizon = data["prediction_horizon"]
        self.direction_threshold = data["direction_threshold"]
        self.training_metadata = data.get("training_metadata", {})
        self._params = data.get("params", self._get_default_params())

        logger.info(f"Model loaded from {path}")
        return self

    @classmethod
    def load_from_path(cls, path: Union[str, Path]) -> "DirectionClassifier":
        """
        Class method to load a model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded DirectionClassifier instance
        """
        classifier = cls()
        classifier.load(path)
        return classifier

    def summary(self) -> str:
        """
        Get a human-readable summary of the model.

        Returns:
            Summary string
        """
        lines = [
            "=" * 50,
            "Direction Classifier Summary",
            "=" * 50,
            f"Prediction Horizon: {self.prediction_horizon} days",
            f"Direction Threshold: {self.direction_threshold:.1%}",
            f"Model Trained: {'Yes' if self.model else 'No'}",
        ]

        if self.model:
            lines.extend([
                f"Number of Features: {len(self.feature_names)}",
            ])

            if self.training_metadata:
                meta = self.training_metadata
                lines.extend([
                    f"Training Samples: {meta.get('n_samples', 'N/A')}",
                    f"Date Range: {meta.get('date_range', {}).get('start', 'N/A')} to "
                    f"{meta.get('date_range', {}).get('end', 'N/A')}",
                    f"Trained At: {meta.get('trained_at', 'N/A')}",
                ])

            if self.feature_importance is not None:
                top_features = self.feature_importance.head(5)
                lines.append("\nTop 5 Features:")
                for _, row in top_features.iterrows():
                    lines.append(f"  {row['rank']}. {row['feature']}: {row['importance']:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing DirectionClassifier...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(500, n_features),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create somewhat predictable labels
    returns = X["feature_0"].rolling(5).mean() + np.random.randn(500) * 0.5
    y = pd.Series(index=dates, dtype=int)
    y[returns < -0.3] = 0  # BEARISH
    y[returns > 0.3] = 2   # BULLISH
    y[(returns >= -0.3) & (returns <= 0.3)] = 1  # NEUTRAL

    # Drop NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    # Train classifier
    classifier = DirectionClassifier()
    metrics = classifier.train(X, y, n_splits=3)

    print(f"\nMetrics:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  F1 Macro: {metrics.f1_macro:.3f}")
    print(f"  AUC OvR: {metrics.auc_ovr:.3f}")

    print(f"\nPer-class metrics:")
    for class_name, class_metrics in metrics.per_class_metrics.items():
        print(f"  {class_name}: precision={class_metrics['precision']:.3f}, "
              f"recall={class_metrics['recall']:.3f}, f1={class_metrics['f1']:.3f}")

    print(f"\nConfusion Matrix:\n{metrics.confusion_matrix}")

    # Test predictions
    sample = X.iloc[-5:]
    preds = classifier.predict_with_confidence(sample)
    print(f"\nSample predictions:")
    for i, pred in enumerate(preds):
        print(f"  {i+1}. {pred.class_name} (confidence: {pred.confidence:.2f})")

    # Show top features
    print(f"\nTop 5 features:")
    print(classifier.get_feature_importance(top_n=5))

    # Print summary
    print(f"\n{classifier.summary()}")
