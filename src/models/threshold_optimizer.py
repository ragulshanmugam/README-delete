"""
Threshold optimization for Direction Classifier.

Finds optimal decision thresholds to maximize accuracy or other metrics,
instead of using the default 0.50 probability cutoff.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdResult:
    """Results from threshold optimization."""
    optimal_threshold: float
    accuracy_at_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    coverage: float  # % of samples above confidence threshold
    baseline_accuracy: float  # accuracy at 0.50 threshold


class ThresholdOptimizer:
    """
    Optimizes decision threshold for binary classification.

    Instead of predicting class 1 when P(class=1) > 0.50,
    finds the optimal threshold that maximizes accuracy or F1.
    """

    def __init__(self, metric: str = "accuracy"):
        """
        Initialize optimizer.

        Args:
            metric: Metric to optimize ("accuracy", "f1", "precision", "recall")
        """
        self.metric = metric
        self.optimal_threshold: Optional[float] = None
        self.threshold_results: Optional[pd.DataFrame] = None

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> ThresholdResult:
        """
        Find optimal threshold for classification.

        Args:
            y_true: True labels (0 or 1)
            y_proba: Predicted probabilities for class 1
            thresholds: Thresholds to evaluate (default: 0.30 to 0.70)

        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        if thresholds is None:
            thresholds = np.arange(0.30, 0.71, 0.01)

        results = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            # Skip if all predictions are same class
            if len(np.unique(y_pred)) < 2:
                continue

            results.append({
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "coverage": 1.0,  # All samples included
            })

        results_df = pd.DataFrame(results)
        self.threshold_results = results_df

        # Handle edge case: no valid thresholds found
        if len(results_df) == 0:
            logger.warning("No valid thresholds found, using default 0.50")
            baseline_pred = (y_proba >= 0.50).astype(int)
            baseline_acc = accuracy_score(y_true, baseline_pred)
            return ThresholdResult(
                optimal_threshold=0.50,
                accuracy_at_threshold=baseline_acc,
                precision_at_threshold=precision_score(y_true, baseline_pred, zero_division=0),
                recall_at_threshold=recall_score(y_true, baseline_pred, zero_division=0),
                f1_at_threshold=f1_score(y_true, baseline_pred, zero_division=0),
                coverage=1.0,
                baseline_accuracy=baseline_acc,
            )

        # Find optimal threshold
        if self.metric == "accuracy":
            best_idx = results_df["accuracy"].idxmax()
        elif self.metric == "f1":
            best_idx = results_df["f1"].idxmax()
        elif self.metric == "precision":
            best_idx = results_df["precision"].idxmax()
        elif self.metric == "recall":
            best_idx = results_df["recall"].idxmax()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        best_row = results_df.loc[best_idx]
        self.optimal_threshold = best_row["threshold"]

        # Calculate baseline (0.50 threshold)
        baseline_pred = (y_proba >= 0.50).astype(int)
        baseline_acc = accuracy_score(y_true, baseline_pred)

        result = ThresholdResult(
            optimal_threshold=best_row["threshold"],
            accuracy_at_threshold=best_row["accuracy"],
            precision_at_threshold=best_row["precision"],
            recall_at_threshold=best_row["recall"],
            f1_at_threshold=best_row["f1"],
            coverage=1.0,
            baseline_accuracy=baseline_acc,
        )

        logger.info(
            f"Optimal threshold: {result.optimal_threshold:.2f} "
            f"(accuracy: {result.accuracy_at_threshold:.3f} vs "
            f"baseline: {result.baseline_accuracy:.3f}, "
            f"improvement: {result.accuracy_at_threshold - result.baseline_accuracy:+.3f})"
        )

        return result

    def find_confidence_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        min_confidence: float = 0.55,
        max_confidence: float = 0.75,
    ) -> Tuple[ThresholdResult, float]:
        """
        Find optimal confidence threshold that filters low-confidence predictions.

        Only predicts when max(P(class=0), P(class=1)) > confidence_threshold.
        This reduces coverage but increases accuracy on traded predictions.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for class 1
            min_confidence: Minimum confidence to consider
            max_confidence: Maximum confidence to consider

        Returns:
            Tuple of (ThresholdResult, confidence_threshold)
        """
        confidence_thresholds = np.arange(min_confidence, max_confidence + 0.01, 0.02)

        best_result = None
        best_confidence = None
        best_score = 0

        for conf_thresh in confidence_thresholds:
            # Max probability for each sample
            max_proba = np.maximum(y_proba, 1 - y_proba)

            # Filter to high-confidence predictions
            mask = max_proba >= conf_thresh

            if mask.sum() < 10:  # Need minimum samples
                continue

            y_true_filtered = y_true[mask]
            y_proba_filtered = y_proba[mask]

            # Find optimal threshold for filtered samples
            result = self.find_optimal_threshold(
                y_true_filtered,
                y_proba_filtered
            )
            result.coverage = mask.mean()

            # Score: balance accuracy and coverage
            score = result.accuracy_at_threshold * np.sqrt(result.coverage)

            if score > best_score:
                best_score = score
                best_result = result
                best_confidence = conf_thresh

        if best_result is None:
            # Fall back to no filtering
            return self.find_optimal_threshold(y_true, y_proba), 0.50

        logger.info(
            f"Best confidence threshold: {best_confidence:.2f} "
            f"(accuracy: {best_result.accuracy_at_threshold:.3f}, "
            f"coverage: {best_result.coverage:.1%})"
        )

        return best_result, best_confidence


class WalkForwardThresholdOptimizer:
    """
    Threshold optimizer that works with walk-forward validation.

    For each fold, finds optimal threshold on validation data,
    then applies it to test data.
    """

    def __init__(self):
        self.fold_thresholds: List[float] = []
        self.fold_results: List[ThresholdResult] = []

    def optimize_fold(
        self,
        y_val_true: np.ndarray,
        y_val_proba: np.ndarray,
        y_test_true: np.ndarray,
        y_test_proba: np.ndarray,
    ) -> Tuple[np.ndarray, ThresholdResult]:
        """
        Find optimal threshold on validation, apply to test.

        Args:
            y_val_true: Validation true labels
            y_val_proba: Validation predicted probabilities
            y_test_true: Test true labels
            y_test_proba: Test predicted probabilities

        Returns:
            Tuple of (test predictions, test result)
        """
        # Find optimal threshold on validation
        optimizer = ThresholdOptimizer(metric="accuracy")
        val_result = optimizer.find_optimal_threshold(y_val_true, y_val_proba)

        optimal_thresh = val_result.optimal_threshold
        self.fold_thresholds.append(optimal_thresh)

        # Apply to test
        y_test_pred = (y_test_proba >= optimal_thresh).astype(int)

        test_result = ThresholdResult(
            optimal_threshold=optimal_thresh,
            accuracy_at_threshold=accuracy_score(y_test_true, y_test_pred),
            precision_at_threshold=precision_score(y_test_true, y_test_pred, zero_division=0),
            recall_at_threshold=recall_score(y_test_true, y_test_pred, zero_division=0),
            f1_at_threshold=f1_score(y_test_true, y_test_pred, zero_division=0),
            coverage=1.0,
            baseline_accuracy=accuracy_score(y_test_true, (y_test_proba >= 0.50).astype(int)),
        )

        self.fold_results.append(test_result)

        return y_test_pred, test_result

    def get_summary(self) -> Dict:
        """Get summary of all folds."""
        if not self.fold_results:
            return {}

        return {
            "avg_threshold": np.mean(self.fold_thresholds),
            "threshold_std": np.std(self.fold_thresholds),
            "avg_accuracy_optimized": np.mean([r.accuracy_at_threshold for r in self.fold_results]),
            "avg_accuracy_baseline": np.mean([r.baseline_accuracy for r in self.fold_results]),
            "avg_improvement": np.mean([
                r.accuracy_at_threshold - r.baseline_accuracy
                for r in self.fold_results
            ]),
            "per_fold": [
                {
                    "threshold": t,
                    "accuracy_optimized": r.accuracy_at_threshold,
                    "accuracy_baseline": r.baseline_accuracy,
                    "improvement": r.accuracy_at_threshold - r.baseline_accuracy,
                }
                for t, r in zip(self.fold_thresholds, self.fold_results)
            ],
        }


def quick_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict:
    """
    Quick analysis of threshold optimization potential.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for class 1

    Returns:
        Dictionary with analysis results
    """
    optimizer = ThresholdOptimizer()

    # Standard threshold optimization
    std_result = optimizer.find_optimal_threshold(y_true, y_proba)

    # Confidence-filtered optimization
    conf_result, conf_thresh = optimizer.find_confidence_threshold(y_true, y_proba)

    return {
        "baseline_accuracy": std_result.baseline_accuracy,
        "optimal_threshold": std_result.optimal_threshold,
        "optimized_accuracy": std_result.accuracy_at_threshold,
        "improvement": std_result.accuracy_at_threshold - std_result.baseline_accuracy,
        "confidence_threshold": conf_thresh,
        "confidence_filtered_accuracy": conf_result.accuracy_at_threshold,
        "confidence_coverage": conf_result.coverage,
        "threshold_curve": optimizer.threshold_results.to_dict("records") if optimizer.threshold_results is not None else [],
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Generate probabilities with some signal
    n = 500
    true_proba = np.random.uniform(0.3, 0.7, n)
    y_true = (np.random.random(n) < true_proba).astype(int)

    # Add noise to probabilities
    y_proba = true_proba + np.random.normal(0, 0.15, n)
    y_proba = np.clip(y_proba, 0.01, 0.99)

    # Shift probabilities to simulate miscalibration
    y_proba = y_proba * 0.8 + 0.1  # Compress toward 0.5

    print("Testing ThresholdOptimizer...")
    print(f"Sample size: {n}")
    print(f"True class distribution: {np.mean(y_true):.2%} positive")
    print()

    results = quick_threshold_analysis(y_true, y_proba)

    print(f"Baseline accuracy (0.50 threshold): {results['baseline_accuracy']:.3f}")
    print(f"Optimal threshold: {results['optimal_threshold']:.2f}")
    print(f"Optimized accuracy: {results['optimized_accuracy']:.3f}")
    print(f"Improvement: {results['improvement']:+.3f}")
    print()
    print(f"With confidence filtering:")
    print(f"  Confidence threshold: {results['confidence_threshold']:.2f}")
    print(f"  Accuracy: {results['confidence_filtered_accuracy']:.3f}")
    print(f"  Coverage: {results['confidence_coverage']:.1%}")
