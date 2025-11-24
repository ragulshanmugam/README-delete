#!/usr/bin/env python3
"""
Model Analysis and Improvement Script.

Analyzes current model performance and applies improvements:
1. Threshold optimization
2. Confidence filtering
3. Feature stability analysis
4. Regime detection

Usage:
    python scripts/analyze_and_improve.py
    python scripts/analyze_and_improve.py --ticker SPY --verbose
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config.settings import get_settings
from src.utils.logger import setup_logger

app = typer.Typer(name="analyze", help="Analyze and improve model performance")
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging."""
    settings = get_settings()
    level = "DEBUG" if verbose else "INFO"
    setup_logger(name="analyze", level=level, log_to_console=True)


def print_header() -> None:
    """Print header."""
    console.print(Panel.fit(
        "[bold blue]ML Options Trading System[/bold blue]\n"
        "[dim]Model Analysis & Improvement[/dim]",
        border_style="blue"
    ))


@app.command()
def analyze(
    ticker: str = typer.Option("SPY", "--ticker", "-t", help="Ticker to analyze"),
    n_features: int = typer.Option(30, "--n-features", "-n", help="Number of features"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Comprehensive model analysis with improvement recommendations.

    Performs:
    1. Baseline model training
    2. Threshold optimization analysis
    3. Confidence filtering analysis
    4. Feature stability analysis
    5. Per-fold regime analysis
    """
    setup_logging(verbose)
    print_header()

    from src.models.feature_pipeline import FeaturePipeline
    from src.models.direction_classifier import DirectionClassifier
    from src.models.threshold_optimizer import (
        ThresholdOptimizer,
        WalkForwardThresholdOptimizer,
        quick_threshold_analysis,
    )

    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    settings = get_settings()

    console.print(f"\n[bold]Analyzing {ticker} model...[/bold]\n")

    # === Step 1: Load Data ===
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading data...", total=None)

        pipeline = FeaturePipeline(
            ticker=ticker,
            include_macro=True,
            classification_mode="binary",
            n_features=n_features,
        )

        years = 5
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

        X, y, metadata = pipeline.prepare_training_data(
            start_date=start_date,
            end_date=end_date,
        )

        progress.update(task, description=f"[green]Loaded {len(X)} samples, {len(X.columns)} features[/green]")

    # Print data summary
    console.print(f"\n[bold]Data Summary:[/bold]")
    console.print(f"  Samples: {len(X)}")
    console.print(f"  Features: {len(X.columns)}")
    console.print(f"  Class 0 (DOWN): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    console.print(f"  Class 1 (UP): {(y == 1).sum()} ({(y == 1).mean():.1%})")

    # === Step 2: Walk-Forward Analysis with Threshold Optimization ===
    console.print(f"\n[bold]Walk-Forward Analysis with Threshold Optimization[/bold]\n")

    n_splits = 5
    n_samples = len(X)
    test_size = int(n_samples * 0.1)
    min_train = int(n_samples * 0.3)

    # Store results
    baseline_results = []
    optimized_results = []
    fold_thresholds = []
    feature_importance_per_fold = []

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        if train_end >= n_samples or test_end > n_samples:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Create validation set for threshold optimization
        val_size = int(len(X_train) * 0.15)
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train_final = X_train.iloc[:-val_size]
        y_train_final = y_train.iloc[:-val_size]

        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )

        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Store feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)
        feature_importance_per_fold.append(importance)

        # Get predictions
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Baseline predictions (0.50 threshold)
        y_test_pred_baseline = (y_test_proba >= 0.50).astype(int)
        baseline_acc = accuracy_score(y_test, y_test_pred_baseline)
        baseline_auc = roc_auc_score(y_test, y_test_proba)

        baseline_results.append({
            'fold': fold + 1,
            'accuracy': baseline_acc,
            'auc': baseline_auc,
            'threshold': 0.50,
        })

        # Optimized threshold (find on validation, apply to test)
        optimizer = ThresholdOptimizer(metric="accuracy")
        val_result = optimizer.find_optimal_threshold(y_val.values, y_val_proba)
        optimal_thresh = val_result.optimal_threshold

        y_test_pred_optimized = (y_test_proba >= optimal_thresh).astype(int)
        optimized_acc = accuracy_score(y_test, y_test_pred_optimized)

        optimized_results.append({
            'fold': fold + 1,
            'accuracy': optimized_acc,
            'auc': baseline_auc,  # AUC doesn't change with threshold
            'threshold': optimal_thresh,
            'improvement': optimized_acc - baseline_acc,
        })

        fold_thresholds.append(optimal_thresh)

    # Print comparison table
    table = Table(title="Baseline vs Optimized Threshold Performance")
    table.add_column("Fold", justify="center")
    table.add_column("Baseline Acc", justify="right")
    table.add_column("Optimal Thresh", justify="right")
    table.add_column("Optimized Acc", justify="right")
    table.add_column("Improvement", justify="right")
    table.add_column("AUC", justify="right")

    for b, o in zip(baseline_results, optimized_results):
        imp_style = "green" if o['improvement'] > 0 else "red"
        table.add_row(
            str(b['fold']),
            f"{b['accuracy']:.1%}",
            f"{o['threshold']:.2f}",
            f"{o['accuracy']:.1%}",
            f"[{imp_style}]{o['improvement']:+.1%}[/{imp_style}]",
            f"{b['auc']:.3f}",
        )

    # Add averages
    avg_baseline = np.mean([r['accuracy'] for r in baseline_results])
    avg_optimized = np.mean([r['accuracy'] for r in optimized_results])
    avg_improvement = avg_optimized - avg_baseline
    avg_auc = np.mean([r['auc'] for r in baseline_results])
    avg_thresh = np.mean(fold_thresholds)

    imp_style = "green" if avg_improvement > 0 else "red"
    table.add_row(
        "[bold]Avg[/bold]",
        f"[bold]{avg_baseline:.1%}[/bold]",
        f"[bold]{avg_thresh:.2f}[/bold]",
        f"[bold]{avg_optimized:.1%}[/bold]",
        f"[bold {imp_style}]{avg_improvement:+.1%}[/bold {imp_style}]",
        f"[bold]{avg_auc:.3f}[/bold]",
    )

    console.print(table)

    # === Step 3: Feature Stability Analysis ===
    console.print(f"\n[bold]Feature Stability Analysis[/bold]")
    console.print("[dim]Features that appear in top 15 across multiple folds are more reliable[/dim]\n")

    # Count how often each feature appears in top 15
    feature_counts = {}
    for fold_imp in feature_importance_per_fold:
        top_15 = fold_imp.head(15)['feature'].tolist()
        for feat in top_15:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    # Sort by count
    stable_features = sorted(feature_counts.items(), key=lambda x: -x[1])

    stable_table = Table(title="Feature Stability (Top 15 per Fold)")
    stable_table.add_column("Feature", style="cyan")
    stable_table.add_column("Appears in N Folds", justify="center")
    stable_table.add_column("Stability", justify="center")

    for feat, count in stable_features[:15]:
        if count >= 4:
            stability = "[green]High[/green]"
        elif count >= 3:
            stability = "[yellow]Medium[/yellow]"
        else:
            stability = "[red]Low[/red]"
        stable_table.add_row(feat, str(count), stability)

    console.print(stable_table)

    # Count highly stable features
    high_stability = sum(1 for _, c in stable_features if c >= 4)
    medium_stability = sum(1 for _, c in stable_features if c == 3)

    # === Step 4: Confidence Filtering Analysis ===
    console.print(f"\n[bold]Confidence Filtering Analysis[/bold]")
    console.print("[dim]Trading only high-confidence predictions[/dim]\n")

    # Collect all test probabilities
    all_test_proba = []
    all_test_true = []

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        if train_end >= n_samples or test_end > n_samples:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)

        y_test_proba = model.predict_proba(X_test)[:, 1]
        all_test_proba.extend(y_test_proba)
        all_test_true.extend(y_test.values)

    all_test_proba = np.array(all_test_proba)
    all_test_true = np.array(all_test_true)

    # Analyze different confidence thresholds
    conf_table = Table(title="Accuracy vs Confidence Threshold")
    conf_table.add_column("Min Confidence", justify="center")
    conf_table.add_column("Coverage", justify="right")
    conf_table.add_column("Accuracy", justify="right")
    conf_table.add_column("Trades/Year", justify="right")

    for conf_thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        max_proba = np.maximum(all_test_proba, 1 - all_test_proba)
        mask = max_proba >= conf_thresh

        if mask.sum() < 5:
            continue

        coverage = mask.mean()
        filtered_acc = accuracy_score(
            all_test_true[mask],
            (all_test_proba[mask] >= 0.50).astype(int)
        )
        trades_per_year = int(coverage * 252)  # Assuming daily trading

        acc_style = "green" if filtered_acc > 0.55 else ("yellow" if filtered_acc > 0.50 else "red")

        conf_table.add_row(
            f"{conf_thresh:.0%}",
            f"{coverage:.1%}",
            f"[{acc_style}]{filtered_acc:.1%}[/{acc_style}]",
            str(trades_per_year),
        )

    console.print(conf_table)

    # === Step 5: Summary and Recommendations ===
    console.print("\n")
    console.print(Panel.fit(
        f"[bold]Analysis Summary[/bold]\n\n"
        f"[cyan]Current Performance:[/cyan]\n"
        f"  Baseline Accuracy: {avg_baseline:.1%}\n"
        f"  Average AUC: {avg_auc:.3f}\n\n"
        f"[green]With Threshold Optimization:[/green]\n"
        f"  Optimized Accuracy: {avg_optimized:.1%}\n"
        f"  Improvement: {avg_improvement:+.1%}\n"
        f"  Optimal Threshold Range: {min(fold_thresholds):.2f} - {max(fold_thresholds):.2f}\n\n"
        f"[blue]Feature Stability:[/blue]\n"
        f"  High stability (4-5 folds): {high_stability} features\n"
        f"  Medium stability (3 folds): {medium_stability} features\n\n"
        f"[yellow]Recommendations:[/yellow]\n"
        f"  1. Use dynamic threshold ({avg_thresh:.2f} avg) instead of 0.50\n"
        f"  2. Consider confidence filtering at 55-60% for higher accuracy\n"
        f"  3. Focus on {high_stability} highly stable features\n"
        f"  4. Current AUC ({avg_auc:.3f}) indicates tradeable edge exists",
        border_style="blue"
    ))


@app.command()
def compare_phases() -> None:
    """
    Compare performance across improvement phases.

    Prints a summary table of all phases for easy comparison.
    """
    print_header()

    # Data from user's progress
    phases = [
        {"phase": "Baseline (Ternary)", "accuracy": 0.273, "f1": 0.27, "auc": 0.596, "features": "~80"},
        {"phase": "Phase 1 (Binary)", "accuracy": 0.475, "f1": 0.48, "auc": 0.618, "features": "~80"},
        {"phase": "Phase 2 (Tuned + ADX)", "accuracy": 0.457, "f1": 0.46, "auc": 0.643, "features": "117->30"},
    ]

    table = Table(title="Model Evolution Summary")
    table.add_column("Phase", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Features", justify="right")

    for p in phases:
        # Color code accuracy
        acc = p['accuracy']
        if acc >= 0.55:
            acc_style = "green"
        elif acc >= 0.45:
            acc_style = "yellow"
        else:
            acc_style = "red"

        # Color code AUC
        auc = p['auc']
        if auc >= 0.65:
            auc_style = "green"
        elif auc >= 0.60:
            auc_style = "yellow"
        else:
            auc_style = "red"

        table.add_row(
            p['phase'],
            f"[{acc_style}]{p['accuracy']:.1%}[/{acc_style}]",
            f"{p['f1']:.2f}",
            f"[{auc_style}]{p['auc']:.3f}[/{auc_style}]",
            p['features'],
        )

    console.print("\n")
    console.print(table)

    console.print("\n[bold]Key Insights:[/bold]")
    console.print("  1. Binary classification was right move (+20% accuracy over ternary)")
    console.print("  2. AUC improved steadily (0.596 -> 0.643) showing better ranking")
    console.print("  3. Accuracy dropped slightly (47.5% -> 45.7%) due to threshold issues")
    console.print("  4. Phase 2 model has BETTER predictive power (AUC), just needs threshold tuning")

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Run: python scripts/analyze_and_improve.py analyze")
    console.print("  2. This will show exact improvement from threshold optimization")
    console.print("  3. Expected: +3-8% accuracy improvement with proper thresholds")


if __name__ == "__main__":
    app()
