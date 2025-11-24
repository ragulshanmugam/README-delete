#!/usr/bin/env python3
"""
Model Diagnostic Script - Identify why AUC is below 0.5.

This script runs comprehensive diagnostics to identify the root cause
of poor model performance and provides actionable recommendations.

Usage:
    python scripts/diagnose_model.py
    python scripts/diagnose_model.py --ticker SPY --n-features 30
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.models.feature_pipeline import FeaturePipeline
from src.models.robust_classifier import (
    diagnose_auc_problem,
    run_model_comparison,
    RobustDirectionClassifier,
    get_safe_features,
    LEVEL_FEATURES,
)

app = typer.Typer(name="diagnose", help="Diagnose model performance issues")
console = Console()


def print_header():
    console.print(Panel.fit(
        "[bold red]MODEL DIAGNOSTIC REPORT[/bold red]\n"
        "[dim]Identifying why AUC < 0.5[/dim]",
        border_style="red"
    ))


@app.command()
def diagnose(
    ticker: str = typer.Option("SPY", "--ticker", "-t"),
    n_features: int = typer.Option(30, "--n-features", "-n"),
    include_macro: bool = typer.Option(True, "--macro/--no-macro"),
    compare_models: bool = typer.Option(True, "--compare/--no-compare"),
):
    """Run comprehensive model diagnostics."""

    print_header()

    # ===== 1. Load Data =====
    console.print("\n[bold]1. Loading Data[/bold]")

    pipeline = FeaturePipeline(
        ticker=ticker,
        include_macro=include_macro,
        classification_mode="binary",
        n_features=n_features,
    )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    X, y, metadata = pipeline.prepare_training_data(
        start_date=start_date,
        end_date=end_date,
    )

    console.print(f"  Samples: {len(X)}")
    console.print(f"  Features: {len(X.columns)}")
    console.print(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")

    # ===== 2. Class Distribution =====
    console.print("\n[bold]2. Class Distribution[/bold]")

    class_dist = y.value_counts(normalize=True).sort_index()
    down_pct = class_dist.get(0, 0)
    up_pct = class_dist.get(1, 0)

    dist_table = Table(show_header=True)
    dist_table.add_column("Class")
    dist_table.add_column("Count")
    dist_table.add_column("Percentage")
    dist_table.add_column("Status")

    imbalance = abs(up_pct - down_pct)
    status_down = "[yellow]Minority[/yellow]" if down_pct < 0.4 else "[green]OK[/green]"
    status_up = "[yellow]Majority[/yellow]" if up_pct > 0.6 else "[green]OK[/green]"

    dist_table.add_row("DOWN (0)", str(int(down_pct * len(y))), f"{down_pct:.1%}", status_down)
    dist_table.add_row("UP (1)", str(int(up_pct * len(y))), f"{up_pct:.1%}", status_up)

    console.print(dist_table)

    if imbalance > 0.2:
        console.print(f"  [yellow]Warning: Class imbalance of {imbalance:.1%}. This biases predictions.[/yellow]")

    # ===== 3. Feature Analysis =====
    console.print("\n[bold]3. Feature Analysis[/bold]")

    # Check for level vs momentum features
    level_features_present = [f for f in X.columns if f.lower() in [l.lower() for l in LEVEL_FEATURES]]
    safe_features = get_safe_features(list(X.columns))

    feat_table = Table(show_header=True)
    feat_table.add_column("Feature Type")
    feat_table.add_column("Count")
    feat_table.add_column("Status")

    feat_table.add_row(
        "Total Features",
        str(len(X.columns)),
        ""
    )
    feat_table.add_row(
        "Level Features (risky)",
        str(len(level_features_present)),
        "[red]Remove these[/red]" if len(level_features_present) > 3 else "[green]OK[/green]"
    )
    feat_table.add_row(
        "Momentum Features (safe)",
        str(len(safe_features)),
        "[green]Keep these[/green]"
    )

    console.print(feat_table)

    if level_features_present:
        console.print(f"\n  [yellow]Level features found (these don't predict direction):[/yellow]")
        for feat in level_features_present[:10]:
            console.print(f"    - {feat}")
        if len(level_features_present) > 10:
            console.print(f"    ... and {len(level_features_present) - 10} more")

    # ===== 4. Correlation Analysis =====
    console.print("\n[bold]4. Feature-Target Correlations[/bold]")

    correlations = X.corrwith(y).abs().sort_values(ascending=False)

    corr_table = Table(show_header=True)
    corr_table.add_column("Feature")
    corr_table.add_column("Correlation")
    corr_table.add_column("Status")

    for feat, corr in correlations.head(10).items():
        if corr > 0.3:
            status = "[red]Possible Leakage![/red]"
        elif corr > 0.1:
            status = "[green]Good Signal[/green]"
        elif corr > 0.05:
            status = "[yellow]Weak Signal[/yellow]"
        else:
            status = "[dim]Very Weak[/dim]"

        corr_table.add_row(feat, f"{corr:.4f}", status)

    console.print(corr_table)

    max_corr = correlations.max()
    if max_corr > 0.3:
        console.print(f"\n  [red]Warning: {correlations.idxmax()} has {max_corr:.2f} correlation.[/red]")
        console.print(f"  [red]This may indicate data leakage - check if this feature uses future data.[/red]")
    elif max_corr < 0.05:
        console.print(f"\n  [yellow]Warning: All correlations are very weak (max={max_corr:.3f}).[/yellow]")
        console.print(f"  [yellow]The features may not contain predictive signal for this target.[/yellow]")

    # ===== 5. Baseline Model Tests =====
    console.print("\n[bold]5. Baseline Model Tests[/bold]")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test 1: Random baseline
    np.random.seed(42)
    random_pred = np.random.random(len(y))
    random_auc = roc_auc_score(y, random_pred)

    # Test 2: Majority class baseline
    majority_acc = max(y.value_counts(normalize=True))

    # Test 3: Simple logistic regression
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_aucs = cross_val_score(lr, X_scaled, y, cv=5, scoring='roc_auc')

    # Test 4: Logistic with safe features only
    X_safe = X[safe_features] if safe_features else X
    X_safe_scaled = scaler.fit_transform(X_safe)
    lr_safe_aucs = cross_val_score(lr, X_safe_scaled, y, cv=5, scoring='roc_auc')

    baseline_table = Table(show_header=True)
    baseline_table.add_column("Model")
    baseline_table.add_column("Metric")
    baseline_table.add_column("Value")
    baseline_table.add_column("Status")

    baseline_table.add_row(
        "Random",
        "AUC",
        f"{random_auc:.3f}",
        "[dim]Reference (~0.50)[/dim]"
    )
    baseline_table.add_row(
        "Majority Class",
        "Accuracy",
        f"{majority_acc:.1%}",
        "[dim]Naive baseline[/dim]"
    )
    baseline_table.add_row(
        "Logistic (all features)",
        "AUC",
        f"{lr_aucs.mean():.3f} +/- {lr_aucs.std():.3f}",
        "[green]> 0.50[/green]" if lr_aucs.mean() > 0.5 else "[red]< 0.50[/red]"
    )
    baseline_table.add_row(
        "Logistic (safe features)",
        "AUC",
        f"{lr_safe_aucs.mean():.3f} +/- {lr_safe_aucs.std():.3f}",
        "[green]> 0.50[/green]" if lr_safe_aucs.mean() > 0.5 else "[red]< 0.50[/red]"
    )

    console.print(baseline_table)

    # Interpretation
    if lr_aucs.mean() < 0.5:
        console.print("\n  [red]CRITICAL: Even simple logistic regression has AUC < 0.5[/red]")
        console.print("  [red]This means the problem is in the FEATURES, not model complexity.[/red]")
    elif lr_aucs.mean() > lr_safe_aucs.mean():
        console.print("\n  [yellow]Note: All features perform better than safe features.[/yellow]")
        console.print("  [yellow]Some 'level' features may actually be useful.[/yellow]")
    else:
        console.print("\n  [green]Good: Safe features perform comparably or better.[/green]")
        console.print("  [green]Removing level features may help reduce overfitting.[/green]")

    # ===== 6. Model Comparison (if enabled) =====
    if compare_models:
        console.print("\n[bold]6. Model Configuration Comparison[/bold]")

        comparison = run_model_comparison(X, y)

        comp_table = Table(show_header=True)
        comp_table.add_column("Model")
        comp_table.add_column("Regularization")
        comp_table.add_column("Safe Features")
        comp_table.add_column("AUC")
        comp_table.add_column("Accuracy")
        comp_table.add_column("vs Baseline")

        for _, row in comparison.iterrows():
            auc_status = "[green]" if row['auc'] > 0.5 else "[red]"
            acc_diff = row['accuracy'] - row['baseline_accuracy']
            acc_status = "[green]" if acc_diff > 0 else "[red]"

            comp_table.add_row(
                row['model_type'],
                row['regularization'],
                "Yes" if row['safe_features'] else "No",
                f"{auc_status}{row['auc']:.3f}[/{auc_status.strip('[')}",
                f"{row['accuracy']:.1%}",
                f"{acc_status}{acc_diff:+.1%}[/{acc_status.strip('[')}",
            )

        console.print(comp_table)

    # ===== 7. Recommendations =====
    console.print("\n")

    recommendations = []

    # AUC-based recommendations
    if lr_aucs.mean() < 0.5:
        recommendations.append(
            "[red]CRITICAL[/red]: Your features are inversely correlated with the target. "
            "Review feature engineering - you may be inadvertently encoding future information "
            "or using features that mean-revert."
        )

    # Feature-based recommendations
    if len(level_features_present) > 5:
        recommendations.append(
            f"[yellow]Remove {len(level_features_present)} level features[/yellow] "
            "(like ema_26, sma_200, bb_lower). These describe WHERE price is, "
            "not WHERE it's going. Use momentum features instead."
        )

    # Imbalance recommendations
    if imbalance > 0.2:
        recommendations.append(
            f"[yellow]Address class imbalance[/yellow] ({imbalance:.1%}). "
            "Your model may be biased toward predicting UP. "
            "Use class_weight='balanced' or adjust the threshold."
        )

    # Overfitting recommendations
    if compare_models and comparison.iloc[0]['auc'] > 0.5:
        best = comparison.iloc[0]
        recommendations.append(
            f"[green]Best configuration[/green]: {best['model_type']} with {best['regularization']} "
            f"regularization, safe_features={best['safe_features']}. AUC={best['auc']:.3f}"
        )

    # General recommendations
    recommendations.append(
        "[blue]Try logistic regression first[/blue]. If a simple model can't beat random (AUC > 0.5), "
        "a complex model like XGBoost will just overfit more."
    )

    recommendations.append(
        "[blue]Focus on feature engineering[/blue]. Create features that measure CHANGES and MOMENTUM, "
        "not absolute levels. Examples: returns_5d, rsi_14, price_vs_sma20."
    )

    console.print(Panel.fit(
        "[bold]RECOMMENDATIONS[/bold]\n\n" +
        "\n\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)]),
        border_style="blue"
    ))

    # ===== 8. Quick Fix =====
    console.print("\n[bold]Quick Fix Commands[/bold]")
    console.print("""
    # 1. Test the robust classifier with safe features:
    python -c "
from src.models.robust_classifier import RobustDirectionClassifier, run_model_comparison
from src.models.feature_pipeline import FeaturePipeline
from datetime import datetime, timedelta

pipeline = FeaturePipeline(ticker='SPY', include_macro=True, classification_mode='binary', n_features=30)
X, y, _ = pipeline.prepare_training_data()

# Run comparison
comparison = run_model_comparison(X, y)
print(comparison)
"

    # 2. Train with recommended settings:
    python -c "
from src.models.robust_classifier import RobustDirectionClassifier
from src.models.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline(ticker='SPY', include_macro=True, classification_mode='binary', n_features=30)
X, y, _ = pipeline.prepare_training_data()

# Use logistic regression with strong regularization and safe features
clf = RobustDirectionClassifier(
    model_type='logistic',
    regularization='strong',
    use_safe_features=True,
)
result = clf.train_walk_forward(X, y)
print(f'AUC: {result.auc:.3f}, Accuracy: {result.accuracy:.1%}')
"
    """)


@app.command()
def quick_test():
    """Quick test of the robust classifier."""

    from src.models.robust_classifier import RobustDirectionClassifier

    console.print("[bold]Quick Test - Robust Classifier[/bold]\n")

    pipeline = FeaturePipeline(
        ticker="SPY",
        include_macro=True,
        classification_mode="binary",
        n_features=30,
    )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    X, y, metadata = pipeline.prepare_training_data(
        start_date=start_date,
        end_date=end_date,
    )

    # Test logistic with safe features
    clf = RobustDirectionClassifier(
        model_type="logistic",
        regularization="strong",
        use_safe_features=True,
    )

    result = clf.train_walk_forward(X, y)

    console.print(f"Model: Logistic Regression (strong regularization, safe features)")
    console.print(f"AUC: {result.auc:.3f}")
    console.print(f"Accuracy: {result.accuracy:.1%} (baseline: {result.baseline_accuracy:.1%})")
    console.print(f"F1: {result.f1:.3f}")

    if result.auc > 0.5:
        console.print("\n[green]SUCCESS: AUC > 0.50. The model has some predictive power.[/green]")
    else:
        console.print("\n[red]ISSUE: AUC still < 0.50. More investigation needed.[/red]")


if __name__ == "__main__":
    app()
