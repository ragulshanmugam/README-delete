#!/usr/bin/env python3
"""
Model training CLI script.

Trains direction classifiers on historical data with MLflow tracking.

Supports two model types:
- logistic: Robust logistic regression with regularization (recommended, AUC ~0.70)
- xgboost: XGBoost classifier (higher variance, prone to overfitting)

Usage:
    # Train with recommended robust settings (logistic + medium regularization + safe features)
    python scripts/train_model.py train --ticker SPY

    # Train with specific model configuration
    python scripts/train_model.py train --ticker SPY --model-type logistic --regularization medium --safe-features

    # Train with XGBoost (original behavior)
    python scripts/train_model.py train --ticker SPY --model-type xgboost --no-safe-features

    # Train with custom settings
    python scripts/train_model.py train --ticker SPY --years 5 --splits 5

    # Train without macro features
    python scripts/train_model.py train --no-macro

    # Skip MLflow tracking
    python scripts/train_model.py train --no-mlflow
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config.settings import get_settings
from src.utils.logger import setup_logger

# Initialize CLI and console
app = typer.Typer(
    name="train_model",
    help="Train direction classifiers for options trading. Supports logistic regression (recommended) and XGBoost.",
)
console = Console()


# Model type enum for CLI
MODEL_TYPES = ["logistic", "xgboost"]
REGULARIZATION_LEVELS = ["strong", "medium", "light"]


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for the script."""
    settings = get_settings()
    level = "DEBUG" if verbose else settings.log_level
    setup_logger(
        name="train_model",
        level=level,
        log_dir=settings.get_log_path(),
        log_to_file=True,
        log_to_console=True,
    )


def print_header() -> None:
    """Print script header."""
    console.print(Panel.fit(
        "[bold blue]ML Options Trading System[/bold blue]\n"
        "[dim]Direction Classifier Training[/dim]",
        border_style="blue"
    ))


def print_metrics_table(metrics, ticker: str) -> None:
    """Print metrics in a formatted table."""
    # Overall metrics table
    table = Table(title=f"Training Results - {ticker}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{metrics.accuracy:.3f}")
    table.add_row("Precision (macro)", f"{metrics.precision_macro:.3f}")
    table.add_row("Recall (macro)", f"{metrics.recall_macro:.3f}")
    table.add_row("F1 Score (macro)", f"{metrics.f1_macro:.3f}")
    table.add_row("AUC (one-vs-rest)", f"{metrics.auc_ovr:.3f}")

    console.print(table)

    # Per-class metrics table
    class_table = Table(title="Per-Class Metrics")
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Precision", justify="right")
    class_table.add_column("Recall", justify="right")
    class_table.add_column("F1", justify="right")
    class_table.add_column("Support", justify="right")

    for class_name, class_metrics in metrics.per_class_metrics.items():
        style = {
            "BEARISH": "red",
            "NEUTRAL": "yellow",
            "BULLISH": "green",
        }.get(class_name, "white")

        class_table.add_row(
            f"[{style}]{class_name}[/{style}]",
            f"{class_metrics['precision']:.3f}",
            f"{class_metrics['recall']:.3f}",
            f"{class_metrics['f1']:.3f}",
            str(class_metrics['support']),
        )

    console.print(class_table)


def print_confusion_matrix(cm, class_names: List[str]) -> None:
    """Print confusion matrix."""
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print("                 Predicted")
    console.print("             " + "  ".join(f"{name[:7]:>7}" for name in class_names))

    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>7}" for val in row)
        label = "Actual" if i == 1 else "      "
        console.print(f"{label} {class_names[i][:7]:>7} [{row_str}]")


def print_top_features(importance_df, n: int = 10) -> None:
    """Print top features table."""
    table = Table(title=f"Top {n} Features")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")

    for _, row in importance_df.head(n).iterrows():
        table.add_row(
            str(int(row['rank'])),
            row['feature'],
            f"{row['importance']:.4f}",
        )

    console.print(table)


def print_fold_metrics(fold_metrics: List[dict]) -> None:
    """Print per-fold metrics."""
    if not fold_metrics:
        return

    table = Table(title="Walk-Forward Validation Results")
    table.add_column("Fold", style="dim", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC", justify="right")

    for i, fold in enumerate(fold_metrics):
        table.add_row(
            str(i + 1),
            f"{fold['accuracy']:.3f}",
            f"{fold['precision_macro']:.3f}",
            f"{fold['recall_macro']:.3f}",
            f"{fold['f1_macro']:.3f}",
            f"{fold['auc_ovr']:.3f}",
        )

    # Add average row
    avg_metrics = {
        key: sum(f[key] for f in fold_metrics) / len(fold_metrics)
        for key in fold_metrics[0].keys()
    }
    table.add_row(
        "[bold]Avg[/bold]",
        f"[bold]{avg_metrics['accuracy']:.3f}[/bold]",
        f"[bold]{avg_metrics['precision_macro']:.3f}[/bold]",
        f"[bold]{avg_metrics['recall_macro']:.3f}[/bold]",
        f"[bold]{avg_metrics['f1_macro']:.3f}[/bold]",
        f"[bold]{avg_metrics['auc_ovr']:.3f}[/bold]",
    )

    console.print(table)


@app.command()
def train(
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker symbol to train on (SPY, QQQ, IWM)"
    ),
    model_type: str = typer.Option(
        "logistic",
        "--model-type", "-m",
        help="Model type: 'logistic' (recommended) or 'xgboost'"
    ),
    regularization: str = typer.Option(
        "medium",
        "--regularization", "-r",
        help="Regularization strength: 'strong', 'medium' (recommended), or 'light'"
    ),
    safe_features: bool = typer.Option(
        True,
        "--safe-features/--no-safe-features",
        help="Filter to momentum features only (recommended to avoid overfitting)"
    ),
    years: int = typer.Option(
        5,
        "--years", "-y",
        help="Years of historical data to use"
    ),
    splits: int = typer.Option(
        5,
        "--splits", "-s",
        help="Number of walk-forward validation splits"
    ),
    include_macro: bool = typer.Option(
        True,
        "--macro/--no-macro",
        help="Include macro features from FRED"
    ),
    include_sentiment: bool = typer.Option(
        False,
        "--sentiment/--no-sentiment",
        help="Include sentiment features from Finnhub/Reddit"
    ),
    include_earnings: bool = typer.Option(
        False,
        "--earnings/--no-earnings",
        help="Include earnings calendar features from Finnhub"
    ),
    binary: bool = typer.Option(
        True,
        "--binary/--ternary",
        help="Use binary (UP/DOWN) or ternary (BULLISH/NEUTRAL/BEARISH) classification"
    ),
    n_features: Optional[int] = typer.Option(
        30,
        "--n-features", "-n",
        help="Number of top features to select (None = use all)"
    ),
    use_mlflow: bool = typer.Option(
        True,
        "--mlflow/--no-mlflow",
        help="Log experiment to MLflow"
    ),
    experiment_name: str = typer.Option(
        "direction_classifier",
        "--experiment", "-e",
        help="MLflow experiment name"
    ),
    save_model: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save trained model to disk"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Directory to save model (default: models/)"
    ),
    params_file: Optional[Path] = typer.Option(
        None,
        "--params-file", "-p",
        help="JSON file with tuned hyperparameters (only for xgboost)"
    ),
    optimize_threshold: bool = typer.Option(
        False,
        "--optimize-threshold/--no-optimize-threshold",
        help="Optimize decision threshold (improves accuracy from ~48% to ~55-60%)"
    ),
    threshold_method: str = typer.Option(
        "f1",
        "--threshold-method",
        help="Threshold optimization method: 'f1', 'youden', 'accuracy', 'cost_sensitive'"
    ),
    threshold_cost_ratio: float = typer.Option(
        2.0,
        "--threshold-cost-ratio",
        help="Cost of missing DOWN vs UP (for cost_sensitive method, higher = more conservative)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Train a direction classifier on historical data.

    This script:
    1. Loads technical features from parquet files
    2. Optionally loads macro features from FRED
    3. Creates target labels (5-day forward returns)
    4. Trains classifier with walk-forward validation
    5. Logs results to MLflow (optional)
    6. Saves the trained model

    RECOMMENDED CONFIGURATION (default):
    - model_type: logistic (more stable, less prone to overfitting)
    - regularization: medium (C=0.1)
    - safe_features: True (filters to momentum features only)

    This configuration achieved AUC ~0.70 in diagnostics.
    """
    # Setup logging
    setup_logging(verbose)

    # Print header
    print_header()

    settings = get_settings()

    # Validate model type and regularization
    if model_type not in MODEL_TYPES:
        console.print(f"[red]Error: Invalid model type '{model_type}'. Choose from: {MODEL_TYPES}[/red]")
        raise typer.Exit(code=1)

    if regularization not in REGULARIZATION_LEVELS:
        console.print(f"[red]Error: Invalid regularization '{regularization}'. Choose from: {REGULARIZATION_LEVELS}[/red]")
        raise typer.Exit(code=1)

    # Determine classification mode
    classification_mode = "binary" if binary else "ternary"
    num_classes = 2 if binary else 3

    # Warn if using non-recommended settings
    if model_type == "xgboost" and not safe_features:
        console.print("[yellow]Warning: XGBoost without safe features is prone to overfitting.[/yellow]")
        console.print("[yellow]Consider using --model-type logistic --safe-features for better results.[/yellow]\n")

    # Configuration summary
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Ticker: {ticker}")
    console.print(f"  Model Type: [cyan]{model_type}[/cyan]")
    console.print(f"  Regularization: [cyan]{regularization}[/cyan]")
    console.print(f"  Safe Features: [cyan]{safe_features}[/cyan]")
    console.print(f"  Data: {years} years")
    console.print(f"  Validation: {splits} walk-forward splits")
    console.print(f"  Classification: {classification_mode.upper()} ({num_classes} classes)")
    console.print(f"  Feature Selection: {'Top ' + str(n_features) if n_features else 'All features'}")
    console.print(f"  Include Macro: {include_macro}")
    console.print(f"  MLflow Tracking: {use_mlflow}")
    console.print(f"  Save Model: {save_model}")
    if optimize_threshold:
        console.print(f"  [green]Threshold Optimization: ENABLED[/green]")
        console.print(f"    Method: {threshold_method}")
        console.print(f"    Cost Ratio: {threshold_cost_ratio}")
    else:
        console.print(f"  Threshold Optimization: disabled")
    console.print()

    # Import modules
    from src.models.feature_pipeline import FeaturePipeline, CLASSIFICATION_BINARY, CLASSIFICATION_TERNARY
    from src.models.direction_classifier import DirectionClassifier
    from src.models.robust_classifier import RobustDirectionClassifier

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Load and prepare data
        task = progress.add_task("Loading and preparing data...", total=None)

        try:
            pipeline = FeaturePipeline(
                ticker=ticker,
                include_macro=include_macro,
                include_sentiment=include_sentiment,
                include_earnings=include_earnings,
                classification_mode=classification_mode,
                n_features=n_features,
            )

            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

            X, y, metadata = pipeline.prepare_training_data(
                start_date=start_date,
                end_date=end_date,
            )

            progress.update(task, description=f"[green]Loaded {len(X)} samples, {len(X.columns)} features[/green]")

        except FileNotFoundError as e:
            progress.update(task, description=f"[red]Data not found: {e}[/red]")
            console.print(f"\n[red]Error: Could not load data for {ticker}[/red]")
            console.print("Run 'python scripts/fetch_data.py --with-features' first to fetch data.")
            raise typer.Exit(code=1)
        except Exception as e:
            progress.update(task, description=f"[red]Error loading data: {e}[/red]")
            raise typer.Exit(code=1)

        # Print data summary
        # Set class names based on mode
        if binary:
            class_name_map = {0: "DOWN", 1: "UP"}
        else:
            class_name_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

        console.print(f"\n[bold]Data Summary:[/bold]")
        console.print(f"  Samples: {metadata['num_samples']}")
        console.print(f"  Features: {metadata['num_features']}")
        console.print(f"  Date Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        console.print(f"  Label Distribution:")
        for label, count in sorted(metadata['label_distribution'].items()):
            class_name = class_name_map.get(label, str(label))
            pct = count / metadata['num_samples'] * 100
            console.print(f"    {class_name}: {count} ({pct:.1f}%)")
        console.print()

        # Step 2: Load tuned params if provided (only for xgboost with DirectionClassifier)
        tuned_params = None
        if params_file and params_file.exists() and model_type == "xgboost" and not safe_features:
            import json
            with open(params_file, 'r') as f:
                params_data = json.load(f)
                tuned_params = params_data.get("best_params", params_data)
            console.print(f"[cyan]Using tuned parameters from {params_file}[/cyan]\n")

        # Step 3: Train model
        task = progress.add_task("Training model...", total=None)

        try:
            # Use RobustDirectionClassifier for logistic or when safe_features is enabled
            # This provides better generalization and avoids overfitting
            use_robust_classifier = model_type == "logistic" or safe_features

            if use_robust_classifier:
                # RobustDirectionClassifier - recommended for production
                classifier = RobustDirectionClassifier(
                    model_type=model_type,
                    regularization=regularization,
                    use_safe_features=safe_features,
                    optimize_threshold=optimize_threshold,
                    threshold_method=threshold_method,
                    threshold_cost_ratio=threshold_cost_ratio,
                )

                if use_mlflow:
                    metrics, run_id = classifier.train_with_mlflow(
                        X, y,
                        experiment_name=experiment_name,
                        n_splits=splits,
                        tags={
                            "ticker": ticker,
                            "years": str(years),
                            "include_macro": str(include_macro),
                            "classification_mode": classification_mode,
                            "n_features": str(n_features) if n_features else "all",
                            "model_type": model_type,
                            "regularization": regularization,
                            "safe_features": str(safe_features),
                            "optimize_threshold": str(optimize_threshold),
                            "threshold_method": threshold_method if optimize_threshold else "none",
                        }
                    )
                    progress.update(task, description=f"[green]Training complete (MLflow run: {run_id[:8]})[/green]")
                else:
                    metrics = classifier.train(X, y, n_splits=splits)
                    run_id = None
                    progress.update(task, description="[green]Training complete[/green]")
            else:
                # DirectionClassifier (XGBoost without safe features) - original behavior
                classifier = DirectionClassifier(num_classes=num_classes)

                # Apply tuned params if available
                if tuned_params:
                    classifier.set_params(**tuned_params)

                if use_mlflow:
                    metrics, run_id = classifier.train_with_mlflow(
                        X, y,
                        experiment_name=experiment_name,
                        n_splits=splits,
                        handle_imbalance=True,
                        tags={
                            "ticker": ticker,
                            "years": str(years),
                            "include_macro": str(include_macro),
                            "classification_mode": classification_mode,
                            "n_features": str(n_features) if n_features else "all",
                            "model_type": "xgboost",
                            "regularization": "default",
                            "safe_features": "False",
                        }
                    )
                    progress.update(task, description=f"[green]Training complete (MLflow run: {run_id[:8]})[/green]")
                else:
                    metrics = classifier.train(
                        X, y,
                        n_splits=splits,
                        handle_imbalance=True,
                    )
                    run_id = None
                    progress.update(task, description="[green]Training complete[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]Training failed: {e}[/red]")
            console.print(f"\n[red]Error during training: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)

        # Step 3: Save model
        model_path = None
        if save_model:
            task = progress.add_task("Saving model...", total=None)

            try:
                output = output_dir or settings.project_root / "models"
                output.mkdir(parents=True, exist_ok=True)

                model_path = classifier.save(output)
                progress.update(task, description=f"[green]Model saved to {model_path}[/green]")

            except Exception as e:
                progress.update(task, description=f"[red]Failed to save model: {e}[/red]")

    # Print results
    console.print("\n")
    print_metrics_table(metrics, ticker)

    # Print confusion matrix
    print_confusion_matrix(
        metrics.confusion_matrix,
        list(classifier.CLASS_NAMES.values())
    )

    # Print fold metrics
    if metrics.fold_metrics:
        console.print("\n")
        print_fold_metrics(metrics.fold_metrics)

    # Print top features
    console.print("\n")
    print_top_features(classifier.get_feature_importance(), n=15)

    # Print feature groups summary
    feature_groups = pipeline.get_feature_groups(list(X.columns))
    console.print("\n[bold]Feature Groups:[/bold]")
    for group, features in sorted(feature_groups.items(), key=lambda x: -len(x[1])):
        console.print(f"  {group}: {len(features)} features")

    # Threshold optimization results
    if optimize_threshold and hasattr(metrics, 'threshold_result') and metrics.threshold_result:
        thresh_result = metrics.threshold_result
        console.print("\n")
        console.print(Panel.fit(
            f"[bold cyan]Threshold Optimization Results[/bold cyan]\n\n"
            f"Optimal Threshold: {thresh_result.optimal_threshold:.3f}\n"
            f"Method: {threshold_method}\n\n"
            f"[bold]Accuracy:[/bold]\n"
            f"  Baseline (0.50): {thresh_result.baseline_accuracy:.1%}\n"
            f"  Optimized:       {thresh_result.accuracy_at_threshold:.1%}\n"
            f"  Improvement:     [green]{thresh_result.accuracy_improvement:+.1%}[/green]\n\n"
            f"[bold]Per-Class Metrics:[/bold]\n"
            f"  DOWN: Precision={thresh_result.down_precision:.3f}, Recall={thresh_result.down_recall:.3f}\n"
            f"  UP:   Precision={thresh_result.up_precision:.3f}, Recall={thresh_result.up_recall:.3f}",
            border_style="cyan"
        ))

    # Summary
    console.print("\n")
    summary_text = (
        f"[bold green]Training Complete[/bold green]\n\n"
        f"Ticker: {ticker}\n"
        f"Model: {model_type} ({regularization} reg, safe_features={safe_features})\n"
        f"Accuracy: {metrics.accuracy:.3f}\n"
        f"F1 Score: {metrics.f1_macro:.3f}\n"
        f"AUC: {metrics.auc_ovr:.3f}\n"
    )
    if optimize_threshold and hasattr(classifier, 'optimal_threshold'):
        summary_text += f"Optimal Threshold: {classifier.optimal_threshold:.3f}\n"
    if model_path:
        summary_text += f"Saved: {model_path}\n"
    if run_id:
        summary_text += f"MLflow Run: {run_id}"

    console.print(Panel.fit(summary_text, border_style="green"))

    # MLflow UI hint
    if use_mlflow:
        console.print(
            "\n[dim]View results in MLflow UI:[/dim]\n"
            f"  mlflow ui --backend-store-uri {settings.mlflow_tracking_uri}"
        )


@app.command()
def train_all(
    tickers: Optional[List[str]] = typer.Option(
        None,
        "--tickers", "-t",
        help="Tickers to train (default: from settings)"
    ),
    model_type: str = typer.Option(
        "logistic",
        "--model-type", "-m",
        help="Model type: 'logistic' (recommended) or 'xgboost'"
    ),
    regularization: str = typer.Option(
        "medium",
        "--regularization", "-r",
        help="Regularization strength: 'strong', 'medium' (recommended), or 'light'"
    ),
    safe_features: bool = typer.Option(
        True,
        "--safe-features/--no-safe-features",
        help="Filter to momentum features only (recommended)"
    ),
    years: int = typer.Option(
        5,
        "--years", "-y",
        help="Years of historical data"
    ),
    include_macro: bool = typer.Option(
        True,
        "--macro/--no-macro",
        help="Include macro features"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Train models for all configured tickers.

    Trains separate Direction Classifier models for each ticker
    and logs all experiments to MLflow.

    Uses the robust classifier configuration by default.
    """
    setup_logging(verbose)
    print_header()

    settings = get_settings()
    tickers_to_train = tickers or settings.tickers

    console.print(f"\n[bold]Training models for: {', '.join(tickers_to_train)}[/bold]")
    console.print(f"  Model: {model_type} ({regularization} reg, safe_features={safe_features})\n")

    results = []

    for ticker in tickers_to_train:
        console.print(f"\n{'='*50}")
        console.print(f"[bold]Training {ticker}[/bold]")
        console.print('='*50)

        try:
            # Import here to avoid circular imports
            from src.models.feature_pipeline import FeaturePipeline
            from src.models.robust_classifier import RobustDirectionClassifier

            # Prepare data
            pipeline = FeaturePipeline(
                ticker=ticker,
                include_macro=include_macro,
                classification_mode="binary",
            )
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

            X, y, metadata = pipeline.prepare_training_data(
                start_date=start_date,
                end_date=end_date,
            )

            # Train with RobustDirectionClassifier
            classifier = RobustDirectionClassifier(
                model_type=model_type,
                regularization=regularization,
                use_safe_features=safe_features,
            )
            metrics, run_id = classifier.train_with_mlflow(
                X, y,
                experiment_name="direction_classifier_batch",
                n_splits=5,
                tags={
                    "ticker": ticker,
                    "batch_run": "true",
                    "model_type": model_type,
                    "regularization": regularization,
                    "safe_features": str(safe_features),
                }
            )

            # Save
            output = settings.project_root / "models"
            model_path = classifier.save(output)

            results.append({
                "ticker": ticker,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1_macro,
                "auc": metrics.auc_ovr,
                "samples": metadata['num_samples'],
                "model_path": str(model_path),
                "status": "success",
            })

            console.print(f"[green]Success: accuracy={metrics.accuracy:.3f}, f1={metrics.f1_macro:.3f}[/green]")

        except Exception as e:
            results.append({
                "ticker": ticker,
                "status": f"failed: {e}",
            })
            console.print(f"[red]Failed: {e}[/red]")

    # Summary table
    console.print(f"\n\n{'='*50}")
    console.print("[bold]Training Summary[/bold]")
    console.print('='*50)

    table = Table()
    table.add_column("Ticker", style="cyan")
    table.add_column("Status")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Samples", justify="right")

    for result in results:
        if result["status"] == "success":
            table.add_row(
                result["ticker"],
                "[green]Success[/green]",
                f"{result['accuracy']:.3f}",
                f"{result['f1']:.3f}",
                f"{result['auc']:.3f}",
                str(result['samples']),
            )
        else:
            table.add_row(
                result["ticker"],
                f"[red]{result['status']}[/red]",
                "-", "-", "-", "-"
            )

    console.print(table)


@app.command()
def evaluate(
    model_path: Path = typer.Argument(
        ...,
        help="Path to saved model file"
    ),
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker to evaluate on"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Evaluate a saved model on test data.

    Loads a previously trained model and evaluates it on
    the most recent data (not seen during training).
    """
    setup_logging(verbose)
    print_header()

    console.print(f"\n[bold]Evaluating model on {ticker}[/bold]")
    console.print(f"Model: {model_path}\n")

    from src.models.feature_pipeline import FeaturePipeline
    from src.models.direction_classifier import DirectionClassifier

    # Load model
    classifier = DirectionClassifier.load_from_path(model_path)
    console.print(classifier.summary())

    # Load recent data
    pipeline = FeaturePipeline(ticker=ticker, include_macro=True)

    # Get last 6 months of data for evaluation
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    try:
        X, y, metadata = pipeline.prepare_training_data(
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(code=1)

    # Evaluate
    console.print(f"\nEvaluating on {len(X)} samples from {start_date} to {end_date}")

    metrics = classifier.evaluate(X, y)

    print_metrics_table(metrics, ticker)
    print_confusion_matrix(
        metrics.confusion_matrix,
        list(classifier.CLASS_NAMES.values())
    )


@app.command()
def predict(
    model_path: Path = typer.Argument(
        ...,
        help="Path to saved model file"
    ),
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker to predict"
    ),
    days: int = typer.Option(
        5,
        "--days", "-d",
        help="Number of recent days to predict"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Make predictions using a saved model.

    Loads a model and makes predictions on the most recent data.
    """
    setup_logging(verbose)
    print_header()

    from src.models.feature_pipeline import FeaturePipeline
    from src.models.direction_classifier import DirectionClassifier

    console.print(f"\n[bold]Predictions for {ticker}[/bold]")

    # Load model
    classifier = DirectionClassifier.load_from_path(model_path)

    # Load recent data
    pipeline = FeaturePipeline(ticker=ticker, include_macro=True)

    try:
        technical_df = pipeline.load_technical_features()
        macro_df = pipeline.load_macro_features()
        merged_df = pipeline.merge_features(technical_df, macro_df)
        X = pipeline.select_features(merged_df)
        X = pipeline.handle_missing_values(X, strategy="fill_forward")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(code=1)

    # Get recent samples
    X_recent = X.tail(days)

    # Predict
    results = classifier.predict_with_confidence(X_recent)

    # Display results
    table = Table(title=f"Predictions for {ticker}")
    table.add_column("Date", style="cyan")
    table.add_column("Prediction")
    table.add_column("Confidence", justify="right")
    table.add_column("P(Bearish)", justify="right")
    table.add_column("P(Neutral)", justify="right")
    table.add_column("P(Bullish)", justify="right")

    for date, result in zip(X_recent.index, results):
        style = {
            "BEARISH": "red",
            "NEUTRAL": "yellow",
            "BULLISH": "green",
        }.get(result.class_name, "white")

        table.add_row(
            str(date.date()),
            f"[{style}]{result.class_name}[/{style}]",
            f"{result.confidence:.1%}",
            f"{result.probabilities['BEARISH']:.1%}",
            f"{result.probabilities['NEUTRAL']:.1%}",
            f"{result.probabilities['BULLISH']:.1%}",
        )

    console.print(table)

    # Latest prediction summary
    latest = results[-1]
    console.print(Panel.fit(
        f"[bold]Latest Prediction ({X_recent.index[-1].date()})[/bold]\n\n"
        f"Direction: [{style}]{latest.class_name}[/{style}]\n"
        f"Confidence: {latest.confidence:.1%}\n\n"
        f"This prediction is for the next {classifier.prediction_horizon} trading days.",
        border_style=style
    ))


@app.command()
def tune(
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker symbol to tune on"
    ),
    n_trials: int = typer.Option(
        50,
        "--trials", "-n",
        help="Number of Optuna trials"
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="Timeout in seconds (optional)"
    ),
    binary: bool = typer.Option(
        True,
        "--binary/--ternary",
        help="Use binary or ternary classification"
    ),
    n_features: Optional[int] = typer.Option(
        30,
        "--n-features",
        help="Number of top features to select"
    ),
    save_params: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save best params to file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Tune hyperparameters using Optuna.

    Runs walk-forward validation to find optimal XGBoost parameters.
    """
    setup_logging(verbose)
    print_header()

    settings = get_settings()
    classification_mode = "binary" if binary else "ternary"
    num_classes = 2 if binary else 3

    console.print(f"\n[bold]Hyperparameter Tuning[/bold]")
    console.print(f"  Ticker: {ticker}")
    console.print(f"  Trials: {n_trials}")
    console.print(f"  Timeout: {timeout}s" if timeout else "  Timeout: None")
    console.print(f"  Mode: {classification_mode.upper()}")
    console.print(f"  Features: {n_features}")
    console.print()

    from src.models.feature_pipeline import FeaturePipeline
    from src.models.hyperparameter_tuning import DirectionClassifierTuner

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data
        task = progress.add_task("Loading data...", total=None)

        try:
            pipeline = FeaturePipeline(
                ticker=ticker,
                include_macro=True,
                classification_mode=classification_mode,
                n_features=n_features,
            )

            X, y, metadata = pipeline.prepare_training_data()
            progress.update(task, description=f"[green]Loaded {len(X)} samples, {len(X.columns)} features[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

        # Run tuning
        task = progress.add_task(f"Running {n_trials} trials...", total=None)

        try:
            tuner = DirectionClassifierTuner(
                X, y,
                n_splits=5,
                num_classes=num_classes,
            )

            best_params = tuner.tune(
                n_trials=n_trials,
                timeout=timeout,
                show_progress=False,  # We have our own progress
            )

            progress.update(task, description=f"[green]Tuning complete! Best AUC: {tuner.study.best_value:.4f}[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]Tuning failed: {e}[/red]")
            raise typer.Exit(code=1)

        # Save params
        if save_params:
            task = progress.add_task("Saving parameters...", total=None)
            params_path = settings.project_root / "models" / f"best_params_{ticker}_{classification_mode}.json"
            tuner.save_best_params(str(params_path))
            progress.update(task, description=f"[green]Saved to {params_path}[/green]")

    # Display results
    console.print("\n[bold green]Best Parameters:[/bold green]")

    table = Table(title="Optimized Hyperparameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")

    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            table.add_row(k, f"{v:.6f}")
        else:
            table.add_row(k, str(v))

    console.print(table)

    # Summary
    console.print(Panel.fit(
        f"[bold green]Tuning Complete[/bold green]\n\n"
        f"Best AUC: {tuner.study.best_value:.4f}\n"
        f"Trials: {len(tuner.study.trials)}\n"
        f"Completed: {len([t for t in tuner.study.trials if t.state.name == 'COMPLETE'])}\n"
        f"Pruned: {len([t for t in tuner.study.trials if t.state.name == 'PRUNED'])}\n\n"
        f"To train with these params:\n"
        f"  python scripts/train_model.py train --ticker {ticker} --params-file {params_path}",
        border_style="green"
    ))


@app.command()
def train_volatility(
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker symbol to train on"
    ),
    mode: str = typer.Option(
        "classification",
        "--mode", "-m",
        help="Model mode: 'classification' (regime) or 'regression' (IV rank value)"
    ),
    years: int = typer.Option(
        5,
        "--years", "-y",
        help="Years of historical data"
    ),
    splits: int = typer.Option(
        5,
        "--splits", "-s",
        help="Number of walk-forward validation splits"
    ),
    save_model: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save trained model"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Train a Volatility Forecaster model.

    Predicts IV regime (LOW/NORMAL/HIGH) 5 days ahead to help
    with options strategy selection.

    Two modes:
    - classification: Predicts regime category (recommended)
    - regression: Predicts actual IV rank value

    Usage:
        python scripts/train_model.py train-volatility --ticker SPY --mode classification
    """
    setup_logging(verbose)
    print_header()

    console.print(Panel.fit(
        "[bold blue]Volatility Forecaster Training[/bold blue]\n"
        "[dim]Predicts IV regime 5 days ahead[/dim]",
        border_style="blue"
    ))

    settings = get_settings()

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Ticker: {ticker}")
    console.print(f"  Mode: [cyan]{mode}[/cyan]")
    console.print(f"  Data: {years} years")
    console.print(f"  Validation: {splits} walk-forward splits")
    console.print(f"  Save Model: {save_model}\n")

    from datetime import datetime, timedelta
    from src.models.feature_pipeline import FeaturePipeline
    from src.models.volatility_forecaster import VolatilityForecaster

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Step 1: Load data
        task = progress.add_task("Loading features...", total=None)

        try:
            pipeline = FeaturePipeline(
                ticker=ticker,
                include_macro=True,
                classification_mode="binary",
            )

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

            X_full, _, _ = pipeline.prepare_training_data(
                start_date=start_date,
                end_date=end_date,
            )

            # Get the full feature dataframe before target creation
            technical_df = pipeline.load_technical_features()
            progress.update(task, description=f"[green]Loaded {len(technical_df)} samples[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]Error loading data: {e}[/red]")
            raise typer.Exit(code=1)

        # Step 2: Train model
        task = progress.add_task("Training volatility forecaster...", total=None)

        try:
            forecaster = VolatilityForecaster(mode=mode)
            X, y = forecaster.prepare_data(technical_df)
            metrics = forecaster.train(X, y, n_splits=splits)

            progress.update(task, description=f"[green]Training complete[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]Training failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)

        # Step 3: Save model
        model_path = None
        if save_model:
            task = progress.add_task("Saving model...", total=None)

            try:
                output = settings.project_root / "models"
                output.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = output / f"volatility_forecaster_{mode}_{ticker}_{timestamp}.joblib"
                forecaster.save(str(model_path))

                progress.update(task, description=f"[green]Model saved to {model_path}[/green]")

            except Exception as e:
                progress.update(task, description=f"[red]Failed to save model: {e}[/red]")

    # Print results
    console.print("\n")

    if mode == "classification":
        # Classification results table
        table = Table(title=f"Volatility Forecaster Results - {ticker}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Accuracy", f"{metrics['accuracy']:.3f}")
        table.add_row("F1 (macro)", f"{metrics['f1_macro']:.3f}")
        table.add_row("Samples", str(metrics['n_samples']))
        table.add_row("Features", str(metrics['n_features']))

        console.print(table)

        # Per-class metrics
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            class_table = Table(title="Per-Class Metrics")
            class_table.add_column("Class", style="cyan")
            class_table.add_column("Precision", justify="right")
            class_table.add_column("Recall", justify="right")
            class_table.add_column("F1", justify="right")
            class_table.add_column("Support", justify="right")

            for class_name in metrics.get('class_names', ['HIGH', 'LOW', 'NORMAL']):
                if class_name in report:
                    cls = report[class_name]
                    class_table.add_row(
                        class_name,
                        f"{cls['precision']:.3f}",
                        f"{cls['recall']:.3f}",
                        f"{cls['f1-score']:.3f}",
                        str(int(cls['support'])),
                    )

            console.print(class_table)

        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            console.print("\n[bold]Confusion Matrix:[/bold]")
            console.print(f"Classes: {metrics.get('class_names', ['HIGH', 'LOW', 'NORMAL'])}")
            for row in cm:
                console.print(f"  {row}")

    else:
        # Regression results table
        table = Table(title=f"Volatility Forecaster Results - {ticker}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("RMSE", f"{metrics['rmse']:.3f}")
        table.add_row("MAE", f"{metrics['mae']:.3f}")
        table.add_row("R2", f"{metrics['r2']:.3f}")
        table.add_row("Samples", str(metrics['n_samples']))
        table.add_row("Features", str(metrics['n_features']))

        console.print(table)

    # Feature importance
    importance_df = forecaster.get_feature_importance()
    console.print("\n[bold]Top 10 Features:[/bold]")
    for i, row in importance_df.head(10).iterrows():
        console.print(f"  {row['feature']}: {row['importance']:.4f}")

    # Summary panel
    if mode == "classification":
        summary = (
            f"[bold green]Training Complete[/bold green]\n\n"
            f"Ticker: {ticker}\n"
            f"Mode: {mode}\n"
            f"Accuracy: {metrics['accuracy']:.3f}\n"
            f"F1 (macro): {metrics['f1_macro']:.3f}"
        )
    else:
        summary = (
            f"[bold green]Training Complete[/bold green]\n\n"
            f"Ticker: {ticker}\n"
            f"Mode: {mode}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"MAE: {metrics['mae']:.3f}\n"
            f"R2: {metrics['r2']:.3f}"
        )

    if model_path:
        summary += f"\nSaved: {model_path}"

    console.print(Panel.fit(summary, border_style="green"))

    # Strategy hint
    console.print("\n[dim]Use with Direction Classifier for strategy selection:[/dim]")
    console.print("[dim]  Direction=UP + IV=LOW -> Buy Calls[/dim]")
    console.print("[dim]  Direction=UP + IV=HIGH -> Sell Put Spreads[/dim]")
    console.print("[dim]  Direction=DOWN + IV=LOW -> Buy Puts[/dim]")
    console.print("[dim]  Direction=DOWN + IV=HIGH -> Sell Call Spreads[/dim]")


@app.command()
def signal(
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker to generate signal for"
    ),
    direction_model: Optional[Path] = typer.Option(
        None,
        "--direction-model", "-d",
        help="Path to direction classifier model (auto-detect if not specified)"
    ),
    volatility_model: Optional[Path] = typer.Option(
        None,
        "--volatility-model", "-v",
        help="Path to volatility forecaster model (auto-detect if not specified)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging"
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Record prediction to database for tracking"
    ),
) -> None:
    """
    Generate trading signal using both models.

    Combines Direction Classifier and Volatility Forecaster
    to recommend a specific options strategy.

    Usage:
        python scripts/train_model.py signal --ticker SPY
        python scripts/train_model.py signal --ticker SPY --record
    """
    setup_logging(verbose)

    console.print(Panel.fit(
        "[bold blue]Trading Signal Generator[/bold blue]\n"
        "[dim]Combining Direction + IV Forecasts[/dim]",
        border_style="blue"
    ))

    settings = get_settings()

    # Find models if not specified
    models_dir = settings.project_root / "models"

    if direction_model is None:
        # Find latest direction classifier
        dir_models = sorted(models_dir.glob("robust_classifier_*.joblib"), reverse=True)
        if not dir_models:
            console.print("[red]No direction classifier found. Train one first.[/red]")
            raise typer.Exit(code=1)
        direction_model = dir_models[0]

    if volatility_model is None:
        # Find latest volatility forecaster
        vol_models = sorted(models_dir.glob("volatility_forecaster_*.joblib"), reverse=True)
        if not vol_models:
            console.print("[red]No volatility forecaster found. Train one first.[/red]")
            raise typer.Exit(code=1)
        volatility_model = vol_models[0]

    console.print(f"\n[bold]Models:[/bold]")
    console.print(f"  Direction: {direction_model.name}")
    console.print(f"  Volatility: {volatility_model.name}")
    console.print(f"  Ticker: {ticker}\n")

    from datetime import datetime, timedelta
    from src.models.feature_pipeline import FeaturePipeline
    from src.models.robust_classifier import RobustDirectionClassifier
    from src.models.volatility_forecaster import VolatilityForecaster
    from src.models.strategy_selector import StrategySelector, format_recommendation

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Load models
        task = progress.add_task("Loading models...", total=None)

        try:
            dir_classifier = RobustDirectionClassifier.load_from_path(str(direction_model))
            vol_forecaster = VolatilityForecaster.load(str(volatility_model))
            progress.update(task, description="[green]Models loaded[/green]")
        except Exception as e:
            progress.update(task, description=f"[red]Failed to load models: {e}[/red]")
            raise typer.Exit(code=1)

        # Load features
        task = progress.add_task("Loading features...", total=None)

        try:
            pipeline = FeaturePipeline(ticker=ticker, include_macro=True)
            features_df = pipeline.load_technical_features()
            latest_features = features_df.iloc[[-1]].copy()

            # Handle missing values
            latest_features = latest_features.ffill().bfill()
            latest_features = latest_features.fillna(0)

            latest_date = features_df.index[-1]
            progress.update(task, description=f"[green]Features loaded (as of {latest_date.date()})[/green]")
        except Exception as e:
            progress.update(task, description=f"[red]Failed to load features: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)

        # Get direction prediction
        task = progress.add_task("Running direction classifier...", total=None)

        try:
            # Get the features the model expects
            model_features = dir_classifier.feature_names
            available_features = [f for f in model_features if f in latest_features.columns]

            if len(available_features) < len(model_features) * 0.5:
                progress.update(task, description="[yellow]Warning: Many features missing[/yellow]")

            # Create feature matrix with available features
            X_dir = latest_features[available_features].copy()
            for f in model_features:
                if f not in X_dir.columns:
                    X_dir[f] = 0

            X_dir = X_dir[model_features]

            # Predict
            dir_proba = dir_classifier.model.predict_proba(
                dir_classifier.scaler.transform(X_dir)
            )[0]

            # Class 1 is typically UP
            up_prob = dir_proba[1] if len(dir_proba) > 1 else dir_proba[0]
            direction = "UP" if up_prob >= dir_classifier.optimal_threshold else "DOWN"
            direction_confidence = max(up_prob, 1 - up_prob)

            progress.update(task, description=f"[green]Direction: {direction} ({direction_confidence:.1%})[/green]")
        except Exception as e:
            progress.update(task, description=f"[red]Direction prediction failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)

        # Get volatility prediction
        task = progress.add_task("Running volatility forecaster...", total=None)

        try:
            # Add lagged IV features that the model expects
            if "iv_rank" in features_df.columns:
                for lag in [5, 10, 20]:
                    latest_features[f"iv_rank_lag_{lag}"] = features_df["iv_rank"].shift(lag).iloc[-1]
                latest_features["iv_rank_change_5d"] = features_df["iv_rank"].iloc[-1] - features_df["iv_rank"].shift(5).iloc[-1]
                latest_features["iv_rank_change_10d"] = features_df["iv_rank"].iloc[-1] - features_df["iv_rank"].shift(10).iloc[-1]

            # Fill any remaining NaNs
            latest_features = latest_features.fillna(50)  # Use median IV rank for missing

            vol_results = vol_forecaster.predict(latest_features)
            vol_pred = vol_results[0]

            iv_regime = vol_pred.predicted_regime
            iv_confidence = vol_pred.confidence
            current_iv_rank = vol_pred.current_iv_rank

            progress.update(task, description=f"[green]IV Regime: {iv_regime} ({iv_confidence:.1%})[/green]")
        except Exception as e:
            progress.update(task, description=f"[red]Volatility prediction failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)

        # Generate strategy recommendation
        task = progress.add_task("Selecting strategy...", total=None)

        try:
            selector = StrategySelector()
            recommendation = selector.select_strategy(
                direction=direction,
                direction_confidence=direction_confidence,
                direction_probability=up_prob,
                iv_regime=iv_regime,
                iv_confidence=iv_confidence,
                current_iv_rank=current_iv_rank,
                ticker=ticker,
            )

            progress.update(task, description=f"[green]Strategy: {recommendation.strategy.value}[/green]")
        except Exception as e:
            progress.update(task, description=f"[red]Strategy selection failed: {e}[/red]")
            raise typer.Exit(code=1)

    # Display recommendation
    console.print("\n")

    # Summary panel
    strategy_color = {
        "Buy Calls": "green",
        "Buy Puts": "red",
        "Bull Call Spread": "green",
        "Bear Put Spread": "red",
        "Bull Put Spread (Credit)": "green",
        "Bear Call Spread (Credit)": "red",
        "Iron Condor": "yellow",
        "No Trade": "dim",
    }.get(recommendation.strategy.value, "white")

    console.print(Panel.fit(
        f"[bold {strategy_color}]{recommendation.strategy.value}[/bold {strategy_color}]\n\n"
        f"[cyan]Direction:[/cyan] {recommendation.direction.value} ({recommendation.direction_confidence:.1%})\n"
        f"[cyan]IV Regime:[/cyan] {recommendation.iv_regime.value} ({recommendation.iv_confidence:.1%})\n"
        f"[cyan]Current IV Rank:[/cyan] {current_iv_rank:.1f}\n"
        f"[cyan]Overall Confidence:[/cyan] {recommendation.overall_confidence:.1%}",
        title=f"Signal for {ticker} - {latest_date.date()}",
        border_style=strategy_color,
    ))

    if recommendation.strategy.value != "No Trade":
        # Trade parameters table
        params_table = Table(title="Trade Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value")

        params_table.add_row("Position Size", recommendation.position_size.upper())
        params_table.add_row("Expiry", recommendation.suggested_expiry)
        params_table.add_row("Strike Selection", recommendation.suggested_delta)
        params_table.add_row("Max Loss", f"{recommendation.max_loss_pct:.1%} of portfolio")
        params_table.add_row("Profit Target", f"{recommendation.profit_target_pct:.0%} of max profit")
        params_table.add_row("Stop Loss", recommendation.stop_loss_trigger)

        console.print(params_table)

        # Reasoning
        console.print(f"\n[bold]Reasoning:[/bold] {recommendation.reasoning}")

    # Cautions
    if recommendation.cautions:
        console.print(f"\n[bold yellow]Cautions:[/bold yellow]")
        for caution in recommendation.cautions:
            console.print(f"  [yellow]![/yellow] {caution}")

    # Market data summary
    console.print(f"\n[dim]Data as of: {latest_date.date()}[/dim]")
    if "close" in features_df.columns:
        console.print(f"[dim]Last Close: ${features_df['close'].iloc[-1]:.2f}[/dim]")
    if "vix_close" in features_df.columns:
        console.print(f"[dim]VIX: {features_df['vix_close'].iloc[-1]:.2f}[/dim]")

    # Record to database if requested
    if record:
        try:
            from dashboard.utils.database import init_database, record_prediction

            init_database()

            pred_id = record_prediction(
                ticker=ticker,
                prediction_date=str(latest_date.date()),
                direction_pred=direction.lower(),
                direction_prob=direction_confidence,
                volatility_pred=iv_regime.lower(),
                volatility_prob=iv_confidence,
                iv_rank=current_iv_rank,
                recommended_strategy=recommendation.strategy.value.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                strategy_confidence=recommendation.overall_confidence,
                underlying_price=features_df['close'].iloc[-1] if 'close' in features_df.columns else None,
                model_version="v0.1",
            )
            console.print(f"\n[green]Prediction recorded to database (ID: {pred_id})[/green]")
        except Exception as e:
            console.print(f"\n[yellow]Warning: Could not record to database: {e}[/yellow]")


@app.command()
def signal_all(
    tickers: Optional[List[str]] = typer.Option(
        None,
        "--tickers", "-t",
        help="Tickers to analyze (default: SPY,QQQ,IWM)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging"
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Record predictions to database for tracking"
    ),
) -> None:
    """
    Generate trading signals for multiple tickers.

    Usage:
        python scripts/train_model.py signal-all --tickers SPY,QQQ,IWM
        python scripts/train_model.py signal-all --record
    """
    setup_logging(verbose)

    console.print(Panel.fit(
        "[bold blue]Multi-Ticker Signal Generator[/bold blue]",
        border_style="blue"
    ))

    tickers_list = tickers or ["SPY", "QQQ", "IWM"]
    recorded_predictions = []  # Collect predictions for recording
    settings = get_settings()
    models_dir = settings.project_root / "models"

    # Find models
    dir_models = sorted(models_dir.glob("robust_classifier_*.joblib"), reverse=True)
    vol_models = sorted(models_dir.glob("volatility_forecaster_*.joblib"), reverse=True)

    if not dir_models or not vol_models:
        console.print("[red]Missing models. Train both direction and volatility models first.[/red]")
        raise typer.Exit(code=1)

    direction_model = dir_models[0]
    volatility_model = vol_models[0]

    from src.models.feature_pipeline import FeaturePipeline
    from src.models.robust_classifier import RobustDirectionClassifier
    from src.models.volatility_forecaster import VolatilityForecaster
    from src.models.strategy_selector import StrategySelector

    # Load models once
    dir_classifier = RobustDirectionClassifier.load_from_path(str(direction_model))
    vol_forecaster = VolatilityForecaster.load(str(volatility_model))
    selector = StrategySelector()

    # Results table
    results_table = Table(title="Trading Signals")
    results_table.add_column("Ticker", style="cyan")
    results_table.add_column("Strategy")
    results_table.add_column("Direction")
    results_table.add_column("IV Regime")
    results_table.add_column("Confidence", justify="right")
    results_table.add_column("Position")

    for ticker in tickers_list:
        try:
            # Load features (same way as the signal command)
            pipeline = FeaturePipeline(ticker=ticker, include_macro=True)
            features_df = pipeline.load_technical_features()

            latest_features = features_df.iloc[[-1]].copy()
            latest_features = latest_features.ffill().bfill().fillna(0)

            # Add lagged IV features for volatility forecaster
            if "iv_rank" in features_df.columns:
                for lag in [5, 10, 20]:
                    latest_features[f"iv_rank_lag_{lag}"] = features_df["iv_rank"].shift(lag).iloc[-1]
                latest_features["iv_rank_change_5d"] = features_df["iv_rank"].iloc[-1] - features_df["iv_rank"].shift(5).iloc[-1]
                latest_features["iv_rank_change_10d"] = features_df["iv_rank"].iloc[-1] - features_df["iv_rank"].shift(10).iloc[-1]
            latest_features = latest_features.fillna(50)  # Use median IV rank for missing

            # Direction prediction
            model_features = dir_classifier.feature_names
            X_dir = latest_features.reindex(columns=model_features, fill_value=0)
            dir_proba = dir_classifier.model.predict_proba(
                dir_classifier.scaler.transform(X_dir)
            )[0]
            up_prob = dir_proba[1] if len(dir_proba) > 1 else dir_proba[0]
            direction = "UP" if up_prob >= dir_classifier.optimal_threshold else "DOWN"
            direction_confidence = max(up_prob, 1 - up_prob)

            # Volatility prediction
            vol_results = vol_forecaster.predict(latest_features)
            vol_pred = vol_results[0]

            # Strategy
            rec = selector.select_strategy(
                direction=direction,
                direction_confidence=direction_confidence,
                direction_probability=up_prob,
                iv_regime=vol_pred.predicted_regime,
                iv_confidence=vol_pred.confidence,
                current_iv_rank=vol_pred.current_iv_rank,
                ticker=ticker,
            )

            # Color coding
            strat_style = {
                "Buy Calls": "green",
                "Buy Puts": "red",
                "Bull Call Spread": "green",
                "Bear Put Spread": "red",
                "Bull Put Spread (Credit)": "green",
                "Bear Call Spread (Credit)": "red",
                "No Trade": "dim",
            }.get(rec.strategy.value, "white")

            results_table.add_row(
                ticker,
                f"[{strat_style}]{rec.strategy.value}[/{strat_style}]",
                f"{rec.direction.value} ({rec.direction_confidence:.0%})",
                f"{rec.iv_regime.value} ({rec.iv_confidence:.0%})",
                f"{rec.overall_confidence:.0%}",
                rec.position_size.upper() if rec.strategy.value != "No Trade" else "-",
            )

            # Collect for recording if requested
            if record:
                recorded_predictions.append({
                    "ticker": ticker,
                    "date": str(features_df.index[-1].date()),
                    "direction": direction.lower(),
                    "direction_confidence": direction_confidence,
                    "iv_regime": vol_pred.predicted_regime.lower(),
                    "iv_confidence": vol_pred.confidence,
                    "iv_rank": vol_pred.current_iv_rank,
                    "strategy": rec.strategy.value.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                    "strategy_confidence": rec.overall_confidence,
                    "underlying_price": features_df['close'].iloc[-1] if 'close' in features_df.columns else None,
                })

        except Exception as e:
            results_table.add_row(
                ticker,
                f"[red]Error[/red]",
                "-", "-", "-", "-"
            )
            if verbose:
                console.print(f"[red]Error for {ticker}: {e}[/red]")

    console.print(results_table)

    # Legend
    console.print("\n[dim]Strategy Legend:[/dim]")
    console.print("[dim]  [green]Green[/green] = Bullish strategies[/dim]")
    console.print("[dim]  [red]Red[/red] = Bearish strategies[/dim]")
    console.print("[dim]  [yellow]Yellow[/yellow] = Neutral strategies[/dim]")

    # Record to database if requested
    if record and recorded_predictions:
        try:
            from dashboard.utils.database import init_database, record_prediction

            init_database()

            for pred in recorded_predictions:
                record_prediction(
                    ticker=pred["ticker"],
                    prediction_date=pred["date"],
                    direction_pred=pred["direction"],
                    direction_prob=pred["direction_confidence"],
                    volatility_pred=pred["iv_regime"],
                    volatility_prob=pred["iv_confidence"],
                    iv_rank=pred["iv_rank"],
                    recommended_strategy=pred["strategy"],
                    strategy_confidence=pred["strategy_confidence"],
                    underlying_price=pred["underlying_price"],
                    model_version="v0.1",
                )

            console.print(f"\n[green]{len(recorded_predictions)} predictions recorded to database[/green]")
        except Exception as e:
            console.print(f"\n[yellow]Warning: Could not record to database: {e}[/yellow]")


if __name__ == "__main__":
    app()
