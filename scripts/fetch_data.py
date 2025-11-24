#!/usr/bin/env python3
"""
Data fetching CLI script.

Fetches historical market data for configured tickers and saves to feature store.

Usage:
    python scripts/fetch_data.py                    # Fetch all configured tickers
    python scripts/fetch_data.py --tickers SPY QQQ # Fetch specific tickers
    python scripts/fetch_data.py --years 3         # Fetch 3 years of data
    python scripts/fetch_data.py --with-features   # Also calculate features

Docker usage:
    docker-compose run app python scripts/fetch_data.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config.settings import get_settings
from src.data.feature_store import FeatureStore, RawDataStore
from src.data.yfinance_loader import YFinanceLoader, DataFetchError
from src.features.technical_indicators import TechnicalIndicators
from src.utils.logger import setup_logger

# Initialize CLI and console
app = typer.Typer(
    name="fetch_data",
    help="Fetch market data for ML options trading system",
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for the script."""
    settings = get_settings()
    level = "DEBUG" if verbose else settings.log_level
    setup_logger(
        name="fetch_data",
        level=level,
        log_dir=settings.get_log_path(),
        log_to_file=True,
        log_to_console=True,
    )


@app.command()
def main(
    tickers: Optional[List[str]] = typer.Option(
        None,
        "--tickers", "-t",
        help="Tickers to fetch (default: from settings)"
    ),
    years: int = typer.Option(
        5,
        "--years", "-y",
        help="Years of historical data to fetch"
    ),
    with_features: bool = typer.Option(
        False,
        "--with-features", "-f",
        help="Calculate technical indicators after fetching"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """
    Fetch historical market data and optionally calculate features.

    This script fetches OHLCV data for configured tickers (SPY, QQQ, IWM)
    and VIX data from yfinance. Data is saved as parquet files.
    """
    # Setup logging
    setup_logging(verbose)

    # Get settings
    settings = get_settings()
    tickers_to_fetch = tickers or settings.tickers

    console.print(f"\n[bold blue]ML Options Trading System - Data Fetch[/bold blue]")
    console.print(f"Tickers: {', '.join(tickers_to_fetch)}")
    console.print(f"Lookback: {years} years")
    console.print(f"Output: {settings.get_data_path('raw')}")
    console.print()

    # Initialize stores
    raw_store = RawDataStore()
    feature_store = FeatureStore()

    # Initialize loader
    loader = YFinanceLoader(tickers=tickers_to_fetch, lookback_years=years)

    # Track results
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Fetch data for each ticker
        for ticker in tickers_to_fetch:
            task = progress.add_task(f"Fetching {ticker}...", total=None)

            try:
                # Fetch OHLCV data
                df = loader.fetch_ticker_data(ticker)

                # Save raw data
                raw_path = raw_store.save(df, f"{ticker}_ohlcv")

                result = {
                    "ticker": ticker,
                    "rows": len(df),
                    "start_date": str(df.index.min().date()),
                    "end_date": str(df.index.max().date()),
                    "status": "success",
                    "raw_path": str(raw_path),
                }

                progress.update(task, description=f"[green]Fetched {ticker}: {len(df)} rows[/green]")

            except DataFetchError as e:
                result = {
                    "ticker": ticker,
                    "rows": 0,
                    "start_date": None,
                    "end_date": None,
                    "status": f"error: {e}",
                    "raw_path": None,
                }
                progress.update(task, description=f"[red]Failed {ticker}: {e}[/red]")

            results.append(result)

        # Fetch VIX data
        task = progress.add_task("Fetching VIX...", total=None)
        try:
            vix_df = loader.fetch_vix_data()
            raw_store.save(vix_df, "VIX_ohlcv")
            results.append({
                "ticker": "VIX",
                "rows": len(vix_df),
                "start_date": str(vix_df.index.min().date()),
                "end_date": str(vix_df.index.max().date()),
                "status": "success",
            })
            progress.update(task, description=f"[green]Fetched VIX: {len(vix_df)} rows[/green]")
        except DataFetchError as e:
            results.append({
                "ticker": "VIX",
                "rows": 0,
                "status": f"error: {e}",
            })
            progress.update(task, description=f"[red]Failed VIX: {e}[/red]")

    # Calculate features if requested
    if with_features:
        console.print("\n[bold]Calculating technical indicators...[/bold]")
        indicators = TechnicalIndicators()

        for result in results:
            if result["status"] == "success" and result["ticker"] != "VIX":
                ticker = result["ticker"]
                console.print(f"  Processing {ticker}...")

                try:
                    # Load raw data
                    df = raw_store.load(f"{ticker}_ohlcv")

                    # Merge with VIX
                    vix_df = raw_store.load("VIX_ohlcv")
                    df = loader.merge_with_vix(df, vix_df)

                    # Calculate indicators
                    features_df = indicators.calculate_all(df)

                    # Save features
                    feature_store.save_features(
                        features_df,
                        ticker=ticker,
                        feature_set="technical",
                        description="Technical indicators and volatility features",
                    )
                    console.print(f"    [green]Saved {len(features_df.columns)} features[/green]")

                except Exception as e:
                    console.print(f"    [red]Error: {e}[/red]")

    # Print summary table
    console.print("\n[bold]Summary[/bold]")
    table = Table()
    table.add_column("Ticker", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Date Range")
    table.add_column("Status")

    for result in results:
        status_style = "green" if result["status"] == "success" else "red"
        date_range = f"{result.get('start_date', 'N/A')} to {result.get('end_date', 'N/A')}"
        table.add_row(
            result["ticker"],
            str(result["rows"]),
            date_range,
            f"[{status_style}]{result['status']}[/{status_style}]"
        )

    console.print(table)

    # Print data locations
    console.print(f"\n[bold]Data Locations:[/bold]")
    console.print(f"  Raw data:  {settings.get_data_path('raw')}")
    if with_features:
        console.print(f"  Features:  {settings.get_data_path('features')}")

    # Exit with error if any failed
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        console.print(f"\n[yellow]Warning: {len(failed)} ticker(s) failed[/yellow]")
        raise typer.Exit(code=1)

    console.print("\n[bold green]Data fetch complete![/bold green]")


@app.command()
def status() -> None:
    """Show current data status."""
    settings = get_settings()

    console.print("\n[bold]Data Status[/bold]")

    # Check raw data
    raw_store = RawDataStore()
    raw_files = raw_store.list_files()

    table = Table(title="Raw Data Files")
    table.add_column("File")
    table.add_column("Exists", justify="center")

    for ticker in settings.tickers + ["VIX"]:
        name = f"{ticker.replace('^', '')}_ohlcv"
        exists = raw_store.exists(name)
        status = "[green]Yes[/green]" if exists else "[red]No[/red]"
        table.add_row(name, status)

    console.print(table)

    # Check features
    feature_store = FeatureStore()
    features = feature_store.list_features()

    if features:
        table = Table(title="Feature Sets")
        table.add_column("Key")
        table.add_column("Rows")
        table.add_column("Features")
        table.add_column("Date Range")

        for f in features:
            date_range = f"{f['date_range']['start']} to {f['date_range']['end']}"
            table.add_row(
                f["key"],
                str(f["num_rows"]),
                str(f["num_features"]),
                date_range
            )

        console.print(table)
    else:
        console.print("\n[yellow]No feature sets found. Run with --with-features to generate.[/yellow]")


@app.command()
def clean() -> None:
    """Remove all data files."""
    settings = get_settings()

    if typer.confirm("Are you sure you want to delete all data files?"):
        import shutil

        for subdir in ["raw", "processed", "features"]:
            path = settings.get_data_path(subdir)
            if path.exists():
                shutil.rmtree(path)
                console.print(f"[yellow]Removed {path}[/yellow]")

        console.print("[green]Data cleaned.[/green]")
    else:
        console.print("Cancelled.")


if __name__ == "__main__":
    app()
