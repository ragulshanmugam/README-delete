#!/usr/bin/env python3
"""
Week 1 Day 1 - Data Exploration Script
=====================================
Run this to explore the data pipeline output.

Usage:
    python notebooks/01_data_exploration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# ============================================================
# SECTION 1: Load and Inspect Data
# ============================================================

print("=" * 60)
print("ML OPTIONS TRADING - DATA EXPLORATION")
print("=" * 60)

# Load raw data
raw_dir = project_root / "data" / "raw"
features_dir = project_root / "data" / "features"

print("\n[1] RAW DATA FILES")
print("-" * 40)

for ticker in ["SPY", "QQQ", "IWM", "VIX"]:
    path = raw_dir / f"{ticker}_ohlcv.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"{ticker}: {len(df)} rows, {df.columns.tolist()[:5]}...")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")

# ============================================================
# SECTION 2: Load Feature Data
# ============================================================

print("\n[2] FEATURE DATA")
print("-" * 40)

# Find latest feature files
feature_files = list(features_dir.glob("*_technical_*.parquet"))
latest_files = {}
for f in feature_files:
    ticker = f.name.split("_")[0]
    if ticker not in latest_files or f.stat().st_mtime > latest_files[ticker].stat().st_mtime:
        latest_files[ticker] = f

# Load features for each ticker
features = {}
for ticker, path in sorted(latest_files.items()):
    df = pd.read_parquet(path)
    features[ticker] = df
    print(f"{ticker}: {len(df)} rows, {len(df.columns)} features")

# Use SPY as primary example
spy = features.get("SPY")
if spy is not None:
    print(f"\nSPY Feature Columns ({len(spy.columns)}):")

    # Group features by category
    categories = {
        "Price/Returns": ["returns", "log_returns", "returns_1d", "returns_5d", "returns_10d", "returns_20d", "returns_60d", "returns_252d"],
        "RSI/Momentum": ["rsi_14", "rsi_28", "stoch_k", "stoch_d", "roc_5", "roc_10", "roc_20", "mfi"],
        "MACD": ["macd", "macd_signal", "macd_histogram", "macd_pct"],
        "Bollinger": ["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position", "bb_distance"],
        "Moving Avg": ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "price_vs_sma20", "price_vs_sma50", "price_vs_sma200"],
        "HV": ["hv_5", "hv_10", "hv_20", "hv_60", "hv_120", "hv_ratio_5_20", "hv_ratio_20_60", "parkinson_hv", "hv_trend"],
        "VIX": ["vix", "vix_rank"],
        "Volume": ["volume_sma_10", "volume_sma_20", "volume_ratio", "volume_surge", "log_volume", "obv", "obv_sma"],
        "52W Range": ["high_52w", "low_52w", "dist_from_high_52w", "dist_from_low_52w", "near_52w_high", "near_52w_low"],
        "ATR": ["atr_14", "atr_pct"],
    }

    for cat, cols in categories.items():
        available = [c for c in cols if c in spy.columns]
        print(f"  {cat}: {available}")

# ============================================================
# SECTION 3: Data Quality Check
# ============================================================

print("\n[3] DATA QUALITY CHECK")
print("-" * 40)

if spy is not None:
    # Check for missing values
    missing = spy.isnull().sum()
    missing_pct = (missing / len(spy) * 100).round(2)

    print("\nMissing Values (columns with any NaN):")
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col in missing_cols.index[:10]:  # Show first 10
            print(f"  {col}: {missing_cols[col]} ({missing_pct[col]}%)")
        if len(missing_cols) > 10:
            print(f"  ... and {len(missing_cols) - 10} more columns")
    else:
        print("  No missing values!")

    # Check for infinite values
    numeric_cols = spy.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(spy[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]

    print("\nInfinite Values:")
    if len(inf_cols) > 0:
        for col in inf_cols.index:
            print(f"  {col}: {inf_cols[col]}")
    else:
        print("  No infinite values!")

# ============================================================
# SECTION 4: Feature Statistics
# ============================================================

print("\n[4] FEATURE STATISTICS (SPY)")
print("-" * 40)

if spy is not None:
    # Key features to examine
    key_features = [
        "returns", "rsi_14", "bb_position", "hv_20", "vix", "volume_ratio", "atr_pct"
    ]

    available_key = [f for f in key_features if f in spy.columns]

    if available_key:
        stats = spy[available_key].describe().T
        stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        print(stats.round(4).to_string())

# ============================================================
# SECTION 5: Target Variable Preview
# ============================================================

print("\n[5] PREVIEW: DIRECTION TARGETS")
print("-" * 40)

if spy is not None and "returns" in spy.columns:
    # Create forward returns (what we'll predict)
    spy_clean = spy.dropna(subset=["returns"])

    # 1-day forward return
    spy_clean = spy_clean.copy()
    spy_clean["fwd_return_1d"] = spy_clean["returns"].shift(-1)

    # Direction (UP = 1, DOWN = 0)
    spy_clean["direction_1d"] = (spy_clean["fwd_return_1d"] > 0).astype(int)

    # Count
    direction_counts = spy_clean["direction_1d"].value_counts()

    print("\n1-Day Direction Distribution:")
    print(f"  UP days (>0%):   {direction_counts.get(1, 0)} ({direction_counts.get(1, 0)/len(spy_clean)*100:.1f}%)")
    print(f"  DOWN days (<=0%): {direction_counts.get(0, 0)} ({direction_counts.get(0, 0)/len(spy_clean)*100:.1f}%)")

    # This is our baseline accuracy (always predict majority class)
    majority = direction_counts.max() / len(spy_clean.dropna(subset=["direction_1d"]))
    print(f"\n  Baseline accuracy (always predict majority): {majority*100:.1f}%")
    print(f"  Our ML model needs to beat: {majority*100:.1f}%")

# ============================================================
# SECTION 6: Correlation Preview
# ============================================================

print("\n[6] FEATURE CORRELATIONS WITH RETURNS")
print("-" * 40)

if spy is not None and "returns" in spy.columns:
    # Calculate correlations with returns
    numeric_cols = spy.select_dtypes(include=[np.number]).columns
    correlations = spy[numeric_cols].corrwith(spy["returns"]).abs().sort_values(ascending=False)

    print("\nTop 10 features correlated with returns:")
    for i, (col, corr) in enumerate(correlations.head(11).items()):
        if col != "returns":  # Skip self-correlation
            print(f"  {i}. {col}: {corr:.4f}")

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETE!")
print("=" * 60)

print("""
NEXT STEPS:
-----------
1. Run: python -m src.models.direction_classifier --help
   (Build the direction prediction model)

2. Or explore more in a Jupyter notebook:
   jupyter notebook notebooks/

3. Key files to know:
   - data/raw/*.parquet       -> Raw OHLCV data
   - data/features/*.parquet  -> Engineered features
   - src/data/feature_store.py -> Load features programmatically
""")
