"""
Feature pipeline for ML model training.

Loads technical features from parquet, macro features from FRED,
merges them correctly, and creates target labels for training.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.config.settings import get_settings
from src.data.feature_store import FeatureStore, RawDataStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Classification modes
CLASSIFICATION_BINARY = "binary"
CLASSIFICATION_TERNARY = "ternary"


class FeaturePipeline:
    """
    Pipeline for preparing features for ML model training.

    Handles:
    - Loading technical features from parquet files
    - Loading macro features from FRED (with graceful degradation)
    - Merging features on date index
    - Creating target labels (5-day forward returns)
    - Handling missing values
    - Returning train-ready X, y DataFrames

    Attributes:
        ticker: Stock ticker symbol
        prediction_horizon: Days ahead for prediction target
        direction_threshold: Return threshold for classification
        include_macro: Whether to include macro features
    """

    # Columns that should never be used as features
    EXCLUDE_COLUMNS = [
        "open", "high", "low", "close", "volume",  # Raw OHLCV
        "dividends", "stock_splits",  # Corporate actions
        "target", "target_returns", "forward_returns",  # Target columns
        "ticker", "symbol",  # Identifier columns (non-numeric)
        "iv_regime",  # Categorical string (use iv_regime_low/high binary flags instead)
    ]

    # Macro columns that are raw series (not engineered)
    MACRO_RAW_COLUMNS = [
        "cpi_yoy", "unemployment_rate",  # Monthly data - may have issues
    ]

    def __init__(
        self,
        ticker: str = "SPY",
        prediction_horizon: Optional[int] = None,
        direction_threshold: Optional[float] = None,
        include_macro: bool = True,
        include_sentiment: bool = False,
        include_earnings: bool = False,
        classification_mode: str = CLASSIFICATION_BINARY,
        n_features: Optional[int] = None,
    ):
        """
        Initialize the feature pipeline.

        Args:
            ticker: Stock ticker symbol
            prediction_horizon: Days ahead for prediction (default from settings)
            direction_threshold: Return threshold for classification (default from settings)
            include_macro: Whether to include macro features (set False if FRED unavailable)
            include_sentiment: Whether to include sentiment features from Finnhub/Reddit
            classification_mode: "binary" (UP/DOWN) or "ternary" (BEARISH/NEUTRAL/BULLISH)
            n_features: Number of top features to select (None = use all)
        """
        settings = get_settings()
        self.ticker = ticker
        self.prediction_horizon = prediction_horizon or settings.prediction_horizon
        self.direction_threshold = direction_threshold or settings.direction_threshold
        self.include_macro = include_macro
        self.include_sentiment = include_sentiment
        self.include_earnings = include_earnings
        self.classification_mode = classification_mode
        self.n_features = n_features

        # Initialize stores
        self.feature_store = FeatureStore()
        self.raw_store = RawDataStore()

        # Cache for macro data
        self._macro_cache: Optional[pd.DataFrame] = None

        # Feature selection state
        self._selected_features: Optional[List[str]] = None
        self._feature_selector = None

        logger.info(
            f"Initialized FeaturePipeline: ticker={ticker}, "
            f"horizon={self.prediction_horizon}d, threshold={self.direction_threshold:.1%}, "
            f"include_macro={include_macro}, mode={classification_mode}, n_features={n_features}"
        )

    def load_technical_features(self) -> pd.DataFrame:
        """
        Load technical features from parquet file.

        Returns:
            DataFrame with technical indicators

        Raises:
            FileNotFoundError: If feature file not found
        """
        logger.info(f"Loading technical features for {self.ticker}")

        try:
            df = self.feature_store.load_features(self.ticker, "technical")
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

            # Add IV features if not already present
            if "iv_rank" not in df.columns:
                df = self._add_iv_features(df)

            return df
        except FileNotFoundError:
            # Try loading from raw data and calculating features
            logger.warning(
                f"No cached features found for {self.ticker}. "
                "Attempting to load raw data and calculate features."
            )
            return self._calculate_technical_features()

    def _calculate_technical_features(self) -> pd.DataFrame:
        """
        Calculate technical features from raw OHLCV data.

        Returns:
            DataFrame with calculated technical indicators
        """
        from src.features.technical_indicators import TechnicalIndicators

        # Load raw data
        raw_df = self.raw_store.load(f"{self.ticker}_ohlcv")

        # Load VIX and merge
        try:
            vix_df = self.raw_store.load("VIX_ohlcv")
            raw_df["vix_close"] = vix_df["close"].reindex(raw_df.index)
            raw_df["vix_close"] = raw_df["vix_close"].ffill()
        except FileNotFoundError:
            logger.warning("VIX data not found, proceeding without it")

        # Calculate indicators
        indicators = TechnicalIndicators()
        features_df = indicators.calculate_all(raw_df)

        # Add IV features
        features_df = self._add_iv_features(features_df)

        # Save for future use
        self.feature_store.save_features(
            features_df,
            ticker=self.ticker,
            feature_set="technical",
            description="Auto-generated technical indicators with IV features",
        )

        return features_df

    def _add_iv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add IV (Implied Volatility) features using VIX as proxy.

        Args:
            df: DataFrame with technical features (must have hv_20 column)

        Returns:
            DataFrame with IV features added
        """
        try:
            from src.features.iv_indicators import IVIndicators

            iv_calc = IVIndicators()
            df = iv_calc.calculate_all(df, self.ticker, hv_column="hv_20")
            logger.info(f"Added IV features for {self.ticker}")

        except Exception as e:
            logger.warning(f"Could not add IV features: {e}")

        return df

    def load_macro_features(
        self,
        start_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load macro features from FRED.

        Args:
            start_date: Start date for FRED data (default: 5 years ago)

        Returns:
            DataFrame with macro features, or None if unavailable
        """
        if not self.include_macro:
            logger.info("Macro features disabled")
            return None

        # Use cached data if available
        if self._macro_cache is not None:
            logger.info("Using cached macro features")
            return self._macro_cache

        # Determine start date
        if start_date is None:
            settings = get_settings()
            years_back = settings.lookback_years
            start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

        logger.info(f"Loading macro features from FRED (start: {start_date})")

        try:
            from src.features.macro_indicators import MacroIndicators

            macro = MacroIndicators()
            macro_df = macro.calculate_all(start_date=start_date)

            # Cache for future use
            self._macro_cache = macro_df

            logger.info(f"Loaded {len(macro_df)} rows, {len(macro_df.columns)} macro features")
            return macro_df

        except ImportError as e:
            logger.warning(f"fredapi not installed, skipping macro features: {e}")
            return None
        except ValueError as e:
            logger.warning(f"FRED API key not configured, skipping macro features: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not load macro features: {e}")
            return None

    def create_target_labels(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create target labels for direction classification.

        Labels:
        - 0 (BEARISH): 5-day return < -1%
        - 1 (NEUTRAL): 5-day return between -1% and +1%
        - 2 (BULLISH): 5-day return > +1%

        Args:
            df: DataFrame with price data
            price_col: Column name for price

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        logger.info(
            f"Creating target labels: horizon={self.prediction_horizon}d, "
            f"threshold={self.direction_threshold:.1%}"
        )

        # Calculate forward returns
        forward_returns = df[price_col].shift(-self.prediction_horizon) / df[price_col] - 1

        # Create labels
        labels = pd.Series(index=df.index, dtype=float)
        labels[forward_returns < -self.direction_threshold] = 0  # BEARISH
        labels[forward_returns > self.direction_threshold] = 2   # BULLISH
        labels[
            (forward_returns >= -self.direction_threshold) &
            (forward_returns <= self.direction_threshold)
        ] = 1  # NEUTRAL

        # Store forward returns for analysis
        df = df.copy()
        df["forward_returns"] = forward_returns

        # Log distribution
        valid_labels = labels.dropna()
        if len(valid_labels) > 0:
            counts = valid_labels.value_counts().sort_index()
            total = len(valid_labels)
            logger.info(
                f"Label distribution: "
                f"BEARISH={counts.get(0, 0)}/{total} ({counts.get(0, 0)/total:.1%}), "
                f"NEUTRAL={counts.get(1, 0)}/{total} ({counts.get(1, 0)/total:.1%}), "
                f"BULLISH={counts.get(2, 0)}/{total} ({counts.get(2, 0)/total:.1%})"
            )

        return df, labels

    def create_target_labels_binary(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create binary target labels for UP/DOWN classification.

        Labels:
        - 0 (DOWN): 5-day return < -threshold (significant down move)
        - 1 (UP): 5-day return > +threshold (significant up move)
        - NaN: Returns between -threshold and +threshold are excluded

        This removes ambiguous "neutral" moves to create cleaner signal.

        Args:
            df: DataFrame with price data
            price_col: Column name for price

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        logger.info(
            f"Creating BINARY target labels: horizon={self.prediction_horizon}d, "
            f"threshold={self.direction_threshold:.1%}"
        )

        # Calculate forward returns
        forward_returns = df[price_col].shift(-self.prediction_horizon) / df[price_col] - 1

        # Create binary labels (exclude ambiguous middle zone)
        labels = pd.Series(index=df.index, dtype=float)
        labels[forward_returns < -self.direction_threshold] = 0  # DOWN
        labels[forward_returns > self.direction_threshold] = 1   # UP
        # Leave ambiguous returns as NaN (will be excluded)

        # Store forward returns for analysis
        df = df.copy()
        df["forward_returns"] = forward_returns

        # Log distribution
        valid_labels = labels.dropna()
        excluded = len(df) - len(valid_labels)
        if len(valid_labels) > 0:
            counts = valid_labels.value_counts().sort_index()
            total = len(valid_labels)
            logger.info(
                f"Binary label distribution: "
                f"DOWN={counts.get(0, 0)}/{total} ({counts.get(0, 0)/total:.1%}), "
                f"UP={counts.get(1, 0)}/{total} ({counts.get(1, 0)/total:.1%}), "
                f"excluded={excluded} ambiguous samples"
            )

        return df, labels

    def merge_features(
        self,
        technical_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge technical and macro features on date index.

        Args:
            technical_df: DataFrame with technical indicators
            macro_df: DataFrame with macro indicators (optional)

        Returns:
            Merged DataFrame
        """
        # Ensure datetime index
        if not isinstance(technical_df.index, pd.DatetimeIndex):
            technical_df = technical_df.copy()
            technical_df.index = pd.to_datetime(technical_df.index)

        if macro_df is None:
            logger.info("No macro features to merge")
            return technical_df

        # Ensure datetime index for macro
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df = macro_df.copy()
            macro_df.index = pd.to_datetime(macro_df.index)

        # Identify columns to merge (avoid duplicates)
        existing_cols = set(technical_df.columns)
        macro_cols = [col for col in macro_df.columns if col not in existing_cols]

        if not macro_cols:
            logger.warning("No unique macro columns to merge")
            return technical_df

        # Merge on date
        logger.info(f"Merging {len(macro_cols)} macro features")
        merged = technical_df.join(macro_df[macro_cols], how="left")

        # Forward-fill any gaps in macro data
        merged[macro_cols] = merged[macro_cols].ffill()

        logger.info(f"Merged DataFrame: {len(merged)} rows, {len(merged.columns)} columns")
        return merged

    def select_features(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Select feature columns for training, excluding raw OHLCV and targets.

        Args:
            df: Full DataFrame
            exclude_cols: Additional columns to exclude

        Returns:
            DataFrame with only feature columns
        """
        # Build exclusion list
        exclude = set(self.EXCLUDE_COLUMNS)
        if exclude_cols:
            exclude.update(exclude_cols)

        # Also exclude any columns ending with specific suffixes
        exclude_suffixes = ("_target", "_label")

        # Select feature columns
        feature_cols = [
            col for col in df.columns
            if col not in exclude
            and not col.endswith(exclude_suffixes)
            and not col.startswith("forward_")
        ]

        logger.info(f"Selected {len(feature_cols)} features from {len(df.columns)} columns")
        return df[feature_cols]

    def select_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top K features using mutual information.

        Uses mutual information to identify features with the strongest
        relationship to the target variable.

        Args:
            X: Feature DataFrame
            y: Labels Series
            k: Number of features to select (default: self.n_features or 30)

        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        k = k or self.n_features or 30  # Default to 30 features

        if k >= len(X.columns):
            logger.info(f"Requested {k} features but only {len(X.columns)} available, using all")
            return X, list(X.columns)

        logger.info(f"Selecting top {k} features using mutual information...")

        # Fit feature selector
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        # Get feature scores for logging
        scores = selector.scores_
        feature_scores = sorted(
            zip(X.columns, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log top features
        logger.info(f"Top {min(10, k)} features by mutual information:")
        for name, score in feature_scores[:10]:
            logger.info(f"  {name}: {score:.4f}")

        # Store for later use
        self._selected_features = selected_features
        self._feature_selector = selector

        return X[selected_features], selected_features

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "smart_fill",
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in feature DataFrame.

        Args:
            df: Feature DataFrame
            strategy: How to handle missing values:
                - "smart_fill": Forward-fill, drop high-NaN cols, median fill, then drop (RECOMMENDED)
                - "drop_rows": Drop rows with any NaN
                - "drop_cols": Drop columns with >50% NaN
                - "fill_forward": Forward-fill NaN values
                - "fill_value": Fill with specified value
                - "fill_median": Fill with column median
            fill_value: Value to use for fill_value strategy

        Returns:
            DataFrame with missing values handled
        """
        initial_rows = len(df)
        initial_nan = df.isna().sum().sum()

        if initial_nan == 0:
            logger.info("No missing values to handle")
            return df

        logger.info(f"Handling {initial_nan} missing values with strategy: {strategy}")

        result = df.copy()

        if strategy == "drop_rows":
            result = result.dropna()
            dropped = initial_rows - len(result)
            logger.info(f"Dropped {dropped} rows with missing values")

        elif strategy == "drop_cols":
            # Drop columns with >50% missing
            threshold = 0.5 * len(result)
            cols_to_drop = result.columns[result.isna().sum() > threshold]
            result = result.drop(columns=cols_to_drop)
            # Then drop remaining rows with NaN
            result = result.dropna()
            logger.info(f"Dropped {len(cols_to_drop)} columns, then remaining NaN rows")

        elif strategy == "fill_forward":
            result = result.ffill()
            # Drop any remaining NaN at the beginning
            result = result.dropna()
            logger.info("Forward-filled missing values")

        elif strategy == "fill_value":
            if fill_value is None:
                fill_value = 0.0
            result = result.fillna(fill_value)
            logger.info(f"Filled missing values with {fill_value}")

        elif strategy == "fill_median":
            for col in result.columns:
                if result[col].isna().any():
                    median_val = result[col].median()
                    result[col] = result[col].fillna(median_val)
            logger.info("Filled missing values with column medians")

        elif strategy == "smart_fill":
            # Smart fill strategy - preserves maximum samples
            # Step 1: Forward-fill (appropriate for time series)
            result = result.ffill()
            filled_by_ffill = initial_nan - result.isna().sum().sum()

            # Step 2: Drop columns with >30% still missing (at start of series)
            missing_pct = result.isna().sum() / len(result)
            cols_to_keep = missing_pct[missing_pct < 0.3].index.tolist()
            cols_dropped = len(result.columns) - len(cols_to_keep)
            result = result[cols_to_keep]

            # Step 3: Fill remaining NaN with column median
            for col in result.columns:
                if result[col].isna().any():
                    median_val = result[col].median()
                    if pd.notna(median_val):
                        result[col] = result[col].fillna(median_val)
                    else:
                        # Column is all NaN, fill with 0
                        result[col] = result[col].fillna(0)

            # Step 4: Drop any rows still with NaN (should be minimal)
            rows_before = len(result)
            result = result.dropna()
            rows_dropped = rows_before - len(result)

            logger.info(
                f"Smart fill: ffill recovered {filled_by_ffill} values, "
                f"dropped {cols_dropped} columns, "
                f"median filled remaining, "
                f"dropped {rows_dropped} rows"
            )

        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        final_nan = result.isna().sum().sum()
        logger.info(f"Missing values: {initial_nan} -> {final_nan}")

        return result

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment features from Finnhub and Reddit.

        Note: For historical data, we use current sentiment as a static proxy
        since historical sentiment data requires premium API access.
        For live trading, sentiment is fetched fresh each day.

        Args:
            df: DataFrame with existing features

        Returns:
            DataFrame with sentiment features added
        """
        try:
            from src.features.sentiment_indicators import calculate_sentiment_features

            logger.info(f"Adding sentiment features for {self.ticker}")
            result = calculate_sentiment_features(df, self.ticker)

            # Count new sentiment features
            sentiment_cols = [c for c in result.columns if c not in df.columns]
            logger.info(f"Added {len(sentiment_cols)} sentiment features")

            return result

        except ImportError:
            logger.warning("Sentiment indicators module not available, skipping")
            return df
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            return df

    def add_earnings_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add earnings calendar features from Finnhub.

        Note: For historical data, we use current earnings calendar as a proxy
        since historical earnings data requires premium API access.
        For live trading, earnings data is fetched fresh.

        Args:
            df: DataFrame with existing features

        Returns:
            DataFrame with earnings features added
        """
        try:
            from src.features.earnings_calendar import calculate_earnings_features

            logger.info(f"Adding earnings features for {self.ticker}")
            result = calculate_earnings_features(df, self.ticker)

            # Count new earnings features
            earnings_cols = [c for c in result.columns if c not in df.columns]
            logger.info(f"Added {len(earnings_cols)} earnings features")

            return result

        except ImportError:
            logger.warning("Earnings calendar module not available, skipping")
            return df
        except Exception as e:
            logger.error(f"Error adding earnings features: {e}")
            return df

    def prepare_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        missing_value_strategy: str = "smart_fill",
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Full pipeline to prepare training data.

        Args:
            start_date: Filter data from this date (optional)
            end_date: Filter data until this date (optional)
            missing_value_strategy: How to handle missing values (default: smart_fill)

        Returns:
            Tuple of (X features, y labels, metadata dict)
        """
        logger.info(f"Preparing training data for {self.ticker} (mode={self.classification_mode})")

        # 1. Load technical features
        technical_df = self.load_technical_features()

        # 2. Load macro features (may return None)
        macro_df = self.load_macro_features()

        # 3. Merge features
        merged_df = self.merge_features(technical_df, macro_df)

        # 3b. Add sentiment features if enabled
        if self.include_sentiment:
            merged_df = self.add_sentiment_features(merged_df)

        # 3c. Add earnings features if enabled
        if self.include_earnings:
            merged_df = self.add_earnings_features(merged_df)

        # 4. Filter date range if specified
        if start_date:
            merged_df = merged_df[merged_df.index >= pd.to_datetime(start_date)]
        if end_date:
            merged_df = merged_df[merged_df.index <= pd.to_datetime(end_date)]

        logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")

        # 5. Create target labels (binary or ternary based on mode)
        if self.classification_mode == CLASSIFICATION_BINARY:
            merged_df, y = self.create_target_labels_binary(merged_df)
        else:
            merged_df, y = self.create_target_labels(merged_df)

        # 6. Select feature columns
        X = self.select_features(merged_df)

        # 7. Handle missing values
        # First, align X and y (remove samples with NaN labels)
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask].astype(int)

        # Then handle missing values in features
        X = self.handle_missing_values(X, strategy=missing_value_strategy)

        # Align y with cleaned X
        y = y.loc[X.index]

        # 8. Feature selection (if n_features is set)
        if self.n_features is not None and self.n_features < len(X.columns):
            X, selected_features = self.select_top_features(X, y, k=self.n_features)
        else:
            selected_features = list(X.columns)

        # 9. Prepare metadata
        metadata = {
            "ticker": self.ticker,
            "prediction_horizon": self.prediction_horizon,
            "direction_threshold": self.direction_threshold,
            "include_macro": self.include_macro,
            "classification_mode": self.classification_mode,
            "n_features_selected": self.n_features,
            "date_range": {
                "start": str(X.index.min()),
                "end": str(X.index.max()),
            },
            "num_samples": len(X),
            "num_features": len(X.columns),
            "feature_names": list(X.columns),
            "label_distribution": y.value_counts().to_dict(),
            "missing_value_strategy": missing_value_strategy,
        }

        logger.info(
            f"Training data prepared: {len(X)} samples, {len(X.columns)} features"
        )

        return X, y, metadata

    def get_feature_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Group features by category for analysis.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping category to list of features
        """
        groups: Dict[str, List[str]] = {
            "returns": [],
            "rsi": [],
            "macd": [],
            "bollinger": [],
            "moving_average": [],
            "volatility": [],
            "iv_features": [],
            "momentum": [],
            "volume": [],
            "price_levels": [],
            "macro_rates": [],
            "macro_events": [],
            "macro_risk": [],
            "other": [],
        }

        for feature in feature_names:
            feature_lower = feature.lower()

            if "returns" in feature_lower or "roc" in feature_lower:
                groups["returns"].append(feature)
            elif "rsi" in feature_lower:
                groups["rsi"].append(feature)
            elif "macd" in feature_lower:
                groups["macd"].append(feature)
            elif "bb_" in feature_lower or "bollinger" in feature_lower:
                groups["bollinger"].append(feature)
            elif "sma" in feature_lower or "ema" in feature_lower or "ma_" in feature_lower:
                groups["moving_average"].append(feature)
            elif any(x in feature_lower for x in ["iv_", "vix_", "vix", "contango"]):
                groups["iv_features"].append(feature)
            elif "hv_" in feature_lower or "volatility" in feature_lower or "atr" in feature_lower:
                groups["volatility"].append(feature)
            elif "stoch" in feature_lower or "mfi" in feature_lower or "momentum" in feature_lower:
                groups["momentum"].append(feature)
            elif "volume" in feature_lower or "obv" in feature_lower:
                groups["volume"].append(feature)
            elif "52w" in feature_lower or "high_" in feature_lower or "low_" in feature_lower:
                groups["price_levels"].append(feature)
            elif any(x in feature_lower for x in ["yield", "fed_funds", "rate", "spread", "inflation"]):
                groups["macro_rates"].append(feature)
            elif any(x in feature_lower for x in ["fomc", "nfp", "cpi", "event"]):
                groups["macro_events"].append(feature)
            elif any(x in feature_lower for x in ["hy_", "risk", "regime"]):
                groups["macro_risk"].append(feature)
            else:
                groups["other"].append(feature)

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}

        return groups


def load_training_data(
    ticker: str = "SPY",
    include_macro: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Convenience function to load training data.

    Args:
        ticker: Stock ticker symbol
        include_macro: Whether to include macro features
        start_date: Filter data from this date
        end_date: Filter data until this date

    Returns:
        Tuple of (X features, y labels, metadata)
    """
    pipeline = FeaturePipeline(ticker=ticker, include_macro=include_macro)
    return pipeline.prepare_training_data(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    # Quick test
    print("Testing FeaturePipeline...")

    # Initialize pipeline
    pipeline = FeaturePipeline(ticker="SPY", include_macro=False)

    # Prepare training data
    X, y, metadata = pipeline.prepare_training_data()

    print(f"\nTraining Data:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    print(f"  Label distribution: {metadata['label_distribution']}")

    # Show feature groups
    groups = pipeline.get_feature_groups(list(X.columns))
    print(f"\nFeature groups:")
    for group, features in groups.items():
        print(f"  {group}: {len(features)} features")

    print(f"\nSample features:\n{X.head()}")
