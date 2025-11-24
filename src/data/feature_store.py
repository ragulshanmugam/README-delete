"""
Feature store for saving and loading feature data.

Provides versioned storage of features as parquet files with metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_dataframe_info

logger = get_logger(__name__)


class FeatureStore:
    """
    Feature store for managing feature data.

    Stores features as parquet files with metadata for versioning
    and reproducibility.

    Attributes:
        base_dir: Base directory for feature storage
        metadata: Metadata for stored features
    """

    METADATA_FILE = "feature_metadata.json"

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the feature store.

        Args:
            base_dir: Base directory for feature storage (default from settings)
        """
        settings = get_settings()
        self.base_dir = base_dir or settings.get_data_path("features")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()
        logger.info(f"Initialized FeatureStore at {self.base_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from file.

        Returns:
            Metadata dictionary
        """
        metadata_path = self.base_dir / self.METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {"features": {}, "created": datetime.now().isoformat()}

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        metadata_path = self.base_dir / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _generate_filename(
        self,
        ticker: str,
        feature_set: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Generate filename for feature file.

        Args:
            ticker: Ticker symbol
            feature_set: Name of feature set (e.g., 'technical', 'volatility')
            version: Version string (default: current timestamp)

        Returns:
            Filename string
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ticker}_{feature_set}_v{version}.parquet"

    def save_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        feature_set: str,
        description: str = "",
        feature_list: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save features to parquet file.

        Args:
            df: DataFrame with features
            ticker: Ticker symbol
            feature_set: Name of feature set
            description: Description of features
            feature_list: List of feature names (auto-detected if None)
            version: Version string (auto-generated if None)

        Returns:
            Path to saved file
        """
        # Generate filename
        filename = self._generate_filename(ticker, feature_set, version)
        file_path = self.base_dir / filename

        # Save parquet
        df.to_parquet(file_path, compression="snappy")

        # Update metadata
        feature_key = f"{ticker}_{feature_set}"
        self.metadata["features"][feature_key] = {
            "filename": filename,
            "ticker": ticker,
            "feature_set": feature_set,
            "description": description,
            "features": feature_list or list(df.columns),
            "num_rows": len(df),
            "num_features": len(df.columns),
            "date_range": {
                "start": str(df.index.min()) if len(df) > 0 else None,
                "end": str(df.index.max()) if len(df) > 0 else None,
            },
            "created_at": datetime.now().isoformat(),
            "version": version or datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        self._save_metadata()

        logger.info(f"Saved features to {file_path}")
        log_dataframe_info(logger, df, f"{ticker}_{feature_set}")

        return file_path

    def load_features(
        self,
        ticker: str,
        feature_set: str,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load features from parquet file.

        Args:
            ticker: Ticker symbol
            feature_set: Name of feature set
            version: Specific version to load (default: latest)

        Returns:
            DataFrame with features

        Raises:
            FileNotFoundError: If feature file not found
        """
        feature_key = f"{ticker}_{feature_set}"

        if feature_key not in self.metadata["features"]:
            raise FileNotFoundError(
                f"No features found for {feature_key}. "
                f"Available: {list(self.metadata['features'].keys())}"
            )

        meta = self.metadata["features"][feature_key]
        file_path = self.base_dir / meta["filename"]

        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        df = pd.read_parquet(file_path)
        logger.info(f"Loaded features from {file_path}")
        log_dataframe_info(logger, df, feature_key)

        return df

    def load_latest(
        self,
        ticker: str,
        feature_set: str,
    ) -> pd.DataFrame:
        """
        Load the latest version of features.

        Args:
            ticker: Ticker symbol
            feature_set: Name of feature set

        Returns:
            DataFrame with features
        """
        return self.load_features(ticker, feature_set)

    def list_features(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available feature sets.

        Args:
            ticker: Filter by ticker (optional)

        Returns:
            List of feature metadata dictionaries
        """
        features = []
        for key, meta in self.metadata["features"].items():
            if ticker is None or meta["ticker"] == ticker:
                features.append({
                    "key": key,
                    **meta
                })
        return features

    def get_feature_info(
        self,
        ticker: str,
        feature_set: str,
    ) -> Dict[str, Any]:
        """
        Get metadata for a feature set.

        Args:
            ticker: Ticker symbol
            feature_set: Name of feature set

        Returns:
            Metadata dictionary

        Raises:
            KeyError: If feature set not found
        """
        feature_key = f"{ticker}_{feature_set}"
        if feature_key not in self.metadata["features"]:
            raise KeyError(f"Feature set not found: {feature_key}")
        return self.metadata["features"][feature_key]

    def delete_features(
        self,
        ticker: str,
        feature_set: str,
    ) -> bool:
        """
        Delete a feature set.

        Args:
            ticker: Ticker symbol
            feature_set: Name of feature set

        Returns:
            True if deleted, False if not found
        """
        feature_key = f"{ticker}_{feature_set}"

        if feature_key not in self.metadata["features"]:
            logger.warning(f"Feature set not found: {feature_key}")
            return False

        meta = self.metadata["features"][feature_key]
        file_path = self.base_dir / meta["filename"]

        # Delete file
        if file_path.exists():
            file_path.unlink()

        # Update metadata
        del self.metadata["features"][feature_key]
        self._save_metadata()

        logger.info(f"Deleted feature set: {feature_key}")
        return True

    def feature_exists(
        self,
        ticker: str,
        feature_set: str,
    ) -> bool:
        """
        Check if a feature set exists.

        Args:
            ticker: Ticker symbol
            feature_set: Name of feature set

        Returns:
            True if exists, False otherwise
        """
        feature_key = f"{ticker}_{feature_set}"
        return feature_key in self.metadata["features"]


class RawDataStore:
    """
    Store for raw market data (OHLCV, VIX).

    Simpler interface than FeatureStore for raw data.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize raw data store.

        Args:
            base_dir: Base directory (default: data/raw)
        """
        settings = get_settings()
        self.base_dir = base_dir or settings.get_data_path("raw")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized RawDataStore at {self.base_dir}")

    def save(self, df: pd.DataFrame, name: str) -> Path:
        """
        Save raw data to parquet.

        Args:
            df: DataFrame to save
            name: Name for the file (without extension)

        Returns:
            Path to saved file
        """
        file_path = self.base_dir / f"{name}.parquet"
        df.to_parquet(file_path, compression="snappy")
        logger.info(f"Saved raw data to {file_path}")
        return file_path

    def load(self, name: str) -> pd.DataFrame:
        """
        Load raw data from parquet.

        Args:
            name: Name of the file (without extension)

        Returns:
            DataFrame

        Raises:
            FileNotFoundError: If file not found
        """
        file_path = self.base_dir / f"{name}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data not found: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded raw data from {file_path}")
        return df

    def exists(self, name: str) -> bool:
        """
        Check if raw data file exists.

        Args:
            name: Name of the file

        Returns:
            True if exists
        """
        file_path = self.base_dir / f"{name}.parquet"
        return file_path.exists()

    def list_files(self) -> List[str]:
        """
        List all raw data files.

        Returns:
            List of file names (without extension)
        """
        files = list(self.base_dir.glob("*.parquet"))
        return [f.stem for f in files]


if __name__ == "__main__":
    # Quick test
    import numpy as np

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "feature_3": np.random.randn(100),
    }, index=dates)

    # Test feature store
    store = FeatureStore()
    path = store.save_features(
        df,
        ticker="SPY",
        feature_set="test",
        description="Test features"
    )
    print(f"Saved to: {path}")

    # Load back
    loaded = store.load_features("SPY", "test")
    print(f"Loaded shape: {loaded.shape}")

    # List features
    print(f"Available features: {store.list_features()}")
