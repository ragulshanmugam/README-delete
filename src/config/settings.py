"""
Configuration management for the ML Options Trading System.

Uses Pydantic for validation and supports environment variables.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with validation.

    Settings are loaded from environment variables with fallback to defaults.
    Use a .env file for local development.
    """

    # Project paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Root directory of the project"
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for storing data files"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )

    # Ticker configuration
    tickers: List[str] = Field(
        default=["SPY", "QQQ", "IWM"],
        description="List of tickers to trade"
    )
    vix_symbol: str = Field(
        default="^VIX",
        description="VIX symbol for yfinance"
    )

    # Data fetching
    lookback_years: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Years of historical data to fetch"
    )
    rate_limit_calls: int = Field(
        default=100,
        description="Max API calls per minute"
    )
    rate_limit_period: int = Field(
        default=60,
        description="Rate limit period in seconds"
    )

    # Feature engineering
    feature_windows: List[int] = Field(
        default=[5, 10, 20, 60, 120, 252],
        description="Window sizes for rolling calculations"
    )

    # Model configuration
    direction_threshold: float = Field(
        default=0.01,
        description="Return threshold for bullish/bearish classification (1%)"
    )
    prediction_horizon: int = Field(
        default=5,
        description="Days ahead for predictions"
    )

    # Risk management
    max_position_size: float = Field(
        default=0.02,
        description="Maximum risk per position (2% of account)"
    )
    max_portfolio_exposure: float = Field(
        default=0.20,
        description="Maximum total portfolio exposure (20%)"
    )
    max_positions: int = Field(
        default=5,
        description="Maximum concurrent positions"
    )

    # Regime thresholds
    vix_low_threshold: float = Field(
        default=15.0,
        description="VIX level for low volatility regime"
    )
    vix_high_threshold: float = Field(
        default=25.0,
        description="VIX level for high volatility regime"
    )
    vix_rank_low: float = Field(
        default=0.30,
        description="VIX rank for low volatility"
    )
    vix_rank_high: float = Field(
        default=0.80,
        description="VIX rank for high volatility"
    )
    hv_iv_contraction: float = Field(
        default=0.60,
        description="HV/IV ratio for IV contraction"
    )
    hv_iv_expansion: float = Field(
        default=0.90,
        description="HV/IV ratio for IV expansion"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )

    # Database (for future use)
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection string"
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="file:./mlruns",
        description="MLflow tracking URI"
    )

    # FRED API (Free: https://fred.stlouisfed.org/docs/api/api_key.html)
    fred_api_key: Optional[str] = Field(
        default=None,
        description="FRED API key for macro data"
    )

    # Broker configuration (for future use)
    broker_paper_mode: bool = Field(
        default=True,
        description="Use paper trading mode"
    )
    broker_port: int = Field(
        default=7497,
        description="Broker API port (7497 for paper, 7496 for live)"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("tickers", mode="before")
    @classmethod
    def parse_tickers(cls, v):
        """Parse tickers from comma-separated string or list."""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @field_validator("feature_windows", mode="before")
    @classmethod
    def parse_feature_windows(cls, v):
        """Parse feature windows from comma-separated string or list."""
        if isinstance(v, str):
            return [int(w.strip()) for w in v.split(",") if w.strip()]
        return v

    @field_validator("data_dir", "log_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Union[str, Path]) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    def get_data_path(self, subdir: str = "") -> Path:
        """
        Get absolute path to data directory.

        Args:
            subdir: Optional subdirectory within data dir

        Returns:
            Absolute path to data directory
        """
        base = self.project_root / self.data_dir
        if subdir:
            base = base / subdir
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_log_path(self) -> Path:
        """
        Get absolute path to log directory.

        Returns:
            Absolute path to log directory
        """
        path = self.project_root / self.log_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_regime_thresholds(self) -> dict:
        """
        Get regime classification thresholds as a dictionary.

        Returns:
            Dictionary of regime thresholds
        """
        return {
            "vix_low": self.vix_low_threshold,
            "vix_high": self.vix_high_threshold,
            "vix_rank_low": self.vix_rank_low,
            "vix_rank_high": self.vix_rank_high,
            "hv_iv_contraction": self.hv_iv_contraction,
            "hv_iv_expansion": self.hv_iv_expansion,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance (cached for performance)
    """
    return Settings()


# Convenience function for accessing settings
def get_setting(key: str, default: any = None) -> any:
    """
    Get a specific setting value.

    Args:
        key: Setting attribute name
        default: Default value if setting doesn't exist

    Returns:
        Setting value or default
    """
    settings = get_settings()
    return getattr(settings, key, default)
