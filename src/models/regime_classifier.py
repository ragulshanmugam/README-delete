"""
Regime classifier for market regime detection.

Rule-based classification of market regimes based on VIX and volatility metrics.
"""

from enum import Enum
from typing import Dict, Optional

import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime enumeration."""

    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"
    IV_EXPANSION = "iv_expansion"
    IV_CONTRACTION = "iv_contraction"
    NEUTRAL = "neutral"


class RegimeClassifier:
    """
    Rule-based market regime classifier.

    Classifies the current market into one of five regimes based on
    VIX levels, VIX rank, and HV/IV ratio.

    Regimes:
    - low_vol: VIX < 15 AND VIX_rank < 30%
    - high_vol: VIX > 25 OR VIX_rank > 80%
    - iv_expansion: HV/IV > 0.9 AND VIX_rank > 50%
    - iv_contraction: HV/IV < 0.6 AND VIX_rank < 50%
    - neutral: None of the above

    Attributes:
        thresholds: Dictionary of regime thresholds
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize the regime classifier.

        Args:
            thresholds: Dictionary of threshold values (default from settings)
        """
        settings = get_settings()
        self.thresholds = thresholds or settings.get_regime_thresholds()

        logger.info(f"Initialized RegimeClassifier with thresholds: {self.thresholds}")

    def classify(
        self,
        vix: float,
        vix_rank: float,
        hv_iv_ratio: float,
    ) -> MarketRegime:
        """
        Classify the current market regime.

        Args:
            vix: Current VIX level
            vix_rank: VIX percentile rank (0-1)
            hv_iv_ratio: Historical volatility / Implied volatility ratio

        Returns:
            MarketRegime enum value
        """
        # Low volatility regime
        if vix < self.thresholds["vix_low"] and vix_rank < self.thresholds["vix_rank_low"]:
            return MarketRegime.LOW_VOL

        # High volatility regime
        if vix > self.thresholds["vix_high"] or vix_rank > self.thresholds["vix_rank_high"]:
            return MarketRegime.HIGH_VOL

        # IV expansion (HV catching up to IV)
        if hv_iv_ratio > self.thresholds["hv_iv_expansion"] and vix_rank > 0.5:
            return MarketRegime.IV_EXPANSION

        # IV contraction (IV overstated relative to HV)
        if hv_iv_ratio < self.thresholds["hv_iv_contraction"] and vix_rank < 0.5:
            return MarketRegime.IV_CONTRACTION

        return MarketRegime.NEUTRAL

    def classify_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify regimes for a DataFrame.

        Args:
            df: DataFrame with columns: vix, vix_rank, hv_20 (or hv_iv_ratio)

        Returns:
            Series with regime classifications
        """
        # Ensure required columns
        required = ["vix", "vix_rank"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate HV/IV ratio if not present
        if "hv_iv_ratio" not in df.columns:
            if "hv_20" in df.columns:
                df = df.copy()
                df["hv_iv_ratio"] = df["hv_20"] / (df["vix"] / 100)
            else:
                raise ValueError("Need either 'hv_iv_ratio' or 'hv_20' column")

        # Classify each row
        regimes = df.apply(
            lambda row: self.classify(
                row["vix"],
                row["vix_rank"],
                row["hv_iv_ratio"]
            ).value,
            axis=1
        )

        return regimes

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime-related features to DataFrame.

        Args:
            df: DataFrame with VIX and volatility data

        Returns:
            DataFrame with regime features added
        """
        result = df.copy()

        # Classify regime
        result["regime"] = self.classify_series(result)

        # One-hot encode regime
        for regime in MarketRegime:
            result[f"regime_{regime.value}"] = (result["regime"] == regime.value).astype(int)

        # Days in current regime
        regime_change = result["regime"] != result["regime"].shift(1)
        regime_groups = regime_change.cumsum()
        result["days_in_regime"] = result.groupby(regime_groups).cumcount() + 1

        # Regime transition in last 5 days
        result["regime_changed_5d"] = (
            result["regime"] != result["regime"].shift(5)
        ).astype(int)

        # VIX level flags
        result["vix_above_20"] = (result["vix"] > 20).astype(int)
        result["vix_above_25"] = (result["vix"] > 25).astype(int)
        result["vix_below_15"] = (result["vix"] < 15).astype(int)

        # HV/IV flags
        if "hv_iv_ratio" in result.columns:
            result["hv_iv_contraction"] = (
                result["hv_iv_ratio"] < self.thresholds["hv_iv_contraction"]
            ).astype(int)
            result["hv_iv_expansion"] = (
                result["hv_iv_ratio"] > self.thresholds["hv_iv_expansion"]
            ).astype(int)

        logger.info(f"Added regime features. Regime distribution:\n{result['regime'].value_counts()}")

        return result

    def get_strategy_recommendation(self, regime: MarketRegime) -> Dict:
        """
        Get strategy recommendation for a regime.

        Args:
            regime: Market regime

        Returns:
            Dictionary with strategy recommendations
        """
        recommendations = {
            MarketRegime.LOW_VOL: {
                "strategy": "debit_spreads",
                "bias": "directional",
                "position_size_mult": 1.0,
                "dte_target": 30,
                "notes": "Low IV, play direction with debit spreads"
            },
            MarketRegime.HIGH_VOL: {
                "strategy": "credit_spreads",
                "bias": "neutral",
                "position_size_mult": 0.5,
                "dte_target": 21,
                "notes": "High IV, reduce size, sell premium"
            },
            MarketRegime.IV_EXPANSION: {
                "strategy": "debit_spreads",
                "bias": "long_vol",
                "position_size_mult": 0.75,
                "dte_target": 45,
                "notes": "IV may rise further, buy options"
            },
            MarketRegime.IV_CONTRACTION: {
                "strategy": "credit_spreads",
                "bias": "short_vol",
                "position_size_mult": 1.0,
                "dte_target": 21,
                "notes": "IV overpriced, sell premium"
            },
            MarketRegime.NEUTRAL: {
                "strategy": "none",
                "bias": "neutral",
                "position_size_mult": 0.75,
                "dte_target": 30,
                "notes": "No clear edge, reduce exposure"
            },
        }

        return recommendations.get(regime, recommendations[MarketRegime.NEUTRAL])


def detect_regime(
    vix: float,
    vix_rank: float,
    hv_iv_ratio: float,
) -> str:
    """
    Convenience function to detect market regime.

    Args:
        vix: Current VIX level
        vix_rank: VIX percentile rank (0-1)
        hv_iv_ratio: HV/IV ratio

    Returns:
        Regime string
    """
    classifier = RegimeClassifier()
    return classifier.classify(vix, vix_rank, hv_iv_ratio).value


if __name__ == "__main__":
    # Quick test
    classifier = RegimeClassifier()

    # Test various conditions
    test_cases = [
        (12, 0.2, 0.5),  # Low vol
        (30, 0.9, 1.0),  # High vol
        (20, 0.6, 1.1),  # IV expansion
        (18, 0.4, 0.5),  # IV contraction
        (18, 0.5, 0.75), # Neutral
    ]

    for vix, vix_rank, hv_iv in test_cases:
        regime = classifier.classify(vix, vix_rank, hv_iv)
        rec = classifier.get_strategy_recommendation(regime)
        print(f"VIX={vix}, rank={vix_rank}, HV/IV={hv_iv} -> {regime.value}")
        print(f"  Strategy: {rec['strategy']}, Size mult: {rec['position_size_mult']}")
