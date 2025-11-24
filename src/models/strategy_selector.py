"""
Strategy Selection Engine.

Combines Direction Classifier and Volatility Forecaster outputs
to recommend specific options strategies for trading.

Strategy Matrix:
- Direction (UP/DOWN) + IV Regime (LOW/NORMAL/HIGH) → Strategy
- Includes position sizing, risk management, and trade parameters
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Direction(Enum):
    """Market direction prediction."""
    UP = "UP"
    DOWN = "DOWN"


class IVRegime(Enum):
    """Implied volatility regime."""
    LOW = "LOW"        # IV Rank 0-30: Options cheap
    NORMAL = "NORMAL"  # IV Rank 30-60: Fair value
    HIGH = "HIGH"      # IV Rank 60-100: Options expensive


class StrategyType(Enum):
    """Options strategy types."""
    BUY_CALLS = "Buy Calls"
    BUY_PUTS = "Buy Puts"
    BULL_CALL_SPREAD = "Bull Call Spread"
    BEAR_PUT_SPREAD = "Bear Put Spread"
    BULL_PUT_SPREAD = "Bull Put Spread (Credit)"
    BEAR_CALL_SPREAD = "Bear Call Spread (Credit)"
    IRON_CONDOR = "Iron Condor"
    NO_TRADE = "No Trade"


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation with trade details."""

    # Core recommendation
    strategy: StrategyType
    direction: Direction
    iv_regime: IVRegime

    # Confidence scores
    direction_confidence: float
    iv_confidence: float
    overall_confidence: float

    # Trade parameters
    position_size: str  # "full", "half", "quarter"
    suggested_expiry: str  # "1-2 weeks", "2-4 weeks", etc.
    suggested_delta: str  # "ATM", "0.30 delta OTM", etc.

    # Risk management
    max_loss_pct: float  # Max % of position to risk
    profit_target_pct: float  # Target profit %
    stop_loss_trigger: str  # When to exit

    # Reasoning
    reasoning: str
    cautions: List[str] = field(default_factory=list)

    # Metadata
    ticker: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "direction": self.direction.value,
            "iv_regime": self.iv_regime.value,
            "direction_confidence": self.direction_confidence,
            "iv_confidence": self.iv_confidence,
            "overall_confidence": self.overall_confidence,
            "position_size": self.position_size,
            "suggested_expiry": self.suggested_expiry,
            "suggested_delta": self.suggested_delta,
            "max_loss_pct": self.max_loss_pct,
            "profit_target_pct": self.profit_target_pct,
            "stop_loss_trigger": self.stop_loss_trigger,
            "reasoning": self.reasoning,
            "cautions": self.cautions,
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
        }


class StrategySelector:
    """
    Strategy Selection Engine.

    Combines direction and volatility forecasts to recommend
    options strategies with specific trade parameters.
    """

    # Minimum confidence thresholds
    MIN_DIRECTION_CONFIDENCE = 0.52  # Must be better than coin flip
    MIN_IV_CONFIDENCE = 0.35  # IV forecaster is less accurate
    MIN_OVERALL_CONFIDENCE = 0.45  # Combined threshold

    # Strategy matrix: (Direction, IV Regime) -> Strategy details
    STRATEGY_MATRIX = {
        # Bullish strategies
        (Direction.UP, IVRegime.LOW): {
            "strategy": StrategyType.BUY_CALLS,
            "reasoning": "Bullish outlook + cheap options = buy directional calls",
            "expiry": "2-4 weeks",
            "delta": "ATM or slightly OTM (0.40-0.50 delta)",
            "max_loss": 0.02,  # 2% of portfolio
            "profit_target": 0.50,  # 50% gain on premium
            "stop_loss": "Close if underlying moves 2% against position",
        },
        (Direction.UP, IVRegime.NORMAL): {
            "strategy": StrategyType.BULL_CALL_SPREAD,
            "reasoning": "Bullish outlook + fair IV = defined risk vertical spread",
            "expiry": "2-3 weeks",
            "delta": "Buy ATM, sell 0.30 delta OTM",
            "max_loss": 0.015,  # 1.5% of portfolio
            "profit_target": 0.75,  # 75% of max profit
            "stop_loss": "Close if spread loses 50% of value",
        },
        (Direction.UP, IVRegime.HIGH): {
            "strategy": StrategyType.BULL_PUT_SPREAD,
            "reasoning": "Bullish outlook + expensive options = sell put spread for credit",
            "expiry": "2-4 weeks",
            "delta": "Sell 0.30 delta, buy 0.15 delta",
            "max_loss": 0.02,  # 2% of portfolio
            "profit_target": 0.50,  # Keep 50% of credit
            "stop_loss": "Close if spread doubles in value against you",
        },

        # Bearish strategies
        (Direction.DOWN, IVRegime.LOW): {
            "strategy": StrategyType.BUY_PUTS,
            "reasoning": "Bearish outlook + cheap options = buy protective puts",
            "expiry": "2-4 weeks",
            "delta": "ATM or slightly OTM (0.40-0.50 delta)",
            "max_loss": 0.02,  # 2% of portfolio
            "profit_target": 0.50,  # 50% gain on premium
            "stop_loss": "Close if underlying moves 2% against position",
        },
        (Direction.DOWN, IVRegime.NORMAL): {
            "strategy": StrategyType.BEAR_PUT_SPREAD,
            "reasoning": "Bearish outlook + fair IV = defined risk vertical spread",
            "expiry": "2-3 weeks",
            "delta": "Buy ATM, sell 0.30 delta OTM",
            "max_loss": 0.015,  # 1.5% of portfolio
            "profit_target": 0.75,  # 75% of max profit
            "stop_loss": "Close if spread loses 50% of value",
        },
        (Direction.DOWN, IVRegime.HIGH): {
            "strategy": StrategyType.BEAR_CALL_SPREAD,
            "reasoning": "Bearish outlook + expensive options = sell call spread for credit",
            "expiry": "2-4 weeks",
            "delta": "Sell 0.30 delta, buy 0.15 delta",
            "max_loss": 0.02,  # 2% of portfolio
            "profit_target": 0.50,  # Keep 50% of credit
            "stop_loss": "Close if spread doubles in value against you",
        },
    }

    def __init__(
        self,
        min_direction_confidence: float = None,
        min_iv_confidence: float = None,
    ):
        """
        Initialize strategy selector.

        Args:
            min_direction_confidence: Override minimum direction confidence
            min_iv_confidence: Override minimum IV confidence
        """
        self.min_direction_confidence = min_direction_confidence or self.MIN_DIRECTION_CONFIDENCE
        self.min_iv_confidence = min_iv_confidence or self.MIN_IV_CONFIDENCE

        logger.info(
            f"Initialized StrategySelector: min_dir_conf={self.min_direction_confidence}, "
            f"min_iv_conf={self.min_iv_confidence}"
        )

    def select_strategy(
        self,
        direction: str,
        direction_confidence: float,
        direction_probability: float,
        iv_regime: str,
        iv_confidence: float,
        current_iv_rank: float,
        ticker: str = "",
    ) -> StrategyRecommendation:
        """
        Select optimal strategy based on model outputs.

        Args:
            direction: "UP" or "DOWN" from direction classifier
            direction_confidence: Confidence score (0-1)
            direction_probability: Raw probability of predicted class
            iv_regime: "LOW", "NORMAL", or "HIGH" from volatility forecaster
            iv_confidence: Confidence score (0-1)
            current_iv_rank: Current IV rank (0-100)
            ticker: Stock ticker

        Returns:
            StrategyRecommendation with complete trade details
        """
        # Convert to enums
        dir_enum = Direction(direction)
        iv_enum = IVRegime(iv_regime)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            direction_confidence, iv_confidence
        )

        # Check confidence thresholds
        cautions = []

        if direction_confidence < self.min_direction_confidence:
            cautions.append(f"Low direction confidence ({direction_confidence:.1%})")

        if iv_confidence < self.min_iv_confidence:
            cautions.append(f"Low IV forecast confidence ({iv_confidence:.1%})")

        # Determine if we should trade
        should_trade = (
            direction_confidence >= self.min_direction_confidence and
            overall_confidence >= self.MIN_OVERALL_CONFIDENCE
        )

        if not should_trade:
            return self._no_trade_recommendation(
                dir_enum, iv_enum, direction_confidence, iv_confidence,
                overall_confidence, cautions, ticker
            )

        # Get strategy from matrix
        key = (dir_enum, iv_enum)
        strategy_details = self.STRATEGY_MATRIX.get(key)

        if not strategy_details:
            return self._no_trade_recommendation(
                dir_enum, iv_enum, direction_confidence, iv_confidence,
                overall_confidence, ["No matching strategy found"], ticker
            )

        # Determine position size based on confidence
        position_size = self._determine_position_size(overall_confidence)

        # Add IV-specific cautions
        if iv_enum == IVRegime.HIGH and current_iv_rank > 80:
            cautions.append("Extremely high IV - expect IV crush after any move")

        if iv_enum == IVRegime.LOW and current_iv_rank < 15:
            cautions.append("Extremely low IV - options very cheap but may stay flat")

        # Build recommendation
        recommendation = StrategyRecommendation(
            strategy=strategy_details["strategy"],
            direction=dir_enum,
            iv_regime=iv_enum,
            direction_confidence=direction_confidence,
            iv_confidence=iv_confidence,
            overall_confidence=overall_confidence,
            position_size=position_size,
            suggested_expiry=strategy_details["expiry"],
            suggested_delta=strategy_details["delta"],
            max_loss_pct=strategy_details["max_loss"],
            profit_target_pct=strategy_details["profit_target"],
            stop_loss_trigger=strategy_details["stop_loss"],
            reasoning=strategy_details["reasoning"],
            cautions=cautions,
            ticker=ticker,
        )

        logger.info(
            f"Strategy selected for {ticker}: {recommendation.strategy.value} "
            f"(dir={direction}, iv={iv_regime}, conf={overall_confidence:.1%})"
        )

        return recommendation

    def _calculate_overall_confidence(
        self,
        direction_confidence: float,
        iv_confidence: float,
    ) -> float:
        """
        Calculate overall confidence score.

        Uses geometric mean to penalize low confidence in either model.
        """
        # Weight direction more heavily (it's more important for P&L)
        weighted_dir = direction_confidence ** 0.6
        weighted_iv = iv_confidence ** 0.4

        return weighted_dir * weighted_iv

    def _determine_position_size(self, confidence: float) -> str:
        """Determine position size based on confidence."""
        if confidence >= 0.65:
            return "full"
        elif confidence >= 0.55:
            return "half"
        else:
            return "quarter"

    def _no_trade_recommendation(
        self,
        direction: Direction,
        iv_regime: IVRegime,
        direction_confidence: float,
        iv_confidence: float,
        overall_confidence: float,
        cautions: List[str],
        ticker: str,
    ) -> StrategyRecommendation:
        """Create a no-trade recommendation."""
        return StrategyRecommendation(
            strategy=StrategyType.NO_TRADE,
            direction=direction,
            iv_regime=iv_regime,
            direction_confidence=direction_confidence,
            iv_confidence=iv_confidence,
            overall_confidence=overall_confidence,
            position_size="none",
            suggested_expiry="N/A",
            suggested_delta="N/A",
            max_loss_pct=0.0,
            profit_target_pct=0.0,
            stop_loss_trigger="N/A",
            reasoning="Confidence below threshold - wait for clearer signal",
            cautions=cautions,
            ticker=ticker,
        )

    def get_strategy_for_iron_condor(
        self,
        iv_regime: str,
        iv_confidence: float,
        current_iv_rank: float,
        ticker: str = "",
    ) -> Optional[StrategyRecommendation]:
        """
        Check if iron condor is appropriate (neutral + high IV).

        Iron condors work best when:
        - No strong directional signal
        - High IV (premium sellers paradise)
        - Expecting IV to contract
        """
        if iv_regime != "HIGH" or iv_confidence < 0.40:
            return None

        return StrategyRecommendation(
            strategy=StrategyType.IRON_CONDOR,
            direction=Direction.UP,  # Neutral, but need a value
            iv_regime=IVRegime.HIGH,
            direction_confidence=0.50,  # Neutral
            iv_confidence=iv_confidence,
            overall_confidence=iv_confidence * 0.8,
            position_size="half",
            suggested_expiry="3-4 weeks",
            suggested_delta="Sell 0.20 delta on both sides, buy wings 0.10 delta",
            max_loss_pct=0.02,
            profit_target_pct=0.25,  # Take 25% of max profit
            stop_loss_trigger="Close if either short strike is breached",
            reasoning="High IV + no clear direction = sell premium on both sides",
            cautions=["Requires active management", "Watch for breakouts"],
            ticker=ticker,
        )


class TradingSignalGenerator:
    """
    Generate trading signals by combining both models.

    Loads trained models and generates daily signals.
    """

    def __init__(
        self,
        direction_model_path: str,
        volatility_model_path: str,
    ):
        """
        Initialize signal generator.

        Args:
            direction_model_path: Path to trained direction classifier
            volatility_model_path: Path to trained volatility forecaster
        """
        from src.models.robust_classifier import RobustDirectionClassifier
        from src.models.volatility_forecaster import VolatilityForecaster

        self.direction_model = RobustDirectionClassifier.load(direction_model_path)
        self.volatility_model = VolatilityForecaster.load(volatility_model_path)
        self.strategy_selector = StrategySelector()

        logger.info("Initialized TradingSignalGenerator with both models")

    def generate_signal(
        self,
        features: pd.DataFrame,
        ticker: str,
    ) -> StrategyRecommendation:
        """
        Generate trading signal from features.

        Args:
            features: DataFrame with features for prediction (single row)
            ticker: Stock ticker

        Returns:
            StrategyRecommendation
        """
        # Get direction prediction
        dir_results = self.direction_model.predict(features)
        dir_pred = dir_results[0]

        direction = "UP" if dir_pred.predicted_class == 1 else "DOWN"
        direction_confidence = dir_pred.confidence
        direction_probability = dir_pred.probability

        # Get volatility prediction
        vol_results = self.volatility_model.predict(features)
        vol_pred = vol_results[0]

        iv_regime = vol_pred.predicted_regime
        iv_confidence = vol_pred.confidence
        current_iv_rank = vol_pred.current_iv_rank

        # Select strategy
        recommendation = self.strategy_selector.select_strategy(
            direction=direction,
            direction_confidence=direction_confidence,
            direction_probability=direction_probability,
            iv_regime=iv_regime,
            iv_confidence=iv_confidence,
            current_iv_rank=current_iv_rank,
            ticker=ticker,
        )

        return recommendation

    def generate_daily_signals(
        self,
        tickers: List[str],
    ) -> Dict[str, StrategyRecommendation]:
        """
        Generate signals for multiple tickers.

        Args:
            tickers: List of tickers to analyze

        Returns:
            Dictionary of ticker -> recommendation
        """
        from src.models.feature_pipeline import FeaturePipeline

        signals = {}

        for ticker in tickers:
            try:
                # Load latest features
                pipeline = FeaturePipeline(ticker=ticker, include_macro=True)
                features = pipeline.load_technical_features()

                # Get most recent row
                latest = features.iloc[[-1]]

                # Generate signal
                recommendation = self.generate_signal(latest, ticker)
                signals[ticker] = recommendation

                logger.info(f"{ticker}: {recommendation.strategy.value}")

            except Exception as e:
                logger.error(f"Failed to generate signal for {ticker}: {e}")

        return signals


def format_recommendation(rec: StrategyRecommendation) -> str:
    """Format recommendation for display."""
    lines = [
        f"{'='*60}",
        f"STRATEGY RECOMMENDATION - {rec.ticker}",
        f"{'='*60}",
        f"",
        f"Strategy: {rec.strategy.value}",
        f"Direction: {rec.direction.value} ({rec.direction_confidence:.1%} confidence)",
        f"IV Regime: {rec.iv_regime.value} ({rec.iv_confidence:.1%} confidence)",
        f"Overall Confidence: {rec.overall_confidence:.1%}",
        f"",
        f"TRADE PARAMETERS",
        f"-" * 40,
        f"Position Size: {rec.position_size.upper()}",
        f"Expiry: {rec.suggested_expiry}",
        f"Strike Selection: {rec.suggested_delta}",
        f"",
        f"RISK MANAGEMENT",
        f"-" * 40,
        f"Max Loss: {rec.max_loss_pct:.1%} of portfolio",
        f"Profit Target: {rec.profit_target_pct:.0%} of max profit",
        f"Stop Loss: {rec.stop_loss_trigger}",
        f"",
        f"REASONING",
        f"-" * 40,
        f"{rec.reasoning}",
    ]

    if rec.cautions:
        lines.extend([
            f"",
            f"CAUTIONS",
            f"-" * 40,
        ])
        for caution in rec.cautions:
            lines.append(f"  - {caution}")

    lines.append(f"{'='*60}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test strategy selector
    print("Testing StrategySelector...\n")

    selector = StrategySelector()

    # Test cases
    test_cases = [
        ("UP", 0.65, 0.65, "LOW", 0.72, 25.0, "SPY"),
        ("UP", 0.60, 0.60, "HIGH", 0.55, 75.0, "SPY"),
        ("DOWN", 0.58, 0.58, "NORMAL", 0.45, 45.0, "QQQ"),
        ("UP", 0.51, 0.51, "LOW", 0.30, 20.0, "IWM"),  # Low confidence
    ]

    for direction, dir_conf, dir_prob, iv_regime, iv_conf, iv_rank, ticker in test_cases:
        rec = selector.select_strategy(
            direction=direction,
            direction_confidence=dir_conf,
            direction_probability=dir_prob,
            iv_regime=iv_regime,
            iv_confidence=iv_conf,
            current_iv_rank=iv_rank,
            ticker=ticker,
        )

        print(format_recommendation(rec))
        print()
