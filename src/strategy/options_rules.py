"""
Rule-based options selection logic.

Determines strategy type, strike selection, DTE, and position sizing
based on regime and model predictions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from src.config.settings import get_settings
from src.models.regime_classifier import MarketRegime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """Options strategy types."""

    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"  # Credit spread
    BEAR_CALL_SPREAD = "bear_call_spread"  # Credit spread
    NO_TRADE = "no_trade"


class Direction(Enum):
    """Predicted market direction."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TradeSignal:
    """
    Trade signal with all parameters.

    Attributes:
        strategy: Type of options strategy
        direction: Market direction
        regime: Current market regime
        confidence: Model confidence (0-1)
        dte_target: Target days to expiration
        delta_long: Target delta for long leg
        delta_short: Target delta for short leg
        spread_width: Width of spread in dollars
        position_size_mult: Position size multiplier
        notes: Additional notes
    """

    strategy: StrategyType
    direction: Direction
    regime: str
    confidence: float
    dte_target: int
    delta_long: float
    delta_short: float
    spread_width: float
    position_size_mult: float
    notes: str = ""


class OptionsRules:
    """
    Rule-based options selection engine.

    Determines optimal options strategy based on:
    - Market regime
    - Direction prediction
    - Volatility forecast
    - Confidence level

    Attributes:
        min_confidence: Minimum confidence to trade
    """

    # Strategy selection matrix: (regime, direction) -> strategy
    STRATEGY_MATRIX = {
        # Low vol regime: play direction with debit spreads
        (MarketRegime.LOW_VOL.value, Direction.BULLISH): StrategyType.BULL_CALL_SPREAD,
        (MarketRegime.LOW_VOL.value, Direction.BEARISH): StrategyType.BEAR_PUT_SPREAD,
        (MarketRegime.LOW_VOL.value, Direction.NEUTRAL): StrategyType.NO_TRADE,

        # High vol regime: sell premium with credit spreads
        (MarketRegime.HIGH_VOL.value, Direction.BULLISH): StrategyType.BULL_PUT_SPREAD,
        (MarketRegime.HIGH_VOL.value, Direction.BEARISH): StrategyType.BEAR_CALL_SPREAD,
        (MarketRegime.HIGH_VOL.value, Direction.NEUTRAL): StrategyType.NO_TRADE,  # Iron condor in Phase 2

        # IV contraction: sell premium (IV will decrease)
        (MarketRegime.IV_CONTRACTION.value, Direction.BULLISH): StrategyType.BULL_PUT_SPREAD,
        (MarketRegime.IV_CONTRACTION.value, Direction.BEARISH): StrategyType.BEAR_CALL_SPREAD,
        (MarketRegime.IV_CONTRACTION.value, Direction.NEUTRAL): StrategyType.BULL_PUT_SPREAD,  # Slight bullish bias

        # IV expansion: buy premium (IV may increase)
        (MarketRegime.IV_EXPANSION.value, Direction.BULLISH): StrategyType.BULL_CALL_SPREAD,
        (MarketRegime.IV_EXPANSION.value, Direction.BEARISH): StrategyType.BEAR_PUT_SPREAD,
        (MarketRegime.IV_EXPANSION.value, Direction.NEUTRAL): StrategyType.NO_TRADE,

        # Neutral regime: trade only with high confidence
        (MarketRegime.NEUTRAL.value, Direction.BULLISH): StrategyType.BULL_CALL_SPREAD,
        (MarketRegime.NEUTRAL.value, Direction.BEARISH): StrategyType.BEAR_PUT_SPREAD,
        (MarketRegime.NEUTRAL.value, Direction.NEUTRAL): StrategyType.NO_TRADE,
    }

    # Delta targets by regime
    DELTA_TARGETS = {
        MarketRegime.LOW_VOL.value: {"long": 0.40, "short": 0.30},
        MarketRegime.HIGH_VOL.value: {"long": 0.30, "short": 0.20},
        MarketRegime.IV_CONTRACTION.value: {"long": 0.25, "short": 0.15},
        MarketRegime.IV_EXPANSION.value: {"long": 0.45, "short": 0.35},
        MarketRegime.NEUTRAL.value: {"long": 0.35, "short": 0.25},
    }

    # Spread widths by regime (in dollars)
    SPREAD_WIDTHS = {
        MarketRegime.LOW_VOL.value: 5,
        MarketRegime.HIGH_VOL.value: 10,
        MarketRegime.IV_CONTRACTION.value: 5,
        MarketRegime.IV_EXPANSION.value: 5,
        MarketRegime.NEUTRAL.value: 5,
    }

    # DTE targets by regime
    DTE_TARGETS = {
        MarketRegime.LOW_VOL.value: 30,
        MarketRegime.HIGH_VOL.value: 21,
        MarketRegime.IV_CONTRACTION.value: 21,
        MarketRegime.IV_EXPANSION.value: 45,
        MarketRegime.NEUTRAL.value: 30,
    }

    # Position size multipliers by regime
    SIZE_MULTIPLIERS = {
        MarketRegime.LOW_VOL.value: 1.0,
        MarketRegime.HIGH_VOL.value: 0.5,
        MarketRegime.IV_CONTRACTION.value: 1.0,
        MarketRegime.IV_EXPANSION.value: 0.75,
        MarketRegime.NEUTRAL.value: 0.75,
    }

    def __init__(self, min_confidence: float = 0.55):
        """
        Initialize options rules engine.

        Args:
            min_confidence: Minimum confidence to generate trade signal
        """
        self.min_confidence = min_confidence
        logger.info(f"Initialized OptionsRules with min_confidence={min_confidence}")

    def generate_signal(
        self,
        regime: str,
        direction_probs: Dict[str, float],
        iv_rank: float,
        iv_forecast: Optional[float] = None,
    ) -> TradeSignal:
        """
        Generate a trade signal based on inputs.

        Args:
            regime: Current market regime
            direction_probs: Dictionary with probabilities for each direction
                             {'bullish': 0.4, 'neutral': 0.35, 'bearish': 0.25}
            iv_rank: Current IV rank (0-1)
            iv_forecast: Forecasted IV rank (optional)

        Returns:
            TradeSignal with all parameters
        """
        # Determine direction from probabilities
        direction, confidence = self._get_direction(direction_probs)

        # Check minimum confidence
        if confidence < self.min_confidence and direction != Direction.NEUTRAL:
            logger.info(
                f"Low confidence ({confidence:.2f} < {self.min_confidence}), "
                f"treating as neutral"
            )
            direction = Direction.NEUTRAL

        # Get strategy from matrix
        strategy = self.STRATEGY_MATRIX.get(
            (regime, direction),
            StrategyType.NO_TRADE
        )

        # Get parameters for regime
        delta_targets = self.DELTA_TARGETS.get(
            regime, self.DELTA_TARGETS[MarketRegime.NEUTRAL.value]
        )
        spread_width = self.SPREAD_WIDTHS.get(
            regime, self.SPREAD_WIDTHS[MarketRegime.NEUTRAL.value]
        )
        dte_target = self.DTE_TARGETS.get(
            regime, self.DTE_TARGETS[MarketRegime.NEUTRAL.value]
        )
        size_mult = self.SIZE_MULTIPLIERS.get(
            regime, self.SIZE_MULTIPLIERS[MarketRegime.NEUTRAL.value]
        )

        # Adjust DTE based on IV rank
        dte_target = self._adjust_dte(dte_target, iv_rank)

        # Adjust size based on confidence
        size_mult *= self._confidence_multiplier(confidence)

        # Generate notes
        notes = self._generate_notes(regime, direction, confidence, iv_rank, iv_forecast)

        signal = TradeSignal(
            strategy=strategy,
            direction=direction,
            regime=regime,
            confidence=confidence,
            dte_target=dte_target,
            delta_long=delta_targets["long"],
            delta_short=delta_targets["short"],
            spread_width=spread_width,
            position_size_mult=size_mult,
            notes=notes,
        )

        logger.info(f"Generated signal: {signal.strategy.value} ({signal.notes})")

        return signal

    def _get_direction(self, probs: Dict[str, float]) -> tuple[Direction, float]:
        """
        Determine direction and confidence from probabilities.

        Args:
            probs: Dictionary with direction probabilities

        Returns:
            Tuple of (Direction, confidence)
        """
        bullish = probs.get("bullish", 0)
        bearish = probs.get("bearish", 0)
        neutral = probs.get("neutral", 0)

        if bullish > bearish and bullish > neutral:
            return Direction.BULLISH, bullish
        elif bearish > bullish and bearish > neutral:
            return Direction.BEARISH, bearish
        else:
            return Direction.NEUTRAL, neutral

    def _adjust_dte(self, base_dte: int, iv_rank: float) -> int:
        """
        Adjust DTE based on IV rank.

        Args:
            base_dte: Base DTE from regime
            iv_rank: Current IV rank

        Returns:
            Adjusted DTE
        """
        if iv_rank > 0.7:
            # High IV: shorter DTE to capture faster theta
            dte = base_dte - 7
        elif iv_rank < 0.3:
            # Low IV: longer DTE
            dte = base_dte + 7
        else:
            dte = base_dte

        # Clamp to valid range
        return max(14, min(45, dte))

    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Calculate position size multiplier from confidence.

        Args:
            confidence: Model confidence (0-1)

        Returns:
            Multiplier (0.5 to 1.0)
        """
        # Linear mapping from confidence to size
        # confidence 0.5 -> mult 0.5
        # confidence 1.0 -> mult 1.0
        return 0.5 + 0.5 * min(1.0, max(0.5, confidence))

    def _generate_notes(
        self,
        regime: str,
        direction: Direction,
        confidence: float,
        iv_rank: float,
        iv_forecast: Optional[float],
    ) -> str:
        """
        Generate human-readable notes for the signal.

        Args:
            regime: Market regime
            direction: Predicted direction
            confidence: Model confidence
            iv_rank: Current IV rank
            iv_forecast: Forecasted IV rank

        Returns:
            Notes string
        """
        notes = []

        notes.append(f"regime={regime}")
        notes.append(f"direction={direction.value}")
        notes.append(f"conf={confidence:.0%}")
        notes.append(f"iv_rank={iv_rank:.0%}")

        if iv_forecast is not None:
            iv_change = iv_forecast - iv_rank
            direction_str = "up" if iv_change > 0 else "down"
            notes.append(f"iv_forecast={direction_str}{abs(iv_change):.0%}")

        return ", ".join(notes)

    def calculate_position_size(
        self,
        account_value: float,
        spread_width: float,
        credit_or_debit: float,
        signal: TradeSignal,
    ) -> int:
        """
        Calculate number of contracts to trade.

        Args:
            account_value: Total account value
            spread_width: Width of spread in dollars
            credit_or_debit: Credit received (positive) or debit paid (negative)
            signal: TradeSignal with parameters

        Returns:
            Number of contracts
        """
        settings = get_settings()

        # Base risk: 2% of account
        base_risk_pct = settings.max_position_size

        # Calculate max loss per spread
        if credit_or_debit > 0:
            # Credit spread: max loss = width - credit
            max_loss_per_spread = (spread_width - credit_or_debit) * 100
        else:
            # Debit spread: max loss = debit paid
            max_loss_per_spread = abs(credit_or_debit) * 100

        # Calculate risk amount
        risk_amount = account_value * base_risk_pct
        risk_amount *= signal.position_size_mult

        # Calculate number of contracts
        num_contracts = int(risk_amount / max_loss_per_spread)

        # Clamp to reasonable range
        return max(1, min(10, num_contracts))


if __name__ == "__main__":
    # Test the rules engine
    rules = OptionsRules()

    # Test various scenarios
    test_cases = [
        {
            "regime": "low_vol",
            "direction_probs": {"bullish": 0.6, "neutral": 0.25, "bearish": 0.15},
            "iv_rank": 0.2,
        },
        {
            "regime": "high_vol",
            "direction_probs": {"bullish": 0.3, "neutral": 0.3, "bearish": 0.4},
            "iv_rank": 0.85,
        },
        {
            "regime": "iv_contraction",
            "direction_probs": {"bullish": 0.35, "neutral": 0.4, "bearish": 0.25},
            "iv_rank": 0.4,
        },
    ]

    for case in test_cases:
        print(f"\nTest case: {case}")
        signal = rules.generate_signal(**case)
        print(f"Result: {signal.strategy.value}")
        print(f"  DTE: {signal.dte_target}, Delta: {signal.delta_long}/{signal.delta_short}")
        print(f"  Size mult: {signal.position_size_mult:.2f}")
        print(f"  Notes: {signal.notes}")
