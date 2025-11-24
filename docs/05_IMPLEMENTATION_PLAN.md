# ML Options Trading System - Implementation Plan

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Active Development
**Timeline:** 3-Week MVP

---

## Table of Contents

1. [Hybrid Approach Architecture](#1-hybrid-approach-architecture)
2. [3-Week MVP Timeline](#2-3-week-mvp-timeline)
3. [Feature Engineering Specifications](#3-feature-engineering-specifications)
4. [Model Specifications](#4-model-specifications)
5. [Rule-Based Options Selection Logic](#5-rule-based-options-selection-logic)
6. [Phase-by-Phase Roadmap](#6-phase-by-phase-roadmap)

---

## 1. Hybrid Approach Architecture

### 1.1 Two-Model System Overview

The system uses a **hybrid approach** combining:
1. **ML Models** for directional prediction and volatility forecasting
2. **Rule-Based Logic** for options selection and risk management

```
                    ┌─────────────────────────────────────────────────┐
                    │             HYBRID ARCHITECTURE                  │
                    └─────────────────────────────────────────────────┘
                                           │
           ┌───────────────────────────────┼───────────────────────────────┐
           │                               │                               │
           ▼                               ▼                               ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   Direction Model   │      │  Volatility Model   │      │   Regime Classifier │
│   (XGBoost/LSTM)    │      │    (XGBoost)        │      │   (Rule-Based)      │
├─────────────────────┤      ├─────────────────────┤      ├─────────────────────┤
│ Predicts:           │      │ Predicts:           │      │ Classifies:         │
│ • Bullish/Bearish   │      │ • Expected RV       │      │ • low_vol           │
│ • 5-day direction   │      │ • IV rank movement  │      │ • high_vol          │
│ • Confidence score  │      │ • Vol regime shift  │      │ • iv_expansion      │
└─────────┬───────────┘      └─────────┬───────────┘      │ • iv_contraction    │
          │                            │                   │ • neutral           │
          │                            │                   └─────────┬───────────┘
          │                            │                             │
          └────────────────────────────┼─────────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │     RULE-BASED OPTIONS ENGINE   │
                        ├─────────────────────────────────┤
                        │ Inputs:                         │
                        │ • Direction prediction          │
                        │ • Volatility forecast           │
                        │ • Current regime                │
                        │                                 │
                        │ Outputs:                        │
                        │ • Strategy type                 │
                        │ • Strike selection              │
                        │ • DTE selection                 │
                        │ • Position sizing               │
                        └─────────────────────────────────┘
```

### 1.2 Why Hybrid?

| Aspect | Pure ML | Rule-Based | Hybrid (Our Approach) |
|--------|---------|------------|----------------------|
| Market prediction | Good | Poor | ML handles this |
| Options selection | Complex | Simple & proven | Rules handle this |
| Interpretability | Low | High | Best of both |
| Backtesting | Difficult | Easy | Tractable |
| Risk management | Black box | Explicit | Rules = explicit |
| Adaptability | High | Low | ML adapts, rules constrain |

### 1.3 Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────┘

Step 1: Data Collection
┌─────────────────┐
│   yfinance      │──► SPY, QQQ, IWM OHLCV (daily)
│   + VIX data    │──► VIX levels (daily)
└─────────────────┘

Step 2: Feature Engineering
┌─────────────────┐
│ Technical       │──► RSI, MACD, Bollinger, MA crossovers
│ Indicators      │
├─────────────────┤
│ Volatility      │──► HV (5/10/20/60d), IV Rank, HV/IV ratio
│ Features        │
├─────────────────┤
│ Regime          │──► VIX rank, regime classification
│ Features        │
└─────────────────┘

Step 3: Model Predictions
┌─────────────────┐
│ Direction Model │──► P(up), P(down), confidence
├─────────────────┤
│ Volatility Model│──► Expected IV rank (5d forward)
└─────────────────┘

Step 4: Options Selection (Rules)
┌─────────────────┐
│ Rule Engine     │──► Strategy, strikes, DTE, size
└─────────────────┘

Step 5: Trade Execution
┌─────────────────┐
│ Broker API      │──► Place orders (paper → live)
└─────────────────┘
```

---

## 2. 3-Week MVP Timeline

### Overview

```
Week 1: Data Pipeline + Feature Store
├── Day 1-2: Project setup, Docker, data fetching
├── Day 3-4: Feature engineering (technical indicators)
└── Day 5-7: Feature store, validation, testing

Week 2: ML Models
├── Day 1-2: Direction classifier (XGBoost)
├── Day 3-4: Volatility forecaster (XGBoost)
├── Day 5: Regime classifier (rule-based)
└── Day 6-7: Model evaluation, backtesting setup

Week 3: Trading Logic + Paper Trading
├── Day 1-2: Rule-based options selection
├── Day 3-4: Risk management, position sizing
├── Day 5: Paper trading integration
└── Day 6-7: End-to-end testing, documentation
```

### Week 1: Data Pipeline + Feature Store

#### Day 1-2: Foundation

**Tasks:**
- [x] Project structure creation
- [x] Docker setup (Dockerfile, docker-compose.yml)
- [x] Data fetching pipeline (yfinance_loader.py)
- [x] Feature store setup (parquet storage)
- [x] Configuration management (settings.py)

**Deliverables:**
- Working `docker-compose run app python scripts/fetch_data.py`
- SPY, QQQ, VIX data fetched and stored
- Logging infrastructure

#### Day 3-4: Feature Engineering

**Tasks:**
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Volatility features (HV multi-window, IV rank)
- Regime features (VIX rank, HV/IV ratio)

**Deliverables:**
- `src/features/technical_indicators.py`
- 50+ features per ticker per day
- Feature validation tests

#### Day 5-7: Feature Store + Testing

**Tasks:**
- Feature store with versioning
- Data quality checks
- Unit tests for all components
- Integration tests

**Deliverables:**
- Feature parquet files with metadata
- pytest suite passing
- Documentation

### Week 2: ML Models

#### Day 1-2: Direction Classifier

**Tasks:**
- Label creation (5-day forward returns)
- XGBoost model training
- Walk-forward validation
- MLflow experiment tracking

**Deliverables:**
- `src/models/direction_classifier.py`
- Model artifacts in MLflow
- AUC > 0.55 target

#### Day 3-4: Volatility Forecaster

**Tasks:**
- Target: IV rank 5 days forward
- XGBoost regression model
- Feature importance analysis
- Model validation

**Deliverables:**
- `src/models/volatility_forecaster.py`
- RMSE < 0.15 target
- Feature importance report

#### Day 5: Regime Classifier

**Tasks:**
- Rule-based regime detection
- VIX thresholds, HV/IV ratio logic
- Regime transition logging

**Deliverables:**
- `src/models/regime_classifier.py`
- Regime history analysis

#### Day 6-7: Model Evaluation

**Tasks:**
- Walk-forward backtesting
- Per-regime performance analysis
- Model selection/ensembling
- Documentation

**Deliverables:**
- Backtest results
- Performance metrics by regime

### Week 3: Trading Logic + Paper Trading

#### Day 1-2: Options Selection Rules

**Tasks:**
- Strategy selection logic
- Strike selection rules
- DTE selection rules
- Position sizing rules

**Deliverables:**
- `src/strategy/options_rules.py`
- Rule documentation

#### Day 3-4: Risk Management

**Tasks:**
- Position limits
- Portfolio Greeks limits
- Stop-loss rules
- Max drawdown monitoring

**Deliverables:**
- `src/strategy/risk_manager.py`
- Risk rules documentation

#### Day 5: Paper Trading Integration

**Tasks:**
- Signal generation pipeline
- Order management
- Position tracking
- P&L calculation

**Deliverables:**
- End-to-end signal → order flow
- Paper trading script

#### Day 6-7: Testing + Documentation

**Tasks:**
- Full system integration tests
- Performance validation
- User documentation
- Deployment guide

**Deliverables:**
- Working MVP
- README with usage instructions

---

## 3. Feature Engineering Specifications

### 3.1 Technical Indicators (25 features)

| Feature | Calculation | Window | Purpose |
|---------|-------------|--------|---------|
| `rsi_14` | RSI | 14 days | Overbought/oversold |
| `rsi_28` | RSI | 28 days | Longer-term momentum |
| `macd` | EMA12 - EMA26 | 12/26 days | Trend momentum |
| `macd_signal` | EMA9 of MACD | 9 days | Signal line |
| `macd_histogram` | MACD - Signal | - | Momentum strength |
| `bb_upper` | SMA20 + 2*std | 20 days | Upper band |
| `bb_lower` | SMA20 - 2*std | 20 days | Lower band |
| `bb_width` | (upper-lower)/mid | 20 days | Volatility measure |
| `bb_position` | (close-lower)/(upper-lower) | 20 days | Position in bands |
| `sma_20` | Simple MA | 20 days | Short-term trend |
| `sma_50` | Simple MA | 50 days | Medium-term trend |
| `sma_200` | Simple MA | 200 days | Long-term trend |
| `ema_12` | Exponential MA | 12 days | Fast EMA |
| `ema_26` | Exponential MA | 26 days | Slow EMA |
| `price_vs_sma20` | (price-sma20)/sma20 | - | Deviation from trend |
| `price_vs_sma50` | (price-sma50)/sma50 | - | Deviation from trend |
| `sma_cross_20_50` | 1 if sma20 > sma50 | - | Trend direction |
| `returns_1d` | Daily return | 1 day | Short momentum |
| `returns_5d` | 5-day return | 5 days | Weekly momentum |
| `returns_20d` | 20-day return | 20 days | Monthly momentum |
| `returns_60d` | 60-day return | 60 days | Quarterly momentum |
| `high_52w` | 52-week high | 252 days | Reference level |
| `low_52w` | 52-week low | 252 days | Reference level |
| `dist_from_high` | (high-close)/high | - | Distance from high |
| `dist_from_low` | (close-low)/low | - | Distance from low |

### 3.2 Volatility Features (20 features)

| Feature | Calculation | Window | Purpose |
|---------|-------------|--------|---------|
| `hv_5` | Std(returns) * sqrt(252) | 5 days | Very short HV |
| `hv_10` | Std(returns) * sqrt(252) | 10 days | Short HV |
| `hv_20` | Std(returns) * sqrt(252) | 20 days | Standard HV |
| `hv_60` | Std(returns) * sqrt(252) | 60 days | Medium HV |
| `hv_120` | Std(returns) * sqrt(252) | 120 days | Long HV |
| `hv_ratio_5_20` | hv_5 / hv_20 | - | Short vs medium |
| `hv_ratio_20_60` | hv_20 / hv_60 | - | Medium vs long |
| `vix` | VIX index level | - | Market IV |
| `vix_sma_10` | SMA of VIX | 10 days | VIX trend |
| `vix_sma_20` | SMA of VIX | 20 days | VIX trend |
| `vix_rank` | Percentile rank | 252 days | IV rank |
| `vix_percentile` | Percentile | 252 days | IV percentile |
| `hv_iv_ratio` | hv_20 / (vix/100) | - | HV vs IV |
| `vix_change_1d` | VIX daily change | 1 day | IV momentum |
| `vix_change_5d` | VIX 5-day change | 5 days | IV momentum |
| `vix_zscore` | (vix-mean)/std | 60 days | Normalized VIX |
| `vol_regime` | Categorical | - | Vol regime |
| `vol_trend` | VIX SMA5 - SMA20 | - | Vol direction |
| `garch_forecast` | GARCH(1,1) | - | Vol forecast |
| `parkinson_hv` | High-low estimator | 20 days | Range-based HV |

### 3.3 Regime Features (10 features)

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `regime` | Rule-based classification | Current regime |
| `regime_encoded` | One-hot encoding | Model input |
| `days_in_regime` | Days since regime change | Regime persistence |
| `regime_change_1w` | Changed in last week | Transition signal |
| `vix_above_20` | VIX > 20 | High vol flag |
| `vix_above_25` | VIX > 25 | Very high vol flag |
| `vix_below_15` | VIX < 15 | Low vol flag |
| `hv_iv_contraction` | hv_iv_ratio < 0.6 | IV overpriced |
| `hv_iv_expansion` | hv_iv_ratio > 0.9 | IV underpriced |
| `trend_regime` | Bull/Bear/Neutral | Price trend |

---

## 4. Model Specifications

### 4.1 Direction Classifier

**Objective:** Predict 5-day forward direction (up/down/neutral)

```python
class DirectionClassifier:
    """
    XGBoost classifier for 5-day forward direction prediction

    Input Features: ~55 technical + volatility + regime features
    Target: 3-class (bullish, bearish, neutral)

    Labels:
    - Bullish: 5-day return > 1%
    - Bearish: 5-day return < -1%
    - Neutral: -1% <= 5-day return <= 1%
    """

    model_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }

    # Walk-forward validation
    train_window = 252 * 2  # 2 years
    test_window = 63  # 3 months
    step_size = 21  # 1 month
```

**Target Metrics:**
- AUC (one-vs-rest): > 0.55
- Accuracy: > 40% (3-class is hard)
- Top confidence trades: > 55% accuracy

### 4.2 Volatility Forecaster

**Objective:** Predict IV rank 5 days forward

```python
class VolatilityForecaster:
    """
    XGBoost regressor for IV rank prediction

    Input Features: ~55 technical + volatility + regime features
    Target: IV rank (0-1) in 5 days
    """

    model_params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42
    }
```

**Target Metrics:**
- RMSE: < 0.15
- MAE: < 0.10
- Directional accuracy: > 55%

### 4.3 Regime Classifier (Rule-Based)

**Objective:** Classify current market regime

```python
class RegimeClassifier:
    """
    Rule-based regime classification

    Regimes:
    1. low_vol: VIX < 15 AND VIX_rank < 30%
    2. high_vol: VIX > 25 OR VIX_rank > 80%
    3. iv_expansion: HV/IV > 0.9 AND VIX_rank > 50%
    4. iv_contraction: HV/IV < 0.6 AND VIX_rank < 50%
    5. neutral: None of above
    """

    thresholds = {
        'vix_low': 15,
        'vix_high': 25,
        'vix_rank_low': 0.3,
        'vix_rank_high': 0.8,
        'hv_iv_contraction': 0.6,
        'hv_iv_expansion': 0.9
    }
```

---

## 5. Rule-Based Options Selection Logic

### 5.1 Strategy Selection Matrix

| Regime | Direction | Strategy | Rationale |
|--------|-----------|----------|-----------|
| `low_vol` | Bullish | Bull Call Spread | Low IV, directional play |
| `low_vol` | Bearish | Bear Put Spread | Low IV, directional play |
| `low_vol` | Neutral | No trade | Not enough edge |
| `high_vol` | Bullish | Bull Put Spread (credit) | High IV, sell premium |
| `high_vol` | Bearish | Bear Call Spread (credit) | High IV, sell premium |
| `high_vol` | Neutral | Iron Condor* | Collect premium both sides |
| `iv_contraction` | Any | Credit Spreads | IV will decline |
| `iv_expansion` | Any | Debit Spreads | IV will increase |
| `neutral` | High conf | Directional spread | Follow direction model |
| `neutral` | Low conf | No trade | Insufficient edge |

*Iron Condor is Phase 2 (beyond MVP)

### 5.2 Strike Selection Rules

```python
def select_strikes(direction: str, regime: str, spot: float) -> dict:
    """
    Rule-based strike selection

    Returns: {
        'strategy': str,
        'buy_strike': float,
        'sell_strike': float,
        'spread_width': float
    }
    """

    # Delta targets based on regime
    delta_targets = {
        'low_vol': {'long': 0.40, 'short': 0.30},  # Closer to ATM
        'high_vol': {'long': 0.30, 'short': 0.20},  # Further OTM
        'iv_contraction': {'long': 0.25, 'short': 0.15},  # Far OTM
        'iv_expansion': {'long': 0.45, 'short': 0.35},  # Near ATM
        'neutral': {'long': 0.35, 'short': 0.25}
    }

    # Spread width based on regime
    spread_widths = {
        'low_vol': 5,  # $5 wide
        'high_vol': 10,  # $10 wide (more cushion)
        'iv_contraction': 5,
        'iv_expansion': 5,
        'neutral': 5
    }

    return {
        'delta_target': delta_targets[regime],
        'spread_width': spread_widths[regime]
    }
```

### 5.3 DTE Selection Rules

```python
def select_dte(regime: str, iv_rank: float) -> int:
    """
    Rule-based DTE selection

    Returns: Target days to expiration
    """

    # Base DTE by regime
    base_dte = {
        'low_vol': 30,  # Longer DTE for theta
        'high_vol': 21,  # Shorter for faster exit
        'iv_contraction': 21,  # Capture IV crush
        'iv_expansion': 45,  # Longer for IV rise
        'neutral': 30
    }

    dte = base_dte[regime]

    # Adjust for IV rank
    if iv_rank > 0.7:
        dte -= 7  # Shorter DTE when IV is high
    elif iv_rank < 0.3:
        dte += 7  # Longer DTE when IV is low

    return max(14, min(45, dte))  # Clamp to 14-45 days
```

### 5.4 Position Sizing Rules

```python
def calculate_position_size(
    account_value: float,
    max_risk: float,
    spread_width: float,
    credit_received: float,
    regime: str,
    confidence: float
) -> int:
    """
    Rule-based position sizing

    Returns: Number of contracts
    """

    # Base risk per trade: 2% of account
    base_risk_pct = 0.02

    # Regime multiplier
    regime_multipliers = {
        'low_vol': 1.0,
        'high_vol': 0.5,  # Half size in high vol
        'iv_contraction': 1.0,
        'iv_expansion': 0.75,
        'neutral': 0.75
    }

    # Confidence multiplier (0.5 to 1.0)
    confidence_mult = 0.5 + 0.5 * confidence

    # Calculate max loss per spread
    max_loss_per_spread = (spread_width - credit_received) * 100

    # Calculate position size
    risk_amount = account_value * base_risk_pct
    risk_amount *= regime_multipliers[regime]
    risk_amount *= confidence_mult

    num_contracts = int(risk_amount / max_loss_per_spread)

    return max(1, min(10, num_contracts))  # 1-10 contracts
```

---

## 6. Phase-by-Phase Roadmap

### Phase 1: MVP (Weeks 1-3) - Current

**Scope:**
- SPY, QQQ, IWM (ETFs only)
- Direction classifier + Volatility forecaster
- Rule-based options selection
- Paper trading

**Deliverables:**
- Working Docker environment
- Data pipeline (yfinance)
- Feature engineering (50+ features)
- ML models (XGBoost)
- Rule-based options logic
- Paper trading script

### Phase 2: Validation (Weeks 4-6)

**Scope:**
- Extended backtesting
- Walk-forward validation
- Parameter optimization
- Live paper trading

**Deliverables:**
- Backtest results (2+ years)
- Optimized parameters
- 30+ paper trades
- Performance analysis

### Phase 3: Mega Caps (Weeks 7-9)

**Scope:**
- Add mega cap stocks (AAPL, MSFT, GOOGL, etc.)
- Earnings blackout handling
- Cross-ticker analysis

**Deliverables:**
- Multi-ticker pipeline
- Earnings calendar integration
- Enhanced feature set

### Phase 4: Earnings Model (Weeks 10-12)

**Scope:**
- Earnings-specific model
- IV crush strategy
- Earnings straddles/strangles

**Deliverables:**
- Earnings model
- IV crush detection
- Earnings trading rules

### Phase 5: Production (Month 4+)

**Scope:**
- Real money deployment (small capital)
- Production monitoring
- Alerting system
- Performance tracking

**Deliverables:**
- Live trading system
- Monitoring dashboard
- Automated reports

---

## Appendix A: Key Files Reference

```
src/
├── config/
│   └── settings.py          # Configuration management
├── data/
│   ├── yfinance_loader.py   # Data fetching
│   └── feature_store.py     # Feature storage
├── features/
│   └── technical_indicators.py  # Feature calculation
├── models/
│   ├── direction_classifier.py  # Direction ML model
│   ├── volatility_forecaster.py # Volatility ML model
│   └── regime_classifier.py     # Regime rules
├── strategy/
│   └── options_rules.py     # Options selection rules
└── utils/
    └── logger.py            # Logging utilities
```

---

## Appendix B: Success Criteria

| Metric | Target | Threshold | Timeline |
|--------|--------|-----------|----------|
| Direction Model AUC | > 0.55 | > 0.52 | Week 2 |
| Volatility Model RMSE | < 0.15 | < 0.20 | Week 2 |
| Backtest Sharpe | > 1.0 | > 0.8 | Week 3 |
| Paper Win Rate | > 55% | > 50% | Phase 2 |
| Paper Profit Factor | > 1.3 | > 1.1 | Phase 2 |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Engineering | Initial implementation plan |
