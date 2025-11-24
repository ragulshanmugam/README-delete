# ML Options Trading System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ML OPTIONS TRADING SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐ │
│   │  Data Layer  │───▶│ Feature Layer│───▶│         ML MODELS            │ │
│   └──────────────┘    └──────────────┘    │  ┌────────────────────────┐  │ │
│          │                   │            │  │  Direction Classifier  │  │ │
│          ▼                   ▼            │  │  (UP/DOWN prediction)  │  │ │
│   ┌──────────────┐    ┌──────────────┐    │  │  AUC: 0.61 | Acc: 57%  │  │ │
│   │ Yahoo Finance│    │  Technical   │    │  └────────────────────────┘  │ │
│   │ FRED API     │    │  Indicators  │    │              │               │ │
│   │ VIX Data     │    │  (83 features│    │              ▼               │ │
│   │ Finnhub API  │    │  IV Features │    │  ┌────────────────────────┐  │ │
│   │ Reddit API   │    │  (18 features│    │  │ Volatility Forecaster  │  │ │
│   └──────────────┘    │  Macro Data  │    │  │ (LOW/NORMAL/HIGH)      │  │ │
│                       │  (34 features│    │  │ Acc: 64% | F1: 0.39    │  │ │
│                       │  Sentiment   │    │  └────────────────────────┘  │ │
│                       │  (12 features│    └──────────────────────────────┘ │
│                       │  Earnings    │                                     │
│                       │  (11 features│                   │                  │
│                       └──────────────┘                   ▼                  │
│                                                  ┌──────────────┐          │
│                                                  │   Strategy   │          │
│                                                  │   Selector   │          │
│                                                  └──────────────┘          │
│                                                          │                  │
│                                                          ▼                  │
│                                                  ┌──────────────┐          │
│                                                  │   Dashboard  │          │
│                                                  │  (Streamlit) │          │
│                                                  │ + SQLite DB  │          │
│                                                  └──────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   RAW DATA  │     │  FEATURES   │     │   TRAINING  │     │  PREDICTION │
│   SOURCES   │────▶│ ENGINEERING │────▶│     DATA    │────▶│    MODEL    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ • OHLCV     │     │ • 83 Tech   │     │ • X: 40 top │     │ • Prob(UP)  │
│ • VIX/VXN   │     │ • 18 IV     │     │   features  │     │ • Direction │
│ • FRED Macro│     │ • 34 Macro  │     │ • y: UP/DOWN│     │ • IV Regime │
│   (10 series)│     │ = 135 total │     │ • 771 samples│    │ • Confidence│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Model Architecture

### Direction Classifier (Current Implementation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DIRECTION CLASSIFIER MODEL                             │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT FEATURES (40 selected from 135)
─────────────────────────────────────
│
├─▶ Momentum Features (Safe)        ├─▶ Macro Features
│   • returns_5d, returns_10d       │   • fed_funds_effective
│   • returns_20d, returns_60d      │   • yield_spread_10y_2y
│   • rsi_14, rsi_28                │   • breakeven_inflation_5y
│   • macd_histogram                │   • real_rate_10y
│   • stoch_k, stoch_d              │
│                                   │
├─▶ Volatility Features             ├─▶ IV Features (NEW)
│   • hv_5, hv_10, hv_20, hv_60     │   • iv_rank (0-100)
│   • hv_ratio_5_20                 │   • iv_hv_spread
│   • bb_width, bb_position         │   • vix_zscore
│   • atr_pct                       │   • iv_regime_low/high
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LOGISTIC REGRESSION (L2 Regularization)                 │
│                                                                             │
│   P(UP) = sigmoid(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b)                        │
│                                                                             │
│   Regularization: C = 0.1 (medium strength)                                 │
│   Solver: lbfgs                                                             │
│   Max iterations: 1000                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THRESHOLD OPTIMIZATION                               │
│                                                                             │
│   Default: Predict UP if P(UP) > 0.50                                      │
│   Optimized: Predict UP if P(UP) > optimal_threshold                       │
│                                                                             │
│   Finding optimal threshold:                                                │
│   - Test thresholds from 0.30 to 0.70                                      │
│   - Select threshold that maximizes F1 score                               │
│   - Current optimal: 0.30 (bias toward UP predictions)                     │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
OUTPUT
──────
• direction: "UP" or "DOWN"
• probability: 0.0 to 1.0
• confidence: max(prob, 1-prob)
```

### Strategy Selector Engine (Phase 4 - Complete)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STRATEGY SELECTOR ENGINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

PURPOSE: Combine Direction + IV predictions into actionable trade signals
─────────────────────────────────────────────────────────────────────────

INPUTS:
───────
│
├─▶ Direction Prediction           ├─▶ Volatility Prediction
│   • direction: UP | DOWN         │   • iv_regime: LOW | NORMAL | HIGH
│   • confidence: 0.0 - 1.0        │   • confidence: 0.0 - 1.0
│   • probability: P(UP)           │   • current_iv_rank: 0-100
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STRATEGY SELECTION MATRIX                            │
│                                                                             │
│   ┌─────────────┬─────────────┬───────────────────────────────────────────┐ │
│   │ Direction   │ IV Regime   │ Strategy                                  │ │
│   ├─────────────┼─────────────┼───────────────────────────────────────────┤ │
│   │ UP          │ LOW         │ Buy Calls (cheap premium, directional)   │ │
│   │ UP          │ NORMAL      │ Bull Call Spread (defined risk/reward)   │ │
│   │ UP          │ HIGH        │ Bull Put Spread (sell premium + upside)  │ │
│   │ DOWN        │ LOW         │ Buy Puts (cheap protection)              │ │
│   │ DOWN        │ NORMAL      │ Bear Put Spread (defined risk)           │ │
│   │ DOWN        │ HIGH        │ Bear Call Spread (sell premium + down)   │ │
│   └─────────────┴─────────────┴───────────────────────────────────────────┘ │
│                                                                             │
│   CONFIDENCE FILTERS:                                                       │
│   • MIN_DIRECTION_CONFIDENCE = 0.52                                         │
│   • MIN_IV_CONFIDENCE = 0.35                                                │
│   • MIN_OVERALL_CONFIDENCE = 0.45                                           │
│   → If confidence below threshold → NO TRADE                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        POSITION SIZING LOGIC                                │
│                                                                             │
│   overall_confidence = direction_conf * 0.6 + iv_conf * 0.4                 │
│                                                                             │
│   Position Size:                                                            │
│   • FULL  (70-100% confidence)  → Standard position size                   │
│   • HALF  (55-70% confidence)   → Reduced position size                    │
│   • NONE  (<55% confidence)     → No trade                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
OUTPUT (StrategyRecommendation)
───────────────────────────────
{
  "strategy": "Buy Calls",
  "direction": "UP",
  "iv_regime": "LOW",
  "direction_confidence": 0.973,
  "iv_confidence": 0.738,
  "overall_confidence": 0.872,
  "position_size": "full",
  "reasoning": "Bullish outlook + cheap options = buy directional calls",
  "trade_parameters": {
    "expiry": "2-4 weeks",
    "strike": "ATM or slightly OTM (0.40-0.50 delta)",
    "max_loss": "2.0% of portfolio",
    "profit_target": "50% of max profit",
    "stop_loss": "Close if underlying moves 2% against"
  }
}
```

### Volatility Forecaster (Phase 3 - Complete)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VOLATILITY FORECASTER MODEL                            │
└─────────────────────────────────────────────────────────────────────────────┘

PURPOSE: Predict IV regime 5 days ahead for strategy selection
─────────────────────────────────────────────────────────────

INPUT FEATURES (27 selected)
─────────────────────────────
│
├─▶ Current IV State               ├─▶ Lagged IV Features
│   • iv_rank (0-100)              │   • iv_rank_lag_5
│   • iv_percentile                │   • iv_rank_lag_10
│   • iv_hv_spread                 │   • iv_rank_lag_20
│   • vix_close, vix_zscore        │   • iv_rank_change_5d
│   • vix_trend, vix_change_1d/5d  │   • iv_rank_change_10d
│
├─▶ Term Structure                 ├─▶ Historical Volatility
│   • vix_term_slope               │   • hv_5, hv_10, hv_20, hv_60
│   • vix_contango                 │   • hv_ratio_5_20, hv_ratio_20_60
│                                  │   • hv_trend
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   LOGISTIC REGRESSION (Multinomial, 3-class)                │
│                                                                             │
│   P(regime) = softmax(W·x + b)                                             │
│                                                                             │
│   Classes: LOW (IV Rank 0-30), NORMAL (30-60), HIGH (60-100)               │
│   Regularization: C = 0.1                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼
OUTPUT
──────
• predicted_regime: "LOW" | "NORMAL" | "HIGH"
• regime_probabilities: {LOW: 0.7, NORMAL: 0.2, HIGH: 0.1}
• expected_iv_direction: "EXPANDING" | "CONTRACTING" | "STABLE"
• confidence: max(probabilities)

PERFORMANCE (SPY, Nov 2024)
────────────────────────────
• Accuracy: 63.9%
• F1 (macro): 0.391
• Best at predicting LOW regime (F1=0.78) - most common state
• Struggles with HIGH regime (F1=0.18) - rare events

TOP FEATURES
────────────
1. iv_rank: 0.439
2. hv_ratio_20_60: 0.329
3. iv_rank_lag_10: 0.284
4. iv_rank_lag_5: 0.278
5. iv_percentile: 0.269
```

---

## Feature Engineering Details

### Technical Indicators (83 features)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TECHNICAL INDICATORS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TREND                    │  MOMENTUM              │  VOLATILITY            │
│  ─────                    │  ────────              │  ──────────            │
│  • SMA (20, 50, 200)      │  • RSI (14, 28)        │  • HV (5,10,20,60,120) │
│  • EMA (12, 26)           │  • MACD + Signal       │  • Bollinger Bands     │
│  • Price vs SMA ratios    │  • Stochastic K/D      │  • ATR (14)            │
│  • MA Crossovers          │  • ROC (5, 10, 20)     │  • Parkinson HV        │
│  • ADX (14)               │  • MFI                 │                        │
│                           │                        │                        │
│  VOLUME                   │  PRICE LEVELS          │  RETURNS               │
│  ──────                   │  ────────────          │  ───────               │
│  • Volume SMA (10, 20)    │  • 52-week High/Low    │  • 1d, 5d, 10d, 20d   │
│  • Volume Ratio           │  • Distance from H/L   │  • 60d, 252d           │
│  • OBV                    │  • Near 52w flags      │  • Log returns         │
│  • Volume Surge flag      │                        │                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### IV Features (18 features) - VIX as Proxy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IV FEATURES (VIX-Based)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VIX PROXY MAPPING:                                                         │
│  ──────────────────                                                         │
│  SPY ──▶ ^VIX  (S&P 500 Volatility Index)                                  │
│  QQQ ──▶ ^VXN  (Nasdaq 100 Volatility Index)                               │
│  IWM ──▶ ^RVX  (Russell 2000 Volatility Index)                             │
│                                                                             │
│  CORE IV METRICS:                                                           │
│  ────────────────                                                           │
│  ┌────────────────┬──────────────────────────────────────────────────────┐ │
│  │ iv_rank        │ (VIX - 52wk_Low) / (52wk_High - 52wk_Low) * 100     │ │
│  │ iv_percentile  │ % of days VIX was lower than current                 │ │
│  │ iv_hv_spread   │ VIX - Historical_Volatility (in % points)            │ │
│  │ iv_premium     │ 1 if iv_hv_spread > 0 else 0                         │ │
│  └────────────────┴──────────────────────────────────────────────────────┘ │
│                                                                             │
│  VIX MOMENTUM:                                                              │
│  ─────────────                                                              │
│  • vix_sma_10, vix_sma_20      (VIX moving averages)                       │
│  • vix_change_1d, vix_change_5d (VIX % change)                             │
│  • vix_zscore                   ((VIX - 60d_mean) / 60d_std)               │
│  • vix_trend                    (vix_sma_10 - vix_sma_20)                  │
│                                                                             │
│  TERM STRUCTURE:                                                            │
│  ───────────────                                                            │
│  • vix_term_slope  = VIX3M - VIX                                           │
│  • vix_contango    = 1 if slope > 0 (normal market)                        │
│                      0 if slope < 0 (fear/backwardation)                   │
│                                                                             │
│  IV REGIME CLASSIFICATION:                                                  │
│  ─────────────────────────                                                  │
│  ┌─────────────┬──────────────┬─────────────────────────────────────────┐  │
│  │ IV Rank     │ Regime       │ Strategy Implication                    │  │
│  ├─────────────┼──────────────┼─────────────────────────────────────────┤  │
│  │ 0 - 25      │ LOW          │ Buy premium (options cheap)             │  │
│  │ 25 - 50     │ NORMAL       │ Neutral / directional plays             │  │
│  │ 50 - 75     │ ELEVATED     │ Consider selling premium                │  │
│  │ 75 - 100    │ HIGH         │ Sell premium (IV crush expected)        │  │
│  └─────────────┴──────────────┴─────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Macro Features (34 features)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MACRO FEATURES (FRED API)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW SERIES (10):                                                           │
│  ────────────────                                                           │
│  • DFF     (Fed Funds Rate)        • T10Y2Y  (10Y-2Y Spread)               │
│  • DGS2    (2-Year Treasury)       • T10YIE  (Breakeven Inflation)         │
│  • DGS10   (10-Year Treasury)      • DFII10  (Real Interest Rate)          │
│  • DGS3MO  (3-Month Treasury)      • UNRATE  (Unemployment)                │
│  • VIXCLS  (VIX Close)             • CPIAUCSL (CPI)                        │
│                                                                             │
│  ENGINEERED FEATURES:                                                       │
│  ────────────────────                                                       │
│  • Yield curve slope & inversions                                          │
│  • Rate momentum (changes over 5d, 20d, 60d)                               │
│  • VIX momentum and regime                                                 │
│  • Real rates (nominal - inflation)                                        │
│  • Fed pivot signals                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sentiment Features (12 features) - Finnhub + Reddit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SENTIMENT FEATURES (Phase 5)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA SOURCES:                                                              │
│  ─────────────                                                              │
│  • Finnhub API (Free Tier) - Market news headlines                         │
│  • ApeWisdom API - Reddit mentions (WSB, stocks subreddits)                │
│                                                                             │
│  NEWS SENTIMENT (Finnhub):                                                  │
│  ─────────────────────────                                                  │
│  ┌──────────────────────┬───────────────────────────────────────────────┐  │
│  │ sentiment_score      │ Bullish% - Bearish% from news (-1 to +1)     │  │
│  │ sentiment_vs_sector  │ Relative to sector average                    │  │
│  │ news_buzz_score      │ News volume intensity (0 to 1)               │  │
│  │ news_article_count   │ Articles in last fetch                       │  │
│  └──────────────────────┴───────────────────────────────────────────────┘  │
│                                                                             │
│  SOCIAL SENTIMENT (Reddit):                                                 │
│  ──────────────────────────                                                 │
│  ┌──────────────────────┬───────────────────────────────────────────────┐  │
│  │ reddit_mentions      │ Mention count in WSB/stocks                   │  │
│  │ reddit_momentum      │ Change from 24h ago                           │  │
│  └──────────────────────┴───────────────────────────────────────────────┘  │
│                                                                             │
│  BINARY FLAGS:                                                              │
│  ─────────────                                                              │
│  • sentiment_bullish      (score > 0.15)                                   │
│  • sentiment_bearish      (score < -0.15)                                  │
│  • sentiment_neutral      (-0.15 to 0.15)                                  │
│  • sentiment_extreme_bullish (score > 0.35)                                │
│  • sentiment_extreme_bearish (score < -0.35)                               │
│  • high_social_attention  (reddit_mentions > 50)                           │
│                                                                             │
│  CROSS-FEATURES (Sentiment + IV):                                          │
│  ─────────────────────────────────                                          │
│  • sentiment_bullish_high_iv  (bullish + IV > 60 → sell premium)           │
│  • sentiment_bearish_low_iv   (bearish + IV < 30 → cheap puts)             │
│  • sentiment_iv_divergence    (contrarian signal)                          │
│                                                                             │
│  NOTE: Free tier provides market-wide sentiment (not per-stock).           │
│        Upgrade to Finnhub Premium ($50/mo) for per-stock sentiment.        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

### Walk-Forward Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WALK-FORWARD CROSS-VALIDATION                          │
│                     (Prevents Look-Ahead Bias)                              │
└─────────────────────────────────────────────────────────────────────────────┘

Timeline: |─────────────────────────────────────────────────────────────────▶

Fold 1:   [========TRAIN========][==TEST==]
Fold 2:   [==========TRAIN==========][==TEST==]
Fold 3:   [============TRAIN============][==TEST==]
Fold 4:   [==============TRAIN==============][==TEST==]
Fold 5:   [================TRAIN================][==TEST==]

Key Parameters:
─────────────────
• n_splits: 5
• test_size: ~20% of available data per fold
• gap: 5 days (prediction horizon) to prevent leakage
• Expanding window: Each fold uses more training data
```

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. LOAD DATA │───▶│ 2. FEATURES  │───▶│ 3. LABELS    │───▶│ 4. TRAIN     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  Load cached        Select top 40       Create binary       Walk-forward
  parquet files      features using      labels:             validation
  + fetch VIX        mutual info         UP if ret > 1%      (5 folds)
  + fetch FRED                           DOWN if ret < -1%

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 5. OPTIMIZE  │───▶│ 6. EVALUATE  │───▶│ 7. SAVE      │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  Find optimal       Calculate:           Save model
  threshold          • AUC                + metadata
  (0.30-0.70)        • Accuracy           to .joblib
                     • F1, P, R           + MLflow
```

---

## Input/Output Specification

### Training Input

```yaml
Input:
  ticker: "SPY" | "QQQ" | "IWM"
  data_period: "5y"

  Raw Data:
    - OHLCV: [open, high, low, close, volume] from Yahoo Finance
    - VIX: [close, high, low] from Yahoo Finance
    - FRED: 10 macro economic series

  Feature Engineering Output:
    - 83 technical indicators
    - 18 IV features
    - 34 macro features
    - Total: 135 features

  Feature Selection:
    - Method: Mutual Information
    - Top N: 40 features selected

  Label Creation:
    - Horizon: 5 trading days
    - Threshold: +/- 1%
    - Classes: UP (return > 1%), DOWN (return < -1%)
    - Ambiguous samples (|return| < 1%) excluded
```

### Training Output

```yaml
Output:
  Model Artifact:
    path: models/robust_classifier_logistic_YYYYMMDD_HHMMSS.joblib
    contents:
      - model: LogisticRegression object
      - feature_names: List[str] (40 features)
      - optimal_threshold: float (e.g., 0.30)
      - scaler: StandardScaler object
      - metadata: training config & metrics

  Metrics:
    - AUC: 0.611 (ranking quality)
    - Accuracy: 54.5% (after threshold optimization)
    - Precision: 0.576
    - Recall: 0.554
    - F1: 0.460

  MLflow Tracking:
    - Run ID
    - Parameters
    - Metrics
    - Model artifact
```

### Prediction Input/Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION INTERFACE                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT (Real-time):
──────────────────
{
  "ticker": "SPY",
  "date": "2024-01-15",
  "features": {
    // 40 pre-computed features
    "returns_5d": 0.023,
    "rsi_14": 58.2,
    "iv_rank": 35.0,
    "iv_hv_spread": 2.5,
    "fed_funds_effective": 5.33,
    ...
  }
}

OUTPUT:
───────
{
  "ticker": "SPY",
  "date": "2024-01-15",
  "prediction": {
    "direction": "UP",
    "probability": 0.65,
    "confidence": 0.65,
    "threshold_used": 0.30
  },
  "iv_context": {
    "iv_rank": 35.0,
    "iv_regime": "normal",
    "iv_hv_spread": 2.5
  },
  "strategy_suggestion": {
    "primary": "Buy Calls (ATM, 2-3 weeks expiry)",
    "reasoning": "Direction=UP + IV_Normal → Directional play"
  }
}
```

---

## Strategy Selection Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STRATEGY SELECTION LOGIC                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────┬─────────────────┬─────────────────┬─────────────────────────┐
│ Direction     │ IV Regime       │ Strategy        │ Rationale               │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ UP            │ LOW (0-25)      │ Buy Calls       │ Cheap options,          │
│               │                 │                 │ directional bet         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ UP            │ NORMAL (25-50)  │ Bull Call Spread│ Balanced cost/reward    │
│               │                 │                 │                         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ UP            │ HIGH (75-100)   │ Sell Puts       │ Collect premium,        │
│               │ (Credit Spread) │                 │ IV crush expected       │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ DOWN          │ LOW (0-25)      │ Buy Puts        │ Cheap protection        │
│               │                 │                 │                         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ DOWN          │ NORMAL (25-50)  │ Bear Put Spread │ Defined risk            │
│               │                 │                 │                         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ DOWN          │ HIGH (75-100)   │ Sell Calls      │ Collect premium +       │
│               │ (Credit Spread) │                 │ directional bias        │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ LOW CONF      │ HIGH            │ Iron Condor     │ Range-bound, sell       │
│ (< 55%)       │                 │                 │ volatility              │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ LOW CONF      │ LOW             │ NO TRADE        │ Wait for better signal  │
│ (< 55%)       │                 │                 │                         │
└───────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```

---

## Project Structure

```
ml-options-trading/
├── src/
│   ├── data/
│   │   ├── fetch_data.py          # Yahoo Finance data fetcher
│   │   ├── feature_store.py       # Parquet file management
│   │   └── macro_indicators.py    # FRED API integration
│   │
│   ├── features/
│   │   ├── technical_indicators.py # 83 technical features
│   │   ├── iv_indicators.py        # 18 IV features (VIX-based)
│   │   └── sentiment_indicators.py # 12 sentiment features (Finnhub/Reddit)
│   │
│   ├── models/
│   │   ├── feature_pipeline.py     # Feature loading & preprocessing
│   │   ├── robust_classifier.py    # Direction classifier (Logistic Regression)
│   │   ├── volatility_forecaster.py # IV regime forecaster
│   │   ├── strategy_selector.py    # Strategy selection engine
│   │   ├── threshold_optimizer.py  # Decision threshold optimization
│   │   └── diagnose_model.py       # Model diagnostics
│   │
│   ├── config/
│   │   └── settings.py             # Configuration management
│   │
│   └── utils/
│       └── logger.py               # Logging utilities
│
├── scripts/
│   ├── fetch_data.py               # Data collection CLI
│   └── train_model.py              # Training CLI
│
├── models/                          # Saved model artifacts
├── data/
│   ├── raw/                         # Raw OHLCV data
│   ├── features/                    # Cached feature parquets
│   └── processed/                   # Processed datasets
│
└── mlruns/                          # MLflow tracking
```

---

## Current Performance Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL PERFORMANCE (Nov 2024)                             │
└─────────────────────────────────────────────────────────────────────────────┘

DIRECTION CLASSIFIER (Binary: UP/DOWN)
──────────────────────────────────────
┌──────────┬───────────┬──────────┬─────────────────┬──────────────────────────┐
│ Ticker   │ AUC       │ Accuracy │ Threshold       │ Notes                    │
├──────────┼───────────┼──────────┼─────────────────┼──────────────────────────┤
│ SPY      │ 0.611     │ 54.5%    │ 0.30            │ Baseline: 47.3%          │
│ QQQ      │ 0.615     │ 62.9%    │ 0.45            │ Similar to SPY           │
│ IWM      │ 0.657     │ 59.1%    │ 0.52            │ Strongest signal!        │
└──────────┴───────────┴──────────┴─────────────────┴──────────────────────────┘

VOLATILITY FORECASTER (3-class: LOW/NORMAL/HIGH)
────────────────────────────────────────────────
┌──────────┬──────────┬───────────┬────────────────────────────────────────────┐
│ Ticker   │ Accuracy │ F1 (macro)│ Per-Class F1                               │
├──────────┼──────────┼───────────┼────────────────────────────────────────────┤
│ SPY      │ 63.9%    │ 0.391     │ LOW: 0.78, NORMAL: 0.22, HIGH: 0.18       │
└──────────┴──────────┴───────────┴────────────────────────────────────────────┘

Key Insights:
• Direction Classifier: IWM shows strongest signal (AUC 0.657)
• Volatility Forecaster: Best at LOW regime (IV spends most time there)
• Combined system enables strategy selection based on both direction + IV
```

---

## Next Steps Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROADMAP                                        │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: Direction Classifier ✅ COMPLETE
─────────────────────────────────────────
• Binary classification (UP/DOWN)
• Walk-forward validation
• Threshold optimization
• Safe feature filtering

PHASE 2: IV Features ✅ COMPLETE
─────────────────────────────────
• VIX-based IV rank/percentile
• IV-HV spread calculation
• Term structure analysis
• IV regime classification

PHASE 3: Volatility Forecaster ✅ COMPLETE
──────────────────────────────────────────
• Predict IV_Rank buckets (LOW/NORMAL/HIGH)
• Input: 27 volatility features + lagged IV
• Output: expected IV regime in 5 days
• Accuracy: 63.9%, F1: 0.391
• Best at LOW regime prediction (F1=0.78)

PHASE 4: Strategy Selection Engine ✅ COMPLETE
──────────────────────────────────────────────
• Combines Direction + IV forecasts → Strategy
• Rules-based strategy matrix (6 strategy types)
• Position sizing (FULL/HALF/NONE based on confidence)
• Risk management rules (max loss, stop loss, profit targets)
• CLI commands: signal, signal-all

PHASE 5: Sentiment Features ✅ COMPLETE
──────────────────────────────────────────────
• Finnhub API integration (market news sentiment)
• ApeWisdom API integration (Reddit mentions)
• 12 sentiment features added
• Cross-features: sentiment + IV interactions
• CLI flag: --sentiment for training

PHASE 6: Earnings Calendar Features ✅ COMPLETE
────────────────────────────────────────────────
• Finnhub earnings calendar integration
• 11 earnings-related features (days to/since earnings, etc.)
• Earnings-IV interaction features
• CLI flag: --earnings for training

PHASE 7: Dashboard & Paper Trading ✅ COMPLETE
────────────────────────────────────────────────
• Streamlit multi-page dashboard
• SQLite database for prediction logging
• Daily Signals page with strategy explanations
• Performance tracking with charts
• Model metrics & PSI drift detection
• CLI: --record flag to save predictions

PHASE 8: Production Deployment 📋 PLANNED
──────────────────────────────────────────
• Tradier/Polygon IV integration (real strike-level IV)
• XGBoost ensemble (model improvements)
• Scheduled daily signal generation
• Alerting system (email/Slack)
```

---

## Quick Reference: CLI Commands

```bash
# Fetch data for a ticker
docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker SPY

# Train Direction Classifier with IV features
docker-compose run --rm app python scripts/train_model.py train \
  --ticker SPY \
  --model-type logistic \
  --safe-features \
  --optimize-threshold \
  --n-features 40

# Train with Sentiment features (requires FINNHUB_API_KEY)
docker-compose run --rm app python scripts/train_model.py train \
  --ticker SPY \
  --sentiment \
  --optimize-threshold

# Train Volatility Forecaster
docker-compose run --rm app python scripts/train_model.py train-volatility \
  --ticker SPY \
  --mode classification

# Train all tickers
docker-compose run --rm app python scripts/train_model.py train-all \
  --tickers SPY,QQQ,IWM

# Generate trading signal for single ticker
docker-compose run --rm app python scripts/train_model.py signal \
  --ticker SPY

# Generate signals for all tickers
docker-compose run --rm app python scripts/train_model.py signal-all \
  --tickers SPY,QQQ,IWM

# Generate and RECORD signals to database
docker-compose run --rm app python scripts/train_model.py signal-all --record

# Train with Earnings features (requires FINNHUB_API_KEY)
docker-compose run --rm app python scripts/train_model.py train \
  --ticker SPY \
  --earnings \
  --sentiment \
  --optimize-threshold

# ===== DASHBOARD COMMANDS =====

# Run Streamlit Dashboard
streamlit run dashboard/app.py

# Or via Docker
docker-compose run --rm -p 8501:8501 app streamlit run dashboard/app.py

# View MLflow dashboard
mlflow ui --backend-store-uri file:./mlruns
```

---

## Two-Model System: Combined Prediction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMBINED PREDICTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   Input Data     │
                         │  (Daily OHLCV)   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
         ┌───────────────────┐       ┌───────────────────┐
         │    Direction      │       │    Volatility     │
         │    Classifier     │       │    Forecaster     │
         │  (127 features)   │       │  (27 features)    │
         └─────────┬─────────┘       └─────────┬─────────┘
                   │                           │
                   ▼                           ▼
         ┌───────────────────┐       ┌───────────────────┐
         │ Direction: UP     │       │ IV Regime: LOW    │
         │ Confidence: 60%   │       │ Confidence: 72%   │
         │ Threshold: 0.30   │       │ Direction: STABLE │
         └─────────┬─────────┘       └─────────┬─────────┘
                   │                           │
                   └───────────┬───────────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │  Strategy Selector │
                    │  (Direction + IV)  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     OUTPUT        │
                    │                   │
                    │ Strategy: BUY     │
                    │   CALLS           │
                    │                   │
                    │ Reason: UP +      │
                    │   LOW IV =        │
                    │   cheap options   │
                    └───────────────────┘
```
