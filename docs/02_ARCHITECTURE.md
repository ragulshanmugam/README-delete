# Adaptive Options Trading System - Architecture Document

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Draft
**Owner:** Engineering Team

---

## 1. Architecture Overview

### 1.1 System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                     External Systems                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Market     │  │   Broker     │  │   VIX        │         │
│  │   Data       │  │   API        │  │   Data       │         │
│  │  (yfinance)  │  │   (IBKR)     │  │  (yfinance)  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Adaptive Options Trading System                  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   Data Layer                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │Fetchers  │  │Processors│  │ Storage  │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                 Feature Layer                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │ Greeks   │  │ Regime   │  │Technical │            │    │
│  │  │Calculator│  │ Detector │  │Indicators│            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   Model Layer                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │ XGBoost  │  │  LSTM    │  │ Ensemble │            │    │
│  │  │  Model   │  │  Model   │  │ Manager  │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                 Strategy Layer                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │IV Rank   │  │Direction-│  │Adaptive  │            │    │
│  │  │Strategy  │  │al Strat. │  │ Router   │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                  Trading Layer                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │  Risk    │  │  Order   │  │Position  │            │    │
│  │  │ Manager  │  │ Manager  │  │ Manager  │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Monitoring & Control Layer                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │Dashboard │  │  Alerts  │  │  Logs    │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Systems                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  PostgreSQL  │  │   Parquet    │  │   MLflow     │         │
│  │  (Trades,    │  │  (Historical │  │   (Models,   │         │
│  │   Positions) │  │    Market)   │  │    Metrics)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture Principles

#### Modularity
- **Principle:** Each layer is independent with well-defined interfaces
- **Benefit:** Can swap components (e.g., change broker) without rewriting system
- **Implementation:** Abstract base classes, dependency injection

#### Testability
- **Principle:** All components can be tested in isolation
- **Benefit:** High confidence in code correctness, fast iteration
- **Implementation:** Mock interfaces, fixtures, pytest framework

#### Observability
- **Principle:** System behavior is transparent and measurable
- **Benefit:** Quick debugging, performance optimization, compliance
- **Implementation:** Structured logging, metrics, tracing

#### Fail-Safe
- **Principle:** System degrades gracefully under failures
- **Benefit:** No catastrophic losses, high availability
- **Implementation:** Circuit breakers, retry logic, fallbacks

#### Configuration-Driven
- **Principle:** Behavior controlled via config files, not code changes
- **Benefit:** Fast parameter tuning, A/B testing, environment separation
- **Implementation:** YAML configs, Pydantic validation

---

### 1.3 Ticker Universe

The system supports a curated universe of liquid ETFs and mega-cap stocks, organized into tiers:

#### Tier 1: Index ETFs (Core - Always Active)

| Symbol | Name | Avg Daily Volume | Options Liquidity | Notes |
|--------|------|------------------|-------------------|-------|
| **SPY** | S&P 500 ETF | ~80M shares | Extremely High | Primary trading instrument |
| **QQQ** | Nasdaq 100 ETF | ~50M shares | Extremely High | Tech-heavy, higher beta |
| **IWM** | Russell 2000 ETF | ~25M shares | High | Small caps, higher volatility |
| **DIA** | Dow Jones ETF | ~3M shares | Medium | Blue chips, lower volatility |

**Characteristics:**
- No earnings risk (ETFs don't report earnings)
- Highly correlated with broad market
- Excellent options liquidity (tight spreads)
- Trade all year without earnings blackouts

#### Tier 2: Mega Caps (Expanded Universe)

| Symbol | Name | Sector | Earnings Months | Avg IV |
|--------|------|--------|-----------------|--------|
| **AAPL** | Apple | Technology | Jan, Apr, Jul, Oct | ~25% |
| **MSFT** | Microsoft | Technology | Jan, Apr, Jul, Oct | ~22% |
| **GOOGL** | Alphabet | Technology | Jan, Apr, Jul, Oct | ~28% |
| **AMZN** | Amazon | Consumer/Tech | Jan, Apr, Jul, Oct | ~30% |
| **NVDA** | NVIDIA | Semiconductors | Feb, May, Aug, Nov | ~45% |
| **TSLA** | Tesla | Automotive/Tech | Jan, Apr, Jul, Oct | ~55% |
| **META** | Meta Platforms | Technology | Jan, Apr, Jul, Oct | ~35% |
| **NFLX** | Netflix | Entertainment | Jan, Apr, Jul, Oct | ~40% |

**Characteristics:**
- Higher IV than ETFs (more premium to sell)
- Earnings 4x per year (requires special handling)
- Individual stock risk (news, events)
- Good options liquidity on all symbols

#### Tier 3: Sector ETFs (Optional Expansion)

| Symbol | Name | Sector | Use Case |
|--------|------|--------|----------|
| **XLF** | Financial Select | Financials | Bank earnings plays |
| **XLE** | Energy Select | Energy | Oil volatility plays |
| **XLK** | Technology Select | Technology | Sector rotation |
| **XLV** | Healthcare Select | Healthcare | Defensive plays |

#### Ticker Configuration

```yaml
# configs/tickers.yaml
ticker_universe:
  tier_1_etfs:
    - symbol: SPY
      name: "S&P 500 ETF"
      type: etf
      always_active: true
      min_dte: 3
      max_dte: 45

    - symbol: QQQ
      name: "Nasdaq 100 ETF"
      type: etf
      always_active: true
      min_dte: 3
      max_dte: 45

    - symbol: IWM
      name: "Russell 2000 ETF"
      type: etf
      always_active: true
      min_dte: 3
      max_dte: 45

  tier_2_mega_caps:
    - symbol: AAPL
      name: "Apple Inc"
      type: stock
      sector: technology
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5  # Days before earnings to stop core model

    - symbol: MSFT
      name: "Microsoft Corp"
      type: stock
      sector: technology
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5

    - symbol: GOOGL
      name: "Alphabet Inc"
      type: stock
      sector: technology
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5

    - symbol: AMZN
      name: "Amazon.com"
      type: stock
      sector: consumer
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5

    - symbol: NVDA
      name: "NVIDIA Corp"
      type: stock
      sector: semiconductors
      earnings_months: [2, 5, 8, 11]
      earnings_blackout_days: 5

    - symbol: TSLA
      name: "Tesla Inc"
      type: stock
      sector: automotive
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5

    - symbol: META
      name: "Meta Platforms"
      type: stock
      sector: technology
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5

    - symbol: NFLX
      name: "Netflix Inc"
      type: stock
      sector: entertainment
      earnings_months: [1, 4, 7, 10]
      earnings_blackout_days: 5
```

#### Ticker Selection Logic

```python
class TickerManager:
    """Manages ticker universe and determines which model to use"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.earnings_calendar = EarningsCalendar()

    def get_active_tickers(self, date: date) -> Dict[str, TickerContext]:
        """
        Get all active tickers and their trading context

        Returns:
            Dict mapping symbol to TickerContext with:
            - model_type: 'core' or 'earnings'
            - days_to_earnings: int or None
            - is_blackout: bool
        """
        active = {}

        # Tier 1 ETFs: Always use core model
        for etf in self.config['tier_1_etfs']:
            active[etf['symbol']] = TickerContext(
                symbol=etf['symbol'],
                model_type='core',
                days_to_earnings=None,
                is_blackout=False,
                is_etf=True
            )

        # Tier 2 Mega Caps: Check earnings proximity
        for stock in self.config['tier_2_mega_caps']:
            days_to_earnings = self.earnings_calendar.days_until_earnings(
                stock['symbol'], date
            )

            if days_to_earnings is not None and days_to_earnings <= 5:
                # Earnings window: Use earnings model
                active[stock['symbol']] = TickerContext(
                    symbol=stock['symbol'],
                    model_type='earnings',
                    days_to_earnings=days_to_earnings,
                    is_blackout=False,
                    is_etf=False
                )
            elif days_to_earnings is not None and days_to_earnings <= stock['earnings_blackout_days']:
                # Blackout period: Skip core model, wait for earnings model
                active[stock['symbol']] = TickerContext(
                    symbol=stock['symbol'],
                    model_type='blackout',
                    days_to_earnings=days_to_earnings,
                    is_blackout=True,
                    is_etf=False
                )
            else:
                # Normal period: Use core model
                active[stock['symbol']] = TickerContext(
                    symbol=stock['symbol'],
                    model_type='core',
                    days_to_earnings=days_to_earnings,
                    is_blackout=False,
                    is_etf=False
                )

        return active
```

#### Phased Rollout Plan

```
Phase 1 (Weeks 1-4):   SPY only
                       └── Build and validate core pipeline

Phase 2 (Weeks 5-8):   SPY + QQQ + IWM
                       └── Validate multi-ticker with similar instruments

Phase 3 (Weeks 9-12):  Add mega caps (exclude earnings windows)
                       └── AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA

Phase 4 (Weeks 13-16): Add earnings model
                       └── Enable earnings plays on mega caps
```

---

## 2. Component Architecture

### 2.1 Data Layer

#### 2.1.1 Data Fetchers

**Purpose:** Acquire raw market data from external sources

**Components:**

```python
# Abstract Interface
class DataFetcher(ABC):
    @abstractmethod
    def fetch_stock_data(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV stock data"""
        pass

    @abstractmethod
    def fetch_options_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Fetch options chain for specific expiration"""
        pass

    @abstractmethod
    def get_available_expirations(self, symbol: str) -> List[str]:
        """Get list of available option expirations"""
        pass

# Concrete Implementations
class YFinanceFetcher(DataFetcher):
    """Free tier data source"""
    pass

class IBKRFetcher(DataFetcher):
    """Interactive Brokers real-time data"""
    pass

class PolygonFetcher(DataFetcher):
    """Polygon.io premium data (future)"""
    pass
```

**Key Features:**
- Rate limiting to avoid API bans
- Retry logic with exponential backoff
- Caching to reduce API calls
- Data validation (check for NaNs, outliers)

**Technology:**
- yfinance: Free market data
- ib_insync: IBKR integration
- requests: HTTP client with retry
- redis: Optional caching layer

---

#### 2.1.2 Data Processors

**Purpose:** Clean, validate, and enrich raw data

**Pipeline:**
```
Raw Data → Validation → Cleaning → Enrichment → Storage
```

**Operations:**
1. **Validation:**
   - Check for missing values
   - Verify data types
   - Check for duplicates
   - Validate timestamp ordering

2. **Cleaning:**
   - Fill forward missing prices
   - Remove outliers (3-sigma rule)
   - Correct bid-ask crosses
   - Filter illiquid options

3. **Enrichment:**
   - Calculate mid-price
   - Compute spread percentage
   - Add moneyness
   - Calculate DTE

**Implementation:**
```python
class DataProcessor:
    def __init__(self, config: ProcessorConfig):
        self.validators = [
            MissingValueValidator(),
            OutlierValidator(sigma=3),
            BidAskValidator()
        ]
        self.enrichers = [
            MidPriceCalculator(),
            MoneynessCalculator(),
            DTECalculator()
        ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validate
        for validator in self.validators:
            df = validator.validate(df)

        # Clean
        df = self._clean_data(df)

        # Enrich
        for enricher in self.enrichers:
            df = enricher.enrich(df)

        return df
```

---

#### 2.1.3 Storage Layer

**Purpose:** Persist data for analysis and backtesting

**Storage Strategy:**

| Data Type | Storage | Reason |
|-----------|---------|--------|
| Market OHLCV | Parquet | Fast columnar queries, compression |
| Options Chains | PostgreSQL | Relational queries, indexes |
| Trades | PostgreSQL | ACID compliance, audit trail |
| Positions | PostgreSQL | Real-time updates, consistency |
| Features | Parquet | Large datasets, analytics |
| Model Artifacts | MLflow | Versioning, experiment tracking |
| Logs | Files + PostgreSQL | Search, retention policies |

**Schema Design:**

```sql
-- Trades Table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(id),
    symbol VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    pnl DECIMAL(10,2),
    pnl_pct DECIMAL(6,4),
    status VARCHAR(20) NOT NULL,
    regime VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_trades_strategy ON trades(strategy);

-- Positions Table
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    symbol VARCHAR(10) NOT NULL,
    leg_type VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    option_type VARCHAR(4) NOT NULL, -- 'call' or 'put'
    strike DECIMAL(10,2) NOT NULL,
    expiration DATE NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,4),
    current_price DECIMAL(10,4),
    delta DECIMAL(6,4),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    unrealized_pnl DECIMAL(10,2),
    status VARCHAR(20) NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);

-- Signals Table
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,
    regime VARCHAR(20) NOT NULL,
    iv_rank DECIMAL(4,3),
    vix DECIMAL(6,2),
    features JSONB,
    model_predictions JSONB,
    executed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_symbol ON signals(symbol);
```

---

### 2.2 Feature Layer

#### 2.2.1 Greeks Calculator

**Purpose:** Calculate option sensitivities using Black-Scholes model

**Key Calculations:**
- Delta: ∂V/∂S (price sensitivity)
- Gamma: ∂²V/∂S² (delta sensitivity)
- Vega: ∂V/∂σ (volatility sensitivity)
- Theta: ∂V/∂t (time decay)
- Rho: ∂V/∂r (interest rate sensitivity)
- Implied Volatility: Solve for σ given market price

**Algorithm:**
```
Input: S (spot), K (strike), T (time), r (rate), market_price
Output: Greeks dictionary

1. Calculate implied volatility:
   - Use Brent's method to solve BS(S,K,T,r,σ) = market_price
   - Bounds: σ ∈ [0.01, 5.0]
   - Tolerance: 1e-5

2. Calculate d1, d2:
   d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
   d2 = d1 - σ√T

3. Calculate Greeks:
   Delta = N(d1) for calls, -N(-d1) for puts
   Gamma = φ(d1) / (Sσ√T)
   Vega = Sφ(d1)√T / 100
   Theta = -Sφ(d1)σ/(2√T) - rKe^(-rT)N(d2) for calls

Where N(x) = standard normal CDF, φ(x) = standard normal PDF
```

**Performance Optimization:**
- Vectorized operations with NumPy
- Cache Greeks for unchanged inputs
- Pre-compute norm.pdf/cdf lookup tables

**Technology:**
- scipy.stats: Normal distribution functions
- scipy.optimize: Root finding for IV
- numba: JIT compilation for speed (optional)

---

#### 2.2.2 Regime Detector

**Purpose:** Classify market conditions for adaptive strategy selection

**Regime Definitions:**

| Regime | Conditions | Strategy Implication |
|--------|------------|---------------------|
| low_vol | VIX < 15 AND VIX_rank < 30% | Sell premium aggressively |
| high_vol | VIX > 25 OR VIX_rank > 80% | Reduce size, defensive |
| iv_expansion | HV/IV > 0.9 AND VIX_rank > 50% | Buy options (IV will rise) |
| iv_contraction | HV/IV < 0.6 AND VIX_rank < 50% | Sell options (IV will fall) |
| neutral | None of above | No new positions |

**Feature Engineering:**
```python
class RegimeDetector:
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime-related features"""

        # Realized volatility (20-day)
        returns = np.log(data['Close'] / data['Close'].shift(1))
        data['rv_20'] = returns.rolling(20).std() * np.sqrt(252)

        # VIX rank (252-day percentile)
        data['vix_rank'] = data['VIX'].rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )

        # HV/IV ratio
        data['hv_iv_ratio'] = data['rv_20'] / (data['VIX'] / 100)

        # VIX term structure (if VIX futures available)
        # data['vix_contango'] = vix_futures_3m - vix_spot

        # Regime classification
        data['regime'] = data.apply(self._classify_regime, axis=1)

        return data

    def _classify_regime(self, row: pd.Series) -> str:
        """Apply regime rules"""
        vix = row['VIX']
        vix_rank = row['vix_rank']
        hv_iv = row['hv_iv_ratio']

        if vix < 15 and vix_rank < 0.3:
            return 'low_vol'
        elif vix > 25 or vix_rank > 0.8:
            return 'high_vol'
        elif hv_iv > 0.9 and vix_rank > 0.5:
            return 'iv_expansion'
        elif hv_iv < 0.6 and vix_rank < 0.5:
            return 'iv_contraction'
        else:
            return 'neutral'
```

**Regime Transition Detection:**
- Track previous regime
- Log regime changes with timestamp
- Alert on transition to high_vol (risk signal)

---

#### 2.2.3 Feature Engineering

**Purpose:** Generate predictive features for ML models

**Feature Categories:**

**A. Options-Specific (40 features)**
```python
# Moneyness
- strike / spot (call & put)
- abs(strike - spot) / spot
- log(strike / spot)

# Greeks
- delta, gamma, vega, theta (call & put)
- delta_strike_ratio = delta / (strike/spot)
- gamma_scaled = gamma * spot^2
- vega_percentage = vega / option_price

# Implied Volatility
- iv_call, iv_put
- iv_skew = iv_put - iv_call (same strike)
- iv_atm, iv_otm25, iv_itm25
- iv_rank, iv_percentile
- iv_term_structure = iv_60dte - iv_30dte

# Time Value
- time_value = option_price - intrinsic_value
- time_value_pct = time_value / option_price
- theta_per_day_normalized = theta / time_value

# Liquidity
- bid_ask_spread_pct
- volume / open_interest
- log(volume), log(open_interest)
```

**B. Volatility Features (30 features)**
```python
# Historical Volatility
- rv_5, rv_10, rv_20, rv_60, rv_120
- rv_ratios: rv_20/rv_60, rv_5/rv_20
- volatility_of_volatility = std(rv_20)

# HV vs IV
- hv_iv_ratio (multiple windows)
- hv_iv_spread = hv - iv
- hv_iv_percentile

# VIX Features
- vix_level, vix_rank, vix_percentile
- vix_change_1d, vix_change_5d
- vix_trend = vix_sma5 - vix_sma20

# Volatility Regime
- regime_low_vol, regime_high_vol (one-hot encoded)
- days_in_current_regime
- regime_transition_probability (predicted)
```

**C. Price & Momentum Features (50 features)**
```python
# Returns
- returns_1d, returns_5d, returns_20d, returns_60d
- log_returns (same windows)
- cumulative_returns_ytd

# Momentum
- rsi_14, rsi_28
- macd, macd_signal, macd_histogram
- stochastic_k, stochastic_d
- adx (directional movement)

# Moving Averages
- sma_20, sma_50, sma_200
- ema_12, ema_26
- price_vs_sma20 = (price - sma20) / sma20
- sma_crossover = (sma_20 > sma_50)

# Price Patterns
- higher_highs, higher_lows (trend detection)
- support_level, resistance_level (technical analysis)
- distance_to_52w_high, distance_to_52w_low
```

**D. Volume & Open Interest (20 features)**
```python
# Volume
- volume, volume_sma20
- volume_ratio = volume / volume_sma20
- volume_surge = (volume > 2 * volume_sma20)
- log(volume)

# Open Interest
- open_interest, oi_change_1d
- oi_ratio = volume / open_interest
- oi_put_call_ratio

# Market Breadth
- put_volume, call_volume
- put_call_volume_ratio
- put_call_oi_ratio
- option_volume_total
```

**E. Temporal Features (15 features)**
```python
# Time
- day_of_week (Monday=1, Friday=5)
- days_to_expiration
- days_to_earnings (if known)
- time_to_expiration_years

# Market Timing
- is_monday, is_friday (one-hot)
- is_opex_week (options expiration week)
- is_earnings_week
- is_fomc_week (Fed meeting)

# Seasonality
- month_of_year
- quarter
```

**F. Derived Features (30 features)**
```python
# Cross-features
- delta * vega (portfolio Greeks products)
- iv_rank * hv_iv_ratio
- volume_ratio * price_momentum

# Lags
- iv_rank_lag1, iv_rank_lag5
- returns_lag1, returns_lag5
- regime_lag1 (previous regime)

# Rolling Statistics
- iv_rank_std_20 (volatility of IV rank)
- returns_skew_60 (skewness of returns)
- returns_kurtosis_60 (tail risk)
```

**Total: ~185 features (extensible to 270+ per Bali et al.)**

**Feature Pipeline:**
```python
class FeatureEngineer:
    def __init__(self):
        self.calculators = [
            OptionsFeatureCalculator(),
            VolatilityFeatureCalculator(),
            PriceFeatureCalculator(),
            VolumeFeatureCalculator(),
            TemporalFeatureCalculator(),
            DerivedFeatureCalculator()
        ]

    def generate_features(
        self,
        market_data: pd.DataFrame,
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate all features for ML models"""

        features = pd.DataFrame(index=market_data.index)

        for calculator in self.calculators:
            new_features = calculator.calculate(market_data, options_data)
            features = features.join(new_features)

        # Handle missing values
        features = self._handle_missing(features)

        # Normalize/standardize if needed
        # (typically done in model training pipeline)

        return features
```

---

### 2.3 Model Layer

#### 2.3.1 Two-Model Architecture

The system uses **two separate models** optimized for different market conditions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TWO-MODEL ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│         CORE MODEL               │    │       EARNINGS MODEL             │
│    (IV Rank Mean Reversion)      │    │     (IV Crush Strategy)          │
├──────────────────────────────────┤    ├──────────────────────────────────┤
│                                  │    │                                  │
│  Tickers:                        │    │  Tickers:                        │
│  • SPY, QQQ, IWM, DIA (ETFs)    │    │  • AAPL, MSFT, GOOGL, AMZN      │
│  • Mega caps (non-earnings)      │    │  • NVDA, TSLA, META, NFLX       │
│                                  │    │  • Only during earnings window   │
│  Training Data:                  │    │                                  │
│  • All days EXCEPT 5 days        │    │  Training Data:                  │
│    before earnings               │    │  • Only earnings windows         │
│  • ~1,200 days/ticker (5 years)  │    │  • ~20 events/ticker/year       │
│                                  │    │  • ~800 events total (5 years)   │
│  Strategy:                       │    │                                  │
│  • IV rank mean reversion        │    │  Strategy:                       │
│  • Sell premium when IV high     │    │  • Play IV crush post-earnings   │
│  • 20-45 DTE options            │    │  • Straddles/strangles           │
│  • Theta decay = profit          │    │  • 0-7 DTE options              │
│                                  │    │  • IV collapse = profit          │
│  ┌───────────┐  ┌───────────┐   │    │                                  │
│  │ XGBoost   │  │   LSTM    │   │    │  ┌───────────────────────────┐  │
│  │  Model    │  │   Model   │   │    │  │   Earnings XGBoost        │  │
│  └─────┬─────┘  └─────┬─────┘   │    │  │   (Specialized Features)  │  │
│        │              │          │    │  └─────────────┬─────────────┘  │
│        └──────┬───────┘          │    │               │                 │
│               ▼                  │    │               ▼                 │
│      ┌─────────────────┐        │    │      ┌─────────────────┐        │
│      │ Core Ensemble   │        │    │      │Earnings Prediction│       │
│      │ Prediction      │        │    │      │ + Direction       │       │
│      └─────────────────┘        │    │      └─────────────────┘        │
│                                  │    │                                  │
└──────────────────────────────────┘    └──────────────────────────────────┘
              │                                        │
              └────────────────┬───────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Model Router      │
                    │                     │
                    │  if earnings_window:│
                    │    → Earnings Model │
                    │  else:              │
                    │    → Core Model     │
                    └─────────────────────┘
```

#### 2.3.2 Core Model (Primary)

**Purpose:** Predict option profitability for IV rank mean reversion strategy

**Training Data:**
- **Tickers:** SPY, QQQ, IWM + Mega caps (excluding earnings windows)
- **Period:** 5 years of historical data
- **Exclusions:** Remove 5 days before each earnings date for stocks
- **Labels:** Binary (1 = profitable trade, 0 = unprofitable)

**Features (185 normalized features):**
```python
core_features = [
    # Normalized across all tickers
    'iv_rank',              # 0-1 percentile for THIS ticker
    'iv_percentile',        # 0-1 percentile vs 252-day history
    'hv_iv_ratio',          # Realized vol / Implied vol
    'moneyness',            # Strike / Spot (comparable across tickers)
    'delta', 'gamma', 'theta', 'vega',  # Greeks (already normalized)
    'dte',                  # Days to expiration

    # Price momentum (relative)
    'returns_5d', 'returns_20d', 'returns_60d',
    'rsi_14', 'rsi_28',
    'price_vs_sma20',       # (Price - SMA20) / SMA20

    # Volume (relative)
    'volume_ratio',         # Volume / 20-day avg volume
    'oi_ratio',             # Open interest ratio

    # Regime
    'vix_level', 'vix_rank',
    'regime_encoded',       # One-hot: low_vol, high_vol, etc.

    # Ticker context
    'is_etf',               # 1 for ETFs, 0 for stocks
    'sector_encoded',       # One-hot: tech, consumer, etc.
    'avg_iv_percentile',    # Ticker's typical IV level
]
```

**Model Architecture:**
```python
class CoreModel:
    """
    Pooled model trained on all tickers (excluding earnings windows)
    Uses normalized features so patterns generalize across tickers
    """

    def __init__(self):
        self.xgboost = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            colsample_bytree=0.8,
            subsample=0.8
        )
        self.lstm = LSTMModel(
            sequence_length=60,
            features=50,
            hidden_units=[128, 64]
        )
        self.ensemble_weights = {'xgboost': 0.6, 'lstm': 0.4}

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train on pooled data from all tickers"""
        # XGBoost on tabular features
        self.xgboost.fit(X, y)

        # LSTM on sequential data
        X_seq = self._create_sequences(X)
        self.lstm.fit(X_seq, y)

    def predict(self, features: pd.DataFrame) -> float:
        """Ensemble prediction"""
        xgb_pred = self.xgboost.predict_proba(features)[:, 1]
        lstm_pred = self.lstm.predict(self._create_sequences(features))

        return (
            self.ensemble_weights['xgboost'] * xgb_pred +
            self.ensemble_weights['lstm'] * lstm_pred
        )
```

#### 2.3.3 Earnings Model (Specialized)

**Purpose:** Predict post-earnings price movement and IV crush profitability

**Training Data:**
- **Tickers:** AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, NFLX
- **Period:** 5 years × 4 earnings/year × 8 stocks = ~160 events
- **Window:** 5 days before earnings through 2 days after
- **Labels:** Binary (1 = profitable IV crush play, 0 = unprofitable)

**Features (Earnings-Specific):**
```python
earnings_features = [
    # IV Context (relative to past earnings)
    'iv_vs_past_earnings_avg',      # Current IV / Avg IV at past 8 earnings
    'iv_percentile_earnings',       # Where is IV vs last 8 earnings?
    'iv_crush_expected',            # Typical IV drop post-earnings (%)

    # Expected Move
    'expected_move_pct',            # Straddle price / Stock price
    'expected_move_vs_historical',  # Expected move vs actual historical moves
    'straddle_price',               # ATM straddle cost

    # Historical Earnings Behavior
    'historical_move_avg',          # Avg absolute move last 8 earnings
    'historical_move_std',          # Volatility of moves
    'beat_rate_last_4',             # How often does company beat?
    'surprise_magnitude_avg',       # Avg earnings surprise %

    # Price Context
    'price_vs_52w_high',            # Distance from 52-week high
    'returns_into_earnings',        # 5-day return leading into earnings
    'gap_fill_tendency',            # Does stock tend to gap and fill?

    # Market Context
    'vix_level',
    'sector_earnings_sentiment',    # How did sector peers do this quarter?

    # Timing
    'days_to_earnings',             # 5, 4, 3, 2, 1, 0
    'is_after_hours',               # Does this company report after hours?
    'is_pre_market',                # Or pre-market?
]
```

**Model Architecture:**
```python
class EarningsModel:
    """
    Specialized model for earnings plays
    Trained only on earnings windows
    """

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,  # Less data, simpler model
            max_depth=4,
            learning_rate=0.1
        )

    def train(self, earnings_data: pd.DataFrame):
        """
        Train on earnings events only

        Data structure:
        - Each row is one earnings event
        - Features calculated 1-5 days before earnings
        - Label is whether IV crush play was profitable
        """
        X = earnings_data[earnings_features]
        y = earnings_data['profitable']

        self.model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Predict earnings play profitability

        Returns:
            - probability: Chance of profitable IV crush
            - suggested_strategy: 'sell_straddle', 'sell_strangle', 'skip'
            - confidence: Model confidence
        """
        prob = self.model.predict_proba(features)[:, 1][0]

        if prob > 0.7:
            strategy = 'sell_straddle'  # High confidence: aggressive
        elif prob > 0.55:
            strategy = 'sell_strangle'  # Medium confidence: wider strikes
        else:
            strategy = 'skip'  # Low confidence: don't trade

        return {
            'probability': prob,
            'suggested_strategy': strategy,
            'confidence': abs(prob - 0.5) * 2  # 0-1 scale
        }
```

#### 2.3.4 Model Router

```python
class ModelRouter:
    """Routes predictions to appropriate model based on context"""

    def __init__(self, core_model: CoreModel, earnings_model: EarningsModel):
        self.core_model = core_model
        self.earnings_model = earnings_model
        self.ticker_manager = TickerManager('configs/tickers.yaml')

    def get_prediction(
        self,
        symbol: str,
        features: pd.DataFrame,
        date: date
    ) -> ModelPrediction:
        """
        Route to appropriate model based on earnings proximity

        Returns:
            ModelPrediction with model_type, prediction, confidence
        """
        context = self.ticker_manager.get_active_tickers(date).get(symbol)

        if context is None:
            raise ValueError(f"Symbol {symbol} not in universe")

        if context.is_blackout:
            # In blackout period - don't trade
            return ModelPrediction(
                model_type='blackout',
                prediction=0.0,
                confidence=0.0,
                should_trade=False,
                reason='Earnings blackout period'
            )

        if context.model_type == 'earnings':
            # Use earnings model
            result = self.earnings_model.predict(features)
            return ModelPrediction(
                model_type='earnings',
                prediction=result['probability'],
                confidence=result['confidence'],
                should_trade=result['suggested_strategy'] != 'skip',
                strategy=result['suggested_strategy'],
                days_to_earnings=context.days_to_earnings
            )

        else:
            # Use core model
            prediction = self.core_model.predict(features)
            confidence = abs(prediction - 0.5) * 2

            return ModelPrediction(
                model_type='core',
                prediction=prediction,
                confidence=confidence,
                should_trade=confidence > 0.3,  # Minimum confidence threshold
                strategy='iv_rank'
            )
```

#### 2.3.5 Training Data Preparation

```python
def prepare_training_data(symbols: List[str], start_date: date, end_date: date):
    """
    Prepare separate datasets for Core and Earnings models

    Returns:
        core_data: DataFrame for core model (excludes earnings windows)
        earnings_data: DataFrame for earnings model (only earnings windows)
    """
    earnings_calendar = EarningsCalendar()
    core_data = []
    earnings_data = []

    for symbol in symbols:
        # Load all data for symbol
        data = load_features(symbol, start_date, end_date)

        # Get earnings dates
        earnings_dates = earnings_calendar.get_earnings_dates(symbol, start_date, end_date)

        for idx, row in data.iterrows():
            current_date = row['date']

            # Find days to nearest earnings
            days_to_earnings = min(
                [(ed - current_date).days for ed in earnings_dates if ed >= current_date],
                default=999
            )

            if days_to_earnings <= 5:
                # Earnings window → Earnings dataset
                row['days_to_earnings'] = days_to_earnings
                earnings_data.append(row)
            elif days_to_earnings > 5:
                # Normal period → Core dataset
                core_data.append(row)
            # Skip blackout period (days 6-10 before earnings) for cleaner data

    return pd.DataFrame(core_data), pd.DataFrame(earnings_data)
```

#### 2.3.6 Multi-Model Ensemble (Within Core Model)

**XGBoost Model:**
- **Task:** Predict option profitability (binary) or return (regression)
- **Input:** 185 engineered features
- **Output:** Probability of profit OR expected return
- **Hyperparameters:**
  - n_estimators: 100-500 (tuned via Optuna)
  - max_depth: 3-7
  - learning_rate: 0.01-0.1
  - colsample_bytree: 0.6-1.0

**LSTM Model:**
- **Task:** Capture temporal patterns in price/volatility
- **Architecture:**
  - Input: Sequence of 60 days × 50 features
  - LSTM layer 1: 128 units, return_sequences=True
  - Dropout: 0.2
  - LSTM layer 2: 64 units
  - Dropout: 0.2
  - Dense: 32 units, ReLU
  - Output: 1 unit, Sigmoid (for probability)
- **Training:** Adam optimizer, binary cross-entropy loss

**Linear Model (Baseline):**
- **Task:** Logistic regression for quick baseline
- **Input:** Top 30 features by importance
- **Purpose:** Validate that complex models add value

**Ensemble Strategy:**
```python
class EnsembleModel:
    def __init__(self, models: Dict[str, BaseModel], weights: Dict[str, float]):
        self.models = models
        self.weights = weights  # Per-regime weights

    def predict(self, features: pd.DataFrame, regime: str) -> float:
        """Ensemble prediction weighted by regime"""

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(features)

        # Regime-specific weights
        regime_weights = self.weights.get(regime, {
            'xgboost': 0.5,
            'lstm': 0.3,
            'linear': 0.2
        })

        # Weighted average
        ensemble_pred = sum(
            predictions[name] * regime_weights.get(name, 0)
            for name in predictions
        )

        return ensemble_pred
```

---

#### 2.3.2 Training Pipeline

**Workflow:**

```
1. Data Preparation
   ├── Load historical market data (5 years)
   ├── Generate features
   └── Create labels (option profitability at expiration)

2. Train/Test Split
   ├── Walk-forward splits (expanding window)
   ├── 70% train, 10% validation, 20% test
   └── Ensure no data leakage

3. Model Training
   ├── Train XGBoost with Optuna hyperparameter tuning
   ├── Train LSTM with early stopping
   ├── Train Linear baseline
   └── Log all experiments to MLflow

4. Validation
   ├── Calculate metrics on validation set
   ├── Feature importance analysis
   ├── Regime-specific performance
   └── Compare to baseline

5. Model Selection
   ├── Choose best model by Sharpe ratio
   ├── Ensemble if multiple models add value
   └── Version and save artifacts

6. Deployment
   ├── Register model in MLflow
   ├── Deploy to production environment
   └── Monitor performance
```

**Walk-Forward Validation:**
```python
def walk_forward_validation(
    data: pd.DataFrame,
    model_class: Type[BaseModel],
    train_window: int = 252 * 3,  # 3 years
    test_window: int = 63  # ~3 months
):
    """
    Walk-forward validation to avoid look-ahead bias

    Example:
    Train: 2020-01-01 to 2022-12-31, Test: 2023-01-01 to 2023-03-31
    Train: 2020-04-01 to 2023-03-31, Test: 2023-04-01 to 2023-06-30
    ...
    """
    results = []

    for i in range(0, len(data) - train_window - test_window, test_window):
        train_data = data.iloc[i:i+train_window]
        test_data = data.iloc[i+train_window:i+train_window+test_window]

        # Train model
        model = model_class()
        model.fit(train_data['features'], train_data['labels'])

        # Test model
        predictions = model.predict(test_data['features'])
        metrics = calculate_metrics(test_data['labels'], predictions)

        results.append({
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'metrics': metrics
        })

    return pd.DataFrame(results)
```

---

### 2.4 Strategy Layer

#### 2.4.1 Base Strategy Interface

```python
class BaseStrategy(ABC):
    """Abstract base class for all strategies"""

    @abstractmethod
    def generate_signal(
        self,
        market_data: Dict,
        options_chain: pd.DataFrame,
        features: pd.DataFrame,
        model_predictions: Optional[Dict] = None
    ) -> Optional[Signal]:
        """
        Generate trading signal

        Returns:
            Signal object with action, strikes, confidence, etc.
            None if no signal
        """
        pass

    @abstractmethod
    def should_close(
        self,
        position: Position,
        market_data: Dict
    ) -> bool:
        """Determine if position should be closed"""
        pass

    def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before execution"""
        # Check strikes are valid
        # Check expiration is valid
        # Check credit/debit is reasonable
        return True
```

**Signal Data Class:**
```python
@dataclass
class Signal:
    timestamp: datetime
    strategy: str
    action: str  # 'buy_call_debit_spread', 'sell_put_credit_spread', etc.
    symbol: str
    buy_strike: Optional[float]
    sell_strike: Optional[float]
    expiration: date
    dte: int
    credit_or_debit: float
    max_profit: float
    max_risk: float
    confidence: float  # 0-1
    regime: str
    iv_rank: float
    features: Dict[str, float]  # Store key features for analysis
    model_predictions: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None
```

---

#### 2.4.2 IV Rank Strategy (Implementation)

```python
class IVRankStrategy(BaseStrategy):
    """
    Mean reversion strategy based on IV rank

    Logic:
    - High IV rank (>70%): Sell credit spreads
    - Low IV rank (<30%): Buy debit spreads (only in specific regimes)
    - Adapts position size to regime
    """

    def __init__(self, config: StrategyConfig):
        self.sell_threshold = config.sell_threshold  # 0.70
        self.buy_threshold = config.buy_threshold  # 0.30
        self.target_dte_min = config.target_dte_min  # 3
        self.target_dte_max = config.target_dte_max  # 7
        self.target_delta = config.target_delta  # 0.25 for sold options
        self.max_spread_width = config.max_spread_width  # $5 or $10

    def generate_signal(
        self,
        market_data: Dict,
        options_chain: pd.DataFrame,
        features: pd.DataFrame,
        model_predictions: Optional[Dict] = None
    ) -> Optional[Signal]:
        """Generate signal based on IV rank"""

        iv_rank = market_data['iv_rank']
        regime = market_data['regime']
        spot = market_data['spot_price']

        # Don't trade in neutral regime
        if regime == 'neutral':
            return None

        # Filter options by DTE
        valid_options = options_chain[
            (options_chain['dte'] >= self.target_dte_min) &
            (options_chain['dte'] <= self.target_dte_max)
        ]

        if len(valid_options) == 0:
            return None

        # High IV rank: Sell premium
        if iv_rank > self.sell_threshold:
            return self._create_sell_signal(
                valid_options, spot, iv_rank, regime, model_predictions
            )

        # Low IV rank: Buy (only if expansion expected)
        elif iv_rank < self.buy_threshold and regime in ['iv_expansion', 'low_vol']:
            return self._create_buy_signal(
                valid_options, spot, iv_rank, regime, model_predictions
            )

        return None

    def _create_sell_signal(
        self,
        options: pd.DataFrame,
        spot: float,
        iv_rank: float,
        regime: str,
        model_predictions: Optional[Dict]
    ) -> Optional[Signal]:
        """Create put credit spread signal"""

        puts = options[options['option_type'] == 'put'].copy()

        # Find strike with target delta
        puts['delta_diff'] = abs(puts['delta'] + self.target_delta)
        sell_put = puts.nsmallest(1, 'delta_diff')

        if len(sell_put) == 0:
            return None

        sell_put = sell_put.iloc[0]

        # Find protection strike
        buy_strike = sell_put['strike'] - self.max_spread_width
        buy_put = puts[puts['strike'] == buy_strike]

        if len(buy_put) == 0:
            return None

        buy_put = buy_put.iloc[0]

        # Calculate credit (conservative: bid - ask)
        credit = sell_put['bid'] - buy_put['ask']

        if credit <= 0:
            return None

        max_risk = self.max_spread_width - credit

        # Confidence score
        confidence = self._calculate_confidence(
            iv_rank, regime, model_predictions, 'sell'
        )

        return Signal(
            timestamp=datetime.now(),
            strategy='iv_rank_sell',
            action='sell_put_credit_spread',
            symbol=sell_put['symbol'],
            sell_strike=sell_put['strike'],
            buy_strike=buy_put['strike'],
            expiration=sell_put['expiration'],
            dte=sell_put['dte'],
            credit_or_debit=credit,
            max_profit=credit,
            max_risk=max_risk,
            confidence=confidence,
            regime=regime,
            iv_rank=iv_rank,
            features={
                'sell_delta': sell_put['delta'],
                'sell_iv': sell_put['calculated_iv'],
                'spread_pct': sell_put['spread_pct']
            },
            model_predictions=model_predictions
        )

    def _calculate_confidence(
        self,
        iv_rank: float,
        regime: str,
        model_predictions: Optional[Dict],
        direction: str
    ) -> float:
        """
        Calculate signal confidence

        Combines:
        - IV rank extremity (farther = higher confidence)
        - Regime appropriateness
        - ML model predictions (if available)
        """

        # Base confidence from IV rank
        if direction == 'sell':
            base_confidence = min(iv_rank, 0.95)
        else:  # buy
            base_confidence = min(1 - iv_rank, 0.95)

        # Regime boost
        regime_multiplier = {
            'low_vol': 1.0 if direction == 'sell' else 0.8,
            'high_vol': 0.7 if direction == 'sell' else 0.9,
            'iv_contraction': 1.2 if direction == 'sell' else 0.6,
            'iv_expansion': 0.6 if direction == 'sell' else 1.2,
            'neutral': 0.5
        }.get(regime, 1.0)

        confidence = base_confidence * regime_multiplier

        # ML model boost (if available and agrees)
        if model_predictions:
            ml_confidence = model_predictions.get('probability', 0.5)
            # Average with ML prediction
            confidence = 0.6 * confidence + 0.4 * ml_confidence

        return min(confidence, 0.98)  # Cap at 98%

    def should_close(self, position: Position, market_data: Dict) -> bool:
        """Determine if position should be closed"""

        # Profit target: 50% of max profit
        if position.unrealized_pnl >= position.max_profit * 0.5:
            return True

        # Stop loss: 2x credit received
        if position.unrealized_pnl <= -position.credit_received * 2:
            return True

        # Close at 80% of time elapsed (avoid gamma risk)
        time_elapsed_pct = 1 - (position.dte / position.initial_dte)
        if time_elapsed_pct > 0.8:
            return True

        # Emergency close in regime shift
        if position.entry_regime == 'low_vol' and market_data['regime'] == 'high_vol':
            return True

        return False
```

---

#### 2.4.3 Earnings Strategy (Specialized)

```python
class EarningsStrategy(BaseStrategy):
    """
    Strategy for playing IV crush around earnings announcements

    Logic:
    - 1-5 days before earnings: Sell straddles/strangles to capture IV crush
    - Uses Earnings Model predictions for entry decisions
    - Exits immediately after earnings announcement
    """

    def __init__(self, config: EarningsStrategyConfig):
        self.min_iv_percentile = config.min_iv_percentile  # 0.6
        self.max_expected_move = config.max_expected_move  # 0.10 (10%)
        self.strangle_width = config.strangle_width  # 0.05 (5% OTM each side)
        self.max_dte = config.max_dte  # 7 days
        self.earnings_model = None  # Injected

    def generate_signal(
        self,
        market_data: Dict,
        options_chain: pd.DataFrame,
        features: pd.DataFrame,
        model_predictions: Optional[Dict] = None
    ) -> Optional[Signal]:
        """Generate earnings play signal"""

        days_to_earnings = market_data.get('days_to_earnings')

        # Only trade 1-5 days before earnings
        if days_to_earnings is None or days_to_earnings > 5 or days_to_earnings < 1:
            return None

        # Check model prediction
        if model_predictions is None:
            return None

        if model_predictions.get('suggested_strategy') == 'skip':
            return None

        # Get spot price and expected move
        spot = market_data['spot_price']
        expected_move = market_data.get('expected_move_pct', 0.05)

        # Find options expiring just after earnings
        earnings_date = market_data['earnings_date']
        valid_options = options_chain[
            (options_chain['expiration'] > earnings_date) &
            (options_chain['dte'] <= self.max_dte)
        ]

        if len(valid_options) == 0:
            return None

        strategy_type = model_predictions.get('suggested_strategy', 'sell_strangle')

        if strategy_type == 'sell_straddle':
            return self._create_straddle_signal(
                valid_options, spot, market_data, model_predictions
            )
        else:  # sell_strangle
            return self._create_strangle_signal(
                valid_options, spot, market_data, model_predictions
            )

    def _create_strangle_signal(
        self,
        options: pd.DataFrame,
        spot: float,
        market_data: Dict,
        model_predictions: Dict
    ) -> Optional[Signal]:
        """Create short strangle signal (sell OTM call + OTM put)"""

        # Find OTM strikes
        call_strike = spot * (1 + self.strangle_width)
        put_strike = spot * (1 - self.strangle_width)

        # Round to nearest strike
        available_strikes = options['strike'].unique()
        call_strike = min(available_strikes, key=lambda x: abs(x - call_strike))
        put_strike = min(available_strikes, key=lambda x: abs(x - put_strike))

        # Get options at these strikes
        sell_call = options[
            (options['strike'] == call_strike) &
            (options['option_type'] == 'call')
        ]
        sell_put = options[
            (options['strike'] == put_strike) &
            (options['option_type'] == 'put')
        ]

        if len(sell_call) == 0 or len(sell_put) == 0:
            return None

        sell_call = sell_call.iloc[0]
        sell_put = sell_put.iloc[0]

        # Calculate credit (conservative: use bids)
        credit = sell_call['bid'] + sell_put['bid']

        if credit <= 0:
            return None

        # Max risk is theoretically unlimited for naked strangle
        # Use expected move to estimate practical max risk
        expected_move_dollars = spot * market_data.get('expected_move_pct', 0.05)
        max_risk = expected_move_dollars * 2  # Rough estimate

        return Signal(
            timestamp=datetime.now(),
            strategy='earnings_strangle',
            action='sell_strangle',
            symbol=market_data['symbol'],
            sell_strike=call_strike,  # Call strike
            buy_strike=put_strike,    # Put strike (overloaded field)
            expiration=sell_call['expiration'],
            dte=sell_call['dte'],
            credit_or_debit=credit,
            max_profit=credit,
            max_risk=max_risk,
            confidence=model_predictions.get('confidence', 0.5),
            regime=market_data.get('regime', 'neutral'),
            iv_rank=market_data.get('iv_rank', 0.5),
            features={
                'days_to_earnings': market_data['days_to_earnings'],
                'expected_move_pct': market_data.get('expected_move_pct'),
                'iv_percentile_earnings': market_data.get('iv_percentile_earnings'),
                'call_iv': sell_call.get('impliedVolatility'),
                'put_iv': sell_put.get('impliedVolatility')
            },
            model_predictions=model_predictions,
            metadata={'strategy_subtype': 'strangle'}
        )

    def _create_straddle_signal(
        self,
        options: pd.DataFrame,
        spot: float,
        market_data: Dict,
        model_predictions: Dict
    ) -> Optional[Signal]:
        """Create short straddle signal (sell ATM call + ATM put)"""

        # Find ATM strike
        available_strikes = options['strike'].unique()
        atm_strike = min(available_strikes, key=lambda x: abs(x - spot))

        # Get ATM options
        sell_call = options[
            (options['strike'] == atm_strike) &
            (options['option_type'] == 'call')
        ]
        sell_put = options[
            (options['strike'] == atm_strike) &
            (options['option_type'] == 'put')
        ]

        if len(sell_call) == 0 or len(sell_put) == 0:
            return None

        sell_call = sell_call.iloc[0]
        sell_put = sell_put.iloc[0]

        credit = sell_call['bid'] + sell_put['bid']

        if credit <= 0:
            return None

        return Signal(
            timestamp=datetime.now(),
            strategy='earnings_straddle',
            action='sell_straddle',
            symbol=market_data['symbol'],
            sell_strike=atm_strike,
            buy_strike=atm_strike,  # Same strike for straddle
            expiration=sell_call['expiration'],
            dte=sell_call['dte'],
            credit_or_debit=credit,
            max_profit=credit,
            max_risk=credit * 3,  # Rough risk estimate
            confidence=model_predictions.get('confidence', 0.5),
            regime=market_data.get('regime', 'neutral'),
            iv_rank=market_data.get('iv_rank', 0.5),
            features={
                'days_to_earnings': market_data['days_to_earnings'],
                'expected_move_pct': market_data.get('expected_move_pct'),
                'straddle_price': credit
            },
            model_predictions=model_predictions,
            metadata={'strategy_subtype': 'straddle'}
        )

    def should_close(self, position: Position, market_data: Dict) -> bool:
        """
        Earnings positions have specific exit rules

        Exit conditions:
        1. Immediately after earnings (IV crush captured)
        2. Profit target hit (50% of credit)
        3. Stock moves beyond expected move (cut losses)
        """

        # Exit after earnings announcement
        if market_data.get('earnings_announced', False):
            return True

        # Profit target: 50% of max profit
        if position.unrealized_pnl >= position.max_profit * 0.5:
            return True

        # Stop loss: Position doubled in value (losing 100% of credit)
        if position.unrealized_pnl <= -position.credit_received:
            return True

        # Time-based: If still open 2 days after earnings, close
        if position.metadata.get('earnings_date'):
            days_since_earnings = (date.today() - position.metadata['earnings_date']).days
            if days_since_earnings >= 2:
                return True

        return False
```

#### 2.4.4 Strategy Router

```python
class StrategyRouter:
    """
    Routes to appropriate strategy based on ticker context

    Integrates with ModelRouter to select correct model + strategy combination
    """

    def __init__(
        self,
        core_strategy: IVRankStrategy,
        earnings_strategy: EarningsStrategy,
        model_router: ModelRouter
    ):
        self.core_strategy = core_strategy
        self.earnings_strategy = earnings_strategy
        self.model_router = model_router

    def get_signals(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict],
        options_chains: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        date: date
    ) -> List[Signal]:
        """
        Generate signals for all symbols using appropriate strategy

        Returns:
            List of signals across all symbols
        """
        signals = []

        for symbol in symbols:
            # Get model prediction and routing
            prediction = self.model_router.get_prediction(
                symbol,
                features.get(symbol),
                date
            )

            if not prediction.should_trade:
                continue

            symbol_data = market_data.get(symbol, {})
            symbol_data['model_predictions'] = {
                'probability': prediction.prediction,
                'confidence': prediction.confidence,
                'suggested_strategy': prediction.strategy
            }

            # Route to appropriate strategy
            if prediction.model_type == 'earnings':
                signal = self.earnings_strategy.generate_signal(
                    symbol_data,
                    options_chains.get(symbol),
                    features.get(symbol),
                    symbol_data['model_predictions']
                )
            else:  # core
                signal = self.core_strategy.generate_signal(
                    symbol_data,
                    options_chains.get(symbol),
                    features.get(symbol),
                    symbol_data['model_predictions']
                )

            if signal is not None:
                signals.append(signal)

        return signals
```

---

### 2.5 Trading Layer

#### 2.5.1 Risk Manager

**Purpose:** Validate trades and enforce risk limits

**Key Responsibilities:**
1. Position sizing
2. Portfolio risk limits
3. Drawdown monitoring
4. Greeks limits

**Implementation:**
```python
class RiskManager:
    def __init__(self, config: RiskConfig):
        self.max_risk_per_trade = config.max_risk_per_trade  # 2%
        self.max_portfolio_risk = config.max_portfolio_risk  # 20%
        self.max_positions = config.max_positions  # 5
        self.max_drawdown = config.max_drawdown  # 25%
        self.max_portfolio_delta = config.max_portfolio_delta  # 100
        self.max_portfolio_gamma = config.max_portfolio_gamma  # 50

        self.peak_equity = 0
        self.current_drawdown = 0

    def validate_trade(
        self,
        signal: Signal,
        account: Account,
        current_positions: List[Position]
    ) -> TradeValidation:
        """
        Validate if trade should be executed

        Returns:
            TradeValidation(is_valid, reason, quantity)
        """

        checks = [
            self._check_drawdown(account),
            self._check_position_count(current_positions),
            self._check_portfolio_risk(signal, account, current_positions),
            self._check_greeks_limits(signal, current_positions),
            self._check_correlation(signal, current_positions)
        ]

        for check in checks:
            if not check.passed:
                return TradeValidation(False, check.reason, 0)

        # Calculate position size
        quantity = self._calculate_position_size(signal, account)

        return TradeValidation(True, "All checks passed", quantity)

    def _calculate_position_size(
        self,
        signal: Signal,
        account: Account
    ) -> int:
        """
        Calculate position size using percentage of capital at risk

        Example:
        - Account equity: $10,000
        - Max risk per trade: 2% = $200
        - Signal max risk: $300 per spread
        - Position size: 200 / 300 = 0.66 → 1 contract (round up to minimum)
        """

        max_dollar_risk = account.equity * self.max_risk_per_trade
        max_risk_per_contract = signal.max_risk * 100  # Each contract = 100 shares

        quantity = int(max_dollar_risk / max_risk_per_contract)
        quantity = max(1, quantity)  # At least 1 contract

        # Regime-based adjustment
        regime_multiplier = {
            'low_vol': 1.0,
            'high_vol': 0.5,
            'iv_expansion': 0.75,
            'iv_contraction': 1.0,
            'neutral': 0.0
        }.get(signal.regime, 1.0)

        quantity = int(quantity * regime_multiplier)

        # Confidence-based adjustment
        if signal.confidence < 0.6:
            quantity = 0  # Don't trade low confidence signals
        elif signal.confidence < 0.75:
            quantity = max(1, int(quantity * 0.5))

        return quantity
```

---

#### 2.5.2 Order Manager

**Purpose:** Execute trades via broker API and track order status

**Order Lifecycle:**
```
Signal → Validate → Create Order → Submit → Monitor → Fill → Update Position
```

**Implementation:**
```python
class OrderManager:
    def __init__(self, broker: BrokerAPI, risk_manager: RiskManager):
        self.broker = broker
        self.risk_manager = risk_manager
        self.pending_orders: Dict[str, Order] = {}

    def execute_signal(
        self,
        signal: Signal,
        account: Account,
        positions: List[Position]
    ) -> ExecutionResult:
        """Execute a trading signal"""

        # Risk validation
        validation = self.risk_manager.validate_trade(signal, account, positions)

        if not validation.is_valid:
            logger.warning(f"Signal rejected: {validation.reason}")
            return ExecutionResult(success=False, reason=validation.reason)

        if validation.quantity == 0:
            logger.info("Signal validated but zero quantity")
            return ExecutionResult(success=False, reason="Zero quantity")

        # Create order
        order = self._create_order_from_signal(signal, validation.quantity)

        # Submit to broker
        try:
            order_id = self.broker.place_order(order)
            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED

            self.pending_orders[order_id] = order

            logger.info(f"Order submitted: {order_id}, {validation.quantity} contracts")

            return ExecutionResult(
                success=True,
                order_id=order_id,
                quantity=validation.quantity
            )

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return ExecutionResult(success=False, reason=str(e))

    def _create_order_from_signal(self, signal: Signal, quantity: int) -> Order:
        """Convert signal to broker order"""

        if signal.action == 'sell_put_credit_spread':
            return Order(
                symbol=signal.symbol,
                strategy='spread',
                legs=[
                    OrderLeg(
                        action='SELL',
                        option_type='PUT',
                        strike=signal.sell_strike,
                        expiration=signal.expiration,
                        quantity=quantity
                    ),
                    OrderLeg(
                        action='BUY',
                        option_type='PUT',
                        strike=signal.buy_strike,
                        expiration=signal.expiration,
                        quantity=quantity
                    )
                ],
                order_type='LIMIT',
                limit_price=signal.credit_or_debit * 0.95,  # Conservative fill
                time_in_force='DAY'
            )

        # Similar for other signal types...

    def monitor_orders(self):
        """Check status of pending orders"""

        for order_id, order in list(self.pending_orders.items()):
            status = self.broker.get_order_status(order_id)

            if status.is_filled:
                logger.info(f"Order filled: {order_id}")
                self._handle_fill(order, status)
                del self.pending_orders[order_id]

            elif status.is_rejected:
                logger.warning(f"Order rejected: {order_id}, reason: {status.reason}")
                del self.pending_orders[order_id]

            elif status.age_seconds > 300:  # 5 minutes
                logger.warning(f"Order unfilled after 5 min, cancelling: {order_id}")
                self.broker.cancel_order(order_id)
                del self.pending_orders[order_id]
```

---

#### 2.5.3 Position Manager

**Purpose:** Track open positions, calculate P&L, manage exits

```python
class PositionManager:
    def __init__(self, broker: BrokerAPI, db: Database):
        self.broker = broker
        self.db = db
        self.positions: Dict[int, Position] = {}

    def update_positions(self):
        """Update all position prices and P&L"""

        for position_id, position in self.positions.items():
            # Get current market price
            current_price = self.broker.get_spread_price(
                position.symbol,
                position.legs
            )

            # Update position
            position.current_price = current_price
            position.unrealized_pnl = self._calculate_pnl(position)
            position.updated_at = datetime.now()

            # Update Greeks
            position.delta = self._calculate_position_delta(position)
            position.gamma = self._calculate_position_gamma(position)

            # Save to database
            self.db.update_position(position)

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate portfolio-level metrics"""

        total_delta = sum(p.delta for p in self.positions.values())
        total_gamma = sum(p.gamma for p in self.positions.values())
        total_vega = sum(p.vega for p in self.positions.values())
        total_theta = sum(p.theta for p in self.positions.values())

        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_exposure = sum(p.notional_value for p in self.positions.values())

        return PortfolioMetrics(
            num_positions=len(self.positions),
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_vega=total_vega,
            total_theta=total_theta,
            total_pnl=total_pnl,
            total_exposure=total_exposure
        )

    def check_exits(self, strategy: BaseStrategy) -> List[int]:
        """Check if any positions should be closed"""

        positions_to_close = []

        for position_id, position in self.positions.items():
            should_close = strategy.should_close(
                position,
                self._get_current_market_data()
            )

            if should_close:
                positions_to_close.append(position_id)
                logger.info(f"Position {position_id} marked for closure: {position.symbol}")

        return positions_to_close

    def close_position(self, position_id: int) -> bool:
        """Close a position"""

        position = self.positions.get(position_id)
        if not position:
            return False

        try:
            # Create closing order (reverse of opening)
            close_order = self._create_close_order(position)
            order_id = self.broker.place_order(close_order)

            logger.info(f"Closing position {position_id}, order: {order_id}")

            # Update position status
            position.status = PositionStatus.CLOSING
            position.close_order_id = order_id

            return True

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            return False
```

---

## 3. Data Flow Architecture

### 3.1 Real-Time Trading Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Trading Cycle                               │
└─────────────────────────────────────────────────────────────────┘

Every 1 hour during market hours:

1. Data Collection (2-5 sec)
   ├── Fetch current SPY price
   ├── Fetch options chains (3-7 DTE)
   ├── Fetch VIX
   └── Store in database

2. Feature Engineering (1-2 sec)
   ├── Calculate Greeks for all options
   ├── Compute IV rank, HV/IV ratio
   ├── Generate 185 features
   └── Detect current regime

3. Signal Generation (1 sec)
   ├── Run IV Rank strategy logic
   ├── Run ML models (if enabled)
   ├── Combine signals via ensemble
   └── Assign confidence scores

4. Risk Validation (< 1 sec)
   ├── Check position limits
   ├── Check drawdown
   ├── Calculate position size
   └── Validate Greeks exposure

5. Order Execution (< 1 sec)
   ├── Create order from signal
   ├── Submit to broker API
   ├── Log order details
   └── Add to pending orders

6. Position Monitoring (continuous)
   ├── Update position prices
   ├── Calculate P&L
   ├── Check exit conditions
   └── Close if criteria met

7. Logging & Metrics (< 1 sec)
   ├── Log all signals (executed or not)
   ├── Update metrics (Prometheus)
   ├── Send alerts if needed
   └── Update dashboard
```

**Total latency target: < 15 seconds from data → order execution**

---

### 3.2 Backtesting Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backtesting Pipeline                          │
└─────────────────────────────────────────────────────────────────┘

Input: Historical data (2020-2024)

1. Data Preparation
   ├── Load market data from Parquet
   ├── Load options data from database
   └── Ensure data quality

2. Feature Generation (vectorized)
   ├── Calculate all features for entire dataset
   ├── ~60 seconds for 5 years of daily data
   └── Store in Parquet for reuse

3. Walk-Forward Loop
   For each period (e.g., every 3 months):
     ├── Train ML models on training window
     ├── Generate signals on test window
     ├── Simulate execution (with slippage)
     ├── Track positions and P&L
     └── Record metrics

4. Results Aggregation
   ├── Combine all periods
   ├── Calculate overall metrics (Sharpe, DD, etc.)
   ├── Analyze by regime
   └── Generate report

5. Visualization
   ├── Equity curve
   ├── Drawdown chart
   ├── Trade distribution
   └── Feature importance

Total runtime: 2-5 minutes for 5 years of data
```

---

## 4. Technology Stack

### 4.1 Core Technologies

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11 | ML ecosystem, rapid development |
| Data Processing | Pandas, NumPy | Industry standard, performant |
| ML Framework | XGBoost, TensorFlow | State-of-art, production-ready |
| Optimization | Optuna | Hyperparameter tuning |
| Backtesting | Vectorbt | Fast vectorized backtesting |
| Database | PostgreSQL | ACID compliance, complex queries |
| Caching | Redis (optional) | Reduce API calls |
| Broker API | IBKR (ib_insync) | Real options support |
| Experiment Tracking | MLflow | Model versioning |
| Monitoring | Prometheus + Grafana | Industry standard observability |
| Logging | Loguru | Clean, powerful logging |
| Configuration | Pydantic + YAML | Type-safe configs |
| Testing | Pytest | Comprehensive test framework |
| CI/CD | GitHub Actions (future) | Automation |

### 4.2 Development Tools

| Tool | Purpose |
|------|---------|
| VS Code / PyCharm | IDE |
| Jupyter Lab | EDA & prototyping |
| Git | Version control |
| Docker | Containerization (optional) |
| Black | Code formatting |
| Mypy | Static type checking |
| Ruff | Linting |

---

## 5. Deployment Architecture

### 5.1 Development Environment

```
Laptop / Workstation
├── PostgreSQL (local)
├── Python app
├── IBKR TWS (paper trading)
├── Jupyter Lab (analysis)
└── Grafana (monitoring)
```

**Specifications:**
- RAM: 16GB+ (32GB recommended for LSTM training)
- Storage: 100GB SSD
- CPU: 4+ cores
- GPU: Optional (CUDA for LSTM)

---

### 5.2 Production Environment (Future)

```
Cloud VM (AWS EC2 / GCP Compute)
├── App Server (t3.large or equivalent)
│   ├── Trading application
│   ├── Scheduled jobs (cron)
│   └── IBKR Gateway (headless)
├── Database (RDS or managed Postgres)
├── Monitoring (Grafana Cloud or self-hosted)
└── Storage (S3 for backups)
```

**Auto-scaling:** Not required initially (single instance)

---

## 6. Security Architecture

### 6.1 Secrets Management

```python
# Environment variables (never commit to git)
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password  # Or use key-based auth
DB_PASSWORD=secure_password
POSTGRES_PASSWORD=another_password
MLFLOW_TRACKING_URI=http://localhost:5000

# Load with python-dotenv
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('IBKR_USERNAME')
```

### 6.2 Network Security

- **Firewall:** Allow only necessary ports (5432 for Postgres, 7497 for IBKR)
- **VPN:** Use VPN when accessing production from remote
- **SSH:** Key-based authentication only
- **HTTPS:** All external APIs use HTTPS

### 6.3 Data Security

- **Backups:** Daily automated backups of PostgreSQL
- **Encryption:** At-rest encryption for sensitive data
- **Audit Logs:** All trades, orders, and manual interventions logged
- **Access Control:** Principle of least privilege

---

## 7. Error Handling & Resilience

### 7.1 Error Categories

| Error Type | Example | Handling |
|------------|---------|----------|
| Transient | API rate limit | Retry with exponential backoff |
| Data Quality | Missing VIX data | Use cached value, alert |
| Broker Rejection | Order rejected | Log, alert, don't retry |
| System Failure | Database down | Graceful shutdown, alert |
| Model Error | NaN prediction | Fall back to rule-based |

### 7.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failures = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = 'OPEN'
```

**Usage:**
```python
broker_circuit = CircuitBreaker(failure_threshold=5, timeout=300)

def place_order_with_circuit_breaker(order):
    return broker_circuit.call(broker.place_order, order)
```

---

## 8. Scalability Considerations

### 8.1 Current Scale (Phase 1-3)

| Metric | Current Target | Notes |
|--------|----------------|-------|
| Underlyings | 1 (SPY) | Single liquid underlying |
| Positions | 5 concurrent | Manageable manually |
| Signals/day | 1-3 | Not high frequency |
| Data volume | ~100 MB/day | Options chains |
| ML training | Weekly | Retrain models |

**Bottlenecks:** None expected at this scale

---

### 8.2 Future Scale (Phase 4+)

| Metric | Future Target | Scaling Strategy |
|--------|---------------|------------------|
| Underlyings | 10+ | Parallel data fetching |
| Positions | 20+ | Better position tracking UI |
| Signals/day | 10+ | Prioritization queue |
| Data volume | ~1 GB/day | Data retention policies |
| ML training | Daily | Incremental learning |

**Potential Bottlenecks:**
- **Data fetching:** Use async I/O (asyncio, aiohttp)
- **Feature generation:** Parallelize with multiprocessing
- **ML inference:** Batch predictions
- **Database queries:** Optimize indexes, consider partitioning

---

## 9. Monitoring & Observability

### 9.1 Metrics to Track

**Trading Metrics:**
```python
# Prometheus metrics
trading_signals_generated = Counter('signals_generated_total', 'Total signals generated', ['strategy', 'regime'])
trading_signals_executed = Counter('signals_executed_total', 'Total signals executed')
trading_orders_rejected = Counter('orders_rejected_total', 'Total orders rejected', ['reason'])
trading_positions_open = Gauge('positions_open', 'Number of open positions')
trading_portfolio_value = Gauge('portfolio_value_usd', 'Portfolio value in USD')
trading_drawdown = Gauge('drawdown_pct', 'Current drawdown percentage')
trading_pnl_daily = Gauge('pnl_daily_usd', 'Daily P&L in USD')
```

**System Metrics:**
```python
# Prometheus metrics
data_fetch_duration_seconds = Histogram('data_fetch_duration_seconds', 'Data fetch latency')
signal_generation_duration_seconds = Histogram('signal_generation_duration_seconds', 'Signal generation latency')
model_prediction_duration_seconds = Histogram('model_prediction_duration_seconds', 'ML model prediction latency')
api_errors_total = Counter('api_errors_total', 'Total API errors', ['source'])
```

### 9.2 Alerts

```yaml
alerts:
  - name: high_drawdown
    condition: drawdown > 0.20
    severity: warning
    channels: [email, slack]

  - name: critical_drawdown
    condition: drawdown > 0.25
    severity: critical
    channels: [email, slack, sms]

  - name: order_rejection_spike
    condition: order_rejection_rate > 0.5
    severity: warning
    channels: [email]

  - name: data_fetch_failure
    condition: data_fetch_failures > 3
    severity: warning
    channels: [slack]

  - name: system_down
    condition: no_heartbeat_for_10_minutes
    severity: critical
    channels: [email, sms]
```

---

## 10. Testing Strategy

### 10.1 Test Pyramid

```
         ┌─────────────┐
         │   E2E Tests │  (5%)
         │  (Manual)   │
         └─────────────┘
       ┌──────────────────┐
       │Integration Tests │ (15%)
       │  (Pytest)        │
       └──────────────────┘
    ┌─────────────────────────┐
    │    Unit Tests           │ (80%)
    │    (Pytest + Mocks)     │
    └─────────────────────────┘
```

### 10.2 Test Coverage Targets

| Module | Target Coverage | Rationale |
|--------|----------------|-----------|
| Greeks Calculator | 95% | Critical for pricing |
| Feature Engineering | 85% | Many edge cases |
| Risk Manager | 90% | Critical for safety |
| Strategies | 80% | Logic-heavy |
| Order Manager | 85% | State machine |
| Data Fetchers | 70% | External dependencies |
| Overall | 80% | Industry standard |

### 10.3 Test Types

**Unit Tests:**
```python
# test_greeks.py
def test_black_scholes_call_atm():
    """Test BS call pricing at-the-money"""
    calc = GreeksCalculator()
    result = calc.calculate_greeks(
        spot=100, strike=100, time_to_expiry=0.25, iv=0.20, option_type='call'
    )
    # ATM call with 20% IV, 3 months should be ~$4
    assert 3.5 < result['price'] < 4.5
    assert 0.48 < result['delta'] < 0.52  # ATM call delta ~0.5
```

**Integration Tests:**
```python
# test_strategy_integration.py
def test_iv_rank_strategy_generates_signal_high_iv():
    """Test that strategy generates sell signal when IV rank is high"""
    strategy = IVRankStrategy(config)

    # Mock data with high IV rank
    market_data = {'iv_rank': 0.85, 'regime': 'iv_contraction', 'spot_price': 450}
    options_chain = create_mock_options_chain()

    signal = strategy.generate_signal(market_data, options_chain, None, None)

    assert signal is not None
    assert signal.action == 'sell_put_credit_spread'
    assert signal.confidence > 0.75
```

**Backtesting Tests:**
```python
# test_backtest.py
def test_backtest_known_profitable_period():
    """Test that backtest identifies known profitable strategy period"""
    data = load_historical_data('2020-01-01', '2020-06-30')  # Known low-vol period
    strategy = IVRankStrategy(config)

    backtest = SimpleBacktester()
    results = backtest.run(data, strategy)

    # In low-vol period, selling premium should be profitable
    assert results['sharpe_ratio'] > 0.5
    assert results['win_rate'] > 0.55
```

---

## 11. Disaster Recovery

### 11.1 Backup Strategy

| Data Type | Backup Frequency | Retention | Storage |
|-----------|-----------------|-----------|---------|
| Database | Daily | 90 days | S3 / Local |
| Model Artifacts | Per training | Unlimited | MLflow / S3 |
| Config Files | On change | Version control | Git |
| Logs | Daily | 30 days | S3 / Local |

### 11.2 Recovery Procedures

**Scenario 1: Database Corruption**
1. Stop trading system
2. Restore from latest backup
3. Verify data integrity
4. Replay trades from audit log if needed
5. Restart system

**Scenario 2: Complete System Failure**
1. Close all positions manually via broker web interface
2. Investigate failure cause
3. Restore system from backups
4. Validate in paper trading
5. Resume live trading

**Scenario 3: Broker API Outage**
1. System detects API failures (circuit breaker)
2. Alert operator
3. Monitor positions via broker web/mobile app
4. Wait for API restoration
5. System auto-resumes when API available

---

## 12. Future Enhancements

### 12.1 Phase 2 Features (Months 7-12)

- **Multi-leg strategies:** Iron condors, butterflies, calendars
- **Earnings strategies:** IV crush plays
- **0DTE optimization:** High-frequency 0DTE signals
- **Portfolio hedging:** Delta-neutral strategies
- **Options on futures:** ES, NQ options

### 12.2 Phase 3 Features (Year 2)

- **Deep RL agents:** PPO, TD3 for hedging
- **Sentiment integration:** Social media, news
- **Multi-broker routing:** IB + Tastyworks
- **Mobile app:** Position monitoring on-the-go
- **Community features:** Share signals (anonymized)

---

## 13. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Engineering | Initial architecture |

---

## 14. Approvals

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Tech Lead | [Your Name] | _________ | _____ |
| Security Review | [Your Name] | _________ | _____ |
| DevOps | [Your Name] | _________ | _____ |

---

**Next Document: Implementation Plan**
