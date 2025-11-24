# Adaptive Options Trading System - Specification Document

**Version:** 1.1
**Date:** 2025-11-21
**Status:** Active Development
**Owner:** Trading System Team

> **Note:** This specification has been updated to reflect the Hybrid ML + Rule-Based approach.
> See [05_IMPLEMENTATION_PLAN.md](./05_IMPLEMENTATION_PLAN.md) for detailed implementation timeline.
> See [06_FUTURE_ROADMAP.md](./06_FUTURE_ROADMAP.md) for future features and roadmap.

---

## 1. Executive Summary

### 1.1 Vision
Build an ML-powered options trading system that adapts to market regimes, generates consistent returns through high-probability trades, and scales from paper trading to live deployment.

### 1.2 Objectives
- **Primary:** Achieve Sharpe Ratio > 1.0 with 20-30% max drawdown
- **Secondary:** 60%+ win rate with average profit > average loss
- **Timeline:** 3-6 months paper trading → small real money deployment
- **Capital:** Start paper, scale to $10k-50k real capital

### 1.3 Success Metrics

| Metric | Target | Threshold | Measurement Period |
|--------|--------|-----------|-------------------|
| Sharpe Ratio | > 1.3 | > 1.0 | Rolling 90 days |
| Annual Return | > 25% | > 15% | Trailing 12 months |
| Max Drawdown | < 20% | < 30% | All-time |
| Win Rate | > 60% | > 55% | Rolling 30 trades |
| Profit Factor | > 1.5 | > 1.3 | Rolling 90 days |
| Avg Win / Avg Loss | > 1.2 | > 1.0 | Rolling 30 trades |

---

## 2. System Requirements

### 2.1 Functional Requirements

#### FR-001: Data Acquisition
- **FR-001.1:** Fetch real-time and historical options chains for configurable underlyings (SPY, QQQ, etc.)
- **FR-001.2:** Calculate accurate Greeks (Delta, Gamma, Vega, Theta) using Black-Scholes
- **FR-001.3:** Compute implied volatility from market prices
- **FR-001.4:** Retrieve VIX and historical volatility data
- **FR-001.5:** Apply liquidity filters (volume > 10, OI > 100, spread < 20%)
- **FR-001.6:** Cache data locally to minimize API calls

#### FR-002: Feature Engineering
- **FR-002.1:** Calculate IV rank and IV percentile (252-day lookback)
- **FR-002.2:** Compute HV/IV ratio (20-day realized vol vs implied)
- **FR-002.3:** Generate technical indicators (RSI, MACD, Bollinger Bands)
- **FR-002.4:** Compute option-specific features (moneyness, time decay rate)
- **FR-002.5:** Calculate historical returns and momentum indicators
- **FR-002.6:** Generate volume and open interest signals

#### FR-003: Regime Detection
- **FR-003.1:** Classify markets into 5 regimes: low_vol, high_vol, iv_expansion, iv_contraction, neutral
- **FR-003.2:** Use VIX levels, VIX rank, and HV/IV ratio as inputs
- **FR-003.3:** Detect regime transitions and log changes
- **FR-003.4:** Maintain regime history for analysis
- **FR-003.5:** Support configurable regime thresholds

#### FR-004: Strategy Engine (Hybrid Approach)
- **FR-004.1:** Implement rule-based options selection (strike, DTE, spread width)
- **FR-004.2:** Generate signals based on ML predictions + regime rules
- **FR-004.3:** Adapt strategy parameters based on regime
- **FR-004.4:** Support credit spreads (high IV) and debit spreads (low IV)
- **FR-004.5:** Implement strategy selection matrix: regime x direction -> strategy type

#### FR-005: ML Models (Two-Model Architecture)
- **FR-005.1:** Train XGBoost direction classifier (bullish/bearish/neutral)
- **FR-005.2:** Train XGBoost volatility forecaster (IV rank prediction)
- **FR-005.3:** Implement rule-based regime classifier (VIX, HV/IV ratio)
- **FR-005.4:** Support walk-forward validation for model training
- **FR-005.5:** Track model performance metrics per regime
- **FR-005.6:** Implement model versioning with MLflow

#### FR-006: Signal Generation
- **FR-006.1:** Combine rule-based and ML signals
- **FR-006.2:** Assign confidence scores to signals (0-1)
- **FR-006.3:** Filter signals by minimum confidence threshold
- **FR-006.4:** Rank signals by expected value
- **FR-006.5:** Support manual signal override (for testing)

#### FR-007: Risk Management
- **FR-007.1:** Calculate position size: max 2% risk per trade
- **FR-007.2:** Limit total portfolio exposure: max 20%
- **FR-007.3:** Limit concurrent positions: max 5 positions
- **FR-007.4:** Enforce maximum drawdown stop: 25%
- **FR-007.5:** Implement regime-based position sizing multipliers
- **FR-007.6:** Track portfolio Greeks (Delta, Gamma, Vega) with limits
- **FR-007.7:** Support Kelly Criterion position sizing (optional)

#### FR-008: Order Execution
- **FR-008.1:** Place vertical spreads (credit and debit) via broker API
- **FR-008.2:** Use limit orders with conservative pricing (95% of theoretical)
- **FR-008.3:** Handle partial fills and order rejections
- **FR-008.4:** Support order cancellation and modification
- **FR-008.5:** Implement smart order routing (if multiple exchanges)
- **FR-008.6:** Track execution quality (slippage, fill rate)

#### FR-009: Position Management
- **FR-009.1:** Track all open positions with real-time P&L
- **FR-009.2:** Implement profit targets (50% of max profit)
- **FR-009.3:** Implement stop losses (2x credit received for credit spreads)
- **FR-009.4:** Support manual position closure
- **FR-009.5:** Auto-close positions at expiration or near-expiration
- **FR-009.6:** Track position Greeks and adjust if limits exceeded

#### FR-010: Backtesting
- **FR-010.1:** Support vectorized backtesting for speed
- **FR-010.2:** Model realistic transaction costs (bid-ask spread, slippage, commissions)
- **FR-010.3:** Implement walk-forward analysis
- **FR-010.4:** Support parameter optimization with out-of-sample validation
- **FR-010.5:** Generate comprehensive performance reports
- **FR-010.6:** Compare multiple strategies side-by-side

#### FR-011: Monitoring & Alerting
- **FR-011.1:** Real-time dashboard showing positions, P&L, metrics
- **FR-011.2:** Alert on drawdown exceeding thresholds
- **FR-011.3:** Alert on position limit breaches
- **FR-011.4:** Alert on failed orders or API errors
- **FR-011.5:** Daily performance summary emails/notifications
- **FR-011.6:** Track system health (data freshness, API latency)

#### FR-012: Logging & Audit
- **FR-012.1:** Log all signals generated with reasoning
- **FR-012.2:** Log all orders placed, filled, or rejected
- **FR-012.3:** Log all position entries and exits with P&L
- **FR-012.4:** Log regime changes and strategy adjustments
- **FR-012.5:** Maintain audit trail for compliance
- **FR-012.6:** Support log search and analysis

### 2.2 Non-Functional Requirements

#### NFR-001: Performance
- **NFR-001.1:** Data fetch latency < 5 seconds
- **NFR-001.2:** Signal generation latency < 2 seconds
- **NFR-001.3:** Order placement latency < 1 second
- **NFR-001.4:** Backtesting 2 years of data < 60 seconds
- **NFR-001.5:** Dashboard refresh rate: 5 seconds

#### NFR-002: Reliability
- **NFR-002.1:** System uptime: 99.5% during market hours
- **NFR-002.2:** Automatic recovery from API failures (retry with exponential backoff)
- **NFR-002.3:** Graceful degradation if ML models unavailable (fall back to rules)
- **NFR-002.4:** Data consistency checks before trading
- **NFR-002.5:** Transaction atomicity for multi-leg orders

#### NFR-003: Scalability
- **NFR-003.1:** Support 1-10 underlyings initially (SPY, QQQ, IWM, DIA, etc.)
- **NFR-003.2:** Support 5-20 concurrent positions
- **NFR-003.3:** Handle 100+ signals evaluated per day
- **NFR-003.4:** Store 5+ years of historical data
- **NFR-003.5:** Support horizontal scaling for ML training

#### NFR-004: Security
- **NFR-004.1:** API keys stored in environment variables (never in code)
- **NFR-004.2:** Database access restricted by IP whitelist
- **NFR-004.3:** Encrypted connections to broker APIs (HTTPS/WSS)
- **NFR-004.4:** Rate limiting on API calls
- **NFR-004.5:** No sensitive data in logs (mask account numbers)

#### NFR-005: Maintainability
- **NFR-005.1:** Code coverage: > 80% for core modules
- **NFR-005.2:** Documentation: Docstrings for all public functions
- **NFR-005.3:** Type hints throughout codebase (Python 3.10+)
- **NFR-005.4:** Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **NFR-005.5:** Configuration via YAML files (no hardcoded parameters)

#### NFR-006: Observability
- **NFR-006.1:** Metrics exported to Prometheus
- **NFR-006.2:** Dashboards in Grafana for real-time monitoring
- **NFR-006.3:** Structured logging for easy parsing
- **NFR-006.4:** Performance profiling for optimization
- **NFR-006.5:** Model performance tracking per regime

---

## 3. System Scope

### 3.1 In Scope

#### Phase 1: Foundation (Weeks 1-4)
- ✅ Data pipeline for SPY options
- ✅ Greeks calculation and IV computation
- ✅ Regime detection (5 regimes)
- ✅ Simple IV Rank strategy
- ✅ Paper trading integration (IBKR or Alpaca)
- ✅ Basic risk management
- ✅ Simple monitoring dashboard

#### Phase 2: ML Integration (Weeks 5-8)
- ✅ Feature engineering (270+ features inspired by Bali et al.)
- ✅ XGBoost model for option return prediction
- ✅ LSTM model for time-series patterns
- ✅ Model training pipeline with MLflow
- ✅ Walk-forward backtesting
- ✅ Strategy optimization

#### Phase 3: Adaptive Trading (Weeks 9-12)
- ✅ Multi-strategy ensemble
- ✅ Regime-based strategy selection
- ✅ Online learning with new data
- ✅ Advanced position management (profit targets, stop losses)
- ✅ Comprehensive backtesting reports
- ✅ Performance attribution analysis

#### Phase 4: Production Deployment (Months 4-6)
- ✅ Expand to 3-5 underlyings (QQQ, IWM, DIA)
- ✅ 0DTE strategy for high-conviction signals
- ✅ Real-money deployment with small capital
- ✅ Automated daily operations
- ✅ Production monitoring and alerting
- ✅ Weekly performance reviews

### 3.2 Out of Scope (Future Enhancements)

#### V2.0 Features
- ❌ Multi-leg strategies beyond verticals (iron condors, butterflies, calendars)
- ❌ Earnings-specific strategies
- ❌ Futures options (SPX, ES, etc.)
- ❌ Portfolio hedging strategies
- ❌ High-frequency trading (sub-second)
- ❌ Custom deep RL agents (PPO, TD3)
- ❌ Multi-broker support and routing
- ❌ Social/sentiment data integration
- ❌ Mobile app for monitoring

### 3.3 Assumptions

#### Market Assumptions
- **ASM-001:** Options markets remain sufficiently liquid (SPY has 1M+ daily volume)
- **ASM-002:** Broker APIs remain stable and accessible
- **ASM-003:** Free data sources (yfinance) continue to be available for paper trading
- **ASM-004:** Market structure doesn't fundamentally change (no major regulatory shifts)
- **ASM-005:** Volatility continues to exhibit mean-reverting behavior

#### Technical Assumptions
- **ASM-006:** Sufficient compute available (local or cloud)
- **ASM-007:** Internet connectivity during market hours
- **ASM-008:** Historical data quality is acceptable for training
- **ASM-009:** Python ecosystem remains stable
- **ASM-010:** Broker paper trading environment accurately reflects live trading

#### Operational Assumptions
- **ASM-011:** 5 hours/day available for development and monitoring
- **ASM-012:** 3-6 months timeline is acceptable
- **ASM-013:** Can monitor system 2-3x per day during market hours
- **ASM-014:** Willing to iterate based on paper trading results

### 3.4 Dependencies

#### External Dependencies
- **DEP-001:** Broker API (Interactive Brokers or Alpaca)
- **DEP-002:** Market data provider (yfinance, Polygon, IBKR)
- **DEP-003:** Cloud compute (optional: AWS, GCP, or local)
- **DEP-004:** Database (PostgreSQL or SQLite)
- **DEP-005:** Python 3.10+ runtime

#### Internal Dependencies
- **DEP-006:** Phase 1 must complete before Phase 2
- **DEP-007:** Backtesting validation before paper trading
- **DEP-008:** Paper trading success before real money

---

## 4. User Stories

### 4.1 As a System Operator

**US-001:** As an operator, I want to start the system each morning with a single command, so that I can begin paper trading quickly.

**Acceptance Criteria:**
- Single CLI command: `python main.py run --mode paper`
- System checks data freshness and connectivity
- Logs indicate successful startup
- Dashboard becomes accessible

---

**US-002:** As an operator, I want to see all open positions on a dashboard, so that I can monitor P&L and risk.

**Acceptance Criteria:**
- Dashboard shows: symbol, strike, expiration, entry price, current price, P&L, Greeks
- Updates every 5 seconds
- Color-coded by P&L (green profit, red loss)
- Shows portfolio totals and risk metrics

---

**US-003:** As an operator, I want to receive alerts when drawdown exceeds 20%, so that I can intervene if necessary.

**Acceptance Criteria:**
- Email or Slack alert sent immediately
- Alert includes current drawdown %, equity, and open positions
- Alert only sent once per threshold crossing
- Can configure alert channels in config

---

**US-004:** As an operator, I want to manually close a position, so that I can exit if I see concerning market conditions.

**Acceptance Criteria:**
- CLI command: `python main.py close-position <position_id>`
- System places market order to close
- Confirms closure and logs P&L
- Updates dashboard immediately

---

### 4.2 As a Strategy Developer

**US-005:** As a developer, I want to backtest a strategy on historical data, so that I can validate it before paper trading.

**Acceptance Criteria:**
- CLI command: `python main.py backtest --strategy iv_rank --start 2020-01-01 --end 2024-12-31`
- Generates performance report with Sharpe, drawdown, win rate
- Creates equity curve visualization
- Outputs trade log CSV

---

**US-006:** As a developer, I want to add a new feature to the ML model, so that I can improve prediction accuracy.

**Acceptance Criteria:**
- Add feature calculation in `src/features/`
- Feature automatically included in model training
- Feature importance tracked in MLflow
- Backtest compares old vs new model

---

**US-007:** As a developer, I want to optimize strategy parameters, so that I can find the best configuration.

**Acceptance Criteria:**
- CLI command: `python main.py optimize --strategy iv_rank --metric sharpe`
- Uses Optuna for Bayesian optimization
- Tests 100+ parameter combinations
- Outputs best parameters and performance

---

### 4.3 As an ML Engineer

**US-008:** As an ML engineer, I want to train a model on recent data, so that it adapts to current market conditions.

**Acceptance Criteria:**
- CLI command: `python main.py train --model xgboost --start 2020-01-01`
- Model trained with walk-forward validation
- Logged to MLflow with metrics
- Automatically versioned and saved

---

**US-009:** As an ML engineer, I want to compare model performance across regimes, so that I can understand when it works best.

**Acceptance Criteria:**
- Report shows Sharpe ratio by regime
- Shows win rate and average return by regime
- Identifies regimes with negative performance
- Suggests regime-specific model tuning

---

### 4.4 As an Analyst

**US-010:** As an analyst, I want to generate a monthly performance report, so that I can assess system profitability.

**Acceptance Criteria:**
- CLI command: `python main.py report --month 2024-11`
- Report includes: total return, Sharpe, max drawdown, win rate, trades list
- Compares to benchmarks (buy-and-hold SPY)
- Exports to PDF

---

**US-011:** As an analyst, I want to understand why a signal was generated, so that I can validate strategy logic.

**Acceptance Criteria:**
- Signal log includes: IV rank, regime, feature values, model predictions
- Explainability via SHAP values for ML signals
- Can query: `python main.py explain-signal <signal_id>`
- Output shows top 10 contributing factors

---

## 5. Data Requirements

### 5.1 Input Data

#### Market Data
| Data Type | Source | Frequency | Retention | Format |
|-----------|--------|-----------|-----------|--------|
| Stock OHLCV | yfinance | Daily | 5 years | Parquet |
| Options Chains | yfinance / IBKR | Intraday (1hr) | 1 year | PostgreSQL |
| VIX | yfinance | Daily | 10 years | Parquet |
| Greeks | Calculated | On-demand | 1 year | PostgreSQL |
| Earnings Dates | yfinance | Daily | 2 years | PostgreSQL |

#### Configuration Data
| Data Type | Source | Format | Version Control |
|-----------|--------|--------|-----------------|
| Strategy Parameters | YAML | configs/strategy.yaml | Git |
| Risk Limits | YAML | configs/risk.yaml | Git |
| Model Hyperparameters | YAML | configs/models.yaml | Git |
| Data Sources | YAML | configs/data.yaml | Git |

### 5.2 Output Data

#### Trading Data
| Data Type | Destination | Frequency | Retention |
|-----------|-------------|-----------|-----------|
| Signals | PostgreSQL | Real-time | Unlimited |
| Orders | PostgreSQL | Real-time | Unlimited |
| Positions | PostgreSQL | Real-time | Unlimited |
| Fills | PostgreSQL | Real-time | Unlimited |
| P&L | PostgreSQL | Daily | Unlimited |

#### ML Data
| Data Type | Destination | Frequency | Retention |
|-----------|-------------|-----------|-----------|
| Features | Parquet | Daily | 5 years |
| Predictions | PostgreSQL | Real-time | 1 year |
| Model Artifacts | MLflow | Per training | All versions |
| Metrics | MLflow | Per training | All runs |

#### Logs
| Data Type | Destination | Frequency | Retention |
|-----------|-------------|-----------|-----------|
| Application Logs | Files + DB | Real-time | 90 days |
| System Metrics | Prometheus | 15s | 30 days |
| Audit Logs | PostgreSQL | Real-time | Unlimited |

---

## 6. Constraints

### 6.1 Technical Constraints
- **CON-001:** Python 3.10+ required (no backwards compatibility)
- **CON-002:** Must run on single machine initially (no distributed computing required)
- **CON-003:** Limited to US equities options (no international)
- **CON-004:** Maximum 60 API calls/minute to yfinance (rate limits)
- **CON-005:** IBKR TWS must be running locally for trading

### 6.2 Business Constraints
- **CON-006:** Paper trading only for first 3-6 months
- **CON-007:** Real money capped at $10k-50k initially
- **CON-008:** Must achieve Sharpe > 1.0 before real money
- **CON-009:** Single operator (no team initially)
- **CON-010:** Must be operable alongside full-time job

### 6.3 Regulatory Constraints
- **CON-011:** Pattern Day Trading rules apply if < $25k account
- **CON-012:** Must comply with broker risk disclosures
- **CON-013:** Tax reporting required for real money profits
- **CON-014:** No insider trading (no material non-public information)

---

## 7. Risk Management

### 7.1 Trading Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Large drawdown (>30%) | Medium | High | Stop trading at 25% drawdown, regime-based sizing |
| Flash crash losses | Low | Critical | Position limits, stop losses, avoid 0DTE in high vol |
| Model overfitting | High | High | Walk-forward validation, regime testing, out-of-sample |
| Black swan event | Low | Critical | Max 20% portfolio exposure, diversify across strategies |
| Broker API failure | Medium | Medium | Automatic retry, fallback to manual, alerts |

### 7.2 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data feed interruption | Medium | High | Data caching, multiple sources, graceful degradation |
| Software bugs | High | Medium | 80% test coverage, staged rollout, paper testing |
| Model degradation | Medium | High | Monitor performance by regime, retrain monthly |
| System downtime | Low | Medium | Monitoring, alerts, auto-restart |
| Database corruption | Low | High | Daily backups, transaction logs |

### 7.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Missed monitoring | Medium | Medium | Automated alerts, daily checklist |
| Incorrect position sizing | Medium | High | Pre-trade validation, risk manager approval |
| Manual intervention error | Medium | Medium | Confirm before close, log all manual actions |
| Configuration error | Medium | High | Version control, config validation, staging env |

---

## 8. Success Criteria

### 8.1 Phase 1 Success (Weeks 1-4)
- ✅ System runs daily on paper account without errors
- ✅ At least 10 paper trades executed
- ✅ Backtested Sharpe > 0.8 (preliminary)
- ✅ All core modules tested (>70% coverage)

### 8.2 Phase 2 Success (Weeks 5-8)
- ✅ ML model trained and deployed
- ✅ Walk-forward backtest Sharpe > 1.0
- ✅ Feature importance aligns with financial theory
- ✅ Model outperforms rule-based baseline

### 8.3 Phase 3 Success (Weeks 9-12)
- ✅ 30+ paper trades with 55%+ win rate
- ✅ Live Sharpe > 1.0 on paper account
- ✅ Max drawdown < 25% on paper account
- ✅ Profit factor > 1.3

### 8.4 Phase 4 Success (Months 4-6)
- ✅ 60+ days of consistent paper trading profitability
- ✅ Real money deployed with same parameters
- ✅ Real money Sharpe > 1.0 after 30 trades
- ✅ System runs autonomously with minimal intervention
- ✅ Ready to scale capital

---

## 9. Open Questions

### 9.1 Strategy Questions
- **Q-001:** Should we focus on 0DTE or 3-7 DTE initially? → Recommend 3-7 DTE for stability
- **Q-002:** What underlyings beyond SPY? → Add QQQ, IWM after SPY proven
- **Q-003:** Should we implement calendar spreads? → Phase 2 enhancement
- **Q-004:** How to handle earnings? → Avoid initially, add as Phase 3 feature

### 9.2 Technical Questions
- **Q-005:** IBKR vs Alpaca for paper trading? → IBKR for real options support
- **Q-006:** Local database vs cloud? → Start local SQLite, migrate to PostgreSQL if needed
- **Q-007:** GPU required for LSTM training? → No, CPU sufficient for dataset size
- **Q-008:** How to handle market data costs at scale? → Use yfinance free tier, upgrade if needed

### 9.3 Operational Questions
- **Q-009:** How many hours per day during live trading? → Target 30 min morning, 30 min EOD
- **Q-010:** When to transition paper → real? → After 60+ days paper profitability
- **Q-011:** Starting real capital amount? → $5k-10k initially
- **Q-012:** Exit strategy if system fails? → Stop at 25% drawdown, reassess

---

## 10. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | System Team | Initial specification |

---

## 11. Approvals

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Owner | [Your Name] | _________ | _____ |
| Tech Lead | [Your Name] | _________ | _____ |
| QA Lead | [Your Name] | _________ | _____ |

---

**Next Steps:**
1. Review and approve specification
2. Create Architecture Document
3. Create Implementation Plan
4. Begin Phase 1 development
