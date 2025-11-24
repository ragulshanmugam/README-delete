# ML Options Trading System - Future Roadmap

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Planning
**Scope:** Post-MVP Features

---

## Table of Contents

1. [Earnings Model](#1-earnings-model)
2. [Mega Caps Expansion Plan](#2-mega-caps-expansion-plan)
3. [Paid Data Sources](#3-paid-data-sources)
4. [Production Deployment](#4-production-deployment)
5. [Advanced Features](#5-advanced-features)

---

## 1. Earnings Model

### 1.1 Overview

The **Earnings Model** is a specialized system for trading options around earnings announcements. It operates separately from the core model and activates only during earnings windows.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EARNINGS MODEL ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────┐
                          │  Earnings Calendar  │
                          │  (Date + Time)      │
                          └──────────┬──────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │ Pre-Earnings│  │ Earnings    │  │ Post-Earn   │
            │ Analysis    │  │ Day Trading │  │ Analysis    │
            └─────────────┘  └─────────────┘  └─────────────┘
                     │               │               │
                     ▼               ▼               ▼
            ┌─────────────────────────────────────────────┐
            │          Earnings Feature Set               │
            ├─────────────────────────────────────────────┤
            │ • Historical earnings moves                  │
            │ • Implied move vs actual move               │
            │ • IV premium before earnings                │
            │ • Analyst estimate dispersion               │
            │ • Options market positioning                │
            └─────────────────────────────────────────────┘
                                     │
                                     ▼
            ┌─────────────────────────────────────────────┐
            │          Earnings XGBoost Model             │
            ├─────────────────────────────────────────────┤
            │ Predicts:                                   │
            │ • Post-earnings move direction              │
            │ • Expected move magnitude                   │
            │ • IV crush magnitude                        │
            │ • Strategy recommendation                   │
            └─────────────────────────────────────────────┘
```

### 1.2 Earnings Features

| Feature | Description | Data Source |
|---------|-------------|-------------|
| `implied_move` | Options-implied expected move | ATM straddle price |
| `actual_move_avg` | Average historical earnings move | Historical data |
| `actual_move_std` | Std dev of historical moves | Historical data |
| `iv_premium` | IV rank before earnings | Options chain |
| `beat_rate` | Historical beat/miss rate | Earnings history |
| `estimate_dispersion` | Analyst estimate range | External API |
| `options_skew` | Put/call skew before earnings | Options chain |
| `days_since_last` | Days since last earnings | Calendar |
| `sector_performance` | Sector earnings trends | Market data |
| `guidance_history` | Historical guidance patterns | Earnings history |

### 1.3 Earnings Strategies

**Strategy 1: Pre-Earnings IV Premium Capture**
```
Timing: 3-5 days before earnings
Strategy: Sell credit spreads
Rationale: IV is elevated, capture premium before event
Exit: Day before earnings OR 50% profit
Risk: Stock moves against position before exit
```

**Strategy 2: Post-Earnings IV Crush**
```
Timing: Day of earnings (before close)
Strategy: Sell strangles/straddles
Rationale: IV will collapse 30-50% after earnings
Exit: Day after earnings
Risk: Stock moves more than implied move
```

**Strategy 3: Earnings Direction Play**
```
Timing: Day of earnings
Strategy: Buy directional spread (if high conviction)
Rationale: Model predicts earnings direction
Exit: Day after earnings
Risk: Direction prediction wrong
```

### 1.4 Implementation Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| Research | Historical earnings data collection | 1 week |
| Features | Earnings feature engineering | 1 week |
| Model | XGBoost model for earnings | 1 week |
| Backtest | Walk-forward validation | 1 week |
| Paper Trade | Paper trading earnings | 4 weeks |

**Estimated Total:** 8 weeks

### 1.5 Required Data Sources

- Earnings calendar (yfinance free)
- Historical earnings data (yfinance free)
- Analyst estimates (paid: Polygon, Alpha Vantage)
- Options chains (yfinance free for EOD)

---

## 2. Mega Caps Expansion Plan

### 2.1 Phased Rollout

```
Phase 1 (MVP):     SPY, QQQ, IWM
                   └── ETFs only, no earnings risk

Phase 2:           Add DIA (Dow Jones ETF)
                   └── Complete ETF coverage

Phase 3:           Add AAPL, MSFT, GOOGL
                   └── Low-volatility mega caps

Phase 4:           Add AMZN, NVDA, META
                   └── Medium-volatility mega caps

Phase 5:           Add TSLA, NFLX
                   └── High-volatility mega caps
```

### 2.2 Mega Cap Characteristics

| Symbol | Avg IV | Earnings Vol | Correlation to SPY | Priority |
|--------|--------|--------------|-------------------|----------|
| AAPL | 25% | Medium | 0.75 | High |
| MSFT | 22% | Low | 0.80 | High |
| GOOGL | 28% | Medium | 0.72 | High |
| AMZN | 30% | Medium | 0.70 | Medium |
| NVDA | 45% | High | 0.65 | Medium |
| META | 35% | High | 0.68 | Medium |
| TSLA | 55% | Very High | 0.55 | Low |
| NFLX | 40% | High | 0.60 | Low |

### 2.3 Earnings Blackout Rules

```python
# Configuration for mega caps
earnings_config = {
    'AAPL': {
        'blackout_days_before': 5,  # Stop core model 5 days before
        'earnings_window_days': 2,  # Use earnings model 2 days before
        'typical_months': [1, 4, 7, 10],  # January, April, July, October
    },
    # Similar for other stocks...
}

def should_trade_core_model(symbol: str, date: date) -> bool:
    """Check if core model should trade this symbol today"""
    days_to_earnings = get_days_to_earnings(symbol, date)

    if days_to_earnings is None:
        return True  # No upcoming earnings

    if symbol in ETFS:
        return True  # ETFs have no earnings

    if days_to_earnings > earnings_config[symbol]['blackout_days_before']:
        return True  # Outside blackout window

    return False  # In blackout period
```

### 2.4 Cross-Ticker Analysis

**Correlation Features:**
- SPY correlation (20-day rolling)
- Sector ETF correlation
- Beta to SPY
- Relative strength vs sector

**Implementation:**
```python
def calculate_cross_ticker_features(symbol: str, date: date) -> dict:
    """Calculate cross-ticker features"""
    return {
        'spy_correlation_20d': correlation(symbol, 'SPY', window=20),
        'spy_beta_60d': beta(symbol, 'SPY', window=60),
        'sector_relative_strength': rs(symbol, sector_etf(symbol)),
        'cross_sector_momentum': sector_momentum(symbol)
    }
```

---

## 3. Paid Data Sources

### 3.1 When to Upgrade

| Data Need | Free Option | Paid Option | When to Upgrade |
|-----------|-------------|-------------|-----------------|
| OHLCV data | yfinance | Polygon.io | Never needed |
| Options EOD | yfinance | ThetaData | When backtesting seriously |
| Options Real-time | IBKR (delayed) | IBKR (live) | Live trading |
| Analyst estimates | None | Polygon/AV | Earnings model |
| Earnings dates | yfinance | Polygon | If reliability issues |
| Fundamentals | yfinance | Polygon | Earnings model |

### 3.2 Polygon.io Integration

**Tier Recommendation:** Developer ($99/mo)

**Features:**
- Real-time options data
- Historical options data
- Analyst estimates
- Corporate actions
- High rate limits

**Integration:**
```python
from polygon import RESTClient

class PolygonFetcher:
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)

    def fetch_options_chain(self, symbol: str, expiration: str):
        """Fetch options chain from Polygon"""
        contracts = self.client.list_options_contracts(
            underlying_ticker=symbol,
            expiration_date=expiration
        )
        return self._process_contracts(contracts)

    def fetch_analyst_estimates(self, symbol: str):
        """Fetch analyst estimates for earnings"""
        # Polygon earnings endpoint
        pass
```

### 3.3 ThetaData Integration

**Tier Recommendation:** Standard ($30/mo)

**Features:**
- 10+ years historical options data
- All strikes, all expirations
- Clean, validated data
- End-of-day only

**Use Case:** Serious backtesting, research

### 3.4 Cost Analysis

| Phase | Data Sources | Monthly Cost |
|-------|--------------|--------------|
| MVP | yfinance only | $0 |
| Paper Trading | yfinance + IBKR | $0-10 |
| Serious Backtest | + ThetaData | $30 |
| Production | + Polygon (optional) | $99-129 |

---

## 4. Production Deployment

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS CLOUD                                       │
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  EC2 Instance (t3.medium)                                          │    │
│   │  ┌─────────────────────────────────────────────────────────────┐  │    │
│   │  │  Trading Application (Python)                                │  │    │
│   │  │  • Data fetching (hourly cron)                              │  │    │
│   │  │  • Feature calculation                                       │  │    │
│   │  │  • Model inference                                           │  │    │
│   │  │  • Signal generation                                         │  │    │
│   │  │  • Order management                                          │  │    │
│   │  └─────────────────────────────────────────────────────────────┘  │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│   ┌─────────────────┐   ┌──────────┴───────────┐   ┌─────────────────┐    │
│   │  RDS PostgreSQL │   │  S3 Storage          │   │  CloudWatch     │    │
│   │  • Trades       │   │  • Model artifacts   │   │  • Metrics      │    │
│   │  • Positions    │   │  • Feature files     │   │  • Alerts       │    │
│   │  • Signals      │   │  • Backups           │   │  • Logs         │    │
│   └─────────────────┘   └──────────────────────┘   └─────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES                                  │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│   │  IBKR Gateway   │   │  Polygon API    │   │  Slack/Email    │         │
│   │  (Live Trading) │   │  (Data)         │   │  (Alerts)       │         │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Monitoring & Alerts

**Critical Alerts (immediate):**
- Drawdown > 15%
- Position limit exceeded
- Order execution failure
- System health check failure

**Warning Alerts (daily summary):**
- Model performance degradation
- Data quality issues
- Unusual volatility

**Dashboard Metrics:**
- Portfolio value (real-time)
- Daily P&L
- Open positions
- Model predictions
- System health

### 4.3 Deployment Checklist

```markdown
Pre-Deployment:
- [ ] 60+ days paper trading
- [ ] Sharpe > 1.0 on paper
- [ ] Win rate > 55%
- [ ] Max drawdown < 20%
- [ ] All tests passing
- [ ] Documentation complete

Infrastructure:
- [ ] EC2 instance provisioned
- [ ] RDS database created
- [ ] S3 buckets configured
- [ ] CloudWatch alerts set up
- [ ] IBKR Gateway installed
- [ ] SSL certificates

Deployment:
- [ ] Application deployed
- [ ] Cron jobs configured
- [ ] Monitoring active
- [ ] Backup verification
- [ ] Manual failover tested

Post-Deployment:
- [ ] Small capital deployed ($5-10k)
- [ ] First live trades executed
- [ ] Daily monitoring routine
- [ ] Weekly performance reviews
```

### 4.4 Scaling Plan

| Capital | Max Positions | Daily Trades | Infrastructure |
|---------|--------------|--------------|----------------|
| $5-10k | 3 | 1-2 | Single EC2 |
| $10-50k | 5 | 2-3 | Single EC2 |
| $50-100k | 10 | 3-5 | EC2 + RDS Multi-AZ |
| $100k+ | 15 | 5-10 | Full HA setup |

---

## 5. Advanced Features

### 5.1 Multi-Leg Strategies

**Phase 2 Strategies:**
- Iron Condors (neutral high IV)
- Iron Butterflies (pinned near strike)
- Calendar Spreads (volatility term structure)

**Implementation:**
```python
class MultiLegStrategy:
    """Base class for multi-leg options strategies"""

    def calculate_greeks(self) -> dict:
        """Calculate net position Greeks"""
        pass

    def calculate_breakevens(self) -> list:
        """Calculate breakeven points"""
        pass

    def probability_of_profit(self) -> float:
        """Calculate POP using delta approximation"""
        pass
```

### 5.2 Portfolio Hedging

**Hedge Strategies:**
- SPY put protection (tail risk)
- VIX call hedges (volatility spikes)
- Collar strategies (upside cap, downside protection)

**Implementation:**
```python
class PortfolioHedger:
    """Manage portfolio-level hedges"""

    def calculate_hedge_ratio(self, portfolio: Portfolio) -> float:
        """Calculate hedge ratio based on portfolio delta"""
        pass

    def recommend_hedge(self, regime: str) -> dict:
        """Recommend hedging strategy based on regime"""
        pass
```

### 5.3 Machine Learning Enhancements

**LSTM Integration:**
- Time-series pattern recognition
- Sequence modeling for regime transitions
- Enhanced volatility forecasting

**Ensemble Methods:**
- Stacking (XGBoost + LSTM)
- Voting ensemble
- Per-regime model selection

**Online Learning:**
- Incremental model updates
- Concept drift detection
- Adaptive thresholds

### 5.4 Alternative Data

**Potential Data Sources:**
- Options flow (unusual activity)
- Short interest data
- Social sentiment (limited value)
- Economic calendar events
- Fed meeting schedules

**Implementation Priority:**
1. Economic calendar (FOMC, CPI, etc.)
2. Options flow (large trades)
3. Short interest (weekly)

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| MVP | 3 weeks | ETFs, basic models, paper trading |
| Validation | 3 weeks | Extended backtests, optimization |
| Mega Caps | 3 weeks | Stock support, earnings blackout |
| Earnings Model | 4 weeks | Earnings-specific trading |
| Production | 4 weeks | Live deployment, monitoring |
| Advanced | Ongoing | Multi-leg, hedging, ML enhancements |

**Total to Production:** ~4 months
**Total to Full System:** ~6 months

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Engineering | Initial roadmap |
