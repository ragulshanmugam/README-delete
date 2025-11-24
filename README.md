# ML Options Trading System

A hybrid ML + rule-based options trading system for ETFs and mega-cap stocks.

## Overview

This system combines machine learning for market prediction with rule-based logic for options selection:

- **Direction Classifier (XGBoost):** Predicts 5-day market direction
- **Volatility Forecaster (XGBoost):** Predicts IV rank movement
- **Regime Classifier (Rule-based):** Classifies market volatility regime
- **Options Rules Engine:** Selects strategy, strikes, DTE based on predictions

## Quick Start

### Prerequisites

- Docker Desktop
- Git

### Setup

```bash
# Clone repository
git clone <repo-url>
cd ml-options-trading

# Copy environment file
cp .env.example .env

# Build Docker image
docker-compose build
```

### Fetch Data

```bash
# Fetch all configured tickers (SPY, QQQ, IWM)
docker-compose run app python scripts/fetch_data.py

# Fetch with technical indicators
docker-compose run app python scripts/fetch_data.py --with-features

# Fetch specific tickers
docker-compose run app python scripts/fetch_data.py --tickers SPY QQQ

# Fetch 3 years of data
docker-compose run app python scripts/fetch_data.py --years 3

# Check data status
docker-compose run app python scripts/fetch_data.py status
```

### Run Tests

```bash
docker-compose run app pytest tests/
```

## Project Structure

```
ml-options-trading/
├── docs/
│   ├── 01_SPECIFICATION.md     # System requirements
│   ├── 02_ARCHITECTURE.md      # Technical architecture
│   ├── 03_IMPLEMENTATION.md    # Implementation details
│   ├── 04_INFRASTRUCTURE.md    # Infrastructure guide
│   ├── 05_IMPLEMENTATION_PLAN.md  # 3-week MVP plan
│   └── 06_FUTURE_ROADMAP.md    # Future features
├── src/
│   ├── config/settings.py      # Configuration
│   ├── data/
│   │   ├── yfinance_loader.py  # Data fetching
│   │   └── feature_store.py    # Feature storage
│   ├── features/
│   │   └── technical_indicators.py
│   ├── models/
│   │   ├── direction_classifier.py
│   │   ├── volatility_forecaster.py
│   │   └── regime_classifier.py
│   └── strategy/
│       └── options_rules.py
├── scripts/
│   └── fetch_data.py           # CLI script
├── tests/
├── data/                       # Data storage (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Architecture

### Two-Model System

```
Direction Model (XGBoost)     Volatility Model (XGBoost)
         |                              |
         v                              v
    [Bullish/Bearish/Neutral]    [IV Rank Forecast]
                    \                /
                     v              v
              Rule-Based Options Engine
                        |
                        v
              [Strategy, Strike, DTE, Size]
```

### Regime Classification

| Regime | Conditions | Strategy |
|--------|------------|----------|
| low_vol | VIX < 15, VIX_rank < 30% | Debit spreads |
| high_vol | VIX > 25 OR VIX_rank > 80% | Credit spreads, smaller size |
| iv_contraction | HV/IV < 0.6 | Sell premium |
| iv_expansion | HV/IV > 0.9 | Buy options |
| neutral | Default | Conservative approach |

## Configuration

Environment variables (`.env`):

```bash
LOG_LEVEL=INFO
LOOKBACK_YEARS=5
TICKERS=SPY,QQQ,IWM
MAX_POSITION_SIZE=0.02
MAX_PORTFOLIO_EXPOSURE=0.20
```

See `.env.example` for all options.

## Development Timeline

### Week 1: Data Pipeline
- [x] Project structure
- [x] Docker setup
- [x] Data fetching (yfinance)
- [x] Feature store (parquet)
- [x] Technical indicators

### Week 2: ML Models
- [ ] Direction classifier (XGBoost)
- [ ] Volatility forecaster (XGBoost)
- [ ] Walk-forward validation
- [ ] Model evaluation

### Week 3: Trading Logic
- [ ] Options selection rules
- [ ] Risk management
- [ ] Paper trading integration
- [ ] End-to-end testing

## Documentation

- [Specification](docs/01_SPECIFICATION.md) - System requirements
- [Architecture](docs/02_ARCHITECTURE.md) - Technical design
- [Implementation Plan](docs/05_IMPLEMENTATION_PLAN.md) - 3-week MVP timeline
- [Infrastructure](docs/04_INFRASTRUCTURE.md) - Setup and deployment
- [Future Roadmap](docs/06_FUTURE_ROADMAP.md) - Planned features

## License

Proprietary - All rights reserved.
