# Adaptive Options Trading System - Implementation Plan

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Ready to Execute
**Timeline:** 16 weeks (4 months)
**Effort:** 5 hours/day, 25-35 hours/week

---

## Table of Contents

1. [Project Timeline](#1-project-timeline)
2. [Sprint Structure](#2-sprint-structure)
3. [Phase 1: Foundation (Weeks 1-4)](#3-phase-1-foundation-weeks-1-4)
4. [Phase 2: ML Integration (Weeks 5-8)](#4-phase-2-ml-integration-weeks-5-8)
5. [Phase 3: Adaptive Trading (Weeks 9-12)](#5-phase-3-adaptive-trading-weeks-9-12)
6. [Phase 4: Production Deployment (Weeks 13-16)](#6-phase-4-production-deployment-weeks-13-16)
7. [Code Module Specifications](#7-code-module-specifications)
8. [Testing Strategy](#8-testing-strategy)
9. [Deployment Checklist](#9-deployment-checklist)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Project Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         16-Week Timeline                                 │
└─────────────────────────────────────────────────────────────────────────┘

Week 1-4: Phase 1 - Foundation
├── Week 1: Infrastructure & Data Pipeline
├── Week 2: Greeks Calculator & Feature Engineering
├── Week 3: Simple Strategy & Paper Trading Setup
└── Week 4: Basic Monitoring & Testing
    └── Milestone: First paper trade executed

Week 5-8: Phase 2 - ML Integration
├── Week 5: Feature Engineering (270+ features)
├── Week 6: XGBoost Model Training
├── Week 7: LSTM Model & Ensemble
└── Week 8: Walk-Forward Backtesting
    └── Milestone: ML model outperforms baseline

Week 9-12: Phase 3 - Adaptive Trading
├── Week 9: Multi-Strategy Implementation
├── Week 10: Advanced Position Management
├── Week 11: Performance Attribution & Optimization
└── Week 12: Comprehensive Testing & Validation
    └── Milestone: Live paper trading profitable for 30+ days

Week 13-16: Phase 4 - Production Deployment
├── Week 13: Multi-Asset Expansion (QQQ, IWM)
├── Week 14: 0DTE Strategy (if applicable)
├── Week 15: Real Money Deployment (small capital)
└── Week 16: Automation & Scaling
    └── Milestone: Autonomous trading with monitoring
```

---

## 2. Sprint Structure

Each week follows this structure:

### Daily Schedule (5 hours/day)

**Morning (2 hours):**
- 30 min: Review previous day's work
- 90 min: Core development (deep work)

**Afternoon (3 hours):**
- 2 hours: Implementation & testing
- 30 min: Documentation & git commits
- 30 min: Planning next day

### Weekly Schedule

| Day | Focus | Hours |
|-----|-------|-------|
| Monday | Planning & setup | 5 |
| Tuesday | Implementation (Part 1) | 5 |
| Wednesday | Implementation (Part 2) | 5 |
| Thursday | Testing & debugging | 5 |
| Friday | Integration & documentation | 5 |

**Total: 25 hours/week minimum**

### Sprint Deliverables

Each week ends with:
- ✅ Working code committed to git
- ✅ Tests passing (>70% coverage)
- ✅ Documentation updated
- ✅ Demo/validation of feature
- ✅ Sprint retrospective notes

---

## 3. Phase 1: Foundation (Weeks 1-4)

### Week 1: Infrastructure & Data Pipeline

**Goal:** Set up project structure and get market data flowing

#### Day 1: Project Setup

**Tasks:**
```bash
# 1. Create project structure
mkdir -p options-ml-trader/{src,tests,notebooks,configs,data,logs,models,docs}
cd options-ml-trader
git init

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python scripts/init_database.py

# 5. Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

**Files to create:**
- `requirements.txt` - Full dependency list
- `.env.example` - Template for environment variables
- `.gitignore` - Ignore venv, .env, data, logs
- `README.md` - Project overview
- `scripts/init_database.py` - Database schema creation

**Validation:**
```bash
# Run setup validation
python scripts/validate_setup.py
# Should print: ✅ All dependencies installed
#               ✅ Database connected
#               ✅ Environment variables loaded
```

**Time estimate: 5 hours**

---

#### Day 2-3: Data Fetchers

**Implementation:**

```python
# src/data/fetchers.py

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import date, datetime
import pandas as pd
import yfinance as yf
from loguru import logger

class DataFetcher(ABC):
    """Abstract base class for data fetchers"""

    @abstractmethod
    def fetch_stock_data(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_available_expirations(
        self,
        symbol: str
    ) -> List[str]:
        pass


class YFinanceFetcher(DataFetcher):
    """
    Free market data via yfinance

    Rate limits: ~2000 requests/hour (unofficial)
    Delay: ~15 minutes for options data
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        self._rate_limiter = RateLimiter(max_calls=100, period=60)

    def fetch_stock_data(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """Fetch OHLCV data for stock"""

        logger.info(f"Fetching {symbol} data from {start} to {end}")

        with self._rate_limiter:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)

        if df.empty:
            raise DataFetchError(f"No data returned for {symbol}")

        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df

    def fetch_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch options chain

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            expiration: Specific expiration date (YYYY-MM-DD) or None for all

        Returns:
            DataFrame with calls and puts combined
        """

        logger.info(f"Fetching options chain for {symbol}, exp={expiration}")

        with self._rate_limiter:
            ticker = yf.Ticker(symbol)

            if expiration:
                chain = ticker.option_chain(expiration)
            else:
                # Get all expirations
                expirations = ticker.options
                chains = []
                for exp in expirations:
                    try:
                        chain = ticker.option_chain(exp)
                        calls = self._process_options(
                            chain.calls, exp, 'call', symbol
                        )
                        puts = self._process_options(
                            chain.puts, exp, 'put', symbol
                        )
                        chains.append(pd.concat([calls, puts]))
                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol} {exp}: {e}")
                        continue

                if not chains:
                    raise DataFetchError(f"No options data for {symbol}")

                return pd.concat(chains, ignore_index=True)

            calls = self._process_options(
                chain.calls, expiration, 'call', symbol
            )
            puts = self._process_options(
                chain.puts, expiration, 'put', symbol
            )

            return pd.concat([calls, puts], ignore_index=True)

    def get_available_expirations(self, symbol: str) -> List[str]:
        """Get list of available option expirations"""

        with self._rate_limiter:
            ticker = yf.Ticker(symbol)
            return list(ticker.options)

    def _process_options(
        self,
        df: pd.DataFrame,
        expiration: str,
        option_type: str,
        symbol: str
    ) -> pd.DataFrame:
        """Add metadata and clean options dataframe"""

        df = df.copy()

        # Add metadata
        df['symbol'] = symbol
        df['expiration'] = pd.to_datetime(expiration)
        df['option_type'] = option_type
        df['fetch_time'] = datetime.now()

        # Calculate DTE
        df['dte'] = (df['expiration'] - pd.Timestamp.now()).dt.days

        # Calculate mid price
        df['mid_price'] = (df['bid'] + df['ask']) / 2

        # Calculate spread percentage
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
        df['spread_pct'] = df['spread_pct'].replace([np.inf, -np.inf], np.nan)

        # Rename columns for consistency
        df = df.rename(columns={
            'contractSymbol': 'contract_symbol',
            'lastTradeDate': 'last_trade_date',
            'lastPrice': 'last_price',
            'openInterest': 'open_interest',
            'impliedVolatility': 'implied_volatility'
        })

        return df


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period  # seconds
        self.calls = []

    def __enter__(self):
        import time

        now = time.time()

        # Remove calls outside window
        self.calls = [t for t in self.calls if now - t < self.period]

        # Check if limit exceeded
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.calls = []

        self.calls.append(now)

    def __exit__(self, *args):
        pass


class DataFetchError(Exception):
    """Raised when data fetch fails"""
    pass
```

**Tests:**

```python
# tests/test_fetchers.py

import pytest
from datetime import date
from src.data.fetchers import YFinanceFetcher

@pytest.fixture
def fetcher():
    return YFinanceFetcher()

def test_fetch_stock_data(fetcher):
    """Test fetching stock data"""
    df = fetcher.fetch_stock_data(
        'SPY',
        start=date(2024, 1, 1),
        end=date(2024, 1, 31)
    )

    assert not df.empty
    assert 'Close' in df.columns
    assert len(df) > 10  # At least 10 trading days

def test_fetch_options_chain(fetcher):
    """Test fetching options chain"""
    expirations = fetcher.get_available_expirations('SPY')
    assert len(expirations) > 0

    # Fetch first expiration
    df = fetcher.fetch_options_chain('SPY', expirations[0])

    assert not df.empty
    assert 'strike' in df.columns
    assert 'option_type' in df.columns
    assert 'call' in df['option_type'].values
    assert 'put' in df['option_type'].values

def test_rate_limiter():
    """Test rate limiter works"""
    import time
    from src.data.fetchers import RateLimiter

    limiter = RateLimiter(max_calls=3, period=1)

    start = time.time()
    for i in range(5):
        with limiter:
            pass

    elapsed = time.time() - start

    # Should have slept at least once
    assert elapsed > 1.0
```

**Validation:**
```bash
python -m pytest tests/test_fetchers.py -v
python scripts/demo_data_fetch.py  # Should print SPY options chain
```

**Time estimate: 10 hours**

---

#### Day 4-5: Data Storage & Processors

**Implementation:**

```python
# src/data/storage.py

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Date, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime

Base = declarative_base()

class OptionsChain(Base):
    """Store options chain snapshots"""
    __tablename__ = 'options_chains'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    expiration = Column(Date, nullable=False, index=True)
    option_type = Column(String(4), nullable=False)  # 'call' or 'put'
    strike = Column(Float, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    mid_price = Column(Float)
    last_price = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    spread_pct = Column(Float)
    dte = Column(Integer)
    fetch_time = Column(DateTime, default=datetime.now, index=True)

class Signal(Base):
    """Store generated signals"""
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    buy_strike = Column(Float)
    sell_strike = Column(Float)
    expiration = Column(Date)
    dte = Column(Integer)
    credit_or_debit = Column(Float)
    max_profit = Column(Float)
    max_risk = Column(Float)
    confidence = Column(Float, nullable=False)
    regime = Column(String(20))
    iv_rank = Column(Float)
    vix = Column(Float)
    executed = Column(Boolean, default=False)
    features = Column(JSON)
    model_predictions = Column(JSON)

class Trade(Base):
    """Store executed trades"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer)
    symbol = Column(String(10), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    entry_time = Column(DateTime, index=True)
    exit_time = Column(DateTime)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    status = Column(String(20), nullable=False)  # 'open', 'closed', 'cancelled'
    regime = Column(String(20))
    metadata = Column(JSON)

class Position(Base):
    """Store current positions"""
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer)
    symbol = Column(String(10), nullable=False, index=True)
    leg_type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    option_type = Column(String(4), nullable=False)
    strike = Column(Float, nullable=False)
    expiration = Column(Date, nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float)
    current_price = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    unrealized_pnl = Column(Float)
    status = Column(String(20), nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Database:
    """Database manager"""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_options_chain(self, df: pd.DataFrame):
        """Save options chain to database"""
        session = self.Session()
        try:
            df.to_sql(
                'options_chains',
                self.engine,
                if_exists='append',
                index=False
            )
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_latest_options_chain(
        self,
        symbol: str,
        min_dte: int = 0,
        max_dte: int = 60
    ) -> pd.DataFrame:
        """Get latest options chain from database"""
        query = f"""
        SELECT * FROM options_chains
        WHERE symbol = '{symbol}'
        AND dte BETWEEN {min_dte} AND {max_dte}
        AND fetch_time = (
            SELECT MAX(fetch_time)
            FROM options_chains
            WHERE symbol = '{symbol}'
        )
        """
        return pd.read_sql(query, self.engine)

    def save_signal(self, signal: dict):
        """Save signal to database"""
        session = self.Session()
        try:
            sig = Signal(**signal)
            session.add(sig)
            session.commit()
            return sig.id
        finally:
            session.close()

    # Similar methods for trades, positions...
```

**Validation:**
```bash
python scripts/test_database.py  # Save and retrieve test data
```

**Time estimate: 10 hours**

**Week 1 Milestone: ✅ Data pipeline working, data stored in database**

---

### Week 2: Greeks Calculator & Feature Engineering

#### Day 6-7: Greeks Calculator

**Implementation:**

```python
# src/features/greeks.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Optional, Literal
from dataclasses import dataclass

@dataclass
class GreeksResult:
    """Container for Greeks calculation results"""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: Optional[float] = None


class GreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model

    Note: Assumes European options. American options have early exercise premium.
    """

    def __init__(self, risk_free_rate: float = 0.045):
        """
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.045 for 4.5%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,  # Years
        iv: float,
        option_type: Literal['call', 'put']
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option

        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years
            iv: Implied volatility (decimal, e.g., 0.20 for 20%)
            option_type: 'call' or 'put'

        Returns:
            GreeksResult with all Greeks
        """

        if time_to_expiry <= 0:
            # Option expired
            if option_type == 'call':
                intrinsic = max(0, spot - strike)
            else:
                intrinsic = max(0, strike - spot)

            return GreeksResult(
                price=intrinsic,
                delta=1.0 if intrinsic > 0 else 0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                iv=iv
            )

        if iv <= 0:
            raise ValueError("Implied volatility must be positive")

        # Calculate d1 and d2
        d1 = self._d1(spot, strike, time_to_expiry, iv)
        d2 = d1 - iv * np.sqrt(time_to_expiry)

        # Calculate price
        if option_type == 'call':
            price = self._call_price(spot, strike, time_to_expiry, iv, d1, d2)
            delta = norm.cdf(d1)
            theta = self._call_theta(spot, strike, time_to_expiry, iv, d1, d2)
        else:
            price = self._put_price(spot, strike, time_to_expiry, iv, d1, d2)
            delta = -norm.cdf(-d1)
            theta = self._put_theta(spot, strike, time_to_expiry, iv, d1, d2)

        # Greeks that are same for calls and puts
        gamma = norm.pdf(d1) / (spot * iv * np.sqrt(time_to_expiry))
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100  # per 1%
        rho = (
            strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) *
            (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))
        ) / 100  # per 1%

        return GreeksResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,  # Already per day
            vega=vega,
            rho=rho,
            iv=iv
        )

    def implied_volatility(
        self,
        option_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: Literal['call', 'put']
    ) -> Optional[float]:
        """
        Calculate implied volatility from market price using Brent's method

        Returns:
            Implied volatility or None if calculation fails
        """

        if time_to_expiry <= 0 or option_price <= 0:
            return None

        # Check for arbitrage
        if option_type == 'call':
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)

        if option_price < intrinsic:
            return None

        def objective(sigma):
            """Error function: theoretical price - market price"""
            try:
                d1 = self._d1(spot, strike, time_to_expiry, sigma)
                d2 = d1 - sigma * np.sqrt(time_to_expiry)

                if option_type == 'call':
                    theo_price = self._call_price(
                        spot, strike, time_to_expiry, sigma, d1, d2
                    )
                else:
                    theo_price = self._put_price(
                        spot, strike, time_to_expiry, sigma, d1, d2
                    )

                return theo_price - option_price
            except:
                return np.inf

        try:
            # Search between 1% and 500% volatility
            iv = brentq(objective, 0.01, 5.0, maxiter=100, xtol=1e-5)
            return iv
        except ValueError:
            # Root not found in range
            return None

    def _d1(self, S, K, T, sigma):
        """Calculate d1 term in Black-Scholes"""
        return (
            np.log(S / K) +
            (self.risk_free_rate + 0.5 * sigma ** 2) * T
        ) / (sigma * np.sqrt(T))

    def _call_price(self, S, K, T, sigma, d1, d2):
        """Calculate call option price"""
        return (
            S * norm.cdf(d1) -
            K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        )

    def _put_price(self, S, K, T, sigma, d1, d2):
        """Calculate put option price"""
        return (
            K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) -
            S * norm.cdf(-d1)
        )

    def _call_theta(self, S, K, T, sigma, d1, d2):
        """Calculate call theta (per day)"""
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
            self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        ) / 365
        return theta

    def _put_theta(self, S, K, T, sigma, d1, d2):
        """Calculate put theta (per day)"""
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
            self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
        ) / 365
        return theta
```

**Tests:**

```python
# tests/test_greeks.py

import pytest
import numpy as np
from src.features.greeks import GreeksCalculator

@pytest.fixture
def calc():
    return GreeksCalculator(risk_free_rate=0.05)

def test_atm_call_price(calc):
    """Test ATM call pricing"""
    result = calc.calculate_greeks(
        spot=100,
        strike=100,
        time_to_expiry=0.25,  # 3 months
        iv=0.20,
        option_type='call'
    )

    # ATM call with 20% IV should be around $3-4
    assert 3.0 < result.price < 4.5
    # ATM call delta should be ~0.5
    assert 0.48 < result.delta < 0.52

def test_implied_volatility_recovery(calc):
    """Test that we can recover IV from price"""
    # Calculate theoretical price
    result = calc.calculate_greeks(
        spot=100, strike=105, time_to_expiry=0.5, iv=0.25, option_type='call'
    )

    # Recover IV from price
    recovered_iv = calc.implied_volatility(
        option_price=result.price,
        spot=100,
        strike=105,
        time_to_expiry=0.5,
        option_type='call'
    )

    assert recovered_iv is not None
    assert abs(recovered_iv - 0.25) < 0.001  # Within 0.1%

def test_put_call_parity(calc):
    """Test put-call parity: C - P = S - K*e^(-rT)"""
    S, K, T, sigma = 100, 100, 1.0, 0.20

    call = calc.calculate_greeks(S, K, T, sigma, 'call')
    put = calc.calculate_greeks(S, K, T, sigma, 'put')

    lhs = call.price - put.price
    rhs = S - K * np.exp(-calc.risk_free_rate * T)

    assert abs(lhs - rhs) < 0.01  # Within 1 cent
```

**Time estimate: 10 hours**

---

#### Day 8-10: Feature Engineering

**Implementation:**

```python
# src/features/regime.py

import pandas as pd
import numpy as np
from typing import Literal

RegimeType = Literal['low_vol', 'high_vol', 'iv_expansion', 'iv_contraction', 'neutral']

class RegimeDetector:
    """Detect market regime based on VIX and volatility metrics"""

    def __init__(self, config: dict):
        self.vix_low = config.get('vix_low_threshold', 15)
        self.vix_high = config.get('vix_high_threshold', 25)
        self.vix_rank_low = config.get('vix_rank_low', 0.3)
        self.vix_rank_high = config.get('vix_rank_high', 0.8)
        self.hv_iv_contraction = config.get('hv_iv_contraction', 0.6)
        self.hv_iv_expansion = config.get('hv_iv_expansion', 0.9)

    def detect_regime(
        self,
        vix: float,
        vix_rank: float,
        hv_iv_ratio: float
    ) -> RegimeType:
        """Classify current market regime"""

        # Low volatility
        if vix < self.vix_low and vix_rank < self.vix_rank_low:
            return 'low_vol'

        # High volatility
        if vix > self.vix_high or vix_rank > self.vix_rank_high:
            return 'high_vol'

        # IV expansion (HV catching up to IV)
        if hv_iv_ratio > self.hv_iv_expansion and vix_rank > 0.5:
            return 'iv_expansion'

        # IV contraction (IV overstated)
        if hv_iv_ratio < self.hv_iv_contraction and vix_rank < 0.5:
            return 'iv_contraction'

        return 'neutral'

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime-related features

        Input columns required:
        - Close: Stock close price
        - VIX: VIX index level

        Returns:
            DataFrame with additional columns:
            - rv_20: 20-day realized volatility
            - vix_rank: VIX percentile rank (252 days)
            - hv_iv_ratio: Realized vol / Implied vol
            - regime: Classified regime
        """

        df = data.copy()

        # Realized volatility (20-day)
        returns = np.log(df['Close'] / df['Close'].shift(1))
        df['rv_20'] = returns.rolling(20).std() * np.sqrt(252)

        # VIX rank (252-day percentile)
        df['vix_rank'] = df['VIX'].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
            if len(x) > 0 and x.max() > x.min() else np.nan
        )

        # HV/IV ratio
        df['hv_iv_ratio'] = df['rv_20'] / (df['VIX'] / 100)

        # Classify regime
        df['regime'] = df.apply(
            lambda row: self.detect_regime(
                row['VIX'],
                row['vix_rank'],
                row['hv_iv_ratio']
            ) if pd.notna(row['vix_rank']) else 'neutral',
            axis=1
        )

        return df
```

**Time estimate: 5 hours**

**Week 2 Milestone: ✅ Greeks calculator working, regime detection implemented**

---

### Week 3: Simple Strategy & Paper Trading

#### Day 11-13: IV Rank Strategy

**Implementation:** (See Architecture doc Section 2.4.2 for full code)

Key files to create:
- `src/strategies/base_strategy.py` - Abstract base class
- `src/strategies/iv_rank.py` - IV rank mean reversion strategy
- `tests/test_iv_rank_strategy.py` - Unit tests

**Time estimate: 15 hours**

---

#### Day 14-15: Broker Integration

**Implementation:**

```python
# src/trading/ibkr_api.py

from ib_insync import *
from typing import List, Dict, Optional
from loguru import logger
import time

class IBKRBroker:
    """Interactive Brokers API wrapper"""

    def __init__(self, paper: bool = True, port: int = None):
        """
        Initialize IBKR connection

        Args:
            paper: Use paper trading account
            port: TWS port (7497 for paper, 7496 for live)
        """
        self.ib = IB()
        self.paper = paper

        if port is None:
            port = 7497 if paper else 7496

        self.port = port
        self.connected = False

    def connect(self, host: str = '127.0.0.1', client_id: int = 1):
        """Connect to IBKR TWS/Gateway"""
        try:
            self.ib.connect(host, self.port, clientId=client_id)
            self.connected = True
            logger.info(f"Connected to IBKR ({'paper' if self.paper else 'live'})")
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise

    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_account_summary(self) -> Dict:
        """Get account information"""
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")

        account_values = self.ib.accountValues()

        summary = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'CashBalance', 'BuyingPower']:
                summary[av.tag] = float(av.value)

        return summary

    def place_vertical_spread(
        self,
        symbol: str,
        expiration: str,
        sell_strike: float,
        buy_strike: float,
        option_type: str,  # 'P' or 'C'
        quantity: int,
        limit_price: float
    ) -> str:
        """
        Place vertical spread order

        Args:
            symbol: Underlying symbol
            expiration: Expiration date (YYYYMMDD format)
            sell_strike: Strike to sell
            buy_strike: Strike to buy
            option_type: 'P' for put, 'C' for call
            quantity: Number of spreads
            limit_price: Limit price per spread

        Returns:
            Order ID
        """

        # Create option contracts
        sell_contract = Option(
            symbol, expiration, sell_strike, option_type, 'SMART'
        )

        buy_contract = Option(
            symbol, expiration, buy_strike, option_type, 'SMART'
        )

        # Qualify contracts (verify they exist)
        self.ib.qualifyContracts(sell_contract, buy_contract)

        # Create combo (spread)
        combo = Contract()
        combo.symbol = symbol
        combo.secType = 'BAG'
        combo.exchange = 'SMART'
        combo.currency = 'USD'

        leg1 = ComboLeg()
        leg1.conId = sell_contract.conId
        leg1.ratio = 1
        leg1.action = 'SELL'
        leg1.exchange = 'SMART'

        leg2 = ComboLeg()
        leg2.conId = buy_contract.conId
        leg2.ratio = 1
        leg2.action = 'BUY'
        leg2.exchange = 'SMART'

        combo.comboLegs = [leg1, leg2]

        # Create limit order
        order = LimitOrder(
            'BUY',  # Buying the spread (net credit or debit)
            quantity,
            limit_price
        )

        # Place order
        trade = self.ib.placeOrder(combo, order)

        logger.info(f"Placed spread order: {trade.order.orderId}")
        return str(trade.order.orderId)

    def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order"""
        trades = self.ib.trades()

        for trade in trades:
            if str(trade.order.orderId) == order_id:
                return {
                    'order_id': order_id,
                    'status': trade.orderStatus.status,
                    'filled': trade.orderStatus.filled,
                    'remaining': trade.orderStatus.remaining,
                    'avg_fill_price': trade.orderStatus.avgFillPrice
                }

        return {'order_id': order_id, 'status': 'NOT_FOUND'}

    def cancel_order(self, order_id: str):
        """Cancel an order"""
        trades = self.ib.trades()

        for trade in trades:
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {order_id}")
                return True

        logger.warning(f"Order {order_id} not found for cancellation")
        return False

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = self.ib.positions()

        return [
            {
                'contract': p.contract.symbol,
                'position': p.position,
                'avg_cost': p.avgCost,
                'market_value': p.marketValue,
                'unrealized_pnl': p.unrealizedPNL
            }
            for p in positions
        ]
```

**Setup Instructions:**

1. Download and install IBKR TWS or IB Gateway
2. Configure for paper trading:
   - Settings → API → Enable ActiveX and Socket Clients
   - Port: 7497 (paper) or 7496 (live)
   - Check "Allow connections from localhost"
3. Test connection:

```python
# scripts/test_ibkr_connection.py

from src.trading.ibkr_api import IBKRBroker

broker = IBKRBroker(paper=True)
broker.connect()

account = broker.get_account_summary()
print(f"Account Equity: ${account['NetLiquidation']:,.2f}")

positions = broker.get_positions()
print(f"Open Positions: {len(positions)}")

broker.disconnect()
```

**Time estimate: 10 hours**

**Week 3 Milestone: ✅ First paper trade placed successfully**

---

### Week 4: Risk Management & Basic Monitoring

#### Day 16-17: Risk Manager

**Implementation:** (See Architecture doc Section 2.5.1 for full code)

Key files:
- `src/trading/risk_manager.py`
- `tests/test_risk_manager.py`

**Time estimate: 10 hours**

---

#### Day 18-20: Monitoring Dashboard

**Simple Dashboard with Dash:**

```python
# src/monitoring/dashboard.py

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from src.data.storage import Database
import pandas as pd

app = dash.Dash(__name__)

db = Database("postgresql://localhost/options_trading")

app.layout = html.Div([
    html.H1("Options Trading Dashboard"),

    html.Div([
        html.Div([
            html.H3("Portfolio Value"),
            html.H2(id='portfolio-value', children="$0")
        ], className='metric-box'),

        html.Div([
            html.H3("Daily P&L"),
            html.H2(id='daily-pnl', children="$0")
        ], className='metric-box'),

        html.Div([
            html.H3("Open Positions"),
            html.H2(id='open-positions', children="0")
        ], className='metric-box'),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    dcc.Graph(id='equity-curve'),

    html.Div(id='positions-table'),

    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
])

@app.callback(
    [Output('portfolio-value', 'children'),
     Output('daily-pnl', 'children'),
     Output('open-positions', 'children'),
     Output('equity-curve', 'figure'),
     Output('positions-table', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch data from database
    query = "SELECT * FROM positions WHERE status = 'open'"
    positions = pd.read_sql(query, db.engine)

    # Calculate metrics
    portfolio_value = positions['current_price'].sum() if len(positions) > 0 else 0
    open_count = len(positions)

    # Equity curve
    query = "SELECT entry_time, pnl FROM trades ORDER BY entry_time"
    trades = pd.read_sql(query, db.engine)

    if len(trades) > 0:
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades['entry_time'],
            y=trades['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L'
        ))
        fig.update_layout(title='Equity Curve')
    else:
        fig = go.Figure()

    # Positions table
    if len(positions) > 0:
        table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in ['Symbol', 'Strike', 'Type', 'P&L']])),
            html.Tbody([
                html.Tr([
                    html.Td(positions.iloc[i]['symbol']),
                    html.Td(f"${positions.iloc[i]['strike']:.2f}"),
                    html.Td(positions.iloc[i]['option_type']),
                    html.Td(f"${positions.iloc[i]['unrealized_pnl']:.2f}",
                           style={'color': 'green' if positions.iloc[i]['unrealized_pnl'] > 0 else 'red'})
                ])
                for i in range(len(positions))
            ])
        ])
    else:
        table = html.P("No open positions")

    return (
        f"${portfolio_value:,.2f}",
        "$0",  # Daily P&L (calculate properly)
        str(open_count),
        fig,
        table
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

**Run dashboard:**
```bash
python src/monitoring/dashboard.py
# Open browser: http://localhost:8050
```

**Time estimate: 15 hours**

**Week 4 Milestone: ✅ Paper trading running daily with monitoring**

---

## 4. Phase 2: ML Integration (Weeks 5-8)

### Week 5: Feature Engineering (270+ Features)

**Goal:** Implement comprehensive feature set inspired by Bali et al.

**Implementation:** (See Architecture doc Section 2.2.3)

Key files:
- `src/features/options_features.py` - Options-specific features (40)
- `src/features/volatility_features.py` - Volatility features (30)
- `src/features/price_features.py` - Price & momentum (50)
- `src/features/volume_features.py` - Volume & OI (20)
- `src/features/temporal_features.py` - Time features (15)
- `src/features/derived_features.py` - Cross features (30)
- `src/features/feature_engineer.py` - Orchestrator

**Testing:**
```bash
python scripts/generate_features.py --symbol SPY --start 2020-01-01 --end 2024-12-31
# Should generate ~185 features for each date
```

**Time estimate: 25 hours**

**Week 5 Milestone: ✅ 185+ features generated for 5 years of data**

---

### Week 6: XGBoost Model

**Goal:** Train XGBoost model to predict option profitability

**Implementation:**

```python
# src/models/xgboost_model.py

import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost
import optuna
from typing import Dict, Tuple
import pandas as pd
import numpy as np

class XGBoostOptionModel:
    """XGBoost model for option return prediction"""

    def __init__(self, params: Dict = None):
        self.params = params or self._default_params()
        self.model = None
        self.feature_names = None

    def _default_params(self) -> Dict:
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_size: float = 0.2
    ) -> Dict:
        """
        Train model with validation split

        Args:
            X: Features
            y: Target (1 for profitable, 0 for unprofitable)
            validation_size: Fraction for validation

        Returns:
            Training metrics
        """

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=42, shuffle=False
        )

        self.feature_names = list(X.columns)

        # Train model
        self.model = xgb.XGBClassifier(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )

        # Calculate metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        val_pred = self.model.predict_proba(X_val)[:, 1]

        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

        metrics = {
            'train_auc': roc_auc_score(y_train, train_pred),
            'val_auc': roc_auc_score(y_val, val_pred),
            'train_accuracy': accuracy_score(y_train, train_pred > 0.5),
            'val_accuracy': accuracy_score(y_val, val_pred > 0.5),
            'val_precision': precision_score(y_val, val_pred > 0.5),
            'val_recall': recall_score(y_val, val_pred > 0.5)
        }

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of profit"""
        if self.model is None:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importances_

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'random_state': 42
            }

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )

            val_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, val_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.params.update(study.best_params)

        return study.best_params

    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained")

        mlflow.xgboost.save_model(self.model, path)

    def load_model(self, path: str):
        """Load model from disk"""
        self.model = mlflow.xgboost.load_model(path)
```

**Training Script:**

```python
# scripts/train_xgboost.py

import sys
sys.path.append('.')

from src.models.xgboost_model import XGBoostOptionModel
from src.data.storage import Database
import pandas as pd
import mlflow

def prepare_dataset():
    """Load features and create labels"""
    # Load features from storage
    features = pd.read_parquet('data/features/spy_features_2020_2024.parquet')

    # Create labels: Did option make money at expiration?
    # (This requires tracking option through expiration - implement based on your data)
    labels = features['profitable'].astype(int)  # Binary: 1 if profitable, 0 otherwise

    # Remove label from features
    X = features.drop(['profitable', 'expiration', 'option_id'], axis=1)
    y = labels

    return X, y

def main():
    # Load data
    print("Loading dataset...")
    X, y = prepare_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Positive samples: {y.sum()} ({y.mean():.1%})")

    # Initialize MLflow
    mlflow.set_experiment("xgboost-options-model")

    with mlflow.start_run():
        # Train model
        model = XGBoostOptionModel()

        print("Training model...")
        metrics = model.train(X, y)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            print(f"{key}: {value:.4f}")

        # Feature importance
        importance = model.get_feature_importance()
        print("\nTop 10 features:")
        print(importance.head(10))

        # Save model
        model.save_model('models/xgboost_v1')
        mlflow.xgboost.log_model(model.model, "model")

        print("\nModel saved!")

if __name__ == '__main__':
    main()
```

**Time estimate: 25 hours**

**Week 6 Milestone: ✅ XGBoost model trained with AUC > 0.65**

---

### Week 7: LSTM Model & Ensemble

*(Similar structure to Week 6, LSTM implementation)*

**Time estimate: 25 hours**

---

### Week 8: Walk-Forward Backtesting

**Goal:** Validate models with realistic walk-forward testing

**Implementation:**

```python
# src/backtesting/walk_forward.py

import pandas as pd
from typing import Dict, List
from src.models.xgboost_model import XGBoostOptionModel
from src.strategies.iv_rank import IVRankStrategy

def walk_forward_validation(
    data: pd.DataFrame,
    train_window_days: int = 252 * 2,  # 2 years
    test_window_days: int = 63,  # 3 months
    step_size_days: int = 21  # 1 month
) -> List[Dict]:
    """
    Perform walk-forward validation

    Process:
    1. Train model on train_window
    2. Test on test_window
    3. Slide forward by step_size
    4. Repeat
    """

    results = []

    for start in range(0, len(data) - train_window_days - test_window_days, step_size_days):
        train_end = start + train_window_days
        test_end = train_end + test_window_days

        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]

        print(f"Training: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Testing:  {test_data.index[0]} to {test_data.index[-1]}")

        # Train model
        X_train = train_data.drop(['profitable'], axis=1)
        y_train = train_data['profitable']

        model = XGBoostOptionModel()
        model.train(X_train, y_train, validation_size=0)

        # Test model
        X_test = test_data.drop(['profitable'], axis=1)
        y_test = test_data['profitable']

        predictions = model.predict_proba(X_test)

        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score

        metrics = {
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'test_auc': roc_auc_score(y_test, predictions),
            'test_accuracy': accuracy_score(y_test, predictions > 0.5),
            'test_samples': len(y_test)
        }

        results.append(metrics)

        print(f"Test AUC: {metrics['test_auc']:.4f}\n")

    return results
```

**Time estimate: 25 hours**

**Week 8 Milestone: ✅ Walk-forward Sharpe > 1.0**

---

## 5. Phase 3: Adaptive Trading (Weeks 9-12)

*(Detailed implementation for multi-strategy, advanced position management, optimization)*

**Time estimate per week: 25 hours**

**Week 12 Milestone: ✅ 30+ days profitable paper trading**

---

## 6. Phase 4: Production Deployment (Weeks 13-16)

*(Multi-asset expansion, 0DTE, real money, automation)*

**Time estimate per week: 25 hours**

**Week 16 Milestone: ✅ Autonomous trading system ready for scaling**

---

## 7. Code Module Specifications

### Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────┐
│                       Dependencies                            │
└──────────────────────────────────────────────────────────────┘

utils/
  ├── config.py (no dependencies)
  ├── logging.py (no dependencies)
  └── monitoring.py (prometheus_client)

data/
  ├── fetchers.py (yfinance, utils)
  ├── processors.py (pandas, fetchers)
  └── storage.py (sqlalchemy, pandas)

features/
  ├── greeks.py (scipy, numpy)
  ├── regime.py (pandas, numpy)
  ├── *_features.py (pandas, numpy, greeks)
  └── feature_engineer.py (all feature modules)

models/
  ├── base_model.py (abc, sklearn)
  ├── xgboost_model.py (xgboost, mlflow, optuna)
  ├── lstm_model.py (tensorflow, mlflow)
  └── ensemble.py (base_model, xgboost, lstm)

strategies/
  ├── base_strategy.py (abc, pandas)
  ├── iv_rank.py (base_strategy, features)
  └── adaptive.py (base_strategy, models, features)

trading/
  ├── risk_manager.py (pandas)
  ├── order_manager.py (broker_api, risk_manager)
  ├── position_manager.py (broker_api, storage)
  └── broker_api.py (ib_insync)

backtesting/
  ├── engine.py (pandas, numpy)
  ├── execution.py (pandas)
  └── metrics.py (pandas, numpy)

monitoring/
  ├── dashboard.py (dash, plotly, storage)
  └── alerts.py (smtplib, slack_sdk)
```

---

## 8. Testing Strategy

### Test Coverage Plan

```
Module                  Target    Priority
--------------------------------
greeks.py               95%       Critical
risk_manager.py         90%       Critical
feature_engineer.py     85%       High
strategies/*            80%       High
order_manager.py        85%       High
models/*                80%       Medium
fetchers.py             70%       Medium
backtesting/*           80%       Medium
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific module
pytest tests/test_greeks.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Integration tests only
pytest tests/integration/ -v -m integration
```

---

## 9. Deployment Checklist

### Phase 1 Deployment (Week 4)

```markdown
- [ ] All unit tests passing (>70% coverage)
- [ ] Data pipeline fetches SPY options daily
- [ ] Greeks calculation verified against known values
- [ ] Regime detection working on historical data
- [ ] IV Rank strategy generates signals
- [ ] IBKR paper trading connection stable
- [ ] First paper trade executed successfully
- [ ] Position tracking working
- [ ] Dashboard accessible and updating
- [ ] Logs being written correctly
- [ ] Database backups configured
```

### Phase 2 Deployment (Week 8)

```markdown
- [ ] 185+ features generated for all historical data
- [ ] XGBoost model trained with AUC > 0.65
- [ ] LSTM model trained (if applicable)
- [ ] Ensemble model outperforms baseline
- [ ] Walk-forward validation Sharpe > 1.0
- [ ] Feature importance aligns with expectations
- [ ] Model artifacts saved in MLflow
- [ ] ML predictions integrated with strategy
- [ ] Backtests run without errors
- [ ] Performance metrics tracked
```

### Phase 3 Deployment (Week 12)

```markdown
- [ ] Multi-strategy selection working
- [ ] Position management with profit targets
- [ ] Stop losses functioning correctly
- [ ] 30+ paper trades executed
- [ ] Win rate > 55%
- [ ] Sharpe ratio > 1.0 on paper account
- [ ] Max drawdown < 25%
- [ ] Performance attribution analysis complete
- [ ] Regime-specific performance validated
- [ ] All alerts working (drawdown, errors, etc.)
```

### Phase 4 Deployment (Week 16)

```markdown
- [ ] Multi-asset expansion (QQQ, IWM) successful
- [ ] 0DTE strategy implemented (if applicable)
- [ ] Real money deployment with $5k-10k
- [ ] 10+ real money trades executed
- [ ] Real money Sharpe > 1.0
- [ ] System runs autonomously
- [ ] Monitoring dashboard comprehensive
- [ ] Weekly performance reviews completed
- [ ] Ready to scale capital
```

---

## 10. Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Data feed failure | Medium | High | Implement retry logic, cache data, multiple sources |
| Model overfitting | High | High | Walk-forward validation, out-of-sample testing |
| Broker API issues | Medium | Medium | Circuit breaker, fallback to manual |
| Database corruption | Low | High | Daily backups, transaction logs |

### Trading Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large drawdown | Medium | Critical | 25% stop, position limits, regime-based sizing |
| Flash crash | Low | Critical | Stop losses, avoid 0DTE in high vol |
| Model degradation | Medium | High | Monitor metrics per regime, retrain monthly |
| Execution slippage | High | Medium | Conservative limit prices, avoid illiquid options |

### Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Feature engineering takes longer | Medium | Low | Start with 50 features, expand later |
| ML model doesn't improve baseline | Medium | Medium | Rule-based strategy is still viable |
| IBKR integration issues | Low | Medium | Test early, use Alpaca as backup |
| Paper trading unprofitable | High | High | Iterate quickly, adjust parameters |

---

## 11. Success Metrics

### Weekly Success Metrics

Track these every week:

```yaml
Week 1:
  - Data fetched for SPY: Yes/No
  - Greeks calculation accurate: Yes/No
  - Database storing data: Yes/No

Week 2:
  - Regime detection working: Yes/No
  - Features calculated correctly: Yes/No
  - Tests passing: >70% coverage

Week 3:
  - IV Rank strategy generates signals: Yes/No
  - First paper trade placed: Yes/No
  - Position tracking working: Yes/No

Week 4:
  - Dashboard accessible: Yes/No
  - Risk manager validating trades: Yes/No
  - System runs daily without errors: Yes/No

Week 5-8:
  - Features generated: 185+ features
  - Model AUC: > 0.65
  - Walk-forward Sharpe: > 1.0

Week 9-12:
  - Paper trades: > 30 trades
  - Win rate: > 55%
  - Sharpe: > 1.0

Week 13-16:
  - Real money deployed: $5k-10k
  - Real Sharpe: > 1.0
  - System autonomous: Yes/No
```

---

## 12. Daily Checklist (During Live Trading)

### Morning Routine (30 minutes)

```markdown
- [ ] Check system health (logs, errors)
- [ ] Review overnight news/events
- [ ] Check open positions
- [ ] Review market regime
- [ ] Ensure IBKR connection active
- [ ] Start trading system
```

### Evening Routine (30 minutes)

```markdown
- [ ] Review day's trades
- [ ] Update P&L tracking
- [ ] Check position Greeks
- [ ] Review signals generated
- [ ] Update metrics dashboard
- [ ] Plan next day
```

### Weekly Review (1 hour)

```markdown
- [ ] Calculate weekly metrics (Sharpe, DD, Win Rate)
- [ ] Analyze losing trades
- [ ] Review strategy performance by regime
- [ ] Check model predictions accuracy
- [ ] Update documentation
- [ ] Git commit all changes
```

---

## 13. Contact & Support

**Project Lead:** [Your Name]
**Email:** [Your Email]
**GitHub:** [Repository URL]

**Office Hours:** Flexible (side project)
**Sprint Reviews:** Friday EOD
**Retrospectives:** End of each phase

---

## 14. Appendices

### A. Environment Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd options-ml-trader

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Setup database
python scripts/init_database.py

# 5. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 6. Download IBKR TWS
# https://www.interactivebrokers.com/en/trading/tws.php

# 7. Verify setup
python scripts/validate_setup.py
```

### B. Useful Commands

```bash
# Run system
python main.py run --mode paper

# Run backtest
python main.py backtest --start 2020-01-01 --end 2024-12-31

# Train model
python scripts/train_xgboost.py

# Generate features
python scripts/generate_features.py --symbol SPY

# Start dashboard
python src/monitoring/dashboard.py

# Run tests
pytest tests/ -v

# Check code quality
black src/
ruff src/
mypy src/
```

### C. Troubleshooting

**Issue: IBKR connection fails**
- Solution: Ensure TWS is running, API enabled, correct port

**Issue: Data fetch times out**
- Solution: Check internet, yfinance rate limit, use caching

**Issue: Model predictions are NaN**
- Solution: Check for missing features, data quality

**Issue: Tests failing**
- Solution: Check dependencies, database connection

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Engineering | Initial implementation plan |

---

**Ready to start? Begin with Week 1, Day 1!** 🚀
