# Adaptive Options Trading System - Infrastructure & Data Sources

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Production Guide

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Training Infrastructure](#2-training-infrastructure)
3. [Database Options](#3-database-options)
4. [Deployment Architectures](#4-deployment-architectures)
5. [Cost Analysis](#5-cost-analysis)
6. [Setup Guides](#6-setup-guides)
7. [Performance Optimization](#7-performance-optimization)

---

## 1. Data Sources

### 1.1 Market Data Sources - Comparison Matrix

| Provider | Type | Cost | Rate Limits | Delay | Options Support | Recommended For |
|----------|------|------|-------------|-------|-----------------|-----------------|
| **yfinance** | Free | $0 | ~2000 req/hr | 15 min | ✅ Yes | Development, Paper Trading |
| **Alpaca** | Free/Paid | $0-99/mo | 200 req/min | Real-time* | ❌ No (stocks only) | Stock data, Paper trading |
| **IBKR** | Broker | $0** | Unlimited*** | Real-time | ✅ Yes | Paper & Live Trading |
| **Polygon.io** | Paid | $29-199/mo | Varies | Real-time | ✅ Yes | Production |
| **Alpha Vantage** | Free/Paid | $0-50/mo | 5 req/min (free) | Real-time | ⚠️ Limited | Historical only |
| **Databento** | Paid | Usage-based | High | Real-time | ✅ Yes | Institutional |
| **ThetaData** | Paid | $30-395/mo | High | End-of-day | ✅ Yes | Historical backtesting |
| **OptionMetrics** | Academic | $$$$ | N/A | N/A | ✅ Yes | Research (via university) |

\* Alpaca free tier has 15-min delay
\** IBKR requires active trading or $10/mo market data
\*** Subject to IBKR API limits

---

### 1.2 Detailed Data Source Analysis

#### **A. yfinance (Recommended for Development)**

**Pros:**
- ✅ Completely free
- ✅ Easy to use (Python library)
- ✅ Options chains with Greeks
- ✅ Historical data (5+ years)
- ✅ No API key required
- ✅ Wide coverage (all US options)

**Cons:**
- ❌ 15-minute delay (unsuitable for live trading)
- ❌ Unofficial API (could break)
- ❌ Rate limits (~2000 requests/hour)
- ❌ No guaranteed uptime
- ❌ Greeks may be stale

**Best for:**
- Phase 1-3: Development & backtesting
- Paper trading validation
- Historical analysis
- Model training

**Code Example:**
```python
import yfinance as yf

# Get options data
ticker = yf.Ticker("SPY")
expirations = ticker.options
chain = ticker.option_chain(expirations[2])  # ~30 days out

calls = chain.calls
puts = chain.puts

# Includes: bid, ask, volume, openInterest, impliedVolatility, delta, gamma, etc.
```

**Cost:** $0/month
**Setup time:** 5 minutes (pip install)

---

#### **B. Interactive Brokers (IBKR) - Recommended for Paper/Live Trading**

**Pros:**
- ✅ Real-time data (with account)
- ✅ Excellent options support
- ✅ Paper trading account available
- ✅ Direct trading integration
- ✅ Full order book depth
- ✅ Historical data via API
- ✅ Professional-grade infrastructure

**Cons:**
- ❌ Requires account setup
- ❌ TWS/Gateway software needed
- ❌ API can be complex
- ❌ Market data fees ($10/mo or waived with activity)
- ❌ Need $500+ to open account

**Best for:**
- Phase 3+: Paper trading
- Phase 4: Live trading
- Real-time data collection
- Production deployment

**Data Costs:**
- Free with paper account (real-time delayed 15 min)
- Free with live account meeting activity requirements
- OR $10/month for real-time quotes
- $4.50/month for additional exchanges

**Setup Requirements:**
1. Open IBKR account (paper or funded)
2. Download TWS or IB Gateway
3. Enable API access in settings
4. Install `ib_insync` Python library

**Code Example:**
```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading port

# Get options chain
contract = Option('SPY', '20250321', 450, 'C', 'SMART')
ib.qualifyContracts(contract)

# Get market data
ticker = ib.reqMktData(contract)
ib.sleep(2)

print(f"Bid: {ticker.bid}, Ask: {ticker.ask}")
```

**Cost:** $0-10/month (market data)
**Setup time:** 30-60 minutes (account + software)

---

#### **C. Polygon.io - Best Paid Alternative**

**Pros:**
- ✅ Real-time & historical data
- ✅ Excellent API documentation
- ✅ WebSocket support
- ✅ Options, stocks, forex, crypto
- ✅ High rate limits
- ✅ Greeks calculations
- ✅ Professional support

**Cons:**
- ❌ Paid only ($29-199/mo)
- ❌ Options data on higher tiers

**Pricing:**
- **Starter:** $29/mo - Stocks only, 5 API calls/min
- **Developer:** $99/mo - Options included, 100 API calls/min
- **Advanced:** $199/mo - Higher limits, WebSocket

**Best for:**
- Production trading (if not using IBKR data)
- High-frequency data collection
- Multiple assets

**Code Example:**
```python
from polygon import RESTClient

client = RESTClient(api_key="YOUR_API_KEY")

# Get options chain
options = client.list_options_contracts(
    underlying_ticker="SPY",
    expiration_date="2025-03-21"
)

for option in options:
    print(option)
```

**Cost:** $99/month (Developer tier for options)
**Setup time:** 10 minutes (API key signup)

---

#### **D. ThetaData - Best for Historical Backtesting**

**Pros:**
- ✅ 10+ years historical options data
- ✅ Every strike, every expiration
- ✅ Tick-level data available
- ✅ One-time download, own forever
- ✅ Clean, quality data

**Cons:**
- ❌ End-of-day data only (no real-time)
- ❌ Expensive for premium tiers
- ❌ Large data files (100+ GB)

**Pricing:**
- **Standard:** $30/mo - End-of-day data
- **Pro:** $100/mo - Intraday data
- **Elite:** $395/mo - Tick data

**Best for:**
- Serious backtesting (Phase 2)
- Research & model validation
- Historical analysis

**Cost:** $30-100/month
**Setup time:** 30 minutes (download client)

---

### 1.3 Recommended Data Strategy by Phase

#### **Phase 1 (Weeks 1-4): Development**
```yaml
Primary: yfinance (free)
Secondary: None needed
Storage: Local database
Cost: $0/month
```

#### **Phase 2 (Weeks 5-8): ML Training**
```yaml
Primary: yfinance (free) for historical
Alternative: ThetaData (optional, $30/mo for better data)
Storage: Local database + Parquet files
Cost: $0-30/month
```

#### **Phase 3 (Weeks 9-12): Paper Trading**
```yaml
Primary: IBKR paper account (free, 15-min delay)
Secondary: yfinance for supplemental data
Storage: Local/Cloud database
Cost: $0/month
```

#### **Phase 4 (Weeks 13-16): Live Trading**
```yaml
Primary: IBKR live account (real-time)
Backup: Polygon.io ($99/mo) OR continue IBKR only
Storage: Cloud database recommended
Cost: $0-10/month (IBKR data fees) + optional $99/mo (Polygon)
```

---

### 1.4 VIX & Reference Data Sources

| Data | Source | Cost | Update Frequency |
|------|--------|------|------------------|
| VIX Index | yfinance | Free | Daily |
| VIX Futures | IBKR | $0-10/mo | Real-time |
| SPY/QQQ/IWM | yfinance/IBKR | Free | Real-time/Daily |
| Risk-free Rate | FRED API | Free | Daily |
| Earnings Dates | yfinance | Free | Daily |
| Economic Calendar | TradingEconomics API | Free/Paid | Daily |

**Risk-Free Rate (FRED API):**
```python
import pandas_datareader as pdr

# Get 3-month Treasury rate
risk_free = pdr.DataReader('DGS3MO', 'fred', start='2020-01-01')
current_rate = risk_free.iloc[-1].values[0] / 100  # Convert to decimal
```

---

## 2. Training Infrastructure

### 2.1 Training Requirements by Model Type

| Model | RAM | CPU/GPU | Training Time | Storage |
|-------|-----|---------|---------------|---------|
| Logistic Regression | 8 GB | CPU (4 cores) | < 1 min | < 1 GB |
| XGBoost | 16 GB | CPU (8 cores) | 5-30 min | 1-5 GB |
| LSTM | 16-32 GB | GPU (8 GB VRAM) | 1-4 hours | 5-10 GB |
| Ensemble | 32 GB | CPU + GPU | 1-5 hours | 10-20 GB |

**Dataset Assumptions:** 5 years daily data, 185 features, 1M samples

---

### 2.2 Local Development Setup

#### **Option A: Laptop/Desktop (Recommended for Start)**

**Minimum Specs:**
- **RAM:** 16 GB (32 GB ideal)
- **CPU:** 4+ cores (8+ cores recommended)
- **Storage:** 100 GB SSD
- **GPU:** Optional (NVIDIA with CUDA for LSTM)
- **OS:** Linux/macOS/Windows (Linux preferred)

**Pros:**
- ✅ No recurring costs
- ✅ Full control
- ✅ Easy debugging
- ✅ No network latency

**Cons:**
- ❌ Limited by hardware
- ❌ No automatic backups
- ❌ Must run manually

**Cost:** $0/month (hardware you own)
**Setup time:** 2-3 hours

**Recommended Laptop:**
- MacBook Pro M2/M3 (16 GB+ RAM) - $2000+
- ThinkPad P-series (32 GB RAM, NVIDIA GPU) - $1500+
- Custom desktop (32 GB RAM, RTX 3060) - $1200+

---

#### **Option B: Local Server (For Serious Development)**

**Build a dedicated ML server:**

| Component | Spec | Cost |
|-----------|------|------|
| CPU | AMD Ryzen 9 5900X (12-core) | $300 |
| RAM | 64 GB DDR4 | $200 |
| GPU | NVIDIA RTX 4060 Ti (16 GB) | $500 |
| Storage | 1 TB NVMe SSD | $100 |
| Motherboard | B550 ATX | $150 |
| PSU | 750W Gold | $100 |
| Case | ATX Mid-tower | $80 |
| **Total** | | **~$1430** |

**Pros:**
- ✅ High performance
- ✅ No monthly costs
- ✅ Can run 24/7
- ✅ Future-proof

**Cons:**
- ❌ Upfront cost
- ❌ Power consumption (~$20-40/mo)
- ❌ Maintenance required

**Operating Costs:** $20-40/month (electricity)

---

### 2.3 Cloud Training Options

#### **Option A: AWS EC2 (Most Flexible)**

**Recommended Instances:**

| Instance Type | vCPU | RAM | GPU | Cost/Hour | Cost/Month (24/7) | Use Case |
|--------------|------|-----|-----|-----------|-------------------|----------|
| **t3.xlarge** | 4 | 16 GB | None | $0.17 | ~$122 | Development |
| **t3.2xlarge** | 8 | 32 GB | None | $0.33 | ~$240 | XGBoost training |
| **g4dn.xlarge** | 4 | 16 GB | T4 (16GB) | $0.53 | ~$382 | LSTM training |
| **c5.4xlarge** | 16 | 32 GB | None | $0.68 | ~$490 | Heavy XGBoost |

**Cost Optimization:**
- Use **Spot Instances** (70% discount, interruptible)
- **Stop instance** when not training (only pay for storage)
- **Reserved Instances** if running 24/7 (40% discount)

**Realistic Usage Pattern:**
```
Development (t3.xlarge): 5 hours/day × 30 days = 150 hrs/mo
Cost: 150 × $0.17 = $25.50/month

Training (g4dn.xlarge): 10 hours/week × 4 = 40 hrs/mo
Cost: 40 × $0.53 = $21.20/month

Storage (100 GB EBS): $10/month

Total: ~$57/month
```

**Pros:**
- ✅ Pay only for usage
- ✅ Scale up/down easily
- ✅ Many instance types
- ✅ Spot instances save money

**Cons:**
- ❌ Can get expensive if left running
- ❌ Setup complexity
- ❌ Data transfer costs

---

#### **Option B: Google Cloud Platform (GCP)**

**Recommended VMs:**

| Machine Type | vCPU | RAM | GPU | Cost/Hour | Use Case |
|--------------|------|-----|-----|-----------|----------|
| **n1-standard-4** | 4 | 15 GB | None | $0.19 | Development |
| **n1-highmem-8** | 8 | 52 GB | None | $0.47 | XGBoost |
| **n1-standard-4 + T4** | 4 | 15 GB | T4 | $0.51 | LSTM |

**Pros:**
- ✅ Good free tier ($300 credit)
- ✅ Easy ML integrations
- ✅ Preemptible VMs (cheaper)

**Cons:**
- ❌ Similar costs to AWS
- ❌ Less mature than AWS

**Cost:** Similar to AWS (~$50-100/month for part-time use)

---

#### **Option C: Paperspace Gradient (ML-Focused)**

**Pre-configured ML environments:**

| Instance | Specs | Cost/Hour | Use Case |
|----------|-------|-----------|----------|
| **Free** | 8 GB RAM, CPU | $0 | Learning |
| **P4000** | 16 GB RAM, 8 GB GPU | $0.51 | Training |
| **P5000** | 30 GB RAM, 16 GB GPU | $0.78 | Heavy training |

**Pros:**
- ✅ Free tier for experiments
- ✅ Pre-installed ML libraries
- ✅ Jupyter notebooks
- ✅ Simple interface

**Cons:**
- ❌ Limited to ML workloads
- ❌ Less control than AWS/GCP

**Cost:** $0-50/month (occasional training)

---

#### **Option D: Colab Pro (Easiest Start)**

**Google Colab:**
- **Free tier:** CPU/GPU access (limited hours)
- **Colab Pro:** $10/month (better GPU, more hours)
- **Colab Pro+:** $50/month (even better resources)

**Pros:**
- ✅ Extremely easy to start
- ✅ Jupyter notebook interface
- ✅ Pre-installed libraries
- ✅ Cheap

**Cons:**
- ❌ Not suitable for production
- ❌ Limited storage
- ❌ Session timeouts

**Cost:** $0-10/month
**Best for:** Experimentation & initial model training

---

### 2.4 Recommended Training Strategy

**Phase 1-2 (Weeks 1-8): Local Development**
- Use your laptop/desktop
- Train XGBoost locally (fast enough)
- Optional: Use Colab Pro for LSTM experiments

**Phase 3 (Weeks 9-12): Hybrid**
- Local for daily operations
- Cloud (AWS Spot or Colab) for intensive training

**Phase 4+ (Production): Cloud**
- Small always-on instance (t3.medium) for trading
- Larger instance (g4dn.xlarge) for weekly retraining

**Estimated Costs:**
- Weeks 1-8: $0-10/month (Colab Pro optional)
- Weeks 9-12: $20-50/month (occasional cloud)
- Production: $50-150/month (depends on scale)

---

## 3. Database Options

### 3.1 Database Comparison

| Database | Type | Cost | Setup | Scalability | Best For |
|----------|------|------|-------|-------------|----------|
| **SQLite** | File-based | Free | 5 min | Single user | Development |
| **PostgreSQL (Local)** | Relational | Free | 30 min | Medium | Phase 1-3 |
| **PostgreSQL (RDS)** | Managed | $15-200/mo | 15 min | High | Production |
| **MySQL** | Relational | Free | 30 min | Medium | Alternative |
| **TimescaleDB** | Time-series | Free/Paid | 45 min | High | Large datasets |
| **MongoDB** | Document | Free/Paid | 30 min | High | Flexible schema |

---

### 3.2 SQLite (Recommended for Phase 1)

**Pros:**
- ✅ Zero setup (file-based)
- ✅ No server needed
- ✅ Perfect for single user
- ✅ Fast for < 100 GB data
- ✅ Built into Python

**Cons:**
- ❌ No concurrent writes
- ❌ Limited for production
- ❌ No remote access

**Setup:**
```python
import sqlite3

conn = sqlite3.connect('data/trading.db')

# Create tables
conn.execute("""
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    entry_time TIMESTAMP,
    pnl REAL
)
""")
```

**Storage Requirements:**
- 1 year options data: ~5 GB
- 5 years: ~25 GB
- Trades/positions: < 1 GB

**Cost:** $0
**Setup time:** 5 minutes

---

### 3.3 PostgreSQL Local (Recommended for Phase 2-3)

**Pros:**
- ✅ Production-grade
- ✅ ACID compliance
- ✅ Complex queries
- ✅ Full-text search
- ✅ JSON support

**Cons:**
- ❌ Requires server setup
- ❌ Maintenance needed

**Setup (Ubuntu/Debian):**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres psql
postgres=# CREATE DATABASE options_trading;
postgres=# CREATE USER trader WITH PASSWORD 'your_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE options_trading TO trader;
postgres=# \q

# Test connection
psql -U trader -d options_trading -h localhost
```

**Setup (macOS):**
```bash
# Install via Homebrew
brew install postgresql@14
brew services start postgresql@14

# Create database
createdb options_trading
```

**Setup (Docker):**
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: options_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: secure_password
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

```bash
docker-compose up -d
```

**Python Connection:**
```python
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://trader:secure_password@localhost:5432/options_trading'
)
```

**Cost:** $0 (self-hosted)
**Setup time:** 30 minutes

---

### 3.4 PostgreSQL on AWS RDS (Recommended for Production)

**Pricing (us-east-1):**

| Instance | vCPU | RAM | Storage | Cost/Month |
|----------|------|-----|---------|------------|
| **db.t3.micro** | 2 | 1 GB | 20 GB | $15 |
| **db.t3.small** | 2 | 2 GB | 50 GB | $35 |
| **db.t3.medium** | 2 | 4 GB | 100 GB | $70 |
| **db.m5.large** | 2 | 8 GB | 100 GB | $140 |

**Pros:**
- ✅ Fully managed (no maintenance)
- ✅ Automatic backups
- ✅ High availability option
- ✅ Automatic scaling

**Cons:**
- ❌ Monthly cost
- ❌ Network latency

**Setup:**
1. AWS Console → RDS → Create Database
2. Choose PostgreSQL 14
3. Select instance size (start with t3.small)
4. Configure security group (allow port 5432)
5. Get endpoint: `your-db.abc123.us-east-1.rds.amazonaws.com`

**Cost:** $35-70/month (recommended for Phase 4)
**Setup time:** 15 minutes

---

### 3.5 TimescaleDB (For Large Time-Series)

**When to use:**
- > 1 TB of historical data
- High-frequency tick data
- Complex time-series queries

**Setup:**
```bash
# Install TimescaleDB extension on PostgreSQL
sudo apt install timescaledb-postgresql-14

# Enable extension
psql -d options_trading
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Convert table to hypertable
SELECT create_hypertable('options_chains', 'fetch_time');
```

**Pros:**
- ✅ Optimized for time-series
- ✅ Compression (10x storage savings)
- ✅ Fast time-range queries

**Cons:**
- ❌ More complex setup
- ❌ PostgreSQL knowledge required

**Cost:** Free (self-hosted) or Timescale Cloud ($25-300/mo)

---

### 3.6 Recommended Database Strategy

**Phase 1 (Weeks 1-4):**
```yaml
Database: SQLite
Storage: Local file (data/trading.db)
Backup: Git + manual copies
Cost: $0
```

**Phase 2-3 (Weeks 5-12):**
```yaml
Database: PostgreSQL (Local or Docker)
Storage: Local server
Backup: pg_dump daily → S3 ($1/mo)
Cost: $0-1/month
```

**Phase 4 (Production):**
```yaml
Database: PostgreSQL on AWS RDS (t3.small)
Storage: 50-100 GB
Backup: Automatic (RDS snapshots)
Cost: $35-70/month
```

---

## 4. Deployment Architectures

### 4.1 Development Environment (Weeks 1-8)

```
┌─────────────────────────────────────────────────┐
│           Your Laptop/Desktop                    │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  Python App                               │  │
│  │  - Data fetchers                          │  │
│  │  - Feature engineering                    │  │
│  │  - Model training                         │  │
│  │  - Backtesting                            │  │
│  └────────┬─────────────────────────────────┘  │
│           │                                      │
│  ┌────────▼─────────┐  ┌──────────────────┐   │
│  │ SQLite Database  │  │ Parquet Files    │   │
│  │ (data/trading.db)│  │ (data/features/) │   │
│  └──────────────────┘  └──────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  Jupyter Lab (Analysis)                  │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │  yfinance   │ (Free data)
    └─────────────┘
```

**Specs:**
- RAM: 16 GB
- Storage: 50 GB
- Network: Home internet

**Cost:** $0/month

---

### 4.2 Paper Trading Environment (Weeks 9-12)

```
┌─────────────────────────────────────────────────┐
│        Your Computer (Local or Cloud)           │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  Trading Application                      │  │
│  │  - Signal generation                      │  │
│  │  - Order management                       │  │
│  │  - Position tracking                      │  │
│  │  - Risk management                        │  │
│  └────┬────────────────────────────┬─────────┘  │
│       │                            │             │
│  ┌────▼─────────┐         ┌────────▼─────────┐ │
│  │ PostgreSQL   │         │  MLflow Server   │ │
│  │ (Local/RDS)  │         │  (Models)        │ │
│  └──────────────┘         └──────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  Monitoring Dashboard (Dash/Grafana)     │  │
│  └──────────────────────────────────────────┘  │
└─────────────┬─────────────────────┬──────────────┘
              │                     │
              ▼                     ▼
      ┌──────────────┐      ┌──────────────┐
      │ IBKR Paper   │      │  yfinance    │
      │  Trading     │      │  (Backup)    │
      └──────────────┘      └──────────────┘
```

**Specs:**
- RAM: 16-32 GB
- Storage: 100 GB
- Database: Local PostgreSQL or RDS (t3.small)

**Cost:** $0-35/month

---

### 4.3 Live Trading Environment (Production)

```
┌─────────────────────────────────────────────────────────┐
│                    AWS Cloud                             │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  EC2 Instance (t3.medium)                        │   │
│  │                                                   │   │
│  │  ┌────────────────────────────────────────────┐ │   │
│  │  │  Trading Application (24/7)                │ │   │
│  │  │  - Cron jobs for data collection          │ │   │
│  │  │  - Signal generation every hour            │ │   │
│  │  │  - Position monitoring                     │ │   │
│  │  │  - Automated trading                       │ │   │
│  │  └────────────────────────────────────────────┘ │   │
│  └───────┬──────────────────────────────┬──────────┘   │
│          │                              │               │
│  ┌───────▼──────────┐          ┌────────▼───────────┐  │
│  │  RDS PostgreSQL  │          │   S3 Storage       │  │
│  │  (t3.small)      │          │   - Backups        │  │
│  │  - Trades        │          │   - Logs           │  │
│  │  - Positions     │          │   - Model versions │  │
│  └──────────────────┘          └────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │  CloudWatch Monitoring + Alerts                   │ │
│  └───────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
       ┌──────────────┐       ┌──────────────┐
       │ IBKR Live    │       │ Polygon.io   │
       │ Trading      │       │ (Optional)   │
       └──────────────┘       └──────────────┘
               │
               ▼
       ┌──────────────┐
       │  Your Phone  │
       │  (Alerts)    │
       └──────────────┘
```

**Infrastructure:**
- **Compute:** EC2 t3.medium (2 vCPU, 4 GB RAM)
- **Database:** RDS t3.small (2 GB RAM, 50 GB storage)
- **Storage:** S3 (backups, logs)
- **Monitoring:** CloudWatch + SNS alerts

**High Availability (Optional):**
- Auto Scaling Group (2+ EC2 instances)
- Read replica for database
- Multi-AZ deployment

**Cost Breakdown:**
- EC2 t3.medium: $30/month (on-demand) or $18/month (reserved)
- RDS t3.small: $35/month
- S3 storage: $5/month
- Data transfer: $5/month
- CloudWatch: $5/month
- **Total: $80-85/month**

With reserved instances and optimizations: **$60-70/month**

---

### 4.4 Alternative: Hybrid Architecture (Recommended)

**Best of both worlds:**

```
┌─────────────────────────────────┐
│   Home Server (Always On)       │
│   - Trading application         │
│   - Local PostgreSQL            │
│   - IBKR Gateway                │
│   Cost: $20/mo (electricity)    │
└────────┬────────────────────────┘
         │
         ├─── Daily backups ────►  AWS S3 ($5/mo)
         │
         └─── ML training  ────►  AWS EC2 Spot ($20/mo occasional)
```

**Pros:**
- ✅ Low monthly cost ($25-30/mo)
- ✅ Low latency (local)
- ✅ Full control
- ✅ Cloud backups

**Cons:**
- ❌ Internet dependent
- ❌ Manual maintenance

---

## 5. Cost Analysis

### 5.1 Total Cost Breakdown by Phase

#### **Phase 1: Development (Weeks 1-4)**

| Item | Option | Cost |
|------|--------|------|
| Data | yfinance | $0 |
| Compute | Your laptop | $0 |
| Database | SQLite | $0 |
| Training | Local | $0 |
| **Total** | | **$0/month** |

---

#### **Phase 2: ML Training (Weeks 5-8)**

| Item | Option | Cost |
|------|--------|------|
| Data | yfinance + optional ThetaData | $0-30/mo |
| Compute | Local + optional Colab Pro | $0-10/mo |
| Database | PostgreSQL (local) | $0 |
| Training | Local/Cloud mix | $0-50/mo |
| **Total** | | **$0-90/month** |

**Recommended: $0-10/month** (yfinance + local compute + optional Colab)

---

#### **Phase 3: Paper Trading (Weeks 9-12)**

| Item | Option | Cost |
|------|--------|------|
| Data | IBKR paper (free) + yfinance | $0 |
| Compute | Local | $0 |
| Database | PostgreSQL (local) | $0 |
| Monitoring | Self-hosted dashboard | $0 |
| **Total** | | **$0/month** |

---

#### **Phase 4: Live Trading (Production)**

**Option A: Cloud (AWS)**

| Item | Cost |
|------|------|
| EC2 t3.medium | $30/mo |
| RDS t3.small | $35/mo |
| S3 storage | $5/mo |
| Monitoring | $5/mo |
| IBKR data | $0-10/mo |
| Optional: Polygon | $0-99/mo |
| **Total (Min)** | **$75-85/month** |
| **Total (Max)** | **$174-184/month** |

**Option B: Hybrid (Home + Cloud)**

| Item | Cost |
|------|------|
| Home server electricity | $20/mo |
| S3 backups | $5/mo |
| Occasional EC2 training | $10/mo |
| IBKR data | $0-10/mo |
| **Total** | **$35-45/month** |

**Option C: Fully Local**

| Item | Cost |
|------|------|
| Electricity | $20/mo |
| IBKR data | $0-10/mo |
| **Total** | **$20-30/month** |

---

### 5.2 Cost Comparison: First Year

| Approach | Setup Cost | Monthly Avg | Year 1 Total |
|----------|-----------|-------------|--------------|
| **Minimal (Local)** | $0 | $5 | $60 |
| **Recommended (Hybrid)** | $0 | $30 | $360 |
| **Cloud (AWS)** | $0 | $80 | $960 |
| **Premium (All paid)** | $0 | $200 | $2,400 |

**Recommended path:** Start minimal ($0-10/mo), scale to hybrid ($30-40/mo) in production

---

### 5.3 Break-Even Analysis

**If trading with $10,000 capital:**
- Target return: 25%/year = $2,500
- Infrastructure cost (hybrid): $360/year
- **Break-even after costs:** $2,140 profit = 21.4% return

**If trading with $50,000 capital:**
- Target return: 25%/year = $12,500
- Infrastructure cost (cloud): $960/year
- **Break-even after costs:** $11,540 profit = 23% return

**Conclusion:** Infrastructure costs are negligible compared to potential returns once capital scales.

---

## 6. Setup Guides

### 6.1 Complete Local Development Setup

#### **Step 1: System Prerequisites (Ubuntu 22.04)**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install build tools
sudo apt install build-essential git curl -y

# Install system libraries for Python packages
sudo apt install libpq-dev python3-dev -y
```

#### **Step 2: Project Setup**

```bash
# Create project directory
mkdir ~/options-ml-trader
cd ~/options-ml-trader

# Initialize git
git init

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Data
yfinance==0.2.38
pandas==2.2.0
numpy==1.26.4
SQLAlchemy==2.0.27

# ML
scikit-learn==1.4.0
xgboost==2.0.3
lightgbm==4.3.0
optuna==3.5.0
mlflow==2.10.2

# Deep Learning (optional)
tensorflow==2.15.0
torch==2.1.2

# Greeks & Math
scipy==1.12.0
py_vollib==1.0.1

# Trading
ib_insync==0.9.86

# Utils
pydantic==2.6.1
python-dotenv==1.0.1
loguru==0.7.2
typer==0.9.0

# Monitoring
prometheus-client==0.19.0
dash==2.14.2
plotly==5.18.0

# Testing
pytest==8.0.0
pytest-cov==4.1.0
EOF

# Install dependencies
pip install -r requirements.txt
```

#### **Step 3: Database Setup**

```bash
# Start PostgreSQL
sudo service postgresql start

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE options_trading;
CREATE USER trader WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE options_trading TO trader;
\q
EOF

# Test connection
psql -U trader -d options_trading -h localhost -c "SELECT version();"
```

#### **Step 4: Environment Variables**

```bash
# Create .env file
cat > .env << 'EOF'
# Database
DATABASE_URL=postgresql://trader:secure_password_here@localhost:5432/options_trading

# IBKR
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IBKR_PORT=7497  # Paper trading

# Risk-free rate (update periodically)
RISK_FREE_RATE=0.045

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns

# Logging
LOG_LEVEL=INFO
EOF

# Add .env to .gitignore
echo ".env" >> .gitignore
```

#### **Step 5: Project Structure**

```bash
# Create directory structure
mkdir -p src/{data,features,models,strategies,trading,backtesting,monitoring,utils}
mkdir -p tests/{unit,integration}
mkdir -p notebooks
mkdir -p configs
mkdir -p data/{cache,features}
mkdir -p logs
mkdir -p models

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;
```

#### **Step 6: Verify Setup**

```bash
# Create verification script
cat > scripts/verify_setup.py << 'EOF'
#!/usr/bin/env python3
import sys

def check_imports():
    """Check all required packages import correctly"""
    required = [
        ('yfinance', 'yf'),
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('xgboost', 'xgb'),
        ('scipy', 'scipy'),
        ('sqlalchemy', 'sqlalchemy'),
        ('mlflow', 'mlflow'),
    ]

    print("Checking package imports...")
    for package, alias in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False

    return True

def check_database():
    """Check database connection"""
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv

    load_dotenv()
    db_url = os.getenv('DATABASE_URL')

    if not db_url:
        print("❌ DATABASE_URL not set in .env")
        return False

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ Database connection")
            return True
    except Exception as e:
        print(f"❌ Database connection: {e}")
        return False

def check_data_fetch():
    """Check data fetching works"""
    import yfinance as yf

    try:
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="5d")
        if len(hist) > 0:
            print("✅ Data fetching (yfinance)")
            return True
        else:
            print("❌ Data fetching: No data returned")
            return False
    except Exception as e:
        print(f"❌ Data fetching: {e}")
        return False

if __name__ == '__main__':
    results = [
        check_imports(),
        check_database(),
        check_data_fetch()
    ]

    if all(results):
        print("\n🎉 All checks passed! Setup is complete.")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)
EOF

chmod +x scripts/verify_setup.py
python scripts/verify_setup.py
```

**Expected output:**
```
Checking package imports...
✅ yfinance
✅ pandas
✅ numpy
✅ sklearn
✅ xgboost
✅ scipy
✅ sqlalchemy
✅ mlflow
✅ Database connection
✅ Data fetching (yfinance)

🎉 All checks passed! Setup is complete.
```

**Setup time:** 1-2 hours
**Cost:** $0

---

### 6.2 AWS Cloud Setup (Production)

#### **Step 1: Create EC2 Instance**

```bash
# Via AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 LTS
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx \
  --user-data file://user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=options-trader}]'
```

**user-data.sh:**
```bash
#!/bin/bash
apt update && apt upgrade -y
apt install python3.11 python3.11-venv postgresql-client git -y

# Auto-start trading app on boot
cat > /etc/systemd/system/trading-app.service << EOF
[Unit]
Description=Options Trading Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/options-ml-trader
ExecStart=/home/ubuntu/options-ml-trader/venv/bin/python main.py run --mode live
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable trading-app
```

#### **Step 2: Create RDS Database**

```bash
# Via AWS CLI
aws rds create-db-instance \
  --db-instance-identifier options-trading-db \
  --db-instance-class db.t3.small \
  --engine postgres \
  --master-username trader \
  --master-user-password SecurePassword123! \
  --allocated-storage 50 \
  --backup-retention-period 7 \
  --vpc-security-group-ids sg-xxxxx \
  --db-subnet-group-name my-subnet-group \
  --publicly-accessible false
```

#### **Step 3: Setup S3 Backups**

```bash
# Create S3 bucket
aws s3 mb s3://options-trader-backups

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket options-trader-backups \
  --versioning-configuration Status=Enabled

# Setup lifecycle policy (delete old backups after 90 days)
cat > lifecycle.json << EOF
{
  "Rules": [{
    "Id": "DeleteOldBackups",
    "Status": "Enabled",
    "Expiration": {
      "Days": 90
    }
  }]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket options-trader-backups \
  --lifecycle-configuration file://lifecycle.json
```

#### **Step 4: Deploy Application**

```bash
# SSH to EC2 instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Clone repository
git clone https://github.com/your-username/options-ml-trader.git
cd options-ml-trader

# Setup (same as local)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure .env with RDS endpoint
cat > .env << EOF
DATABASE_URL=postgresql://trader:SecurePassword123!@options-trading-db.abc123.us-east-1.rds.amazonaws.com:5432/options_trading
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IBKR_PORT=7496  # Live trading
EOF

# Start service
sudo systemctl start trading-app
sudo systemctl status trading-app
```

**Setup time:** 2-3 hours
**Cost:** $75-85/month

---

## 7. Performance Optimization

### 7.1 Database Optimization

**Indexes for fast queries:**
```sql
-- Options chains
CREATE INDEX idx_options_symbol_exp ON options_chains(symbol, expiration);
CREATE INDEX idx_options_fetch_time ON options_chains(fetch_time);
CREATE INDEX idx_options_dte ON options_chains(dte);

-- Trades
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_trades_status ON trades(status);

-- Positions
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
```

**Query optimization:**
```sql
-- Use EXPLAIN to check query plans
EXPLAIN ANALYZE
SELECT * FROM options_chains
WHERE symbol = 'SPY'
AND dte BETWEEN 3 AND 7
AND fetch_time = (SELECT MAX(fetch_time) FROM options_chains);
```

**Partitioning (for large datasets):**
```sql
-- Partition options_chains by month
CREATE TABLE options_chains_2024_01 PARTITION OF options_chains
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

---

### 7.2 Data Fetching Optimization

**Parallel fetching:**
```python
from concurrent.futures import ThreadPoolExecutor

def fetch_multiple_symbols(symbols):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_options, sym) for sym in symbols]
        results = [f.result() for f in futures]
    return results
```

**Caching:**
```python
import redis
from functools import wraps

cache = redis.Redis(host='localhost', port=6379)

def cache_result(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            cached = cache.get(key)
            if cached:
                return pickle.loads(cached)

            result = func(*args, **kwargs)
            cache.setex(key, ttl, pickle.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(ttl=3600)  # Cache for 1 hour
def fetch_options_chain(symbol, expiration):
    # Expensive operation
    pass
```

---

### 7.3 ML Training Optimization

**Use Parquet for fast feature loading:**
```python
# Save features
df.to_parquet('data/features/spy_features.parquet', compression='snappy')

# Load features (10x faster than CSV)
df = pd.read_parquet('data/features/spy_features.parquet')
```

**Parallel XGBoost training:**
```python
params = {
    'n_jobs': -1,  # Use all CPU cores
    'tree_method': 'hist',  # Faster histogram method
}
```

**GPU acceleration for LSTM:**
```python
import tensorflow as tf

# Use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

---

## 8. Monitoring & Alerts

### 8.1 System Health Monitoring

**CloudWatch Metrics (if using AWS):**
- CPU utilization
- Memory usage
- Disk I/O
- Network traffic
- Application errors

**Custom Metrics:**
```python
from prometheus_client import Gauge, Counter, Histogram

# Define metrics
portfolio_value = Gauge('portfolio_value_usd', 'Portfolio value in USD')
signals_generated = Counter('signals_generated_total', 'Total signals generated')
order_latency = Histogram('order_latency_seconds', 'Order placement latency')

# Update metrics
portfolio_value.set(10500.00)
signals_generated.inc()
order_latency.observe(0.15)
```

### 8.2 Alerts Configuration

**Email alerts via SES:**
```python
import boto3

ses = boto3.client('ses', region_name='us-east-1')

def send_alert(subject, message):
    ses.send_email(
        Source='alerts@yourdomain.com',
        Destination={'ToAddresses': ['your@email.com']},
        Message={
            'Subject': {'Data': subject},
            'Body': {'Text': {'Data': message}}
        }
    )

# Alert on drawdown
if current_drawdown > 0.20:
    send_alert(
        subject="⚠️ High Drawdown Alert",
        message=f"Drawdown reached {current_drawdown:.1%}"
    )
```

---

## Appendix: Quick Reference

### Data Sources Quick Pick

| Phase | Data Source | Cost | Why |
|-------|-------------|------|-----|
| Dev (1-8 weeks) | yfinance | $0 | Free, good enough |
| Paper (9-12 weeks) | IBKR Paper | $0 | Real trading simulation |
| Live (13+ weeks) | IBKR Live | $0-10/mo | Required for trading |

### Infrastructure Quick Pick

| Phase | Compute | Database | Cost |
|-------|---------|----------|------|
| Dev | Your laptop | SQLite | $0 |
| Paper | Your laptop | PostgreSQL (local) | $0 |
| Live | Home server OR AWS t3.medium | PostgreSQL (local/RDS) | $20-80/mo |

### Recommended Budget

| Item | Monthly Cost |
|------|--------------|
| Development (Weeks 1-8) | $0-10 |
| Paper Trading (Weeks 9-12) | $0 |
| Live Trading (Production) | $30-80 |

**Total Year 1:** $300-600 (with scale-up to production)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Infrastructure Team | Initial infrastructure guide |

---

**Next: Start Week 1 Setup** 🚀
