# ML Options Trading System - How To Guide

## Quick Start Checklist

### Initial Setup

- [ ] **Clone repository**
  ```bash
  git clone <repo-url>
  cd ml-options-trading
  ```

- [ ] **Create `.env` file** with API keys
  ```bash
  # Required for sentiment & earnings features
  FINNHUB_API_KEY=your_key_here

  # Required for macro features (yield curves, inflation, etc.)
  FRED_API_KEY=your_key_here

  # Optional (for future use)
  # TRADIER_API_KEY=your_key_here
  # POLYGON_API_KEY=your_key_here
  ```

  Get free keys:
  - FINNHUB: https://finnhub.io/register
  - FRED: https://fred.stlouisfed.org/docs/api/api_key.html

- [ ] **Build Docker image**
  ```bash
  docker-compose build
  ```

- [ ] **Fetch initial data**
  ```bash
  docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker SPY
  docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker QQQ
  docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker IWM
  ```

- [ ] **Train models** (first time only, ~5 min per ticker)
  ```bash
  # Direction classifiers
  docker-compose run --rm app python scripts/train_model.py train --ticker SPY
  docker-compose run --rm app python scripts/train_model.py train --ticker QQQ
  docker-compose run --rm app python scripts/train_model.py train --ticker IWM

  # Volatility forecasters
  docker-compose run --rm app python scripts/train_model.py train-volatility --ticker SPY
  docker-compose run --rm app python scripts/train_model.py train-volatility --ticker QQQ
  docker-compose run --rm app python scripts/train_model.py train-volatility --ticker IWM
  ```

---

## Daily Operations

### Generate Trading Signals

```bash
# Generate signals for all tickers (view only)
docker-compose run --rm app python scripts/train_model.py signal-all

# Generate AND record to database
docker-compose run --rm app python scripts/train_model.py signal-all --record

# Single ticker with verbose output
docker-compose run --rm app python scripts/train_model.py signal --ticker SPY --record
```

### Run the Dashboard

```bash
# Option 1: Local Python (if dependencies installed)
streamlit run dashboard/app.py

# Option 2: Via Docker
docker-compose run --rm -p 8501:8501 app streamlit run dashboard/app.py

# Open in browser: http://localhost:8501
```

### Update Data (run weekly or before retraining)

```bash
docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker SPY
docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker QQQ
docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker IWM
```

---

## Database Reference

### Location
| Item | Path |
|------|------|
| SQLite Database | `data/predictions.db` |
| Auto-created | Yes, on first use |

### Tables

**predictions** - Daily ML signals
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| ticker | TEXT | SPY, QQQ, IWM |
| prediction_date | DATE | Signal date |
| direction_pred | TEXT | up, down, neutral |
| direction_prob | REAL | Confidence (0-1) |
| volatility_pred | TEXT | low, medium, high |
| iv_rank | REAL | IV rank (0-100) |
| recommended_strategy | TEXT | e.g., bull_put_spread |
| underlying_price | REAL | Price at prediction |
| created_at | TIMESTAMP | When recorded |

**outcomes** - Actual results (manual entry)
| Column | Type | Description |
|--------|------|-------------|
| prediction_id | INTEGER | Links to predictions |
| actual_direction | TEXT | What actually happened |
| actual_return | REAL | % return |
| trade_taken | INTEGER | 1 if traded, 0 if paper |
| pnl_dollars | REAL | Profit/loss |

**model_metrics** - Drift tracking
| Column | Type | Description |
|--------|------|-------------|
| metric_date | DATE | Measurement date |
| psi_score | REAL | Population Stability Index |
| drift_detected | INTEGER | 1 if PSI > 0.25 |

### Query Examples

```bash
# View recent predictions
sqlite3 data/predictions.db "SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT 10;"

# Check accuracy
sqlite3 data/predictions.db "
SELECT
  p.ticker,
  COUNT(*) as total,
  SUM(CASE WHEN o.actual_direction = p.direction_pred THEN 1 ELSE 0 END) as correct
FROM predictions p
JOIN outcomes o ON p.id = o.prediction_id
GROUP BY p.ticker;
"
```

---

## Training Options

### Basic Training
```bash
docker-compose run --rm app python scripts/train_model.py train --ticker SPY
```

### With All Features
```bash
docker-compose run --rm app python scripts/train_model.py train \
  --ticker SPY \
  --sentiment \
  --earnings \
  --optimize-threshold \
  --safe-features
```

### CLI Flags Reference
| Flag | Description |
|------|-------------|
| `--ticker` | SPY, QQQ, or IWM |
| `--sentiment` | Include Finnhub/Reddit sentiment features |
| `--earnings` | Include earnings calendar features |
| `--macro/--no-macro` | Include/exclude FRED macro data |
| `--safe-features` | Use only non-leaky features |
| `--optimize-threshold` | Find optimal classification threshold |
| `--n-features N` | Limit to top N features |
| `--years N` | Years of training data (default: 3) |

---

## Model Drift Detection

### PSI Score Interpretation
| PSI Value | Status | Action |
|-----------|--------|--------|
| < 0.10 | OK | No action needed |
| 0.10 - 0.25 | Monitor | Watch closely |
| > 0.25 | Drift | Retrain models |

### Check Drift via Dashboard
1. Open dashboard: `http://localhost:8501`
2. Navigate to "Model Metrics" page
3. Click "Run Drift Analysis"

### Check Drift via CLI
```bash
docker-compose run --rm app python -c "
from dashboard.utils.database import get_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute('SELECT * FROM model_metrics ORDER BY metric_date DESC LIMIT 5')
for row in cursor.fetchall():
    print(dict(row))
"
```

---

## Scheduled Signal Generation (GitHub Actions)

### Automated Workflows

**Daily Signal Generation** - Runs twice per trading day:
- **6:30 AM ET** (Mon-Fri) - Pre-market signals using previous day's close
- **5:00 PM ET** (Mon-Fri) - Post-market signals using same-day data
- Skips US market holidays automatically
- Commits predictions to `data/predictions.db`

**Weekly Model Retrain** - Runs every Sunday:
- **8:00 AM ET** (Sunday)
- Checks for model drift using PSI (Population Stability Index)
- Retrains models if PSI > 0.20 or no models exist
- Fetches latest data and commits trained models

### GitHub Secrets Setup (Required)

Before workflows can run, add these secrets to your repository:

```bash
# From your terminal (requires gh CLI)
gh secret set FINNHUB_API_KEY --body "your_finnhub_key"
gh secret set FRED_API_KEY --body "your_fred_key"
```

Or manually:
1. Go to repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add `FINNHUB_API_KEY` and `FRED_API_KEY`

### Manual Trigger
Go to Actions tab → Select workflow → "Run workflow"

### Check Action Status
1. Go to repository → Actions tab
2. View workflow runs and logs
3. Check for warnings or errors (especially FRED/model-related)

---

## Troubleshooting

### "Model not found" error
```bash
# Check if models exist
ls -la models/*.joblib

# If missing, train them
docker-compose run --rm app python scripts/train_model.py train --ticker SPY
```

### "API KEY not set" warnings
```bash
# Check .env file has both keys
cat .env | grep -E "FINNHUB|FRED"

# Make sure docker-compose loads them
docker-compose run --rm app env | grep -E "FINNHUB|FRED"

# Verify GitHub Actions secrets (if using workflows)
gh secret list
```

**FRED_API_KEY missing** = No macro features (yield curves, inflation)
**FINNHUB_API_KEY missing** = No sentiment features

### Dashboard won't start
```bash
# Check streamlit is installed
pip install streamlit plotly

# Or rebuild Docker
docker-compose build --no-cache
```

### Database locked error
```bash
# Only one process can write at a time
# Close dashboard before running --record commands
```

### Stale data
```bash
# Re-fetch latest price data
docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker SPY
```

---

## File Locations Reference

| Purpose | Location | Git Tracked? |
|---------|----------|--------------|
| Price data (parquet) | `data/raw/` | No (too large) |
| Feature data | `data/features/` | No |
| **Trained models** | `models/*.joblib` | **Yes** (committed by workflows) |
| Predictions database | `data/predictions.db` | Yes (committed daily) |
| MLflow experiments | `mlruns/` | No |
| Logs | `logs/` | No |
| Dashboard app | `dashboard/app.py` | Yes |
| Config | `src/config/settings.py` | Yes |

**Note**: Models (`*.joblib`, `*.json`) are now tracked in git so GitHub Actions workflows persist trained models automatically.

---

## Retraining Checklist

When accuracy drops or drift is detected:

- [ ] Fetch latest data
  ```bash
  docker-compose run --rm app python scripts/fetch_data.py fetch-all --ticker SPY
  ```

- [ ] Retrain direction classifier
  ```bash
  docker-compose run --rm app python scripts/train_model.py train \
    --ticker SPY --sentiment --earnings --optimize-threshold
  ```

- [ ] Retrain volatility forecaster
  ```bash
  docker-compose run --rm app python scripts/train_model.py train-volatility --ticker SPY
  ```

- [ ] Verify new model performance
  ```bash
  docker-compose run --rm app python scripts/train_model.py evaluate models/spy_direction_model.joblib
  ```

- [ ] Generate test signal
  ```bash
  docker-compose run --rm app python scripts/train_model.py signal --ticker SPY
  ```

---

## Next Steps (Roadmap)

1. **Tradier IV Integration** - Real options IV data per strike
2. **XGBoost Ensemble** - Combine multiple models
3. **Email/Slack Alerts** - Notifications for high-confidence signals
4. **Backtesting Engine** - Historical strategy performance
