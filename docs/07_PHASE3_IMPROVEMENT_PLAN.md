# Phase 3: Model Improvement Plan

## Current Situation Analysis

### Performance Summary (Post Phase 2)

| Metric | Baseline | Phase 1 | Phase 2 | Target |
|--------|----------|---------|---------|--------|
| Accuracy | 27.3% | 47.5% | **45.7%** | 60%+ |
| F1 Score | 0.27 | 0.48 | 0.46 | 0.55+ |
| AUC | 0.596 | 0.618 | **0.643** | 0.65+ |

### The Paradox Explained

**Why did accuracy drop while AUC improved?**

1. **AUC measures ranking ability** - The model got better at assigning higher probabilities to actual UP moves
2. **Accuracy uses a fixed 0.50 threshold** - This threshold is arbitrary and often suboptimal
3. **Per-fold variance is high** - Fold 4 shows 20.8% accuracy but 0.727 AUC

**Key Insight:** Your model's *predictive power* improved (AUC), but the *decision threshold* needs calibration.

### Per-Fold Analysis

| Fold | Accuracy | AUC | Interpretation |
|------|----------|-----|----------------|
| 1 | 44.2% | 0.589 | Marginal signal |
| 2 | 44.2% | 0.623 | Moderate signal |
| 3 | 51.9% | 0.575 | Good accuracy, weak ranking |
| 4 | 20.8% | **0.727** | Threshold completely wrong |
| 5 | 67.5% | **0.702** | Strong performance |

**Fold 4 Deep Dive:**
- 0.727 AUC is excellent ranking ability
- 20.8% accuracy means 50% threshold is catastrophically wrong
- Likely a different market regime requiring different threshold

---

## Phase 3 Improvement Plan

### Phase 3A: Threshold Optimization (IMPLEMENTED)

**Files Created:**
- `src/models/threshold_optimizer.py` - Threshold optimization utilities
- `scripts/analyze_and_improve.py` - Analysis and improvement script

**Expected Improvement:** +5-8% accuracy

**How It Works:**
1. Instead of fixed 0.50 threshold, find optimal per-fold
2. Use validation set to find threshold, apply to test
3. Implement confidence filtering for higher-accuracy trades

**Usage:**
```bash
python scripts/analyze_and_improve.py analyze --ticker SPY
```

### Phase 3B: Probability Calibration

**Status:** Planned

**Expected Improvement:** +3-5% accuracy

**Implementation:**
1. Add Platt scaling (logistic regression on probabilities)
2. Alternative: Isotonic regression calibration
3. Calibrate within each walk-forward fold

**Code Snippet:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap XGBoost with calibration
calibrated_model = CalibratedClassifierCV(
    base_estimator=xgb_model,
    method='sigmoid',  # Platt scaling
    cv='prefit'
)
calibrated_model.fit(X_val, y_val)
```

### Phase 3C: Feature Stability Analysis

**Status:** Implemented in analyze script

**Expected Improvement:** +2-3% accuracy, reduced variance

**Approach:**
1. Track which features appear in top-K across all folds
2. Use only "stable" features (appear in 4+ of 5 folds)
3. Reduces overfitting to fold-specific patterns

### Phase 3D: Regime-Aware Predictions

**Status:** Planned

**Expected Improvement:** Reduced variance, better risk management

**Approach:**
1. Cluster market regimes using VIX + ADX
2. Identify which regime each fold belongs to
3. Either:
   - Train separate models per regime, OR
   - Adjust thresholds per regime, OR
   - Add regime as a feature

**Regime Detection Example:**
```python
# Simple regime detection
def detect_regime(vix, adx):
    if vix > 25 and adx > 25:
        return "trending_volatile"
    elif vix > 25:
        return "range_volatile"
    elif adx > 25:
        return "trending_calm"
    else:
        return "range_calm"
```

### Phase 3E: Expand Training Data

**Status:** Planned

**Expected Improvement:** +3-5% accuracy, more stable results

**Options:**
1. **Add tickers:** Train on SPY + QQQ + IWM together
2. **Extend history:** Use 7+ years instead of 5
3. **Data augmentation:** Add noise to features for regularization

---

## Realistic Expectations

### What's Achievable

| Scenario | Accuracy | AUC | Probability |
|----------|----------|-----|-------------|
| Current baseline | 45.7% | 0.643 | - |
| With threshold optimization | 50-53% | 0.643 | High |
| With confidence filtering | 55-60% | 0.643 | Medium (lower coverage) |
| With all Phase 3 improvements | 53-58% | 0.65-0.68 | Medium |

### What's NOT Achievable

- **60%+ accuracy with full coverage** - Very difficult for 5-day predictions
- **Consistent performance across all regimes** - Market regimes differ fundamentally
- **No losing periods** - Even good models have drawdowns

### Trading System Viability

**For profitability, you need:**
1. Edge (accuracy > 50% or good risk/reward)
2. Position sizing to survive drawdowns
3. Low transaction costs

**Your current AUC (0.643) suggests:**
- Edge exists (better than random)
- With proper threshold: 52-55% accuracy is realistic
- Combined with options leverage, this can be profitable

---

## Immediate Action Items

### Today (1-2 hours)

1. Run analysis script:
   ```bash
   python scripts/analyze_and_improve.py analyze --ticker SPY
   ```

2. Review threshold optimization results

3. Identify stable features

### This Week

1. Implement probability calibration
2. Add regime detection
3. Test multi-ticker training

### Validation Before Live Trading

1. **Paper trade for 30+ days**
2. Track:
   - Predicted direction vs actual
   - Predicted probability vs actual win rate
   - Performance by regime
3. Only go live if paper results match backtest

---

## Key Takeaways

1. **45.7% accuracy is misleading** - Your model improved (AUC 0.618 -> 0.643)

2. **Threshold optimization is low-hanging fruit** - Expected +5-8% accuracy

3. **55% accuracy is a realistic target** - Professional level for this problem

4. **AUC 0.643 is tradeable** - With proper implementation, edge exists

5. **Confidence filtering is powerful** - Trade less, win more

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/models/threshold_optimizer.py` | Threshold optimization utilities |
| `scripts/analyze_and_improve.py` | Analysis and improvement CLI |
| `src/models/direction_classifier.py` | Main classifier (existing) |
| `src/models/hyperparameter_tuning.py` | Optuna tuning (existing) |
