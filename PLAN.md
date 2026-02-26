# Algorithmic Trading System — Implementation Plan

## Overview

A **hybrid swing-trading system** for broad equities (S&P 500 / TSX Composite),
combining machine learning, classic quantitative signals, and macro/fundamental
overlays. Trades are held for days to weeks. The system begins as a backtesting
framework and can later connect to a brokerage for paper or live trading.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (main.py)                │
│  - Scheduling / cron loop                               │
│  - Coordinates all modules below                        │
└────┬──────────┬──────────────┬──────────────┬───────────┘
     │          │              │              │
     v          v              v              v
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│  DATA   │ │ SIGNALS  │ │  MODEL   │ │  EXECUTION │
│ PIPELINE│ │ ENGINE   │ │ (ML)     │ │  & RISK    │
└─────────┘ └──────────┘ └──────────┘ └────────────┘
     │          │              │              │
     v          v              v              v
┌─────────────────────────────────────────────────────────┐
│                 BACKTESTING ENGINE                       │
│  - Simulates strategy over historical data              │
│  - Performance metrics, drawdown, Sharpe, etc.          │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Data Pipeline (`data/`)

**Goal:** Reliable, cached access to price, fundamental, and macro data.

### 1a. Price Data
- **Source:** `yfinance` (free, covers US & Canadian equities)
- **Granularity:** Daily OHLCV bars
- **Universe:** S&P 500 constituents (or a subset like top 100 by market cap)
- **Storage:** Local Parquet files (fast reads, columnar, compact)

### 1b. Fundamental Data
- **Source:** `yfinance` (quarterly financials, P/E, EPS, sector)
- **Fields:** P/E ratio, EPS growth, revenue growth, debt-to-equity, sector classification
- **Update frequency:** Quarterly (after earnings)

### 1c. Macro / Economic Data
- **Source:** FRED API via `fredapi` (free with API key)
- **Indicators:**
  - US 10-Year Treasury yield
  - VIX (volatility index)
  - Unemployment rate
  - CPI / inflation
  - Fed Funds Rate
  - Crude oil prices (WTI) — ties back to Alberta resource revenue context
- **Update frequency:** Daily or as released

### 1d. Data Manager Module
- Unified interface: `DataManager.get_prices(ticker, start, end)`
- Caching layer (avoid re-downloading)
- Data validation (gap detection, split/dividend adjustments)

**Files to create:**
```
data/
  __init__.py
  price_loader.py       # yfinance wrapper + Parquet cache
  fundamental_loader.py # Quarterly financials
  macro_loader.py       # FRED API wrapper
  data_manager.py       # Unified interface
  cache/                # Local Parquet storage (gitignored)
```

**Dependencies:** `yfinance`, `fredapi`, `pandas`, `pyarrow`

---

## Phase 2 — Signal Engine (`signals/`)

**Goal:** Compute a library of trading signals (features) from raw data.

### 2a. Technical Signals
| Signal              | Description                         | Lookback    |
|---------------------|-------------------------------------|-------------|
| SMA crossover       | 10-day vs 50-day moving average     | 50 days     |
| RSI                 | Relative Strength Index (14-day)    | 14 days     |
| MACD                | Moving Average Convergence/Divergence| 26 days    |
| Bollinger Bands     | Price relative to 2σ bands          | 20 days     |
| ATR                 | Average True Range (volatility)     | 14 days     |
| Momentum            | N-day return (5, 10, 20 day)        | 20 days     |
| Volume profile      | Volume vs 20-day average            | 20 days     |
| Mean reversion      | Z-score of price vs 20-day mean     | 20 days     |

### 2b. Fundamental Signals
| Signal              | Description                         |
|---------------------|-------------------------------------|
| Earnings surprise   | Actual EPS vs consensus estimate    |
| Value score         | Composite of P/E, P/B, dividend yield |
| Quality score       | ROE, debt-to-equity, earnings stability |
| Growth score        | Revenue + EPS growth rate           |

### 2c. Macro Regime Signals
| Signal              | Description                         |
|---------------------|-------------------------------------|
| Yield curve slope   | 10Y - 2Y treasury spread            |
| VIX regime          | Low / Normal / High volatility       |
| Risk-on / Risk-off  | Composite macro sentiment indicator  |
| Oil price trend     | WTI momentum (Alberta-relevant)      |

### 2d. Signal Combiner
- Normalizes all signals to comparable scales (z-score or [0,1])
- Produces a feature matrix: rows = (date, ticker), columns = signals
- Handles NaN/missing data gracefully

**Files to create:**
```
signals/
  __init__.py
  technical.py        # TA-Lib or manual implementations
  fundamental.py      # Value, quality, growth scores
  macro.py            # Regime detection
  combiner.py         # Normalize + merge into feature matrix
```

**Dependencies:** `ta` (technical analysis library) or `pandas-ta`, `numpy`

---

## Phase 3 — ML Model (`models/`)

**Goal:** Train models that predict forward returns (or up/down classification)
from the feature matrix.

### 3a. Target Variable
- **Primary:** 5-day forward return (classification: up > 1%, down < -1%, flat)
- **Secondary:** 10-day and 20-day forward returns for multi-horizon signals

### 3b. Model Candidates (ranked by priority)
1. **LightGBM / XGBoost** — gradient-boosted trees, strong baseline, handles
   mixed features well, fast to train. Start here.
2. **Random Forest** — robust, less prone to overfitting, good for
   feature importance analysis.
3. **Ridge / Lasso Regression** — simple linear baseline for comparison.
4. **LSTM (stretch goal)** — if tree models plateau, explore sequence models
   for capturing temporal dependencies.

### 3c. Training Pipeline
- **Walk-forward validation** (critical for time series — no future leakage):
  - Train on 2 years of data
  - Validate on next 3 months
  - Roll forward, retrain
- **Feature selection:** Recursive feature elimination + importance ranking
- **Hyperparameter tuning:** Optuna or simple grid search
- **Metrics:** Sharpe ratio of predictions, accuracy, precision/recall for
  directional calls, profit factor

### 3d. Model Registry
- Save trained models with metadata (training window, features used, metrics)
- Compare models over time
- Simple versioning (timestamp-based)

**Files to create:**
```
models/
  __init__.py
  features.py          # Feature engineering (lag, rolling, interactions)
  trainer.py           # Walk-forward training loop
  predict.py           # Generate predictions from trained model
  registry.py          # Save/load/compare models
  saved/               # Serialized models (gitignored)
```

**Dependencies:** `lightgbm`, `scikit-learn`, `optuna` (optional)

---

## Phase 4 — Strategy & Risk Management (`strategy/`)

**Goal:** Convert model predictions + signals into actual trade decisions
with proper risk controls.

### 4a. Portfolio Construction
- **Signal aggregation:** Weighted combination of ML prediction + quant signals
  + macro regime overlay
- **Position sizing:** Volatility-scaled (ATR-based) — larger positions in
  low-volatility names, smaller in high-vol
- **Sector limits:** Max 25% in any single sector
- **Max positions:** 10-20 concurrent positions

### 4b. Entry Rules
- ML model predicts positive forward return with confidence > threshold
- At least 2 of 3 signal categories (technical, fundamental, macro) agree
- Macro regime is not "risk-off" (unless signal is very strong)

### 4c. Exit Rules
- **Take profit:** Close at +5% gain (configurable)
- **Stop loss:** Close at -3% loss (configurable)
- **Time stop:** Close after 15 trading days if neither target hit
- **Trailing stop:** Once +3% is reached, trail stop at -1.5% from peak
- **Signal reversal:** Close if model prediction flips to negative

### 4d. Risk Limits
| Limit                    | Value          |
|--------------------------|----------------|
| Max portfolio drawdown   | -10% (pause)   |
| Max single-position size | 10% of capital  |
| Max sector exposure      | 25% of capital  |
| Max correlated positions | 5 in same group |
| Daily loss limit         | -2% (stop trading for day) |

**Files to create:**
```
strategy/
  __init__.py
  portfolio.py        # Position sizing, allocation
  entry_exit.py       # Entry/exit rule engine
  risk_manager.py     # Enforce limits, circuit breakers
```

---

## Phase 5 — Backtesting Engine (`backtest/`)

**Goal:** Simulate the full strategy on historical data to measure performance
before risking real capital.

### 5a. Backtester Design
- **Event-driven** (not vectorized) for realistic simulation:
  - Process one day at a time
  - No lookahead bias
  - Account for slippage (0.05% default) and commissions
- **Walk-forward:** Re-train model at each rebalance window

### 5b. Performance Metrics
| Metric                  | Target         |
|-------------------------|----------------|
| Annual return           | > 12%          |
| Sharpe ratio            | > 1.5          |
| Max drawdown            | < 15%          |
| Win rate                | > 55%          |
| Profit factor           | > 1.5          |
| Avg holding period      | 5-15 days      |
| Turnover                | Report monthly |

### 5c. Reporting
- Equity curve plot
- Drawdown chart
- Monthly/annual return heatmap
- Trade log with entry/exit reasons
- Feature importance over time
- Comparison vs buy-and-hold S&P 500

**Files to create:**
```
backtest/
  __init__.py
  engine.py            # Core event-driven backtester
  metrics.py           # Sharpe, drawdown, win rate, etc.
  report.py            # Generate charts + summary
  results/             # Saved backtest results (gitignored)
```

**Dependencies:** `matplotlib`, `seaborn` (for reporting)

---

## Phase 6 — Execution Layer (`execution/`) [Future]

**Goal:** Connect to a brokerage API for paper and eventually live trading.

### 6a. Broker Abstraction
- Common interface: `place_order()`, `get_positions()`, `get_account()`
- Implementations for:
  - **Paper trading** (simulated, in-memory)
  - **Alpaca** (commission-free, good API, paper mode built-in)
  - **Interactive Brokers** (broader asset coverage, professional)

### 6b. Order Management
- Limit orders preferred (avoid market orders in low-liquidity names)
- Order timeout / cancel-replace logic
- Fill tracking and reconciliation

### 6c. Monitoring & Alerts
- Dashboard: current positions, P&L, signals
- Alerts: Slack/email on trade execution, risk limit breach, errors

**Files to create (future):**
```
execution/
  __init__.py
  broker_base.py       # Abstract broker interface
  paper_broker.py      # Simulated execution
  alpaca_broker.py     # Alpaca API integration
  monitor.py           # Position monitoring + alerts
```

---

## Project Structure (Final)

```
AB-Budget-Analysis/
├── README.md
├── PLAN.md                 # This file
├── requirements.txt
├── config.yaml             # API keys, parameters, universe definition
├── main.py                 # Orchestrator / entry point
│
├── data/
│   ├── __init__.py
│   ├── price_loader.py
│   ├── fundamental_loader.py
│   ├── macro_loader.py
│   ├── data_manager.py
│   └── cache/              # (gitignored)
│
├── signals/
│   ├── __init__.py
│   ├── technical.py
│   ├── fundamental.py
│   ├── macro.py
│   └── combiner.py
│
├── models/
│   ├── __init__.py
│   ├── features.py
│   ├── trainer.py
│   ├── predict.py
│   ├── registry.py
│   └── saved/              # (gitignored)
│
├── strategy/
│   ├── __init__.py
│   ├── portfolio.py
│   ├── entry_exit.py
│   └── risk_manager.py
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py
│   ├── metrics.py
│   ├── report.py
│   └── results/            # (gitignored)
│
├── execution/              # Future phase
│   ├── __init__.py
│   ├── broker_base.py
│   ├── paper_broker.py
│   └── alpaca_broker.py
│
├── tests/
│   ├── test_data.py
│   ├── test_signals.py
│   ├── test_models.py
│   ├── test_strategy.py
│   └── test_backtest.py
│
└── notebooks/              # Exploratory analysis
    ├── 01_data_exploration.ipynb
    └── 02_signal_research.ipynb
```

---

## Implementation Order

| Step | Phase                    | Est. Complexity | Depends On |
|------|--------------------------|-----------------|------------|
| 1    | Data Pipeline (Phase 1)  | Medium          | —          |
| 2    | Signal Engine (Phase 2)  | Medium          | Phase 1    |
| 3    | Backtesting Engine (Phase 5) | High        | Phase 1    |
| 4    | ML Model (Phase 3)       | High           | Phase 1, 2 |
| 5    | Strategy & Risk (Phase 4)| Medium          | Phase 2, 3 |
| 6    | Integration & Tuning     | High           | All above  |
| 7    | Execution Layer (Phase 6)| Medium          | Phase 4, 5 |

> **Note:** Phases 2 and 5 can be developed in parallel since they share
> only the data layer dependency.

---

## Key Dependencies

```
# requirements.txt (initial)
yfinance>=0.2.31
fredapi>=0.5.1
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
lightgbm>=4.0
scikit-learn>=1.3
ta>=0.11.0          # Technical analysis indicators
matplotlib>=3.7
seaborn>=0.13
pyyaml>=6.0
optuna>=3.4         # Optional: hyperparameter tuning
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting ML model to historical data | Strategy fails live | Walk-forward validation, out-of-sample testing, regularization |
| Look-ahead bias in backtesting | Inflated results | Event-driven backtester, careful feature timestamp alignment |
| Data quality issues (splits, survivorship bias) | Wrong signals | Use adjusted prices, test with S&P 500 *current* constituents only (accept survivorship bias initially) |
| yfinance rate limits / API changes | Data pipeline breaks | Local caching, fallback to Alpha Vantage |
| Slippage underestimation | Live results worse than backtest | Conservative slippage assumptions (0.05-0.1%), limit orders |
| Regime change (strategy stops working) | Losses | Macro regime overlay, drawdown-based circuit breaker, regular model retraining |

---

## Success Criteria (Backtest)

Before considering live deployment, the strategy must demonstrate:
1. **Sharpe ratio > 1.5** over the full backtest period (2015-2025)
2. **Max drawdown < 15%**
3. **Consistent performance** across at least 3 different market regimes
   (bull, bear, sideways)
4. **Out-of-sample performance** within 70% of in-sample performance
5. **Robust to parameter changes** (no cliff-edge sensitivity)
