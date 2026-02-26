# Algorithmic Trading System — Implementation Plan

## Overview

A **hybrid swing-trading system** for broad equities (S&P 500 / TSX Composite),
combining machine learning, classic quantitative signals, and macro/fundamental
overlays. Trades are held for days to weeks.

The system is delivered as a **native iOS app (SwiftUI)** for personal use that
presents strong buy and sell signals with full explainability — tap any ticker
to see exactly *why* the model recommends the trade. A Python backend runs the
ML pipeline daily and serves recommendations via a REST API, with push
notifications for high-confidence signals.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              iPhone App (SwiftUI)                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Signal List (Home Screen)                         │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐                 │  │
│  │  │  AAPL  │ │  MSFT  │ │  JPM   │  ...            │  │
│  │  │  BUY   │ │  SELL  │ │  BUY   │                 │  │
│  │  │  78%   │ │  82%   │ │  71%   │                 │  │
│  │  └───┬────┘ └────────┘ └────────┘                 │  │
│  │      │ tap                                         │  │
│  │      v                                             │  │
│  │  Detail View                                       │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ AAPL — Strong Buy (78% confidence)           │  │  │
│  │  │                                              │  │  │
│  │  │ Technical:  RSI oversold, MACD bullish cross │  │  │
│  │  │ Fundamental: P/E below sector, EPS beat +8%  │  │  │
│  │  │ Macro:  Risk-on regime, VIX declining        │  │  │
│  │  │ ML Model: +2.3% predicted 5-day return       │  │  │
│  │  │                                              │  │  │
│  │  │ Entry: $204.20  Stop: $198.10  Target: $214  │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│  + Push notifications for high-confidence signals (APNs) │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTPS (REST API)
                       v
┌──────────────────────────────────────────────────────────┐
│            Backend API (FastAPI / Python)                 │
│            Hosted on VPS / Railway / Fly.io              │
│                                                          │
│  GET  /signals          → current buy/sell list (JSON)   │
│  GET  /signals/{ticker} → full detail + explainability   │
│  POST /device-token     → register APNs device token     │
│                                                          │
│  Cron: daily at 4:30 PM ET (after market close)          │
│    1. Fetch latest data                                  │
│    2. Re-score all tickers                               │
│    3. Store recommendations in DB                        │
│    4. Push notifications for strong signals              │
└──────────────────────┬───────────────────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────────────────┐
│              ML + Quant Engine (Python)                   │
│                                                          │
│  Data Pipeline → Signals → ML Model → Strategy/Risk      │
│       │              │          │            │            │
│       v              v          v            v            │
│  ┌─────────┐  ┌──────────┐ ┌────────┐ ┌──────────────┐  │
│  │ yfinance│  │Technical │ │LightGBM│ │ Explainability│  │
│  │ FRED API│  │Fundament.│ │XGBoost │ │ Engine        │  │
│  └─────────┘  │Macro     │ │        │ │ (human-       │  │
│               └──────────┘ └────────┘ │  readable why) │  │
│                                       └──────────────┘  │
│                                                          │
│  Backtesting Engine (validate before going live)         │
└──────────────────────────────────────────────────────────┘
```

### External APIs Used: 2

| # | API | What It Provides | Cost |
|---|-----|-----------------|------|
| 1 | **Yahoo Finance** (`yfinance`) | Daily OHLCV prices, quarterly financials (P/E, EPS, revenue, debt ratios), sector data | Free |
| 2 | **FRED** (`fredapi`) | Macro indicators: VIX, yield curve, unemployment, CPI, Fed Funds Rate, oil prices (WTI) | Free (API key) |

All technical signals (RSI, MACD, Bollinger Bands, etc.) and ML features are
computed locally from the downloaded data — no additional APIs needed.

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

## Phase 6 — Explainability Engine (`explainability/`)

**Goal:** For every buy/sell signal, produce a human-readable explanation of
*what* the system recommends and *why* — this is what the user sees when they
tap a ticker in the iOS app.

### 6a. Signal Decomposition
- Break the final score into contributions from each signal category
- Use LightGBM's built-in **SHAP values** to attribute prediction to features
- Rank the top 3-5 contributing factors per recommendation

### 6b. Narrative Generator
- Convert raw signal values into plain-English sentences:
  - `RSI = 28` → "RSI is oversold at 28 (below 30 threshold)"
  - `PE_vs_sector = -0.23` → "P/E ratio is 23% below the sector average"
  - `macro_regime = risk_on` → "Macro environment favors risk-on positioning"
- Categorize explanations under: Technical, Fundamental, Macro, ML Model

### 6c. Recommendation Output Schema
Each recommendation produced by the engine follows this structure:
```json
{
  "ticker": "AAPL",
  "action": "BUY",
  "confidence": 0.78,
  "predicted_return_5d": 0.023,
  "entry_price": 204.20,
  "stop_loss": 198.10,
  "take_profit": 214.40,
  "trailing_stop_trigger": 210.30,
  "time_stop_days": 15,
  "position_size_pct": 4.2,
  "sector": "Technology",
  "explanation": {
    "technical": [
      "RSI oversold at 28 — historically rebounds from this level",
      "Price touching lower Bollinger Band with contracting bandwidth",
      "MACD bullish crossover forming on daily chart"
    ],
    "fundamental": [
      "P/E of 24.1 is 23% below Technology sector average of 31.2",
      "Last quarter EPS beat consensus by 8%",
      "Quality score: A (high ROE, low debt-to-equity)"
    ],
    "macro": [
      "Yield curve normalizing — favors equity risk",
      "VIX declining from 22 to 16 — entering low-volatility regime",
      "WTI crude trending up — risk-on signal"
    ],
    "ml_model": {
      "predicted_return": "+2.3% over 5 days",
      "confidence_percentile": "Top 8% of all current signals",
      "top_features": ["RSI_14", "PE_vs_sector", "VIX_regime"]
    }
  },
  "risk_context": {
    "sector_exposure_after": "Technology: 22% (limit: 25%)",
    "portfolio_positions_after": "12 of 20 max",
    "correlation_note": "Low correlation with existing holdings"
  },
  "generated_at": "2026-02-26T21:30:00Z"
}
```

### 6d. Historical Performance Context
- Show how similar signals performed historically:
  "In the last 50 times RSI dropped below 30 for AAPL, the stock gained
  an average of 3.2% over the following 10 days (68% win rate)"
- Backtest hit rate for the specific combination of active signals

**Files to create:**
```
explainability/
  __init__.py
  decomposer.py       # SHAP-based signal attribution
  narrator.py         # Convert signals to plain-English text
  historical.py       # Similar-signal historical performance
  schema.py           # Recommendation dataclass / JSON schema
```

**Dependencies:** `shap`

---

## Phase 7 — Backend API (`server/`)

**Goal:** A lightweight Python API that runs the ML pipeline on schedule,
stores recommendations, and serves them to the iOS app.

### 7a. API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/signals` | Current buy/sell list with summary info |
| GET | `/signals/{ticker}` | Full recommendation detail + explainability |
| GET | `/signals/history` | Past recommendations + outcomes |
| GET | `/health` | Server health check |
| POST | `/device-token` | Register APNs device token for push |

### 7b. Daily Pipeline Scheduler
- **Cron trigger:** Daily at 4:30 PM ET (after US market close)
- **Pipeline steps:**
  1. Fetch latest price + fundamental + macro data
  2. Compute all signals
  3. Run ML model predictions
  4. Apply strategy rules + risk filters
  5. Generate explainability narratives
  6. Store recommendations in database
  7. Send push notifications for strong signals (confidence > 75%)

### 7c. Database
- **SQLite** (personal use — no need for PostgreSQL)
- Tables:
  - `recommendations` — daily signals with full JSON payload
  - `outcomes` — actual returns after recommendation (track accuracy)
  - `device_tokens` — APNs tokens for push notifications
  - `model_metadata` — trained model versions and performance

### 7d. Push Notifications (APNs)
- Register device token from iOS app on first launch
- Send push when a new strong signal (confidence > 75%) is generated
- Notification payload: ticker, action (BUY/SELL), confidence, one-line reason
- Use `aioapns` or `PyAPNs2` library
- Requires Apple Developer account + APNs certificate

### 7e. Hosting
- **Railway** or **Fly.io** (simple deploys, free/cheap tiers)
- Alternatively: any $5/month VPS (DigitalOcean, Linode)
- Dockerized for easy deployment

**Files to create:**
```
server/
  __init__.py
  app.py               # FastAPI application
  scheduler.py         # APScheduler daily pipeline trigger
  database.py          # SQLite models + queries
  push.py              # APNs push notification sender
  Dockerfile           # Container for deployment
```

**Dependencies:** `fastapi`, `uvicorn`, `apscheduler`, `aioapns`, `sqlalchemy`

---

## Phase 8 — iOS App (`ios/SignalBoard/`)

**Goal:** A native SwiftUI iPhone app that displays buy/sell recommendations
with full drill-down explainability and push notification support.

### 8a. App Screens

**Screen 1 — Signal List (Home)**
- Cards for each active recommendation, sorted by confidence
- Each card shows: ticker symbol, company name, BUY/SELL badge, confidence %
- Color-coded: green for buy, red for sell, intensity scales with confidence
- Pull-to-refresh for latest signals
- Last-updated timestamp at top
- Empty state when no signals are active

**Screen 2 — Signal Detail (Tap a ticker)**
- Header: ticker, action, confidence, predicted return
- Sections (expandable):
  - **Technical Analysis** — bullet points from narrator
  - **Fundamental Analysis** — value/quality/growth breakdown
  - **Macro Environment** — regime context
  - **ML Model Insight** — prediction, top features, SHAP summary
  - **Risk Parameters** — entry, stop, target, position size
  - **Historical Context** — how similar setups performed
- Mini price chart (last 30 days) with entry/stop/target levels marked

**Screen 3 — Settings**
- Backend URL configuration
- Push notification toggle
- Confidence threshold filter (e.g., only show signals > 70%)

### 8b. Networking Layer
- Swift `async/await` with `URLSession`
- Calls backend `/signals` and `/signals/{ticker}` endpoints
- JSON decoding into Swift structs matching the recommendation schema
- Error handling + offline state

### 8c. Push Notifications
- Request notification permission on first launch
- Register device token with backend via `/device-token`
- Handle incoming notifications: deep-link to Signal Detail for the ticker

### 8d. Distribution
- **Personal use only** — no App Store submission needed
- Install via Xcode directly to your iPhone, or
- Use TestFlight (free, up to 25 internal testers)
- Requires Apple Developer account ($99/year)

**Files to create:**
```
ios/
  SignalBoard/
    SignalBoardApp.swift        # App entry point
    Models/
      Signal.swift              # Data models matching API schema
    Views/
      SignalListView.swift      # Home screen — card list
      SignalCardView.swift      # Individual ticker card
      SignalDetailView.swift    # Full explainability drill-down
      PriceChartView.swift      # Mini price chart
      SettingsView.swift        # Configuration screen
    Services/
      APIClient.swift           # Backend REST client
      NotificationManager.swift # APNs registration + handling
    Assets.xcassets/            # App icon, colors
    Info.plist
  SignalBoard.xcodeproj/
```

**Requirements:** Xcode 15+, iOS 17+, Apple Developer account

---

## Phase 9 — Execution Layer (`execution/`) [Future]

**Goal:** Connect to a brokerage API for paper and eventually live trading.

### 9a. Broker Abstraction
- Common interface: `place_order()`, `get_positions()`, `get_account()`
- Implementations for:
  - **Paper trading** (simulated, in-memory)
  - **Alpaca** (commission-free, good API, paper mode built-in)
  - **Interactive Brokers** (broader asset coverage, professional)

### 9b. Order Management
- Limit orders preferred (avoid market orders in low-liquidity names)
- Order timeout / cancel-replace logic
- Fill tracking and reconciliation

### 9c. App Integration
- Add "Execute Trade" button to Signal Detail view (Phase 8)
- Confirmation dialog before placing orders
- Position tracking view in the app

**Files to create (future):**
```
execution/
  __init__.py
  broker_base.py       # Abstract broker interface
  paper_broker.py      # Simulated execution
  alpaca_broker.py     # Alpaca API integration
```

---

## Project Structure (Final)

```
AB-Budget-Analysis/
├── README.md
├── PLAN.md                     # This file
├── requirements.txt            # Python dependencies
├── config.yaml                 # API keys, parameters, universe definition
├── main.py                     # Orchestrator / entry point
├── .gitignore
│
├── data/                       # Phase 1 — Data Pipeline
│   ├── __init__.py
│   ├── price_loader.py         # yfinance wrapper + Parquet cache
│   ├── fundamental_loader.py   # Quarterly financials
│   ├── macro_loader.py         # FRED API wrapper
│   ├── data_manager.py         # Unified interface
│   └── cache/                  # (gitignored)
│
├── signals/                    # Phase 2 — Signal Engine
│   ├── __init__.py
│   ├── technical.py            # RSI, MACD, Bollinger, etc.
│   ├── fundamental.py          # Value, quality, growth scores
│   ├── macro.py                # Regime detection
│   └── combiner.py             # Normalize + merge into feature matrix
│
├── models/                     # Phase 3 — ML Model
│   ├── __init__.py
│   ├── features.py             # Feature engineering
│   ├── trainer.py              # Walk-forward training loop
│   ├── predict.py              # Generate predictions
│   ├── registry.py             # Save/load/compare models
│   └── saved/                  # (gitignored)
│
├── strategy/                   # Phase 4 — Strategy & Risk
│   ├── __init__.py
│   ├── portfolio.py            # Position sizing, allocation
│   ├── entry_exit.py           # Entry/exit rule engine
│   └── risk_manager.py         # Enforce limits, circuit breakers
│
├── backtest/                   # Phase 5 — Backtesting
│   ├── __init__.py
│   ├── engine.py               # Core event-driven backtester
│   ├── metrics.py              # Sharpe, drawdown, win rate
│   ├── report.py               # Generate charts + summary
│   └── results/                # (gitignored)
│
├── explainability/             # Phase 6 — Explainability
│   ├── __init__.py
│   ├── decomposer.py           # SHAP-based signal attribution
│   ├── narrator.py             # Signals → plain-English text
│   ├── historical.py           # Similar-signal historical perf
│   └── schema.py               # Recommendation dataclass / JSON
│
├── server/                     # Phase 7 — Backend API
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── scheduler.py            # Daily pipeline trigger
│   ├── database.py             # SQLite models + queries
│   ├── push.py                 # APNs push notifications
│   └── Dockerfile              # Container for deployment
│
├── ios/                        # Phase 8 — iOS App
│   └── SignalBoard/
│       ├── SignalBoardApp.swift
│       ├── Models/
│       │   └── Signal.swift
│       ├── Views/
│       │   ├── SignalListView.swift
│       │   ├── SignalCardView.swift
│       │   ├── SignalDetailView.swift
│       │   ├── PriceChartView.swift
│       │   └── SettingsView.swift
│       ├── Services/
│       │   ├── APIClient.swift
│       │   └── NotificationManager.swift
│       ├── Assets.xcassets/
│       └── Info.plist
│
├── execution/                  # Phase 9 — Execution (future)
│   ├── __init__.py
│   ├── broker_base.py
│   ├── paper_broker.py
│   └── alpaca_broker.py
│
├── tests/                      # Unit + integration tests
│   ├── test_data.py
│   ├── test_signals.py
│   ├── test_models.py
│   ├── test_strategy.py
│   ├── test_backtest.py
│   ├── test_explainability.py
│   └── test_server.py
│
└── notebooks/                  # Exploratory analysis
    ├── 01_data_exploration.ipynb
    └── 02_signal_research.ipynb
```

---

## Implementation Order

| Step | Phase                          | Est. Complexity | Depends On     |
|------|--------------------------------|-----------------|----------------|
| 1    | Data Pipeline (Phase 1)        | Medium          | —              |
| 2    | Signal Engine (Phase 2)        | Medium          | Phase 1        |
| 3    | Backtesting Engine (Phase 5)   | High            | Phase 1        |
| 4    | ML Model (Phase 3)             | High            | Phase 1, 2     |
| 5    | Strategy & Risk (Phase 4)      | Medium          | Phase 2, 3     |
| 6    | Explainability Engine (Phase 6)| Medium          | Phase 2, 3, 4  |
| 7    | Integration & Backtest Tuning  | High            | Phases 1-6     |
| 8    | Backend API (Phase 7)          | Medium          | Phases 1-6     |
| 9    | iOS App (Phase 8)              | Medium-High     | Phase 7        |
| 10   | Execution Layer (Phase 9)      | Medium          | Phase 7, 8     |

> **Notes:**
> - Phases 2 and 5 can be developed in parallel (share only the data layer).
> - The iOS app (Phase 8) can begin scaffolding with mock data while the
>   backend (Phase 7) is being built.
> - The strategy must pass backtest success criteria (Step 7) before
>   deploying the backend for live signal generation.

---

## Key Dependencies

### Python (Backend + ML Engine)
```
# requirements.txt
# --- Data Pipeline ---
yfinance>=0.2.31
fredapi>=0.5.1
pandas>=2.0
numpy>=1.24
pyarrow>=14.0

# --- Signals & ML ---
ta>=0.11.0              # Technical analysis indicators
lightgbm>=4.0
scikit-learn>=1.3
shap>=0.44              # Model explainability (SHAP values)
optuna>=3.4             # Optional: hyperparameter tuning

# --- Backtesting & Reporting ---
matplotlib>=3.7
seaborn>=0.13

# --- Backend API ---
fastapi>=0.109
uvicorn>=0.27
sqlalchemy>=2.0
apscheduler>=3.10
aioapns>=3.0            # Apple Push Notification service

# --- Config ---
pyyaml>=6.0
```

### iOS App
- Xcode 15+
- iOS 17+ deployment target
- SwiftUI (built-in)
- Swift Charts (built-in, for price charts)
- Apple Developer account ($99/year — for APNs + device install)

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
| Backend server downtime | No daily signals | Health check endpoint, simple uptime monitoring (e.g., UptimeRobot free tier) |
| APNs certificate expiry | Push notifications stop | Calendar reminder to renew; log push failures in DB |
| Stale signals shown in app | User acts on outdated info | Show last-updated timestamp prominently; grey out signals older than 24h |

---

## Success Criteria (Backtest)

Before considering live deployment, the strategy must demonstrate:
1. **Sharpe ratio > 1.5** over the full backtest period (2015-2025)
2. **Max drawdown < 15%**
3. **Consistent performance** across at least 3 different market regimes
   (bull, bear, sideways)
4. **Out-of-sample performance** within 70% of in-sample performance
5. **Robust to parameter changes** (no cliff-edge sensitivity)
