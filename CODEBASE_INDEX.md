# CODEBASE INDEX: SignalBoard
> Auto-generated 2026-02-26. Optimized for AI assistant context.

## Overview
A hybrid swing-trading system for broad equities (S&P 500 / TSX Composite) combining machine learning, quantitative signals, and macro/fundamental overlays. Trades are held for days to weeks.

**Stats:** 78 files | .py: 63 | .swift: 8 | : 2 | .md: 2 | .yaml: 1 | .ini: 1 | .txt: 1

## Quick Reference
**Entry points:** main.py (SignalBoard — main orchestrator.), server/app.py (FastAPI application — REST API for the iOS app.)
**Dependencies:** yfinance, fredapi, pandas, numpy, pyarrow, ta, lightgbm, xgboost, scikit-learn, shap, optuna, scipy, matplotlib, seaborn, fastapi

## Directory Tree
```
SignalBoard/
├── backtest/
│   └── results/
├── data/
│   └── cache/
├── execution/
├── explainability/
├── ios/
│   └── SignalBoard/
│       ├── Models/
│       ├── Services/
│       └── Views/
├── models/
│   └── saved/
├── notebooks/
├── server/
├── signals/
├── strategy/
└── tests/
```

## Files

### Root
- **main.py** (11.3 KB)
  - Docstring: SignalBoard — main orchestrator.
  - Functions: cmd_train(config), cmd_backtest(config), cmd_predict(config), cmd_analyze(config), cmd_serve(config), cmd_pipeline(config), main() (+1 private)
  - Imports: asyncio, backtest, config_loader, data, models, numpy, pandas, server, signals
- **README.md** (2.7 KB)
  - Headings: SignalBoard → Architecture → Quick Start → Install dependencies → Copy and edit config → Edit config.yaml with your FRED API key → Train the model → Run a backtest → Generate predictions → Start the API server → Run tests → Project Structure → External APIs
- **config_loader.py** (747 B)
  - Docstring: Load and validate configuration from config.yaml.
  - Functions: load_config(path), get_config()
  - Imports: yaml
- **requirements.txt** (403 B)
  - Packages: yfinance, fredapi, pandas, numpy, pyarrow, ta, lightgbm, xgboost, scikit-learn, shap, optuna, scipy, matplotlib, seaborn, fastapi, uvicorn, sqlalchemy, apscheduler, aioapns, pyyaml
- **.gitignore** (585 B)
- **config.example.yaml** (1.8 KB)
  - Top-level keys: fred, universe, strategy, model, backtest, server, apns
- **PLAN.md** (31.4 KB)
  - Headings: Algorithmic Trading System — Implementation Plan → Overview → Architecture → External APIs Used: 2 → Phase 1 — Data Pipeline (`data/`) → 1a. Price Data → 1b. Fundamental Data → 1c. Macro / Economic Data → 1d. Data Manager Module → Phase 2 — Signal Engine (`signals/`) → 2a. Technical Signals → 2b. Fundamental Signals → 2c. Macro Regime Signals → 2d. Signal Combiner → Phase 3 — ML Model (`models/`)
- **pytest.ini** (125 B)
