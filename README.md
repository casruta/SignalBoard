# SignalBoard

A hybrid swing-trading system for broad equities (S&P 500 / TSX Composite) combining machine learning, quantitative signals, and macro/fundamental overlays. Trades are held for days to weeks.

The system generates buy/sell recommendations with full explainability and delivers them via a native iOS app (SwiftUI) backed by a Python REST API.

## Architecture

```
iPhone App (SwiftUI)  ←→  Backend API (FastAPI)  ←→  ML + Quant Engine (Python)
```

- **Data Pipeline** — yfinance (prices, fundamentals) + FRED API (macro indicators)
- **Signal Engine** — 100+ features across technical, fundamental, macro, microstructure, statistical, calendar, cross-sectional, and network categories
- **ML Model** — LightGBM with walk-forward validation, purged CV, and temporal ensemble
- **Strategy** — ATR-based position sizing, multi-exit rules, correlation-aware risk management
- **Backtesting** — Event-driven simulator with slippage/commission modeling
- **Explainability** — SHAP-based signal decomposition with plain-English narratives

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config.example.yaml config.yaml
# Edit config.yaml with your FRED API key

# Train the model
python main.py train --config config.yaml

# Run a backtest
python main.py backtest --config config.yaml

# Generate predictions
python main.py predict --config config.yaml

# Start the API server
python main.py serve --config config.yaml

# Run tests
pytest tests/ -v
```

## Project Structure

```
SignalBoard/
├── main.py                  # CLI entry point (train/backtest/predict/serve/pipeline/analyze)
├── config_loader.py         # YAML config loader
├── config.example.yaml      # Example configuration
├── requirements.txt         # Python dependencies
├── data/                    # Data pipeline (yfinance, FRED)
├── signals/                 # Signal engine (100+ features)
├── models/                  # ML models (LightGBM, ensemble, uncertainty)
├── strategy/                # Portfolio construction, entry/exit, risk management
├── backtest/                # Event-driven backtester + reporting
├── explainability/          # SHAP decomposition + narrative generation
├── server/                  # FastAPI backend + push notifications
├── ios/                     # SwiftUI iPhone app
└── tests/                   # Unit + integration tests
```

## External APIs

| API | Purpose | Cost |
|-----|---------|------|
| Yahoo Finance (`yfinance`) | Daily OHLCV, fundamentals, options chains | Free |
| FRED (`fredapi`) | Macro indicators (VIX, yield curve, CPI, etc.) | Free (API key) |
