"""FastAPI application — REST API + web frontend."""

import asyncio
import json
import logging
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config_loader import load_config
from server.database import Database
from server.mock_financials import generate_mock_report
from server.scheduler import run_daily_pipeline

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps succeeds."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


app = FastAPI(
    title="SignalBoard API",
    description="Algorithmic trading signal recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Personal use — open CORS is fine
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ────────────────────────────────────────────────────────

_config = None
_db = None
_scheduler = None


class DeviceTokenRequest(BaseModel):
    token: str


# ── Lifecycle ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _config, _db, _scheduler
    _config = load_config()
    _db = Database(_config["server"]["database_path"])

    # Schedule daily pipeline
    _scheduler = AsyncIOScheduler()
    server_cfg = _config["server"]
    _scheduler.add_job(
        lambda: asyncio.ensure_future(run_daily_pipeline(_config)),
        "cron",
        hour=server_cfg["daily_run_hour"],
        minute=server_cfg["daily_run_minute"],
        timezone=server_cfg["timezone"],
        id="daily_pipeline",
    )
    _scheduler.start()
    logger.info(
        "Scheduler started: daily pipeline at %02d:%02d %s",
        server_cfg["daily_run_hour"],
        server_cfg["daily_run_minute"],
        server_cfg["timezone"],
    )


# ── Helpers ──────────────────────────────────────────────────────

def _build_signal_from_screened(screened: dict) -> dict:
    """Translate a screened stock dict into the signal dict shape
    that ``generate_mock_report()`` expects.

    The screened stock DB row has top-level fields (ticker, sector, etc.)
    plus an ``analysis`` sub-dict with detailed metrics.  The mock report
    generator reads ``fundamental.points`` text blobs through
    ``_parse_anchors()`` regex, so we synthesise anchor-bearing strings
    from the numeric values available.
    """
    analysis = screened.get("analysis") or {}
    ticker = screened["ticker"]
    sector = screened.get("sector") or analysis.get("sector") or "Industrials"
    price = screened.get("current_price") or analysis.get("current_price") or 50.0

    def _safe(val, default=0):
        """Return default if val is None or NaN."""
        if val is None:
            return default
        try:
            if math.isnan(val) or math.isinf(val):
                return default
        except TypeError:
            pass
        return val

    # Derive confidence from DCF upside (capped 0.55–0.90)
    dcf_upside = _safe(screened.get("dcf_upside_pct") or analysis.get("dcf_upside_pct"), 0)
    confidence = min(max(0.55 + abs(dcf_upside) * 0.3, 0.55), 0.90)

    # Build fundamental.points with anchored text for _parse_anchors()
    points = []
    fcf_yield = _safe(screened.get("fcf_yield") or analysis.get("fcf_yield"), None)
    if fcf_yield is not None:
        points.append(f"FCF yield {fcf_yield * 100:.1f}%")
    wacc = _safe(screened.get("wacc") or analysis.get("wacc"), None)
    if wacc is not None:
        points.append(f"{wacc * 100:.1f}% WACC")
    mos = _safe(screened.get("margin_of_safety") or analysis.get("dcf_margin_of_safety"), None)
    if mos is not None:
        points.append(f"Margin of safety {mos * 100:.0f}%")
    if dcf_upside:
        points.append(f"DCF upside {abs(dcf_upside) * 100:.0f}%")
    iv = _safe(screened.get("intrinsic_value") or analysis.get("intrinsic_value_per_share"), None)
    if iv is not None:
        points.append(f"Intrinsic value ${iv:.2f} per share")
    mc = _safe(screened.get("market_cap") or analysis.get("market_cap"), None)
    if mc is not None:
        mc_label = f"${mc / 1e9:.1f}B" if mc >= 1e9 else f"${mc / 1e6:.0f}M"
        points.append(f"{mc_label} market cap")
    roe = _safe(analysis.get("roe"), None)
    if roe is not None:
        points.append(f"ROIC {roe * 100:.1f}%")
    de = _safe(analysis.get("debt_to_equity"), None)
    if de is not None:
        points.append(f"Debt-to-equity {de:.2f}")
    pe = _safe(analysis.get("pe_ratio"), None)
    if pe is not None:
        points.append(f"P/E ratio {pe:.1f}")
    ev_fcf = _safe(analysis.get("ev_to_fcf"), None)
    if ev_fcf is not None:
        points.append(f"EV/FCF {ev_fcf:.1f}x")

    # Reasons as macro points
    reasons = analysis.get("reasons") or []

    return {
        "ticker": ticker,
        "short_name": screened.get("short_name") or "",
        "sector": sector,
        "action": "BUY" if dcf_upside > 0 else "HOLD",
        "confidence": confidence,
        "entry_price": price,
        "fundamental": {"points": points},
        "macro": {"points": reasons[:3]},
        "technical": {"points": []},
        "ml_insight": (f"Screened stock with {dcf_upside * 100:+.0f}% DCF upside and "
                       f"{mos * 100:.0f}% margin of safety." if mos
                       else f"Screened stock with {dcf_upside * 100:+.0f}% DCF upside."),
        "risk_context": "Valuation based on DCF model assumptions. "
                        "Actual results may differ materially from projections.",
    }


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/signals")
async def get_signals():
    """Get current buy/sell signal list."""
    signals = _db.get_latest_signals()
    # Return summaries for the list view
    summaries = []
    for s in signals:
        summaries.append({
            "ticker": s.get("ticker"),
            "short_name": s.get("short_name", ""),
            "action": s.get("action"),
            "confidence": s.get("confidence"),
            "predicted_return_5d": s.get("predicted_return_5d"),
            "entry_price": s.get("entry_price"),
            "sector": s.get("sector"),
            "generated_at": s.get("generated_at"),
        })
    return {"signals": summaries, "count": len(summaries)}


@app.get("/signals/history")
async def get_signal_history(limit: int = 100):
    """Get historical recommendations."""
    history = _db.get_signal_history(limit=limit)
    return {"history": history, "count": len(history)}


@app.get("/signals/{ticker}")
async def get_signal_detail(ticker: str):
    """Get full recommendation detail for a ticker."""
    detail = _db.get_signal_detail(ticker.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No signal found for {ticker}")
    return detail


@app.post("/device-token")
async def register_device_token(req: DeviceTokenRequest):
    """Register an APNs device token for push notifications."""
    _db.save_device_token(req.token)
    return {"status": "registered"}


@app.get("/screened")
async def get_screened_stocks(limit: int = 20):
    """Get dynamically screened stocks ranked by DCF undervaluation."""
    stocks = _db.get_screened_stocks(limit=limit)
    return _sanitize_for_json({"stocks": stocks, "count": len(stocks)})


@app.get("/screened/{ticker}")
async def get_screened_stock_detail(ticker: str):
    """Get full analysis detail for a screened stock."""
    detail = _db.get_screened_stock_detail(ticker.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No screened data for {ticker}")
    return _sanitize_for_json(detail)


@app.post("/pipeline/run")
async def trigger_pipeline():
    """Manually trigger the daily pipeline (for testing)."""
    results = await run_daily_pipeline(_config)
    return {"status": "complete", "signals_generated": len(results)}


@app.get("/investor-report/{ticker}")
async def get_investor_report(ticker: str):
    """Generate a concise investor report in markdown for a ticker."""
    ticker_upper = ticker.upper()
    screened = _db.get_screened_stock_detail(ticker_upper)
    if screened is None:
        raise HTTPException(status_code=404, detail="No screened data found")
    a = screened.get("analysis", {})
    price = screened.get("current_price", 0)
    iv = screened.get("intrinsic_value", 0)
    mos = screened.get("margin_of_safety", 0)
    upside = ((iv / price - 1) * 100) if price and iv and price > 0 else 0
    mkt_cap = screened.get("market_cap", 0)
    cap_str = f"${mkt_cap / 1e9:.1f}B" if mkt_cap and mkt_cap >= 1e9 else f"${(mkt_cap or 0) / 1e6:.0f}M"
    name = screened.get("short_name", ticker_upper)
    sector = screened.get("sector", "")
    industry = screened.get("industry", "")
    piotroski = a.get("piotroski_f_score")
    fcf_yield = a.get("fcf_yield")
    wacc = a.get("wacc")
    roic_spread = a.get("roic_vs_wacc_spread")
    altman_z = a.get("altman_z_score")
    pe = a.get("pe_ratio")
    roe = a.get("roe")
    dte = a.get("debt_to_equity")
    int_cov = a.get("interest_coverage")
    curr_ratio = a.get("current_ratio")
    ev_fcf = a.get("ev_to_fcf")

    def _f(v, fmt="pct"):
        if v is None:
            return "N/A"
        if fmt == "pct":
            return f"{v * 100:.1f}%"
        if fmt == "x":
            return f"{v:.1f}x"
        if fmt == "score":
            return f"{v:.0f}/9"
        if fmt == "num":
            return f"{v:.1f}"
        return str(v)

    md = f"# {name} ({ticker_upper})\n\n"
    md += f"**Rating: Strong Buy | Price Target: ${iv:.2f} | Current Price: ${price:.2f} | Upside: +{upside:.0f}%**\n\n"
    md += f"*Sector: {sector} | Industry: {industry} | Market Cap: {cap_str}*\n\n---\n\n"
    md += "## Investment Thesis\n\n"
    md += f"{name} trades at ${price:.2f}, a {_f(mos)} discount to our DCF intrinsic value of ${iv:.2f}. "
    md += f"The stock scores {_f(piotroski, 'score')} on the Piotroski F-Score, signalling strong financial health. "
    if fcf_yield:
        md += f"A {_f(fcf_yield)} free cash flow yield "
    if pe:
        md += f"and {pe:.1f}x trailing P/E "
    md += "indicate the company is generating substantial cash relative to its valuation.\n\n"
    md += "## Valuation\n\n"
    md += f"Our Damodaran-aligned FCFF discounted cash flow model produces an intrinsic value of ${iv:.2f} per share. "
    if wacc:
        md += f"WACC is {_f(wacc)}. "
    md += f"Margin of safety: {_f(mos)}. "
    if ev_fcf:
        md += f"EV/FCF: {_f(ev_fcf, 'x')}. "
    if roic_spread is not None:
        md += f"ROIC exceeds cost of capital by {_f(roic_spread)}, confirming value creation above cost of capital.\n\n"
    else:
        md += "\n\n"
    md += "## Financial Profile\n\n"
    if roe:
        md += f"Return on equity: {_f(roe)}. "
    if curr_ratio:
        md += f"Current ratio: {_f(curr_ratio, 'x')}. "
    if int_cov:
        md += f"Interest coverage: {_f(int_cov, 'x')}. "
    if dte is not None:
        md += f"Debt-to-equity: {dte:.0f}%. "
    if altman_z:
        label = "safe zone" if altman_z > 3 else "grey zone" if altman_z > 1.8 else "distress zone"
        md += f"Altman Z-Score: {altman_z:.2f} ({label})."
    md += "\n\n"
    md += "## Key Risks\n\n"
    reasons = a.get("verification", {})
    md += f"Market cap of {cap_str} introduces position sizing considerations. "
    if dte and dte > 100:
        md += f"Elevated leverage ({dte:.0f}% D/E) requires monitoring. "
    md += "Government contract concentration, budget sequestration risk, and procurement delays are sector-wide concerns.\n\n"
    md += "---\n\n*Generated by SignalBoard Analytics Engine. For research purposes only.*\n"
    return {"ticker": ticker_upper, "markdown": md}


@app.get("/report/{ticker}")
async def get_report(ticker: str):
    """Generate a full financial report for a ticker."""
    ticker_upper = ticker.upper()
    signal_data = _db.get_signal_detail(ticker_upper)
    if signal_data is not None:
        report = generate_mock_report(signal_data)
        return _sanitize_for_json(report)
    # Fallback: adapt screened stock data into signal format
    screened = _db.get_screened_stock_detail(ticker_upper)
    if screened is None:
        raise HTTPException(status_code=404, detail="No data found")
    signal_data = _build_signal_from_screened(screened)
    report = generate_mock_report(signal_data)
    return _sanitize_for_json(report)


# ── Web frontend ─────────────────────────────────────────────

_web_dir = Path(__file__).resolve().parent.parent / "web"


@app.get("/detail.html")
async def detail_page():
    return FileResponse(_web_dir / "detail.html")


@app.get("/about.html")
async def about_page():
    return FileResponse(_web_dir / "about.html")


# Mount static files (CSS, JS) — must come after API routes
app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")


@app.get("/")
async def index_page():
    return FileResponse(_web_dir / "index.html")


# ── Run directly ─────────────────────────────────────────────────

def main():
    import uvicorn
    config = load_config()
    server_cfg = config["server"]
    uvicorn.run(
        "server.app:app",
        host=server_cfg["host"],
        port=server_cfg["port"],
        reload=False,
    )


if __name__ == "__main__":
    main()
