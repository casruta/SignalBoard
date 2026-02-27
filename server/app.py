"""FastAPI application — REST API + web frontend."""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config_loader import load_config
from server.database import Database
from server.scheduler import run_daily_pipeline

logger = logging.getLogger(__name__)

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
    """Get dynamically screened stocks ranked by composite quality score."""
    stocks = _db.get_screened_stocks(limit=limit)
    return {"stocks": stocks, "count": len(stocks)}


@app.get("/screened/{ticker}")
async def get_screened_stock_detail(ticker: str):
    """Get full analysis detail for a screened stock."""
    detail = _db.get_screened_stock_detail(ticker.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No screened data for {ticker}")
    return detail


@app.post("/pipeline/run")
async def trigger_pipeline():
    """Manually trigger the daily pipeline (for testing)."""
    results = await run_daily_pipeline(_config)
    return {"status": "complete", "signals_generated": len(results)}


# ── Web frontend ─────────────────────────────────────────────

_web_dir = Path(__file__).resolve().parent.parent / "web"


@app.get("/detail.html")
async def detail_page():
    return FileResponse(_web_dir / "detail.html")


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
