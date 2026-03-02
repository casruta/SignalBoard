"""SQLite database for storing recommendations and device tokens."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Text,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class RecommendationRow(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # BUY, SELL
    confidence = Column(Float, nullable=False)
    predicted_return_5d = Column(Float)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    sector = Column(String(50))
    short_name = Column(String(100))
    payload = Column(Text)  # Full recommendation JSON
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)


class OutcomeRow(Base):
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(Integer, index=True)
    ticker = Column(String(10))
    actual_return_5d = Column(Float)
    actual_return_10d = Column(Float)
    measured_at = Column(DateTime, default=datetime.utcnow)


class ScreenedStockRow(Base):
    __tablename__ = "screened_stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    short_name = Column(String(100))
    sector = Column(String(50))
    industry = Column(String(100))
    market_cap = Column(Float)
    composite_score = Column(Float)
    rank = Column(Integer)
    # DCF-focused component scores (0-1 each)
    dcf_upside_score = Column(Float)
    fcf_yield_score = Column(Float)
    roic_spread_score = Column(Float)
    # Key DCF metrics for display
    intrinsic_value = Column(Float)
    market_price = Column(Float)
    margin_of_safety_pct = Column(Float)
    fcf_yield_pct = Column(Float)
    roic_spread_pct = Column(Float)
    wacc_pct = Column(Float)
    # Full analysis payload
    payload = Column(Text)  # Full analysis JSON (DCF, deep fundamentals, etc.)
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)


class DeviceTokenRow(Base):
    __tablename__ = "device_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token = Column(String(200), unique=True, nullable=False)
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)


class Database:
    """Manages the SQLite database."""

    def __init__(self, db_path: str = "server/signals.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def _session(self) -> Session:
        return self._Session()

    # ── Recommendations ──────────────────────────────────────────

    def save_recommendations(self, recommendations: list[dict]) -> None:
        """Save a batch of recommendations."""
        with self._session() as session:
            for rec in recommendations:
                row = RecommendationRow(
                    ticker=rec["ticker"],
                    action=rec["action"],
                    confidence=rec["confidence"],
                    predicted_return_5d=rec.get("predicted_return_5d"),
                    entry_price=rec.get("entry_price"),
                    stop_loss=rec.get("stop_loss"),
                    take_profit=rec.get("take_profit"),
                    sector=rec.get("sector"),
                    short_name=rec.get("short_name", ""),
                    payload=json.dumps(rec),
                    generated_at=datetime.fromisoformat(rec.get("generated_at", datetime.utcnow().isoformat())),
                )
                session.add(row)
            session.commit()

    def get_latest_signals(self) -> list[dict]:
        """Get the most recent batch of recommendations."""
        with self._session() as session:
            latest_time = session.query(func.max(RecommendationRow.generated_at)).scalar()
            if latest_time is None:
                return []
            # Get all recs from the latest batch (within 1 hour of max time)
            cutoff = latest_time - timedelta(hours=1)
            rows = (
                session.query(RecommendationRow)
                .filter(RecommendationRow.generated_at >= cutoff)
                .filter(RecommendationRow.action.in_(["BUY", "SELL"]))
                .order_by(RecommendationRow.confidence.desc())
                .all()
            )
            return [json.loads(r.payload) for r in rows]

    def get_signal_detail(self, ticker: str) -> dict | None:
        """Get the latest recommendation for a specific ticker."""
        with self._session() as session:
            row = (
                session.query(RecommendationRow)
                .filter(RecommendationRow.ticker == ticker)
                .order_by(RecommendationRow.generated_at.desc())
                .first()
            )
            if row is None:
                return None
            return json.loads(row.payload)

    def get_signal_history(self, limit: int = 100) -> list[dict]:
        """Get historical recommendations."""
        with self._session() as session:
            rows = (
                session.query(RecommendationRow)
                .order_by(RecommendationRow.generated_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "ticker": r.ticker,
                    "action": r.action,
                    "confidence": r.confidence,
                    "generated_at": r.generated_at.isoformat() if r.generated_at else None,
                }
                for r in rows
            ]

    def clear_all_recommendations(self) -> None:
        """Delete all recommendations from the database."""
        with self._session() as session:
            session.query(RecommendationRow).delete()
            session.commit()

    # ── Screened Stocks ────────────────────────────────────────────

    def save_screened_stocks(self, stocks: list[dict]) -> None:
        """Save a batch of screened stocks (replaces previous batch)."""
        with self._session() as session:
            # Clear previous screened stocks
            session.query(ScreenedStockRow).delete()
            for s in stocks:
                row = ScreenedStockRow(
                    ticker=s["ticker"],
                    short_name=s.get("short_name", ""),
                    sector=s.get("sector", ""),
                    industry=s.get("industry", ""),
                    market_cap=s.get("market_cap"),
                    composite_score=s.get("composite_score"),
                    rank=s.get("rank"),
                    dcf_upside_score=s.get("dcf_upside_score"),
                    fcf_yield_score=s.get("fcf_yield_score"),
                    roic_spread_score=s.get("roic_spread_score"),
                    intrinsic_value=s.get("intrinsic_value"),
                    market_price=s.get("market_price"),
                    margin_of_safety_pct=s.get("margin_of_safety_pct"),
                    fcf_yield_pct=s.get("fcf_yield_pct"),
                    roic_spread_pct=s.get("roic_spread_pct"),
                    wacc_pct=s.get("wacc_pct"),
                    payload=json.dumps(s.get("analysis", {})),
                    generated_at=datetime.utcnow(),
                )
                session.add(row)
            session.commit()

    def get_screened_stocks(self, limit: int = 20) -> list[dict]:
        """Get the current screened stocks, ranked by composite score."""
        with self._session() as session:
            rows = (
                session.query(ScreenedStockRow)
                .order_by(ScreenedStockRow.composite_score.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "ticker": r.ticker,
                    "short_name": r.short_name,
                    "sector": r.sector,
                    "industry": r.industry,
                    "market_cap": r.market_cap,
                    "composite_score": r.composite_score,
                    "rank": r.rank,
                    "dcf_upside_score": r.dcf_upside_score,
                    "fcf_yield_score": r.fcf_yield_score,
                    "roic_spread_score": r.roic_spread_score,
                    "intrinsic_value": r.intrinsic_value,
                    "market_price": r.market_price,
                    "margin_of_safety_pct": r.margin_of_safety_pct,
                    "fcf_yield_pct": r.fcf_yield_pct,
                    "roic_spread_pct": r.roic_spread_pct,
                    "wacc_pct": r.wacc_pct,
                    "generated_at": r.generated_at.isoformat() if r.generated_at else None,
                }
                for r in rows
            ]

    def get_screened_stock_detail(self, ticker: str) -> dict | None:
        """Get full screened stock analysis for a ticker."""
        with self._session() as session:
            row = (
                session.query(ScreenedStockRow)
                .filter(ScreenedStockRow.ticker == ticker.upper())
                .order_by(ScreenedStockRow.generated_at.desc())
                .first()
            )
            if row is None:
                return None
            result = {
                "ticker": row.ticker,
                "short_name": row.short_name,
                "sector": row.sector,
                "industry": row.industry,
                "market_cap": row.market_cap,
                "composite_score": row.composite_score,
                "rank": row.rank,
                "dcf_upside_score": row.dcf_upside_score,
                "fcf_yield_score": row.fcf_yield_score,
                "roic_spread_score": row.roic_spread_score,
                "intrinsic_value": row.intrinsic_value,
                "market_price": row.market_price,
                "margin_of_safety_pct": row.margin_of_safety_pct,
                "fcf_yield_pct": row.fcf_yield_pct,
                "roic_spread_pct": row.roic_spread_pct,
                "wacc_pct": row.wacc_pct,
                "generated_at": row.generated_at.isoformat() if row.generated_at else None,
            }
            if row.payload:
                result["analysis"] = json.loads(row.payload)
            return result

    # ── Device Tokens ────────────────────────────────────────────

    def save_device_token(self, token: str) -> None:
        with self._session() as session:
            existing = session.query(DeviceTokenRow).filter_by(token=token).first()
            if existing:
                existing.last_used = datetime.utcnow()
            else:
                session.add(DeviceTokenRow(token=token))
            session.commit()

    def get_device_tokens(self) -> list[str]:
        with self._session() as session:
            rows = session.query(DeviceTokenRow).all()
            return [r.token for r in rows]
