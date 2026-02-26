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
