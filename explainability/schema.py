"""Recommendation data schema — the JSON structure served to the iOS app."""

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ExplanationSection:
    """One category of explanation (technical, fundamental, macro, ml)."""
    points: list[str] = field(default_factory=list)


@dataclass
class MLInsight:
    """ML model-specific explanation."""
    predicted_return: str
    confidence_percentile: str
    top_features: list[str] = field(default_factory=list)


@dataclass
class RiskContext:
    """Risk parameters for the recommendation."""
    sector_exposure_after: str
    portfolio_positions_after: str
    correlation_note: str = ""


@dataclass
class Recommendation:
    """Full recommendation payload — what the app receives for each signal."""
    ticker: str
    action: str                 # BUY or SELL
    confidence: float           # 0-1
    predicted_return_5d: float
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop_trigger: float
    time_stop_days: int
    position_size_pct: float
    sector: str
    short_name: str

    technical: ExplanationSection = field(default_factory=ExplanationSection)
    fundamental: ExplanationSection = field(default_factory=ExplanationSection)
    macro: ExplanationSection = field(default_factory=ExplanationSection)
    ml_insight: MLInsight | None = None
    risk_context: RiskContext | None = None
    historical_context: list[str] = field(default_factory=list)

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary(self) -> dict:
        """Short version for the signal list view."""
        return {
            "ticker": self.ticker,
            "short_name": self.short_name,
            "action": self.action,
            "confidence": self.confidence,
            "predicted_return_5d": self.predicted_return_5d,
            "sector": self.sector,
            "generated_at": self.generated_at,
        }
