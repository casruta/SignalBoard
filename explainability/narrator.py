"""Convert raw signal values into human-readable explanations."""

import numpy as np


# ── Technical signal narrations ──────────────────────────────────

_TECHNICAL_NARRATORS = {}


def _tech(name):
    """Decorator to register a technical signal narrator."""
    def decorator(fn):
        _TECHNICAL_NARRATORS[name] = fn
        return fn
    return decorator


@_tech("rsi_14")
def _narrate_rsi(value, **kw):
    if value < 30:
        return f"RSI oversold at {value:.0f} (below 30 threshold) — historically a bounce zone"
    elif value > 70:
        return f"RSI overbought at {value:.0f} (above 70 threshold) — potential pullback"
    return None


@_tech("sma_crossover")
def _narrate_sma_cross(value, **kw):
    if value == 1:
        return "10-day SMA crossed above 50-day SMA — bullish trend signal"
    elif value == 0:
        return "10-day SMA below 50-day SMA — bearish trend"
    return None


@_tech("macd_histogram")
def _narrate_macd(value, **kw):
    if value > 0:
        return f"MACD histogram positive ({value:.3f}) — bullish momentum"
    elif value < 0:
        return f"MACD histogram negative ({value:.3f}) — bearish momentum"
    return None


@_tech("bb_pct")
def _narrate_bb(value, **kw):
    if value < 0.1:
        return f"Price near lower Bollinger Band (%B = {value:.2f}) — potential mean reversion"
    elif value > 0.9:
        return f"Price near upper Bollinger Band (%B = {value:.2f}) — extended above average"
    return None


@_tech("zscore_20")
def _narrate_zscore(value, **kw):
    if value < -2:
        return f"Price z-score at {value:.1f} — significantly below 20-day mean"
    elif value > 2:
        return f"Price z-score at {value:.1f} — significantly above 20-day mean"
    return None


@_tech("momentum_5")
def _narrate_mom5(value, **kw):
    return f"5-day momentum: {value:+.1%}"


@_tech("momentum_20")
def _narrate_mom20(value, **kw):
    return f"20-day momentum: {value:+.1%}"


@_tech("volume_ratio")
def _narrate_volume(value, **kw):
    if value > 1.5:
        return f"Volume {value:.1f}x above 20-day average — elevated interest"
    elif value < 0.5:
        return f"Volume {value:.1f}x below 20-day average — low activity"
    return None


# ── Fundamental signal narrations ────────────────────────────────

def narrate_fundamental(feature: str, value, fundamentals: dict | None = None) -> str | None:
    """Generate explanation for a fundamental feature."""
    if feature == "fund_pe_vs_sector" and value is not None and not np.isnan(value):
        if value < -0.1:
            return f"P/E ratio is {abs(value):.0%} below sector average — potential value"
        elif value > 0.1:
            return f"P/E ratio is {value:.0%} above sector average — trading at a premium"

    if feature == "fund_value_score" and value is not None:
        if value > 0.7:
            return f"Value score: {value:.2f} (top quartile) — attractive valuation"
        elif value < 0.3:
            return f"Value score: {value:.2f} (bottom quartile) — expensive relative to peers"

    if feature == "fund_quality_score" and value is not None:
        if value > 0.7:
            return f"Quality score: {value:.2f} — strong fundamentals (high ROE, low debt)"
        elif value < 0.3:
            return f"Quality score: {value:.2f} — weaker fundamentals"

    if feature == "fund_growth_score" and value is not None:
        if value > 0.7:
            return f"Growth score: {value:.2f} — strong earnings and revenue growth"

    if feature == "fund_quality_grade" and value is not None:
        return f"Quality grade: {value}"

    return None


# ── Macro signal narrations ──────────────────────────────────────

def narrate_macro(feature: str, value) -> str | None:
    """Generate explanation for a macro feature."""
    if feature == "macro_yield_curve_slope":
        if value is not None and not np.isnan(value):
            if value < 0:
                return f"Yield curve inverted ({value:.2f}%) — historically signals recession risk"
            elif value > 0.5:
                return f"Yield curve normalizing ({value:.2f}%) — favors equity risk"

    if feature == "macro_vix_regime":
        if value == 0:
            return "VIX in low-volatility regime (< 15) — calm markets, risk-on"
        elif value == 1:
            return "VIX in normal regime (15-25)"
        elif value == 2:
            return "VIX in high-volatility regime (> 25) — elevated fear"

    if feature == "macro_vix_trend":
        if value is not None and not np.isnan(value):
            if value < -0.1:
                return f"VIX declining ({value:+.1%} over 10 days) — fear subsiding"
            elif value > 0.1:
                return f"VIX rising ({value:+.1%} over 10 days) — fear increasing"

    if feature == "macro_risk_regime":
        if value == 1:
            return "Macro regime: Risk-On — favorable for equities"
        elif value == -1:
            return "Macro regime: Risk-Off — caution warranted"
        elif value == 0:
            return "Macro regime: Neutral"

    if feature == "macro_oil_momentum_20":
        if value is not None and not np.isnan(value):
            return f"Oil (WTI) 20-day momentum: {value:+.1%}"

    return None


# ── Main narration function ──────────────────────────────────────

def narrate_signal(
    feature: str,
    value,
    fundamentals: dict | None = None,
) -> str | None:
    """Generate a human-readable explanation for any feature."""
    # Technical
    if feature in _TECHNICAL_NARRATORS:
        return _TECHNICAL_NARRATORS[feature](value)

    # Fundamental
    if feature.startswith("fund_"):
        return narrate_fundamental(feature, value, fundamentals)

    # Macro
    if feature.startswith("macro_"):
        return narrate_macro(feature, value)

    return None


def build_explanation(
    feature_contributions: dict[str, list[tuple[str, float]]],
    feature_values: dict[str, float],
    fundamentals: dict | None = None,
    max_per_category: int = 3,
) -> dict[str, list[str]]:
    """Build complete categorized explanation from SHAP decomposition.

    Parameters
    ----------
    feature_contributions : {category: [(feature, shap_value), ...]}
        from decomposer.decompose_by_category()
    feature_values : {feature: raw_value} for the prediction row
    max_per_category : max explanation points per category

    Returns
    -------
    {category: [human-readable string, ...]}
    """
    explanation = {}
    for category, features in feature_contributions.items():
        points = []
        for feat, shap_val in features:
            raw_val = feature_values.get(feat)
            text = narrate_signal(feat, raw_val, fundamentals)
            if text:
                points.append(text)
            if len(points) >= max_per_category:
                break
        explanation[category] = points
    return explanation
