"""Model verification for DCF and DDM valuation outputs.

Validates that model outputs are reasonable, catching edge cases where
the models produce nonsensical results (e.g. negative WACC, intrinsic
values that are orders of magnitude away from market price).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of a single model verification pass."""

    model_name: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics_checked: int = 0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics_checked": self.metrics_checked,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid(value) -> bool:
    """Return True if *value* is a finite number (not None/NaN/inf)."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _get_price(info: dict | None) -> float | None:
    """Extract current market price from the info dict."""
    if info is None:
        return None
    price = info.get("currentPrice")
    if _is_valid(price) and price > 0:
        return float(price)
    return None


# ---------------------------------------------------------------------------
# DCF verification
# ---------------------------------------------------------------------------

def verify_dcf(dcf_result: dict, info: dict) -> VerificationResult:
    """Verify that DCF model outputs fall within reasonable bounds.

    Parameters
    ----------
    dcf_result : dict
        Output of ``compute_dcf_valuation``.
    info : dict
        The yfinance ``.info`` dict (needs ``currentPrice`` at minimum).

    Returns
    -------
    VerificationResult
    """
    result = VerificationResult(model_name="DCF", passed=True)
    price = _get_price(info)

    # 1. WACC bounds
    wacc = dcf_result.get("wacc")
    if _is_valid(wacc):
        result.metrics_checked += 1
        if wacc < 0.05 or wacc > 0.30:
            result.errors.append(
                f"WACC {wacc:.4f} outside hard bounds [0.05, 0.30]"
            )
        elif wacc < 0.07:
            result.warnings.append(f"WACC {wacc:.4f} is unusually low (< 7%)")
        elif wacc > 0.20:
            result.warnings.append(f"WACC {wacc:.4f} is unusually high (> 20%)")

    # 2. Intrinsic value vs market price (tightened from [0.1x, 10x])
    iv = dcf_result.get("intrinsic_value_per_share")
    if _is_valid(iv) and price is not None:
        result.metrics_checked += 1
        if iv < 0.2 * price or iv > 5 * price:
            result.errors.append(
                f"Intrinsic value ${iv:.2f} is outside "
                f"[0.2x, 5x] of market price ${price:.2f}"
            )
        elif iv < 0.3 * price or iv > 3 * price:
            result.warnings.append(
                f"Intrinsic value ${iv:.2f} is outside "
                f"[0.3x, 3x] of market price ${price:.2f}"
            )

    # 3. DCF upside percentage
    upside = dcf_result.get("dcf_upside_pct")
    if _is_valid(upside):
        result.metrics_checked += 1
        if upside < -0.90 or upside > 5.0:
            result.errors.append(
                f"DCF upside {upside:.2%} outside bounds [-90%, +500%]"
            )
        elif upside > 2.0:
            result.warnings.append(
                f"DCF upside {upside:.2%} is very high (> 200%)"
            )
        elif upside < -0.5:
            result.warnings.append(
                f"DCF upside {upside:.2%} is very negative (< -50%)"
            )

    # 4. FCF yield
    fcf_yield = dcf_result.get("fcf_yield")
    if _is_valid(fcf_yield):
        result.metrics_checked += 1
        if fcf_yield < -0.10 or fcf_yield > 0.30:
            result.warnings.append(
                f"FCF yield {fcf_yield:.2%} outside expected range [-10%, 30%]"
            )
        elif fcf_yield < 0:
            result.warnings.append(f"FCF yield is negative ({fcf_yield:.2%})")

    # 5. Implied growth rate
    implied_growth = dcf_result.get("implied_growth_rate")
    if _is_valid(implied_growth):
        result.metrics_checked += 1
        # Hard error: implied growth >= WACC means Gordon Growth diverges
        if _is_valid(wacc) and implied_growth >= wacc:
            result.errors.append(
                f"Implied growth rate ({implied_growth:.2%}) >= WACC ({wacc:.2%}) "
                f"— Gordon Growth Model cannot converge; valuation unreliable"
            )
        elif implied_growth < 0:
            result.warnings.append(
                f"Implied growth rate is negative ({implied_growth:.2%})"
            )
        elif implied_growth > 0.25:
            result.warnings.append(
                f"Implied growth rate {implied_growth:.2%} is very high (> 25%)"
            )

    # 6. ROIC vs WACC spread
    spread = dcf_result.get("roic_vs_wacc_spread")
    if _is_valid(spread):
        result.metrics_checked += 1
        if spread < -0.20 or spread > 0.50:
            result.warnings.append(
                f"ROIC-WACC spread {spread:.2%} outside expected range "
                f"[-20%, 50%]"
            )

    # 7. Margin of safety
    mos = dcf_result.get("margin_of_safety")
    if _is_valid(mos):
        result.metrics_checked += 1
        if mos < -2.0 or mos > 0.95:
            result.warnings.append(
                f"Margin of safety {mos:.2%} outside expected range "
                f"[-200%, 95%]"
            )

    # 8. Terminal value divergence (exit multiple cross-check)
    tv_div = dcf_result.get("tv_divergence_pct")
    if _is_valid(tv_div):
        result.metrics_checked += 1
        if tv_div > 1.0:
            result.errors.append(
                f"Terminal value divergence {tv_div:.0%} exceeds 100% "
                f"— Gordon Growth and exit multiple drastically disagree"
            )
        elif tv_div > 0.5:
            result.warnings.append(
                f"Terminal value divergence {tv_div:.0%} exceeds 50%"
            )

    # 9. Implied reinvestment rate consistency
    reinv = dcf_result.get("implied_reinvestment_rate")
    if _is_valid(reinv):
        result.metrics_checked += 1
        if reinv > 1.5:
            result.warnings.append(
                f"Implied reinvestment rate {reinv:.1%} > 150% "
                f"— projected growth may exceed what reinvestment supports"
            )

    # 10. Scenario range (model uncertainty)
    scenario_range = dcf_result.get("scenario_range_pct")
    if _is_valid(scenario_range):
        result.metrics_checked += 1
        if scenario_range > 1.0:
            result.warnings.append(
                f"Scenario range {scenario_range:.0%} > 100% "
                f"— valuation highly sensitive to assumptions"
            )

    # Final pass/fail: errors mean failure; warnings are acceptable.
    result.passed = len(result.errors) == 0

    for w in result.warnings:
        logger.warning("[DCF verification] %s", w)
    for e in result.errors:
        logger.error("[DCF verification] %s", e)

    return result


# ---------------------------------------------------------------------------
# DDM verification
# ---------------------------------------------------------------------------

def verify_ddm(ddm_result: dict, info: dict) -> VerificationResult:
    """Verify that DDM model outputs fall within reasonable bounds.

    Parameters
    ----------
    ddm_result : dict
        Output of the DDM valuation module.
    info : dict
        The yfinance ``.info`` dict (needs ``currentPrice`` at minimum).

    Returns
    -------
    VerificationResult
    """
    result = VerificationResult(model_name="DDM", passed=True)

    # 1. Check applicability first
    applicable = ddm_result.get("ddm_applicable")
    if applicable is False:
        result.metrics_checked += 1
        result.warnings.append("DDM not applicable (no dividends)")
        return result

    price = _get_price(info)

    # 2. Gordon Growth Model constraint: required_return > growth_rate_used
    required_return = ddm_result.get("required_return")
    growth_used = ddm_result.get("growth_rate_used")
    if _is_valid(required_return) and _is_valid(growth_used):
        result.metrics_checked += 1
        if growth_used >= required_return:
            result.errors.append(
                f"Gordon Growth Model violated: growth_rate_used "
                f"({growth_used:.4f}) >= required_return ({required_return:.4f})"
            )

    # 3. DDM intrinsic value vs market price
    ddm_iv = ddm_result.get("ddm_intrinsic_value")
    if _is_valid(ddm_iv) and price is not None:
        result.metrics_checked += 1
        if ddm_iv < 0.1 * price or ddm_iv > 10 * price:
            result.errors.append(
                f"DDM intrinsic value ${ddm_iv:.2f} is outside "
                f"[0.1x, 10x] of market price ${price:.2f}"
            )

    # 4. Sustainable growth rate
    sust_growth = ddm_result.get("sustainable_growth_rate")
    if _is_valid(sust_growth):
        result.metrics_checked += 1
        if sust_growth < 0:
            result.warnings.append(
                f"Sustainable growth rate is negative ({sust_growth:.2%})"
            )
        elif sust_growth > 0.20:
            result.warnings.append(
                f"Sustainable growth rate {sust_growth:.2%} is very high (> 20%)"
            )
        elif sust_growth > 0.15:
            result.warnings.append(
                f"Sustainable growth rate {sust_growth:.2%} is elevated (> 15%)"
            )

    # 5. Dividend stability (coefficient of variation)
    div_stability = ddm_result.get("dividend_stability")
    if _is_valid(div_stability):
        result.metrics_checked += 1
        if div_stability > 0.5:
            result.warnings.append(
                f"Dividend stability CV {div_stability:.3f} is high (> 0.5); "
                f"DDM inputs may be unreliable"
            )

    # 6. DDM upside percentage
    ddm_upside = ddm_result.get("ddm_upside_pct")
    if _is_valid(ddm_upside):
        result.metrics_checked += 1
        if ddm_upside < -0.90 or ddm_upside > 5.0:
            result.warnings.append(
                f"DDM upside {ddm_upside:.2%} outside expected range "
                f"[-90%, +500%]"
            )
        elif ddm_upside > 2.0:
            result.warnings.append(
                f"DDM upside {ddm_upside:.2%} is very high (> 200%)"
            )
        elif ddm_upside < -0.5:
            result.warnings.append(
                f"DDM upside {ddm_upside:.2%} is very negative (< -50%)"
            )

    # 7. growth_rate_used should be positive and < required_return
    if _is_valid(growth_used):
        result.metrics_checked += 1
        if _is_valid(required_return) and growth_used >= required_return:
            # Already captured as error in check 2, avoid duplicate
            pass
        elif growth_used < 0:
            result.warnings.append(
                f"growth_rate_used is negative ({growth_used:.4f})"
            )

    result.passed = len(result.errors) == 0

    for w in result.warnings:
        logger.warning("[DDM verification] %s", w)
    for e in result.errors:
        logger.error("[DDM verification] %s", e)

    return result


# ---------------------------------------------------------------------------
# Combined verification
# ---------------------------------------------------------------------------

def verify_all(
    dcf_result: dict,
    ddm_result: dict | None,
    info: dict,
) -> dict:
    """Run both DCF and DDM verifiers and return a combined report.

    Parameters
    ----------
    dcf_result : dict
        Output of ``compute_dcf_valuation``.
    ddm_result : dict or None
        Output of the DDM valuation module, or None if DDM was not run.
    info : dict
        The yfinance ``.info`` dict.

    Returns
    -------
    dict with keys: dcf_verification, ddm_verification, all_passed,
    total_warnings, total_errors, summary.
    """
    dcf_ver = verify_dcf(dcf_result, info)

    ddm_ver: VerificationResult | None = None
    if ddm_result is not None:
        ddm_ver = verify_ddm(ddm_result, info)

    total_warnings = len(dcf_ver.warnings)
    total_errors = len(dcf_ver.errors)
    if ddm_ver is not None:
        total_warnings += len(ddm_ver.warnings)
        total_errors += len(ddm_ver.errors)

    all_passed = dcf_ver.passed and (ddm_ver is None or ddm_ver.passed)

    # Build human-readable summary
    parts: list[str] = []

    dcf_status = "passed" if dcf_ver.passed else "FAILED"
    w_count = len(dcf_ver.warnings)
    if w_count:
        parts.append(f"DCF {dcf_status} with {w_count} warning{'s' if w_count != 1 else ''}")
    else:
        parts.append(f"DCF {dcf_status}")

    if ddm_ver is not None:
        ddm_status = "passed" if ddm_ver.passed else "FAILED"
        w_count = len(ddm_ver.warnings)
        if w_count:
            parts.append(f"DDM {ddm_status} with {w_count} warning{'s' if w_count != 1 else ''}")
        else:
            parts.append(f"DDM {ddm_status}")
    else:
        parts.append("DDM not run")

    summary = ". ".join(parts) + "."

    return {
        "dcf_verification": dcf_ver.to_dict(),
        "ddm_verification": ddm_ver.to_dict() if ddm_ver is not None else None,
        "all_passed": all_passed,
        "total_warnings": total_warnings,
        "total_errors": total_errors,
        "summary": summary,
    }


def log_verification_summary(verification: dict, ticker: str) -> None:
    """Log the verification results at appropriate severity levels.

    Parameters
    ----------
    verification : dict
        Output of ``verify_all``.
    ticker : str
        Stock ticker symbol for log context.
    """
    summary = verification.get("summary", "")
    total_errors = verification.get("total_errors", 0)
    total_warnings = verification.get("total_warnings", 0)

    if total_errors > 0:
        logger.error(
            "[%s] Model verification FAILED (%d error%s, %d warning%s): %s",
            ticker,
            total_errors,
            "s" if total_errors != 1 else "",
            total_warnings,
            "s" if total_warnings != 1 else "",
            summary,
        )
    elif total_warnings > 0:
        logger.warning(
            "[%s] Model verification passed with %d warning%s: %s",
            ticker,
            total_warnings,
            "s" if total_warnings != 1 else "",
            summary,
        )
    else:
        logger.info("[%s] Model verification passed: %s", ticker, summary)
