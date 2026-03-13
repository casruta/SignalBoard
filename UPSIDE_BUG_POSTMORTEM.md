# Upside Data Inconsistency -- Postmortem

**Date**: 2026-03-12
**Status**: Resolved

## Bug

The Screen page showed a different DCF upside percentage than the detail/article page for the same stock. For example, CADE displayed 354% upside on the Screen page but ~200% on the detail page.

## Root Cause

Two independent calculations of upside existed:

1. **Screen page**: reads `dcf_upside_pct` directly from the `screened_stocks` DB table (the real screener value, capped to [-0.5, 2.0]).
2. **Detail page**: calls `/report/{ticker}`, which runs `generate_mock_report()`. This back-solves a new DCF from anchor text, producing a different implied price and upside because:
   - Share count was re-derived as `market_cap / price` (may differ from actual diluted shares)
   - Net debt defaulted to `market_cap * 0.15` instead of using the screener's real calculation
   - A minimum price floor of `price * 0.15` distorted highly undervalued stocks
   - The screener's [-0.5, 2.0] upside cap was not applied to the back-solved value

An existing override at `mock_financials.py:725` checked for `signal.get("intrinsic_value_per_share")`, but `_build_signal_from_screened()` in `app.py` never passed that key -- so the override never fired for screened stocks.

## Fix

Three changes across two files:

### 1. `server/app.py` -- `_build_signal_from_screened()`

Added two keys to the returned signal dict:
- `intrinsic_value_per_share`: feeds the existing implied-price override in `mock_financials.py`
- `real_dcf_upside_pct`: the real capped upside from the screener

### 2. `server/mock_financials.py` -- DCF output upside

When `real_dcf_upside_pct` is present in the signal, use it directly instead of recalculating from the back-solved implied price.

### 3. `server/mock_financials.py` -- header upside

Same fix applied to the report header block so `header.upside_pct` also uses the real screener value.

## Prevention

The single-source-of-truth for upside is the screener's `dcf_upside_pct`. Any new code path that displays upside must pass the real value through rather than recalculating it.
