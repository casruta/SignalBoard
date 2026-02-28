"""Dynamic universe builder for small-mid cap stock discovery."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UniverseConfig:
    """Configuration for universe construction parameters."""

    min_market_cap: float = 300_000_000  # $300M
    max_market_cap: float = 20_000_000_000  # $20B
    min_daily_volume: int = 100_000
    min_history_years: int = 2
    exclude_mega_caps: bool = True


# ---------------------------------------------------------------------------
# Curated sector tickers -- ~200 real US-traded small-to-mid cap names
# Selection criteria:
#   * $300M-$20B market cap range (approximate at time of curation)
#   * Files 10-K / 10-Q with the SEC (no SPACs, no pre-revenue biotech)
#   * Reasonable daily volume for data availability
#   * Broad sector diversification
# ---------------------------------------------------------------------------

_SECTOR_TICKERS: Dict[str, List[str]] = {
    "technology": [
        # Mid-cap SaaS, infrastructure, cybersecurity, semiconductor
        "DOCN", "APPF", "CALX", "VERX", "PCOR", "DT", "SITM", "CWAN",
        "BRZE", "ALTR", "TENB", "RBRK", "VRNS", "QTWO", "PAYO",
        "PRGS", "SMAR", "JAMF", "QLYS", "PING", "WK", "CCCS",
        "LITE", "DIOD", "AMBA", "ONTO", "FORM", "CEVA", "OLED",
    ],
    "healthcare": [
        # Medical devices, diagnostics, healthcare services, biotech w/ revenue
        "GKOS", "PODD", "EXAS", "CORT", "NUVB", "HIMS", "OSCR",
        "NVCR", "INSP", "IRTC", "TNDM", "NVST", "OMCL", "PRCT",
        "RVMD", "CRNX", "NTRA", "AZTA", "LNTH", "TCMD", "SDGR",
        "ENSG", "AMED", "ACHC", "SGRY", "CERT",
    ],
    "industrials": [
        # Precision manufacturing, infrastructure, defense, building products
        "ITT", "BLDR", "GFF", "AGCO", "RRX", "WMS", "ATKR",
        "BWXT", "HXL", "ESAB", "KTOS", "TDG", "AVAV", "AXON",
        "CSWI", "RBC", "GGG", "NPO", "AIT", "PIPR", "SITE",
        "SPSC", "EPAC", "ZWS", "MBC",
    ],
    "energy": [
        # E&P, midstream, oilfield services
        "CEIX", "MTDR", "SM", "CTRA", "AR", "TRGP", "VNOM",
        "DINO", "MGY", "GPOR", "CHRD", "PTEN", "HP", "RRC",
        "NOG", "ESTE", "REPX", "CIVI", "LBRT",
    ],
    "consumer_discretionary": [
        # Retail, restaurants, brands, leisure
        "BOOT", "WING", "SHAK", "DKS", "DECK", "TXRH", "EAT",
        "PLAY", "GRMN", "FOXF", "SIG", "BURL", "WSM", "ODP",
        "CROX", "PLNT", "WOOF", "CAVA", "BROS", "LULU",
    ],
    "consumer_staples": [
        # Specialty food, personal care, packaged goods
        "ELF", "FRPT", "IPAR", "CELH", "HAIN", "THS", "USFD",
        "PFGC", "SPB", "SJM",
    ],
    "financials": [
        # Specialty finance, asset management, insurance, regional banks
        "STEP", "HOMB", "EVER", "IBKR", "LPLA", "PIPR",
        "HLNE", "VCTR", "GBCI", "CADE", "FNB", "PNFP",
        "KNSL", "RLI", "PLMR", "HLI", "EVR", "MC",
    ],
    "materials": [
        # Steel, specialty chemicals, building materials
        "STLD", "CMC", "ATI", "UFPI", "TREX", "CLF",
        "HUN", "IOSP", "CBT", "HAYN", "SXT", "KWR",
    ],
    "real_estate": [
        # Residential, manufactured housing, industrial REITs
        "INVH", "SUI", "ELS", "REXR", "NNN", "STAG",
        "IIPR", "FRPH", "UMH", "PLYM",
    ],
    "communication_services": [
        # Digital media, publishing, entertainment
        "ZD", "CARG", "CARS", "CXM", "MGNI", "PUBM",
        "TRMR", "DLO",
    ],
    "utilities": [
        # Small-mid cap utilities and clean energy
        "NOVA", "RUN", "ARRY", "NRG", "OGS", "UTL",
    ],
}

# Mega-cap tickers to always exclude, regardless of source
_MEGA_CAP_EXCLUSIONS: frozenset[str] = frozenset({
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B",
    "BRK-A", "JPM", "JNJ", "V", "UNH", "XOM", "PG", "MA", "HD",
    "CVX", "MRK", "ABBV", "PEP", "KO", "COST", "AVGO", "LLY",
    "WMT", "MCD", "CSCO", "ACN", "TMO", "ABT", "DHR", "TSLA",
    "CRM", "NFLX", "ADBE", "AMD", "INTC", "ORCL", "QCOM",
})


class UniverseBuilder:
    """Builds a curated stock universe of small-to-mid cap names.

    Designed to surface under-followed companies with solid fundamentals
    data availability (SEC filers with quarterly earnings).
    """

    def __init__(self, config: Optional[UniverseConfig] = None) -> None:
        self._config = config or UniverseConfig()

    @property
    def config(self) -> UniverseConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_universe(self) -> List[str]:
        """Return deduplicated, sorted list of curated small-mid cap tickers."""
        tickers: set[str] = set()
        for sector, names in _SECTOR_TICKERS.items():
            tickers.update(names)
            logger.debug("Loaded %d tickers from sector '%s'", len(names), sector)

        if self._config.exclude_mega_caps:
            before = len(tickers)
            tickers -= _MEGA_CAP_EXCLUSIONS
            removed = before - len(tickers)
            if removed:
                logger.info("Removed %d mega-cap tickers from universe", removed)

        universe = sorted(tickers)
        logger.info(
            "Built universe of %d tickers (cap range $%.0fM-$%.0fB)",
            len(universe),
            self._config.min_market_cap / 1e6,
            self._config.max_market_cap / 1e9,
        )
        return universe

    def get_sector_map(self) -> Dict[str, str]:
        """Return mapping of ticker -> sector name for all curated tickers."""
        mapping: Dict[str, str] = {}
        for sector, names in _SECTOR_TICKERS.items():
            for ticker in names:
                mapping[ticker] = sector
        return mapping

    def filter_by_data_availability(
        self,
        tickers: List[str],
        min_history_years: Optional[int] = None,
    ) -> List[str]:
        """Remove tickers lacking sufficient yfinance price history.

        Downloads a minimal slice of historical data for each ticker and
        keeps only those with at least ``min_history_years`` of records.

        Args:
            tickers: Candidate ticker symbols to check.
            min_history_years: Override for minimum years of history required.
                Defaults to ``config.min_history_years``.

        Returns:
            Filtered list of tickers with adequate price history.
        """
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "yfinance not installed; skipping data-availability filter"
            )
            return tickers

        years = min_history_years or self._config.min_history_years
        cutoff_date = datetime.now() - timedelta(days=years * 365)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        passed: List[str] = []
        failed: List[str] = []

        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(
                    start=cutoff_str, end=datetime.now().strftime("%Y-%m-%d")
                )
                if hist is not None and len(hist) >= years * 200:
                    passed.append(ticker)
                else:
                    failed.append(ticker)
                    logger.debug(
                        "Ticker %s has insufficient history (%d rows, need ~%d)",
                        ticker,
                        len(hist) if hist is not None else 0,
                        years * 200,
                    )
            except Exception:
                failed.append(ticker)
                logger.debug("Failed to fetch history for %s", ticker, exc_info=True)

        logger.info(
            "Data availability filter: %d/%d tickers passed (%d removed)",
            len(passed),
            len(tickers),
            len(failed),
        )
        return passed

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def sector_counts(self) -> Dict[str, int]:
        """Return count of tickers per sector."""
        return {sector: len(names) for sector, names in _SECTOR_TICKERS.items()}

    def __repr__(self) -> str:
        total = sum(len(v) for v in _SECTOR_TICKERS.values())
        return (
            f"UniverseBuilder(sectors={len(_SECTOR_TICKERS)}, "
            f"total_curated={total}, "
            f"cap_range=${self._config.min_market_cap/1e6:.0f}M"
            f"-${self._config.max_market_cap/1e9:.0f}B)"
        )
