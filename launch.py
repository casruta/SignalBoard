"""SignalBoard Launcher — industry-selectable one-click startup.

Presents a menu of industry groups, fetches fresh data for the chosen
group, seeds the database, starts the FastAPI server, and opens the
browser.

Usage:
    python launch.py              Interactive menu
    python launch.py --industry 1 Skip menu, run Defence & Aerospace
"""

import argparse
import copy
import logging
import os
import sys
import time
import webbrowser
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signalboard.launcher")

# ── Industry ticker groups ───────────────────────────────────────────

INDUSTRIES: dict[int, dict] = {
    1: {
        "name": "Defence & Aerospace",
        "tickers": [
            "KTOS",   # Kratos Defense — drones, unmanned systems
            "MRCY",   # Mercury Systems — defense electronics
            "AVAV",   # AeroVironment — small UAS and loitering munitions
            "BWXT",   # BWX Technologies — naval nuclear reactors
            "CW",     # Curtiss-Wright — defense/nuclear electronics
            "HXL",    # Hexcel — aerospace composites
            "MOG-A",  # Moog — precision motion control for military
            "ESLT",   # Elbit Systems — UAVs, EW, C4ISR (ADR)
            "DRS",    # Leonardo DRS — defense electronics, sensing
            "WWD",    # Woodward — aerospace fuel systems, turbine controls
            "SAIC",   # Science Applications Intl — defense IT, cyber
            "CACI",   # CACI International — intelligence, EW, cyber
            "VSEC",   # V2X — defense facility mgmt, IT, logistics
            "SPR",    # Spirit AeroSystems — aerostructures
            "COHR",   # Coherent — photonics/laser tech for defense
        ],
    },
    2: {
        "name": "Semiconductors & Chip Equipment",
        "tickers": [
            "SWKS",   # Skyworks — analog/mixed-signal for mobile, auto
            "QRVO",   # Qorvo — RF solutions for mobile, defense
            "LSCC",   # Lattice Semiconductor — low-power FPGAs
            "MKSI",   # MKS Instruments — process control for fabs
            "ENTG",   # Entegris — materials purity for chipmaking
            "DIOD",   # Diodes Inc — analog/discrete semiconductors
            "SLAB",   # Silicon Labs — IoT wireless chipsets
            "FORM",   # FormFactor — wafer probe cards
            "ACLS",   # Axcelis — ion implantation equipment
            "CRUS",   # Cirrus Logic — audio/signal-processing chips
            "SMTC",   # Semtech — analog/mixed-signal, LoRa IoT
            "POWI",   # Power Integrations — power conversion ICs
            "COHU",   # Cohu — semiconductor test equipment
        ],
    },
    3: {
        "name": "Medical Devices & Diagnostics",
        "tickers": [
            "ALGN",   # Align Technology — Invisalign, scanners
            "NVCR",   # NovoCure — TTFields oncology
            "GMED",   # Globus Medical — spine surgery robots
            "MMSI",   # Merit Medical — cardiology/radiology devices
            "NEOG",   # Neogen — food/animal safety diagnostics
            "NTRA",   # Natera — cell-free DNA testing
            "INSP",   # Inspire Medical — neurostimulation for sleep apnea
            "IART",   # Integra LifeSciences — tissue, neuro-critical care
            "LNTH",   # Lantheus — diagnostic imaging agents
            "IRTC",   # iRhythm — cardiac monitoring (Zio patch)
            "TNDM",   # Tandem Diabetes — insulin pump systems
            "NVST",   # Envista — dental products, orthodontics
            "OMCL",   # Omnicell — medication management automation
        ],
    },
    4: {
        "name": "Specialty Industrials & Infrastructure",
        "tickers": [
            "WMS",    # Advanced Drainage — stormwater pipes
            "SITE",   # SiteOne Landscape — wholesale distribution
            "GNRC",   # Generac — backup power, energy storage
            "AAON",   # AAON — commercial HVAC, near-zero debt
            "AIT",    # Applied Industrial Technologies — distribution
            "WDFC",   # WD-40 — maintenance products, brand moat
            "EXPO",   # Exponent — engineering consulting
            "ROAD",   # Construction Partners — asphalt, road construction
            "STRL",   # Sterling Infrastructure — e-infra, building
            "PRIM",   # Primoris Services — utilities, pipelines
            "DY",     # Dycom — telecom/utility infrastructure
            "ZWS",    # Zurn Elkay — water management, drainage
            "FIX",    # Comfort Systems USA — mechanical contractor
        ],
    },
}

MENU_WIDTH = 50


def show_menu() -> int:
    """Display industry selection menu and return choice."""
    print()
    print("=" * MENU_WIDTH)
    print("  SignalBoard  —  Industry Selector")
    print("=" * MENU_WIDTH)
    print()
    for key, group in INDUSTRIES.items():
        count = len(group["tickers"])
        print(f"  [{key}]  {group['name']}  ({count} stocks)")
    print()
    print(f"  [0]  Original universe (all {_original_count()} stocks)")
    print()
    print("-" * MENU_WIDTH)

    while True:
        try:
            choice = int(input("  Select industry [0-4]: ").strip())
            if 0 <= choice <= len(INDUSTRIES):
                return choice
        except (ValueError, EOFError):
            pass
        print("  Invalid choice. Enter a number 0-4.")


def _original_count() -> int:
    """Count tickers in the original config."""
    config = load_config()
    return len(config.get("universe", {}).get("tickers", []))


def run(industry_choice: int) -> None:
    """Seed database with chosen industry, start server, open browser."""
    config = load_config()
    run_config = copy.deepcopy(config)

    if industry_choice > 0:
        group = INDUSTRIES[industry_choice]
        run_config["universe"]["tickers"] = group["tickers"]
        logger.info(
            "Selected: %s (%d tickers)",
            group["name"], len(group["tickers"]),
        )
    else:
        logger.info(
            "Selected: Original universe (%d tickers)",
            len(run_config["universe"]["tickers"]),
        )

    # ── Seed with fresh data ─────────────────────────────────────
    from server.seed import seed_live
    db_path = run_config["server"]["database_path"]

    logger.info("Fetching data and running screener...")
    seed_live(db_path, run_config)
    logger.info("Database seeded with fresh data.")

    # ── Start server and open browser ────────────────────────────
    host = run_config["server"].get("host", "0.0.0.0")
    port = run_config["server"].get("port", 9000)
    url = f"http://localhost:{port}"

    logger.info("Starting server on %s", url)

    import threading
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    from server.app import app
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="SignalBoard Launcher")
    parser.add_argument(
        "--industry", "-i",
        type=int,
        default=None,
        help="Industry choice (1-4) or 0 for original universe. Omit for interactive menu.",
    )
    args = parser.parse_args()

    if args.industry is not None:
        choice = args.industry
        if not (0 <= choice <= len(INDUSTRIES)):
            print(f"Invalid industry choice: {choice}. Must be 0-{len(INDUSTRIES)}.")
            sys.exit(1)
    else:
        choice = show_menu()

    if choice > 0:
        group = INDUSTRIES[choice]
        print(f"\n  >> {group['name']} ({len(group['tickers'])} stocks)")
    else:
        print(f"\n  >> Original universe")

    print("  >> Fetching data & running analysis...\n")
    run(choice)


if __name__ == "__main__":
    main()
