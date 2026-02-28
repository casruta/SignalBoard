"""Seed the database with realistic mock trading recommendations."""

import logging
from datetime import datetime

from server.database import Database

logger = logging.getLogger(__name__)

MOCK_SIGNALS = [
    # ── BUY Signals (8) — Small-Mid Cap Focus ────────────────────────
    {
        "ticker": "CEIX",
        "short_name": "CONSOL Energy",
        "sector": "Energy",
        "action": "BUY",
        "confidence": 0.91,
        "predicted_return_5d": 0.038,
        "entry_price": 102.50,
        "stop_loss": 96.00,
        "take_profit": 112.00,
        "technical": {
            "points": [
                "RSI at 39 rebounding off oversold with bullish divergence",
                "Volume accumulation pattern over last 10 sessions",
                "Price reclaiming 20-day MA after pullback to support at $96",
            ]
        },
        "fundamental": {
            "points": [
                "Piotroski F-Score 8/9 — top-decile financial strength",
                "FCF yield 15% on $4B market cap, deeply undervalued vs peers",
                "Trading at 0.5x book value with ROIC 28% vs 10% WACC",
                "Only 5 analysts covering — classic blindspot for under-covered stock",
            ]
        },
        "macro": {
            "points": [
                "Thermal coal export pricing resilient on Asian demand",
                "Energy security concerns keeping domestic production favored",
            ]
        },
        "ml_insight": "Ensemble model 89% BUY probability. Dominant features: extreme FCF yield, Piotroski score, and low analyst coverage blindspot signal.",
        "risk_context": "$4B market cap small-mid cap energy name. Commodity price sensitivity is primary risk. Position at 5% given sector vol. Coal transition risk is long-term overhang.",
        "historical_context": "CEIX at sub-0.6x book with Piotroski 8+ has rallied 4.2% median over 5 days in last 2 years (n=6, win rate 83%).",
    },
    {
        "ticker": "CALX",
        "short_name": "Calix Inc.",
        "sector": "Technology",
        "action": "BUY",
        "confidence": 0.86,
        "predicted_return_5d": 0.032,
        "entry_price": 38.50,
        "stop_loss": 35.50,
        "take_profit": 42.00,
        "technical": {
            "points": [
                "Double bottom pattern at $35 support confirmed",
                "MACD crossing above signal line with histogram expanding",
                "Bollinger Band squeeze resolving to the upside",
            ]
        },
        "fundamental": {
            "points": [
                "Revenue growth 20% YoY driven by broadband infrastructure buildout",
                "Gross margins expanding from 52% to 56% on software mix shift",
                "Piotroski F-Score 7/9, DCF upside 35% at $3B market cap",
                "Only 7 analysts covering — institutional ownership still building",
            ]
        },
        "macro": {
            "points": [
                "Federal broadband subsidies ($42B BEAD program) accelerating rural deployments",
                "Fiber-to-the-home penetration still below 40% in US",
            ]
        },
        "ml_insight": "Model assigns 84% BUY probability. Key drivers: revenue acceleration, margin expansion, and undiscovered value signal from low coverage.",
        "risk_context": "$3B market cap tech name. Government funding timeline risk. Position at 5%. Beta 1.3. Earnings in 4 weeks adds event risk.",
        "historical_context": "CALX double bottoms with expanding margins have produced 3.5% median 5-day return (n=5, win rate 80%).",
    },
    {
        "ticker": "BOOT",
        "short_name": "Boot Barn Holdings",
        "sector": "Consumer Discretionary",
        "action": "BUY",
        "confidence": 0.88,
        "predicted_return_5d": 0.029,
        "entry_price": 148.00,
        "stop_loss": 140.00,
        "take_profit": 160.00,
        "technical": {
            "points": [
                "Breakout above 50-day MA on 2x average volume",
                "RSI at 56 with room to run before overbought",
                "Price clearing descending trendline resistance from Q4 highs",
            ]
        },
        "fundamental": {
            "points": [
                "Same-store sales growth 15% — best in specialty retail",
                "Insider cluster buying: CEO and CFO purchased $2.1M in last 30 days",
                "ROIC 24% vs 9% WACC on $4B market cap, Piotroski 7/9",
                "Store count growth 10% YoY with 3-year payback per unit",
            ]
        },
        "macro": {
            "points": [
                "Western and work-wear categories outperforming broader retail",
                "Rural consumer spending resilient relative to urban discretionary",
            ]
        },
        "ml_insight": "88% BUY signal. Insider cluster buying combined with volume breakout is the top-weighted feature. Quality momentum interaction triggered.",
        "risk_context": "$4B market cap retail stock. Consumer spending slowdown is key risk. Insider buying provides downside floor. Position at 6%.",
        "historical_context": "BOOT breakouts with concurrent insider buying have returned 3.1% median over 5 days (n=7, win rate 86%).",
    },
    {
        "ticker": "HIMS",
        "short_name": "Hims & Hers Health",
        "sector": "Healthcare",
        "action": "BUY",
        "confidence": 0.84,
        "predicted_return_5d": 0.041,
        "entry_price": 52.00,
        "stop_loss": 47.50,
        "take_profit": 58.00,
        "technical": {
            "points": [
                "Bull flag consolidation after 30% earnings gap-up",
                "RSI cooling from 72 to 58 — healthy pullback",
                "Volume profile shows strong support at $50 level",
            ]
        },
        "fundamental": {
            "points": [
                "Revenue growth 40% YoY, subscriber base crossing 2M",
                "Approaching GAAP profitability — operating leverage inflecting",
                "DCF upside 28% on $6B market cap with conservative 25% growth assumption",
                "Only 9 analysts covering despite accelerating fundamentals",
            ]
        },
        "macro": {
            "points": [
                "Telehealth adoption permanently elevated post-pandemic",
                "GLP-1 compounding opportunity adds TAM expansion optionality",
            ]
        },
        "ml_insight": "84% BUY probability. Revenue acceleration + approaching profitability inflection are dominant features. Undiscovered value signal active.",
        "risk_context": "$6B market cap healthcare disruptor. Regulatory risk around compounding pharmacy rules. High beta (1.9). Position at 4% given volatility.",
        "historical_context": "HIMS bull flags post-earnings have resolved upward with 4.5% median 5-day return (n=4, win rate 75%).",
    },
    {
        "ticker": "ITT",
        "short_name": "ITT Inc.",
        "sector": "Industrials",
        "action": "BUY",
        "confidence": 0.87,
        "predicted_return_5d": 0.022,
        "entry_price": 152.00,
        "stop_loss": 146.00,
        "take_profit": 161.00,
        "technical": {
            "points": [
                "Ascending triangle breakout with volume confirmation",
                "On-balance volume hitting new highs ahead of price",
                "All major moving averages aligned bullishly (10>20>50>200)",
            ]
        },
        "fundamental": {
            "points": [
                "ROIC 22% vs 8% WACC — exceptional value creation at $12B cap",
                "30 consecutive years of dividend increases, payout ratio only 18%",
                "Piotroski F-Score 8/9, FCF conversion rate above 110%",
                "Organic growth 7% with margin expansion in motion technology segment",
            ]
        },
        "macro": {
            "points": [
                "Industrial reshoring driving domestic capex in precision components",
                "Aerospace aftermarket demand robust with aging fleet dynamics",
            ]
        },
        "ml_insight": "87% BUY probability. Quality compounding signal: high ROIC spread + dividend consistency + Piotroski strength. Classic quality-momentum setup.",
        "risk_context": "$12B market cap precision manufacturer. Cyclical industrial exposure. Low beta (0.9). Position at 6%. Recession risk is primary concern.",
        "historical_context": "ITT ascending triangle breakouts with OBV confirmation have yielded 2.4% median 5-day return (n=8, win rate 75%).",
    },
    {
        "ticker": "STEP",
        "short_name": "StepStone Group",
        "sector": "Financials",
        "action": "BUY",
        "confidence": 0.85,
        "predicted_return_5d": 0.027,
        "entry_price": 58.00,
        "stop_loss": 54.00,
        "take_profit": 63.00,
        "technical": {
            "points": [
                "Cup and handle forming on weekly with handle near completion",
                "RSI at 52 with bullish momentum divergence",
                "Volume drying up on handle pullback — classic accumulation",
            ]
        },
        "fundamental": {
            "points": [
                "AUM growing 25% YoY to $170B, fee-related earnings accelerating",
                "Insider ownership 35% — management heavily aligned at $7B cap",
                "ROIC 19% with asset-light model, Piotroski 7/9",
                "Only 6 analysts covering — significantly under-followed vs peers",
            ]
        },
        "macro": {
            "points": [
                "Institutional allocation to alternatives still increasing globally",
                "Private credit and infrastructure fundraising at record levels",
            ]
        },
        "ml_insight": "85% BUY probability. High insider ownership + low analyst coverage + AUM growth create strong undiscovered value signal.",
        "risk_context": "$7B market cap alt asset manager. Fundraising cycle risk and market-sensitive carry revenue. Position at 5%. Liquidity adequate.",
        "historical_context": "STEP cup-and-handle completions have produced 2.9% median 5-day return (n=5, win rate 80%).",
    },
    {
        "ticker": "GKOS",
        "short_name": "Glaukos Corp.",
        "sector": "Healthcare",
        "action": "BUY",
        "confidence": 0.83,
        "predicted_return_5d": 0.035,
        "entry_price": 118.00,
        "stop_loss": 110.00,
        "take_profit": 130.00,
        "technical": {
            "points": [
                "Inverse head and shoulders forming with neckline at $120",
                "RSI at 48 recovering from oversold territory",
                "Volume expanding on up days for 3 consecutive weeks",
            ]
        },
        "fundamental": {
            "points": [
                "Disruptive iDose TR glaucoma device gaining rapid share",
                "Gross margins 90% — best-in-class for med-tech at $5B cap",
                "Revenue growth 18% with expanding surgeon adoption curve",
                "DCF upside 42% — Street underestimates iDose TAM expansion",
            ]
        },
        "macro": {
            "points": [
                "Aging population driving glaucoma prevalence higher globally",
                "Med-tech M&A multiples expanding as large-caps seek growth",
            ]
        },
        "ml_insight": "83% BUY probability. High DCF upside + low analyst coverage (8 analysts) triggers undiscovered value interaction. Margin quality score in top decile.",
        "risk_context": "$5B market cap med-tech. Single-product concentration risk around iDose. FDA regulatory risk. Position at 4%. Binary outcomes possible.",
        "historical_context": "GKOS inverse H&S patterns with expanding volume have returned 3.8% median over 5 days (n=4, win rate 75%).",
    },
    {
        "ticker": "STLD",
        "short_name": "Steel Dynamics",
        "sector": "Materials",
        "action": "BUY",
        "confidence": 0.86,
        "predicted_return_5d": 0.024,
        "entry_price": 135.00,
        "stop_loss": 129.00,
        "take_profit": 144.00,
        "technical": {
            "points": [
                "Price bouncing off 200-day MA with hammer reversal candle",
                "MACD histogram turning positive after extended bearish run",
                "Relative strength vs XLB improving for 2 consecutive weeks",
            ]
        },
        "fundamental": {
            "points": [
                "Lowest-cost US steel producer, FCF yield 9% on $18B market cap",
                "Piotroski F-Score 8/9, buyback retiring 5% of shares annually",
                "ROIC 18% vs 9% WACC through full cycle, balance sheet net cash",
                "New aluminum flat-roll mill adds diversification and growth runway",
            ]
        },
        "macro": {
            "points": [
                "Infrastructure spending and reshoring driving domestic steel demand",
                "Section 232 tariffs maintaining pricing floor for US producers",
            ]
        },
        "ml_insight": "86% BUY probability. FCF machine signal + Piotroski quality + buyback yield create strong composite. Quality momentum interaction active.",
        "risk_context": "$18B market cap steel producer at upper end of small-mid cap range. Commodity cyclicality risk. Beta 1.2. Position at 5%.",
        "historical_context": "STLD bounces off 200-day MA with Piotroski 8+ have yielded 2.6% median 5-day return (n=9, win rate 78%).",
    },
    # ── SELL Signals (4) — Small-Mid Cap Deteriorating Names ─────────
    {
        "ticker": "RUN",
        "short_name": "Sunrun Inc.",
        "sector": "Utilities",
        "action": "SELL",
        "confidence": 0.82,
        "predicted_return_5d": -0.028,
        "entry_price": 12.50,
        "stop_loss": 14.00,
        "take_profit": 10.50,
        "technical": {
            "points": [
                "Descending channel intact, price rejected at upper channel line",
                "RSI at 44 with no oversold bounce momentum",
                "Death cross: 50-day MA below 200-day MA, gap widening",
            ]
        },
        "fundamental": {
            "points": [
                "Unit economics deteriorating as customer acquisition costs rise 22%",
                "Piotroski F-Score 3/9 — financial health in bottom quartile",
                "FCF negative with $6.5B in net debt on $3B market cap",
                "Subscriber churn ticking up as NEM 3.0 reduces rooftop solar economics",
            ]
        },
        "macro": {
            "points": [
                "Higher interest rates directly impair solar lease financing model",
                "IRA subsidy uncertainty adding policy risk to residential solar",
            ]
        },
        "ml_insight": "SELL probability 80%. Negative FCF + weak Piotroski + rising rates interaction dominate the model. Deterioration trend signal triggered.",
        "risk_context": "$3B market cap solar name. Heavily shorted (18% SI). Short squeeze risk is real — stop tight at 12%. Position at 3%.",
        "historical_context": "RUN in descending channels with Piotroski below 4 has declined -3.1% median over 5 days (n=6, win rate 67%).",
    },
    {
        "ticker": "BYND",
        "short_name": "Beyond Meat",
        "sector": "Consumer Staples",
        "action": "SELL",
        "confidence": 0.88,
        "predicted_return_5d": -0.035,
        "entry_price": 5.80,
        "stop_loss": 6.50,
        "take_profit": 4.80,
        "technical": {
            "points": [
                "Price in persistent downtrend below all major moving averages",
                "Each rally attempt fails at lower levels — textbook distribution",
                "Volume spiking on down days with no buying interest on bounces",
            ]
        },
        "fundamental": {
            "points": [
                "Revenue declining 18% YoY — consumer rejection of plant-based premium",
                "Cash burn $50M/quarter with only $150M remaining, dilution imminent",
                "Piotroski F-Score 2/9 — near-distress financial profile",
                "Management turnover: 3rd CFO in 2 years, no insider buying at $500M cap",
            ]
        },
        "macro": {
            "points": [
                "Plant-based meat category shrinking as novelty fades",
                "Consumer trade-down favoring value protein over premium alternatives",
            ]
        },
        "ml_insight": "Strong SELL at 88%. Cash burn acceleration + Piotroski distress score + management instability form a toxic combination. Model sees continued erosion.",
        "risk_context": "$500M micro-cap. Extremely high volatility. Potential acquisition target creates upside tail risk. Position at 2% max. Tight stop at 12%.",
        "historical_context": "BYND in persistent downtrend with Piotroski 2 has seen -3.8% median 5-day return (n=8, win rate 75%).",
    },
    {
        "ticker": "LCID",
        "short_name": "Lucid Group",
        "sector": "Consumer Discretionary",
        "action": "SELL",
        "confidence": 0.84,
        "predicted_return_5d": -0.031,
        "entry_price": 3.20,
        "stop_loss": 3.60,
        "take_profit": 2.60,
        "technical": {
            "points": [
                "Bear flag forming after breakdown below $3.50 support",
                "RSI at 38 but no bullish divergence to suggest reversal",
                "Volume profile shows heavy resistance from $3.40 to $3.80",
            ]
        },
        "fundamental": {
            "points": [
                "Production 2,200 units/quarter vs 10,000 capacity — massive underutilization",
                "Operating loss $700M/quarter, widening despite revenue growth",
                "Piotroski F-Score 2/9, $6B market cap but burning $2.8B/year cash",
                "Repeated equity offerings diluting shareholders — 15% dilution in last 12 months",
            ]
        },
        "macro": {
            "points": [
                "EV competition intensifying from legacy OEMs with scale advantages",
                "Higher rates making luxury EV financing less attractive to consumers",
            ]
        },
        "ml_insight": "84% SELL probability. Production miss + cash burn + dilution cycle are the dominant negative signals. No positive interaction features triggered.",
        "risk_context": "$6B market cap EV startup. PIF sovereign wealth backing provides floor but also enables continued dilution. Position at 2%. Meme-stock squeeze risk.",
        "historical_context": "LCID bear flags with production misses have led to -3.4% median 5-day return (n=5, win rate 80%).",
    },
    {
        "ticker": "BLNK",
        "short_name": "Blink Charging",
        "sector": "Industrials",
        "action": "SELL",
        "confidence": 0.80,
        "predicted_return_5d": -0.026,
        "entry_price": 2.40,
        "stop_loss": 2.75,
        "take_profit": 1.95,
        "technical": {
            "points": [
                "Lower highs and lower lows on weekly — confirmed long-term downtrend",
                "50-day MA acting as resistance, price rejected 4 consecutive times",
                "MACD histogram entrenched in negative territory",
            ]
        },
        "fundamental": {
            "points": [
                "Revenue growing 15% but operating losses expanding faster at 25%",
                "Stock-based compensation 40% of revenue — massive shareholder dilution",
                "Piotroski F-Score 3/9 on $300M market cap, no path to profitability visible",
                "Insider selling accelerating: CFO sold 30% of holdings last quarter",
            ]
        },
        "macro": {
            "points": [
                "EV charging buildout slowing as automakers delay EV targets",
                "Federal NEVI funding disbursement slower than expected",
            ]
        },
        "ml_insight": "80% SELL probability. Insider selling + SBC dilution + unprofitable growth create negative composite. Quality deterioration signal active.",
        "risk_context": "$300M micro-cap EV infrastructure name. Extremely illiquid, wide spreads. Potential penny-stock volatility. Position at 2% max.",
        "historical_context": "BLNK in confirmed downtrends with insider selling has seen -2.8% median 5-day return (n=7, win rate 71%).",
    },
]


MOCK_SCREENED = [
    {"ticker": "CEIX", "short_name": "CONSOL Energy", "sector": "Energy", "industry": "Coal", "market_cap": 4_000_000_000, "composite_score": 0.91, "rank": 1, "piotroski_score": 0.89, "cash_flow_score": 0.93, "dcf_score": 0.88, "balance_sheet_score": 0.85, "blindspot_score": 0.92, "margin_score": 0.78, "roic_spread_score": 0.95},
    {"ticker": "CALX", "short_name": "Calix Inc.", "sector": "Technology", "industry": "Networking", "market_cap": 3_000_000_000, "composite_score": 0.86, "rank": 2, "piotroski_score": 0.78, "cash_flow_score": 0.72, "dcf_score": 0.85, "balance_sheet_score": 0.80, "blindspot_score": 0.88, "margin_score": 0.82, "roic_spread_score": 0.74},
    {"ticker": "BOOT", "short_name": "Boot Barn Holdings", "sector": "Consumer Discretionary", "industry": "Specialty Retail", "market_cap": 4_000_000_000, "composite_score": 0.88, "rank": 3, "piotroski_score": 0.78, "cash_flow_score": 0.81, "dcf_score": 0.76, "balance_sheet_score": 0.83, "blindspot_score": 0.70, "margin_score": 0.85, "roic_spread_score": 0.90},
    {"ticker": "HIMS", "short_name": "Hims & Hers Health", "sector": "Healthcare", "industry": "Telehealth", "market_cap": 6_000_000_000, "composite_score": 0.84, "rank": 4, "piotroski_score": 0.65, "cash_flow_score": 0.58, "dcf_score": 0.82, "balance_sheet_score": 0.60, "blindspot_score": 0.85, "margin_score": 0.72, "roic_spread_score": 0.55},
    {"ticker": "ITT", "short_name": "ITT Inc.", "sector": "Industrials", "industry": "Precision Components", "market_cap": 12_000_000_000, "composite_score": 0.87, "rank": 5, "piotroski_score": 0.89, "cash_flow_score": 0.91, "dcf_score": 0.73, "balance_sheet_score": 0.92, "blindspot_score": 0.55, "margin_score": 0.88, "roic_spread_score": 0.92},
    {"ticker": "STEP", "short_name": "StepStone Group", "sector": "Financials", "industry": "Asset Management", "market_cap": 7_000_000_000, "composite_score": 0.85, "rank": 6, "piotroski_score": 0.78, "cash_flow_score": 0.82, "dcf_score": 0.79, "balance_sheet_score": 0.75, "blindspot_score": 0.90, "margin_score": 0.80, "roic_spread_score": 0.85},
    {"ticker": "GKOS", "short_name": "Glaukos Corp.", "sector": "Healthcare", "industry": "Medical Devices", "market_cap": 5_000_000_000, "composite_score": 0.83, "rank": 7, "piotroski_score": 0.60, "cash_flow_score": 0.55, "dcf_score": 0.90, "balance_sheet_score": 0.65, "blindspot_score": 0.82, "margin_score": 0.92, "roic_spread_score": 0.60},
    {"ticker": "STLD", "short_name": "Steel Dynamics", "sector": "Materials", "industry": "Steel", "market_cap": 18_000_000_000, "composite_score": 0.86, "rank": 8, "piotroski_score": 0.89, "cash_flow_score": 0.87, "dcf_score": 0.78, "balance_sheet_score": 0.90, "blindspot_score": 0.45, "margin_score": 0.82, "roic_spread_score": 0.88},
]


def seed(db_path: str) -> None:
    """Clear existing data and insert mock recommendations and screened stocks."""
    db = Database(db_path)
    db.clear_all_recommendations()

    now = datetime.utcnow().isoformat()
    for signal in MOCK_SIGNALS:
        signal["generated_at"] = now

    db.save_recommendations(MOCK_SIGNALS)
    logger.info("Seeded %d mock recommendations", len(MOCK_SIGNALS))

    db.save_screened_stocks(MOCK_SCREENED)
    logger.info("Seeded %d mock screened stocks", len(MOCK_SCREENED))
