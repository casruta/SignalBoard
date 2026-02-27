"""Seed the database with realistic mock trading recommendations."""

import logging
from datetime import datetime

from server.database import Database

logger = logging.getLogger(__name__)

MOCK_SIGNALS = [
    {
        "ticker": "AAPL",
        "short_name": "Apple Inc.",
        "sector": "Technology",
        "action": "BUY",
        "confidence": 0.92,
        "predicted_return_5d": 0.034,
        "entry_price": 227.50,
        "stop_loss": 220.50,
        "take_profit": 239.00,
        "technical": {
            "points": [
                "RSI at 42 rebounding from oversold territory",
                "MACD histogram turning positive after bearish crossover",
                "Price holding above 200-day moving average at $218",
                "Bollinger Bands squeezing — breakout imminent",
            ]
        },
        "fundamental": {
            "points": [
                "Services revenue grew 14% YoY, now 26% of total revenue",
                "Gross margin expanded to 46.6%, highest in a decade",
                "iPhone 16 cycle showing stronger-than-expected ASP uplift",
            ]
        },
        "macro": {
            "points": [
                "Fed signaling rate cuts support growth-stock valuations",
                "Consumer spending resilient despite inflation concerns",
            ]
        },
        "ml_insight": "Ensemble model shows 89% probability of positive 5-day return. Feature importance dominated by momentum reversal signals and options flow imbalance.",
        "risk_context": "Max drawdown risk estimated at 3.1%. Position sized at 8% of portfolio given sector concentration limits. VIX at 16 suggests low implied vol environment.",
        "historical_context": "Similar setups in AAPL (RSI bounce + MACD turn) have produced 3.2% median 5-day return over the past 3 years (n=14, win rate 78%).",
    },
    {
        "ticker": "NVDA",
        "short_name": "NVIDIA Corp.",
        "sector": "Technology",
        "action": "BUY",
        "confidence": 0.89,
        "predicted_return_5d": 0.042,
        "entry_price": 875.00,
        "stop_loss": 845.00,
        "take_profit": 920.00,
        "technical": {
            "points": [
                "Breakout above consolidation range with volume confirmation",
                "RSI at 58, room to run before overbought",
                "20-day MA crossed above 50-day MA (golden cross on daily)",
            ]
        },
        "fundamental": {
            "points": [
                "Data center revenue up 409% YoY, AI demand accelerating",
                "Blackwell GPU ramp ahead of schedule per management",
                "Gross margins stable at 75% despite supply scaling",
            ]
        },
        "macro": {
            "points": [
                "Global AI capex forecasts revised up 30% for 2026",
                "Semiconductor cycle entering expansion phase per SIA data",
            ]
        },
        "ml_insight": "Model assigns 86% BUY probability. Key drivers: earnings momentum z-score (+2.4), sector relative strength, and institutional accumulation signal.",
        "risk_context": "High beta (1.8) stock — position capped at 6% of portfolio. Implied vol elevated at 52% annualized. Earnings in 3 weeks adds event risk.",
        "historical_context": "Post-consolidation breakouts in NVDA with volume have returned 4.8% median over 5 days in last 2 years (n=9, win rate 67%).",
    },
    {
        "ticker": "JPM",
        "short_name": "JPMorgan Chase",
        "sector": "Financials",
        "action": "BUY",
        "confidence": 0.85,
        "predicted_return_5d": 0.021,
        "entry_price": 198.00,
        "stop_loss": 192.00,
        "take_profit": 207.00,
        "technical": {
            "points": [
                "Price reclaiming 50-day MA after pullback",
                "On-balance volume divergence suggests accumulation",
                "Support confirmed at $192 level (tested 3 times)",
            ]
        },
        "fundamental": {
            "points": [
                "Net interest income guidance raised by $2B for FY2026",
                "Credit quality remains strong — NCO ratio at 0.6%",
                "Buyback program accelerated, $4B remaining in authorization",
            ]
        },
        "macro": {
            "points": [
                "Yield curve normalizing — positive for bank net interest margins",
                "Loan growth picking up as corporate confidence improves",
            ]
        },
        "ml_insight": "Model confidence 82%. Financials sector rotation signal triggered. Interest rate sensitivity feature contributing most to the BUY call.",
        "risk_context": "Moderate risk profile. Beta of 1.1. Position sized at 7% given diversified financials exposure. Downside scenario: unexpected credit deterioration.",
        "historical_context": "JPM bouncing off 50-day MA with positive OBV divergence has yielded 2.3% median 5-day return (n=11, win rate 73%).",
    },
    {
        "ticker": "LLY",
        "short_name": "Eli Lilly",
        "sector": "Healthcare",
        "action": "BUY",
        "confidence": 0.87,
        "predicted_return_5d": 0.028,
        "entry_price": 785.00,
        "stop_loss": 760.00,
        "take_profit": 820.00,
        "technical": {
            "points": [
                "Cup and handle pattern completing on weekly chart",
                "RSI divergence — price flat while RSI rising",
                "Volume dry-up on pullback suggests selling exhaustion",
            ]
        },
        "fundamental": {
            "points": [
                "Mounjaro/Zepbound revenue trajectory exceeding Street estimates",
                "Pipeline readouts in Q1 for Alzheimer's and oncology",
                "Earnings growth rate of 40%+ for next 3 years",
            ]
        },
        "macro": {
            "points": [
                "GLP-1 market size estimates revised up to $150B by 2030",
                "Healthcare sector defensive in uncertain macro environment",
            ]
        },
        "ml_insight": "Strong BUY signal (87%). Pharma momentum factor and earnings revision breadth are the top contributing features.",
        "risk_context": "Single-product concentration risk (GLP-1 class). Position at 5% given healthcare sector cap. Binary event risk from pipeline data.",
        "historical_context": "Cup-and-handle completions in large-cap pharma have 3.1% median 5-day return (n=8, win rate 75%).",
    },
    {
        "ticker": "XOM",
        "short_name": "Exxon Mobil",
        "sector": "Energy",
        "action": "SELL",
        "confidence": 0.81,
        "predicted_return_5d": -0.019,
        "entry_price": 112.00,
        "stop_loss": 116.50,
        "take_profit": 106.00,
        "technical": {
            "points": [
                "Bearish engulfing candle on weekly chart",
                "RSI at 68 with bearish divergence forming",
                "Price rejected at resistance zone $113-114 twice",
            ]
        },
        "fundamental": {
            "points": [
                "Refining margins compressing as global capacity comes online",
                "Capex guidance raised 12% — potential drag on FCF",
                "Pioneer integration costs weighing on near-term EPS",
            ]
        },
        "macro": {
            "points": [
                "Oil demand growth forecasts trimmed by IEA for 2026",
                "OPEC+ compliance weakening, supply overhang risk",
            ]
        },
        "ml_insight": "Model assigns 78% probability of negative 5-day return. Energy sector momentum turning negative. Mean reversion signal triggered.",
        "risk_context": "Short position risk: energy stocks can gap on geopolitical events. Stop loss tight at 4%. Position sized at 5%.",
        "historical_context": "Bearish engulfing at resistance in XOM has preceded -2.1% median 5-day return (n=7, win rate 71%).",
    },
    {
        "ticker": "AMZN",
        "short_name": "Amazon.com",
        "sector": "Consumer Discretionary",
        "action": "BUY",
        "confidence": 0.83,
        "predicted_return_5d": 0.025,
        "entry_price": 192.00,
        "stop_loss": 185.00,
        "take_profit": 202.00,
        "technical": {
            "points": [
                "Ascending triangle forming with base at $188",
                "Volume increasing on up days, decreasing on down days",
                "MACD about to cross signal line from below",
            ]
        },
        "fundamental": {
            "points": [
                "AWS revenue reaccelerating to 19% growth",
                "Operating margins expanding as cost discipline continues",
                "Advertising business now $14B/quarter run rate",
            ]
        },
        "macro": {
            "points": [
                "E-commerce penetration still growing post-normalization",
                "Cloud spending budgets increasing across enterprise surveys",
            ]
        },
        "ml_insight": "BUY probability 80%. AWS reacceleration signal and consumer discretionary sector rotation are key model drivers.",
        "risk_context": "Large cap with moderate vol. Beta 1.2. Position at 7%. Risk is cloud growth deceleration surprise.",
        "historical_context": "AMZN ascending triangle breakouts have 2.8% median 5-day return (n=6, win rate 67%).",
    },
    {
        "ticker": "PG",
        "short_name": "Procter & Gamble",
        "sector": "Consumer Staples",
        "action": "SELL",
        "confidence": 0.77,
        "predicted_return_5d": -0.015,
        "entry_price": 168.00,
        "stop_loss": 172.00,
        "take_profit": 162.00,
        "technical": {
            "points": [
                "Death cross: 50-day MA crossed below 200-day MA",
                "Price below both key moving averages",
                "Declining volume on rally attempts",
            ]
        },
        "fundamental": {
            "points": [
                "Volume growth turning negative in key categories",
                "Private label share gains pressuring pricing power",
                "Input cost inflation reaccelerating (palm oil, packaging)",
            ]
        },
        "macro": {
            "points": [
                "Consumer trade-down trend accelerating in staples",
                "Sector rotation away from defensives into cyclicals",
            ]
        },
        "ml_insight": "SELL probability 74%. Defensive sector underperformance signal and negative earnings revision momentum are primary features.",
        "risk_context": "Low-beta (0.5) defensive name — short squeeze risk is low. Position at 4%. Safe-haven flows could provide temporary support.",
        "historical_context": "Death crosses in PG have been followed by -1.8% median 5-day return (n=5, win rate 60%).",
    },
    {
        "ticker": "META",
        "short_name": "Meta Platforms",
        "sector": "Communication Services",
        "action": "BUY",
        "confidence": 0.86,
        "predicted_return_5d": 0.031,
        "entry_price": 580.00,
        "stop_loss": 560.00,
        "take_profit": 610.00,
        "technical": {
            "points": [
                "Flag pattern forming after strong earnings gap up",
                "RSI cooling to 55 from overbought — healthy reset",
                "Support at $570 from prior resistance turned support",
            ]
        },
        "fundamental": {
            "points": [
                "Reels monetization efficiency improving rapidly",
                "AI-driven ad targeting boosting ROAS metrics",
                "Reality Labs losses narrowing ahead of schedule",
            ]
        },
        "macro": {
            "points": [
                "Digital ad spend growing 12% YoY globally",
                "Strong engagement trends across Instagram and WhatsApp",
            ]
        },
        "ml_insight": "86% BUY probability. Post-earnings momentum signal and digital advertising growth factor are top contributors.",
        "risk_context": "Position at 6%. Regulatory overhang (EU DMA compliance) is a tail risk. High capex guidance could pressure sentiment.",
        "historical_context": "META flag patterns post-earnings have resolved upward 80% of the time with 3.4% median 5-day gain (n=10).",
    },
    {
        "ticker": "MRK",
        "short_name": "Merck & Co.",
        "sector": "Healthcare",
        "action": "SELL",
        "confidence": 0.79,
        "predicted_return_5d": -0.022,
        "entry_price": 125.00,
        "stop_loss": 129.00,
        "take_profit": 119.00,
        "technical": {
            "points": [
                "Head and shoulders pattern completing on daily chart",
                "Neckline break at $126 with volume confirmation",
                "All major moving averages sloping downward",
            ]
        },
        "fundamental": {
            "points": [
                "Keytruda patent cliff approaching (2028) weighing on valuation",
                "Pipeline diversification efforts behind peers",
                "Recent acquisition integration adding $800M in costs",
            ]
        },
        "macro": {
            "points": [
                "Drug pricing reform pressure on pharma margins",
                "Sector rotation favoring growth over value healthcare",
            ]
        },
        "ml_insight": "SELL probability 76%. Patent cliff risk factor and negative technical momentum are the dominant model inputs.",
        "risk_context": "Defensive pharma name — position at 4%. Risk of snap-back if pipeline data surprises positively.",
        "historical_context": "H&S completions in large-cap pharma have -2.5% median 5-day return (n=6, win rate 67%).",
    },
    {
        "ticker": "BA",
        "short_name": "Boeing Co.",
        "sector": "Industrials",
        "action": "SELL",
        "confidence": 0.83,
        "predicted_return_5d": -0.032,
        "entry_price": 178.00,
        "stop_loss": 185.00,
        "take_profit": 165.00,
        "technical": {
            "points": [
                "Breakdown below ascending channel support",
                "RSI at 38 and falling — no oversold bounce yet",
                "Volume spike on down move confirms distribution",
            ]
        },
        "fundamental": {
            "points": [
                "Production rate recovery slower than guided",
                "Free cash flow still negative; burn rate of $1.2B/quarter",
                "Quality control issues leading to delivery delays",
            ]
        },
        "macro": {
            "points": [
                "Airline capex cycle peaking as travel growth normalizes",
                "Defense budget under scrutiny — contract award uncertainty",
            ]
        },
        "ml_insight": "Strong SELL signal (83%). Negative FCF trend and industrial sector weakness are primary features. Model sees continued downside.",
        "risk_context": "High-vol name (annualized vol 42%). Position at 4%. Short squeeze risk from retail sentiment. Stop tight at 4%.",
        "historical_context": "Channel breakdowns in BA have led to -3.5% median 5-day return (n=5, win rate 80%).",
    },
    {
        "ticker": "MSFT",
        "short_name": "Microsoft Corp.",
        "sector": "Technology",
        "action": "BUY",
        "confidence": 0.88,
        "predicted_return_5d": 0.026,
        "entry_price": 420.00,
        "stop_loss": 408.00,
        "take_profit": 440.00,
        "technical": {
            "points": [
                "Bounce off 100-day MA with hammer candle",
                "Accumulation/distribution line trending up",
                "Fibonacci 38.2% retracement holding as support",
            ]
        },
        "fundamental": {
            "points": [
                "Azure revenue growth reaccelerating to 33%",
                "Copilot monetization adding $2B incremental annual revenue",
                "Operating leverage improving — margins at 44%",
            ]
        },
        "macro": {
            "points": [
                "Enterprise IT spending stable despite macro uncertainty",
                "AI infrastructure buildout benefiting hyperscalers disproportionately",
            ]
        },
        "ml_insight": "88% BUY probability. Cloud acceleration signal, AI monetization factor, and technical support bounce are key drivers.",
        "risk_context": "Core holding — position at 8%. Low downside risk given diversified revenue. Antitrust headline risk is manageable.",
        "historical_context": "MSFT bounces off 100-day MA have yielded 2.9% median 5-day return (n=12, win rate 83%).",
    },
    {
        "ticker": "CVX",
        "short_name": "Chevron Corp.",
        "sector": "Energy",
        "action": "SELL",
        "confidence": 0.75,
        "predicted_return_5d": -0.017,
        "entry_price": 155.00,
        "stop_loss": 159.50,
        "take_profit": 148.00,
        "technical": {
            "points": [
                "Lower highs and lower lows on daily — confirmed downtrend",
                "50-day MA acting as resistance, rejected twice",
                "MACD histogram deepening into negative territory",
            ]
        },
        "fundamental": {
            "points": [
                "Hess acquisition regulatory uncertainty dragging on stock",
                "Permian Basin production growth slowing QoQ",
                "Dividend yield attractive but FCF coverage thinning",
            ]
        },
        "macro": {
            "points": [
                "Natural gas prices depressed — hurts integrated margins",
                "ESG fund outflows from energy sector continuing",
            ]
        },
        "ml_insight": "SELL probability 72%. Energy sector momentum is negative. Downtrend persistence signal triggered.",
        "risk_context": "Position at 4%. Energy stocks subject to geopolitical supply shocks (tail risk). Dividend support may slow decline.",
        "historical_context": "CVX in confirmed downtrends has seen -1.9% median 5-day return (n=8, win rate 63%).",
    },
]


def seed(db_path: str) -> None:
    """Clear existing data and insert mock recommendations."""
    db = Database(db_path)
    db.clear_all_recommendations()

    now = datetime.utcnow().isoformat()
    for signal in MOCK_SIGNALS:
        signal["generated_at"] = now

    db.save_recommendations(MOCK_SIGNALS)
    logger.info("Seeded %d mock recommendations", len(MOCK_SIGNALS))
