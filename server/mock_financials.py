"""Generate internally-consistent mock financial reports from seed signal data."""
from __future__ import annotations

import re, random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from data.universe_builder import _SECTOR_TICKERS

_NAMES = {
    "RRC": "Range Resources", "NOG": "Northern Oil & Gas", "AR": "Antero Resources",
    "TRGP": "Targa Resources", "MTDR": "Matador Resources", "SM": "SM Energy",
    "FOXF": "Fox Factory", "PLNT": "Planet Fitness", "BROS": "Dutch Bros",
    "DDOG": "Datadog", "NET": "Cloudflare", "BILL": "BILL Holdings",
    "HALO": "Halozyme", "NBIX": "Neurocrine Bio", "RARE": "Ultragenyx",
    "TREX": "Trex Company", "GNRC": "Generac", "FIX": "Comfort Systems",
    "STEP": "StepStone Group", "PIPR": "Piper Sandler", "IBKR": "Interactive Brokers",
    "CLF": "Cleveland-Cliffs", "ATI": "ATI Inc", "CRS": "Carpenter Technology",
    "CALM": "Cal-Maine Foods", "FLO": "Flowers Foods",
    "SKY": "Skyline Champion", "GRBK": "Green Brick Partners",
    "ONTO": "Onto Innovation", "AEIS": "Advanced Energy",
    "UFPI": "UFP Industries", "GMS": "GMS Inc", "EPAM": "EPAM Systems",
    "LNTH": "Lantheus Holdings", "BOOT": "Boot Barn Holdings", "CEIX": "CONSOL Energy",
    "ENSG": "Ensign Group", "DOCN": "DigitalOcean",
    "DINO": "HF Sinclair", "GPOR": "Gulfport Energy", "CTRA": "Coterra Energy",
    "HP": "Helmerich & Payne", "DVN": "Devon Energy", "MUR": "Murphy Oil",
    "OVV": "Ovintiv", "CHRD": "Chord Energy", "PR": "Permian Resources",
    "PTEN": "Patterson-UTI", "WHD": "Cactus Inc", "LBRT": "Liberty Energy",
    "CROX": "Crocs Inc", "DKS": "Dick's Sporting", "DECK": "Deckers Outdoor",
    "BURL": "Burlington Stores", "AEO": "American Eagle", "ANF": "Abercrombie & Fitch",
    "MDB": "MongoDB", "SNOW": "Snowflake", "CRWD": "CrowdStrike", "ZS": "Zscaler",
    "VEEV": "Veeva Systems", "GLOB": "Globant", "WDAY": "Workday",
    "MEDP": "Medpace", "EXAS": "Exact Sciences", "IONS": "Ionis Pharma",
    "AZEK": "AZEK Company", "BLDR": "Builders FirstSource", "VMC": "Vulcan Materials",
    "MLM": "Martin Marietta", "EXP": "Eagle Materials",
    "CVNA": "Carvana", "DASH": "DoorDash", "RBLX": "Roblox",
    "HCKT": "The Hackett Group", "JJSF": "J&J Snack Foods", "EPAC": "Enerpac Tool Group",
}

# Sector defaults: gm, om, nm, ev/rev, tax, d&a%, capex%, div_yield, beta
_SD = {
    "Energy":       (.35,.18,.12, 1.8,.22,.08,.12,.015, 1.2),
    "Technology":   (.55,.15,.10, 5.0,.21,.05,.04, 0.0, 1.3),
    "Consumer Discretionary": (.42,.12,.07, 2.0,.23,.04,.05,.005, 1.1),
    "Healthcare":   (.65,.10,.06, 6.0,.20,.04,.03, 0.0, 1.1),
    "Industrials":  (.35,.14,.09, 2.5,.22,.05,.05,.012, 1.0),
    "Financials":   (.70,.30,.22, 4.0,.21,.02,.02,.01,  1.15),
    "Materials":    (.25,.12,.08, 1.2,.23,.06,.07,.015, 1.2),
    "Consumer Staples": (.35,.08,.04, 1.0,.23,.04,.04,.02, 0.8),
    "Consumer Defensive": (.35,.08,.04, 1.0,.23,.04,.04,.02, 0.8),
    "Utilities":    (.30,.10,.05, 2.0,.22,.06,.15,.03, 0.9),
}
_SD_KEYS = ("gm","om","nm","ev_rev","tax","dna","capex","div","beta")
_DFLT = (.40,.12,.07, 2.5,.22,.05,.05,.01, 1.0)
_SMAP = {"Energy":"energy","Technology":"technology","Consumer Discretionary":"consumer_discretionary",
    "Healthcare":"healthcare","Industrials":"industrials","Financials":"financials",
    "Materials":"materials","Consumer Staples":"consumer_staples","Consumer Defensive":"consumer_staples",
    "Utilities":"utilities","Communication Services":"communication_services","Real Estate":"real_estate"}


@dataclass
class _Anchors:
    market_cap: Optional[float] = None; fcf_yield: Optional[float] = None
    roic: Optional[float] = None; wacc: Optional[float] = None
    piotroski: Optional[int] = None; rev_growth: Optional[float] = None
    gross_margin: Optional[float] = None; gross_margin_high: Optional[float] = None
    book_multiple: Optional[float] = None; analyst_count: Optional[int] = None
    dcf_upside: Optional[float] = None; net_debt: Optional[float] = None
    beta: Optional[float] = None; sbc_pct_rev: Optional[float] = None
    op_loss_quarterly: Optional[float] = None; short_interest: Optional[float] = None
    buyback_yield: Optional[float] = None; fcf_conversion: Optional[float] = None
    dilution_pct: Optional[float] = None
    dividend_yield: Optional[float] = None; payout_ratio: Optional[float] = None
    rsi: Optional[float] = None


def _dollar_val(m) -> float:
    return float(m.group(1)) * (1e9 if m.group(2).upper() == "B" else 1e6)


def _m(name: str, formula: str, inputs: list[tuple], result: tuple, note: str | None = None) -> dict:
    """Build a math breakdown dict for the explain panel."""
    return {
        "name": name,
        "formula": formula,
        "inputs": [{"label": l, "value": v, "fmt": f} for l, v, f in inputs],
        "result": {"label": result[0], "value": result[1], "fmt": result[2]},
        "note": note,
    }


def _parse_anchors(signal: dict) -> _Anchors:
    a = _Anchors()
    parts = signal.get("fundamental",{}).get("points",[]) + signal.get("macro",{}).get("points",[])
    blob = " ".join(parts + [signal.get("risk_context",""), signal.get("ml_insight","")])
    def _s(pat):
        m = re.search(pat, blob, re.I)
        return m.group(1) if m else None
    # Market cap (must be near "cap" keyword to avoid matching "$50M/quarter")
    m = re.search(r'\$(\d+(?:\.\d+)?)\s*([BM])\s*(?:market\s*)?cap', blob, re.I)
    if m: a.market_cap = _dollar_val(m)
    v = _s(r'FCF\s*yield\s*(\d+(?:\.\d+)?)%');
    if v: a.fcf_yield = float(v)/100
    v = _s(r'ROIC\s*(\d+(?:\.\d+)?)%');
    if v: a.roic = float(v)/100
    v = _s(r'(\d+(?:\.\d+)?)%\s*WACC');
    if v: a.wacc = float(v)/100
    v = _s(r'Piotroski\s*(?:F-Score\s*)?(\d)/9');
    if v: a.piotroski = int(v)
    v = _s(r'[Rr]evenue\s*(?:growth|growing|declining)\s*(\d+(?:\.\d+)?)%')
    if v:
        a.rev_growth = float(v)/100
        if "declining" in blob.lower(): a.rev_growth = -a.rev_growth
    gm = re.search(r'[Gg]ross\s*margins?\s*(\d+(?:\.\d+)?)%\s*(?:to\s*(\d+(?:\.\d+)?)%)?', blob)
    if gm:
        a.gross_margin = float(gm.group(1))/100
        if gm.group(2): a.gross_margin_high = float(gm.group(2))/100
    v = _s(r'(\d+(?:\.\d+)?)x\s*book');
    if v: a.book_multiple = float(v)
    v = _s(r'(\d+)\s*analysts?\s*covering');
    if v: a.analyst_count = int(v)
    v = _s(r'DCF\s*upside\s*(\d+(?:\.\d+)?)%');
    if v: a.dcf_upside = float(v)/100
    m = re.search(r'\$(\d+(?:\.\d+)?)\s*([BM])\s*(?:in\s*)?net\s*debt', blob, re.I)
    if m: a.net_debt = _dollar_val(m)
    v = _s(r'[Bb]eta\s*\(?(\d+(?:\.\d+)?)');
    if v: a.beta = float(v)
    v = _s(r'SBC\s*(\d+(?:\.\d+)?)%\s*of\s*revenue');
    if v: a.sbc_pct_rev = float(v)/100
    m = re.search(r'[Oo]perating\s*loss\s*\$(\d+(?:\.\d+)?)\s*([BM])/quarter', blob)
    if m: a.op_loss_quarterly = _dollar_val(m)
    v = _s(r'(\d+(?:\.\d+)?)%\s*SI\b');
    if v: a.short_interest = float(v)/100
    v = _s(r'buyback.*?(\d+(?:\.\d+)?)%');
    if v: a.buyback_yield = float(v)/100
    v = _s(r'FCF\s*conversion.*?(\d+(?:\.\d+)?)%');
    if v: a.fcf_conversion = float(v)/100
    v = _s(r'(\d+(?:\.\d+)?)%\s*dilution');
    if v: a.dilution_pct = float(v)/100
    # Dividend-related metrics (require "dividend" prefix to avoid matching FCF yield)
    v = _s(r'[Dd]ividend\s+yield\s*(\d+(?:\.\d+)?)%')
    if v: a.dividend_yield = float(v)/100
    v = _s(r'payout\s*ratio\s*(\d+(?:\.\d+)?)%')
    if v: a.payout_ratio = float(v)/100
    # RSI — search technical points (not in blob which is fundamental+macro only)
    tech_blob = " ".join(signal.get("technical", {}).get("points", []))
    rm = re.search(r'RSI\s*(?:at\s*)?(\d+(?:\.\d+)?)', tech_blob, re.I)
    if rm: a.rsi = float(rm.group(1))
    return a


def _sd(sector: str) -> dict:
    vals = _SD.get(sector, _DFLT)
    return dict(zip(_SD_KEYS, vals))


def _resolve(a: _Anchors, sector: str) -> dict:
    sd = _sd(sector)
    gm = a.gross_margin_high or a.gross_margin or sd["gm"]
    mc = a.market_cap or 3e9
    beta = a.beta or sd["beta"]
    wacc = a.wacc or 0.045 + beta * 0.055
    roic = a.roic or (wacc + 0.05 if (a.piotroski or 5) >= 5 else wacc * 0.7)
    rg = a.rev_growth if a.rev_growth is not None else 0.08
    fy = a.fcf_yield or (0.06 if rg >= 0 else -0.04)
    return {"mc":mc,"gm":gm,"om":sd["om"],"nm":sd["nm"],"ev_rev":sd["ev_rev"],"tax":sd["tax"],
            "dna":sd["dna"],"capex":sd["capex"],"div":sd["div"],"beta":beta,
            "wacc":wacc,"roic":roic,"rg":rg,"fy":fy}


def _build_is(p: dict, a: _Anchors, sh: float) -> List[dict]:
    yr0 = datetime.now().year
    rev0 = p["mc"] / p["ev_rev"]
    rows = []
    for i, off in enumerate([-2, -1, 0]):
        rev = rev0 / (1 + p["rg"]) ** (-off) if off < 0 else rev0
        gm = p["gm"] - 0.02*(2+off)
        om = p["om"] - 0.015*(2+off)
        if a.op_loss_quarterly and off == 0:
            op = -a.op_loss_quarterly * 4; om = op/rev if rev else -.5; gm = max(om+.10, .05)
        cogs = rev*(1-gm); gp = rev*gm
        sbc = rev*(a.sbc_pct_rev or 0.03)
        rd = rev*(0.12 if gm > 0.5 else 0.03)
        sga = max(gp - rev*om - rd, rev*0.08)
        op = gp - rd - sga
        ie = (a.net_debt or p["mc"]*0.15)*0.055 if (a.net_debt or 0) > 0 else p["mc"]*0.01
        pt = op - ie; tx = pt*p["tax"] if pt > 0 else 0; ni = pt - tx
        ebitda = op + rev*p["dna"] + sbc
        yoy = p["rg"] if off > -2 else p["rg"] * 0.8
        math = {
            "revenue": _m("Revenue", "Market Cap / EV/Revenue", [
                ("Market Cap", p["mc"], "millions"), ("EV/Revenue Multiple", p["ev_rev"], "x")],
                ("Revenue", round(rev, 0), "millions"),
                "Back-solved from current market cap and sector EV/Revenue multiple" if off == 0 else f"Adjusted by {yoy:.1%} growth rate"),
            "gross_margin": _m("Gross Margin", "Gross Profit / Revenue", [
                ("Revenue", round(rev, 0), "millions"), ("Gross Profit", round(gp, 0), "millions")],
                ("Gross Margin", round(gm, 4), "pct")),
            "ebitda": _m("EBITDA", "Operating Income + D&A + SBC", [
                ("Operating Income", round(op, 0), "millions"),
                ("D&A", round(rev * p["dna"], 0), "millions"),
                ("Stock-Based Comp", round(sbc, 0), "millions")],
                ("EBITDA", round(ebitda, 0), "millions")),
            "ebitda_margin": _m("EBITDA Margin", "EBITDA / Revenue", [
                ("EBITDA", round(ebitda, 0), "millions"), ("Revenue", round(rev, 0), "millions")],
                ("EBITDA Margin", round(ebitda / rev if rev else 0, 4), "pct")),
            "net_income": _m("Net Income", "Pretax Income - Tax Expense", [
                ("Operating Income", round(op, 0), "millions"),
                ("Interest Expense", round(ie, 0), "millions"),
                ("Pretax Income", round(pt, 0), "millions"),
                ("Tax Rate", round(p["tax"], 4), "pct"),
                ("Tax Expense", round(tx, 0), "millions")],
                ("Net Income", round(ni, 0), "millions")),
            "diluted_eps": _m("Diluted EPS", "Net Income / Shares Outstanding", [
                ("Net Income", round(ni, 0), "millions"), ("Shares Outstanding", round(sh, 0), "raw")],
                ("Diluted EPS", round(ni / sh if sh else 0, 2), "usd")),
        }
        rows.append({"year":f"FY{yr0+off}","revenue":round(rev,0),
            "yoy_growth":round(yoy,4),
            "cogs":round(cogs,0),"gross_profit":round(gp,0),"gross_margin":round(gm,4),
            "rd_expense":round(rd,0),"sga_expense":round(sga,0),
            "operating_income":round(op,0),"operating_margin":round(op/rev if rev else 0,4),
            "interest_expense":round(ie,0),"pretax_income":round(pt,0),
            "tax_expense":round(tx,0),"tax_rate":round(p["tax"],4),
            "net_income":round(ni,0),"net_margin":round(ni/rev if rev else 0,4),
            "diluted_eps":round(ni/sh if sh else 0,2),
            "ebitda":round(ebitda,0),"ebitda_margin":round(ebitda/rev if rev else 0,4),
            "sbc":round(sbc,0),"_math":math})
    return rows


def _build_bs(p: dict, a: _Anchors, sh: float, isd: List[dict]) -> List[dict]:
    rows = []
    for i, inc in enumerate(isd):
        rev = inc["revenue"]
        nd = a.net_debt if a.net_debt else p["mc"]*0.15
        ltd = max(nd*0.8, 0) if nd > 0 else 0; std = max(nd*0.2, 0) if nd > 0 else 0
        cash = max(rev*0.12 - min(nd,0), rev*0.05) if nd <= 0 else rev*0.05
        recv = rev*0.12; inv = rev*(0.06 if p["gm"]<0.6 else 0.02)
        tca = cash+recv+inv; ppe = rev*p["capex"]*5; ap = rev*0.08; tcl = ap+std
        eq = (p["mc"]/(a.book_multiple or 2.5)) * (0.9+0.05*i)
        tl = tcl + ltd; ta = eq + tl; gw = max(ta - tca - ppe, 0)
        bm = a.book_multiple or 2.5
        inv_pct = 0.06 if p["gm"] < 0.6 else 0.02
        math = {
            "cash": _m("Cash & Equivalents", "Revenue x 5%" if nd > 0 else "Revenue x 12% - min(Net Debt, 0)", [
                ("Revenue", round(rev, 0), "millions"), ("Net Debt", round(nd, 0), "millions")],
                ("Cash", round(cash, 0), "millions")),
            "total_assets": _m("Total Assets", "Total Equity + Total Liabilities", [
                ("Total Equity", round(eq, 0), "millions"), ("Total Liabilities", round(tl, 0), "millions")],
                ("Total Assets", round(ta, 0), "millions")),
            "total_debt": _m("Total Debt", "Long-Term Debt + Short-Term Debt", [
                ("Long-Term Debt", round(ltd, 0), "millions"), ("Short-Term Debt", round(std, 0), "millions")],
                ("Total Debt", round(ltd + std, 0), "millions"),
                f"Net debt split 80/20 LT/ST from {'anchor' if a.net_debt else 'Market Cap x 15%'}"),
            "total_equity": _m("Total Equity", "Market Cap / Book Multiple x Adj", [
                ("Market Cap", p["mc"], "millions"), ("Book Multiple", bm, "x"),
                ("Year Adj", round(0.9 + 0.05 * i, 2), "raw")],
                ("Total Equity", round(eq, 0), "millions")),
            "book_value_per_share": _m("Book Value / Share", "Total Equity / Shares", [
                ("Total Equity", round(eq, 0), "millions"), ("Shares Outstanding", round(sh, 0), "raw")],
                ("BV/Share", round(eq / sh if sh else 0, 2), "usd")),
        }
        rows.append({"year":inc["year"],"cash":round(cash,0),"receivables":round(recv,0),
            "inventory":round(inv,0),"total_current_assets":round(tca,0),
            "ppe_net":round(ppe,0),"goodwill":round(gw,0),"total_assets":round(ta,0),
            "accounts_payable":round(ap,0),"short_term_debt":round(std,0),
            "total_current_liabilities":round(tcl,0),"long_term_debt":round(ltd,0),
            "total_liabilities":round(tl,0),"total_equity":round(eq,0),
            "book_value_per_share":round(eq/sh if sh else 0,2),"_math":math})
    return rows


def _build_cf(p: dict, a: _Anchors, sh: float, isd: List[dict]) -> List[dict]:
    rows = []
    for inc in isd:
        rev, ni = inc["revenue"], inc["net_income"]
        dna = rev*p["dna"]; sbc = inc["sbc"]; wc = rev*random.uniform(-0.02, 0.02)
        cfo = round(ni+dna+sbc+wc, 0); cap = round(-rev*p["capex"], 0)
        fcf = cfo + cap  # exact on rounded values
        bb = -p["mc"]*(a.buyback_yield or 0) if (a.buyback_yield or 0) > 0 else 0
        div = -p["mc"]*p["div"]; cff = bb+div
        math = {
            "cfo": _m("Cash from Operations", "Net Income + D&A + SBC + WC Change", [
                ("Net Income", round(ni, 0), "millions"), ("D&A", round(dna, 0), "millions"),
                ("SBC", round(sbc, 0), "millions"), ("WC Change", round(wc, 0), "millions")],
                ("CFO", cfo, "millions")),
            "capex": _m("Capital Expenditures", "-Revenue x CapEx %", [
                ("Revenue", round(rev, 0), "millions"), ("CapEx % of Rev", round(p["capex"], 4), "pct")],
                ("CapEx", cap, "millions")),
            "fcf": _m("Free Cash Flow", "CFO + CapEx", [
                ("CFO", cfo, "millions"), ("CapEx", cap, "millions")],
                ("FCF", fcf, "millions")),
            "fcf_margin": _m("FCF Margin", "FCF / Revenue", [
                ("FCF", fcf, "millions"), ("Revenue", round(rev, 0), "millions")],
                ("FCF Margin", round(fcf / rev if rev else 0, 4), "pct")),
            "fcf_yield": _m("FCF Yield", "FCF / Market Cap", [
                ("FCF", fcf, "millions"), ("Market Cap", p["mc"], "millions")],
                ("FCF Yield", round(fcf / p["mc"] if p["mc"] else 0, 4), "pct")),
            "fcf_per_share": _m("FCF / Share", "FCF / Shares Outstanding", [
                ("FCF", fcf, "millions"), ("Shares Outstanding", round(sh, 0), "raw")],
                ("FCF/Share", round(fcf / sh if sh else 0, 2), "usd")),
        }
        rows.append({"year":inc["year"],"net_income":round(ni,0),"dna":round(dna,0),
            "sbc":round(sbc,0),"working_capital_change":round(wc,0),"cfo":cfo,"capex":cap,
            "acquisitions":0,"cfi":cap,"debt_change":0,"buybacks":round(bb,0),
            "dividends":round(div,0),"cff":round(cff,0),"fcf":fcf,
            "fcf_margin":round(fcf/rev if rev else 0,4),
            "fcf_yield":round(fcf/p["mc"] if p["mc"] else 0,4),
            "fcf_per_share":round(fcf/sh if sh else 0,2),"_math":math})
    return rows


def _build_dcf(p: dict, a: _Anchors, sh: float, isd: List[dict], action: str, price: float) -> dict:
    rev_base = isd[-1]["revenue"]; em_term = isd[-1]["ebitda_margin"]
    w = p["wacc"]; rf = 0.045; erp = 0.055; b = p["beta"]
    coe = rf + b*erp; pcod = 0.055; acod = pcod*(1-p["tax"])
    nd = a.net_debt if a.net_debt and a.net_debt > 0 else p["mc"]*0.15
    ew = p["mc"]/(p["mc"]+max(nd,0)); dw = 1-ew; cw = coe*ew + acod*dw
    # Target price from anchor or action
    tp = price*(1+(a.dcf_upside or 0)) if a.dcf_upside else price*(1.20 if action=="BUY" else 0.75)
    em_boost = 0.04 if action=="BUY" else -0.02
    cagr = min(max(p["rg"], -0.10), 0.30)
    projs = []
    yr0 = datetime.now().year
    for i in range(1, 6):
        gf = cagr*(1-0.10*i); rv = rev_base*(1+gf)**i
        em = em_term + em_boost*(i/5); eb = rv*em; dn = rv*p["dna"]
        ebit = eb-dn; tx = ebit*p["tax"] if ebit>0 else 0; nopat = ebit-tx
        cx = -rv*p["capex"]; nwc = -rv*0.01; uf = nopat+dn+cx+nwc
        proj_math = {
            "revenue": _m("Projected Revenue", "Base Revenue x (1 + Growth)^Year", [
                ("Base Revenue", round(rev_base, 0), "millions"),
                ("Growth Factor", round(gf, 4), "pct"), ("Year", i, "raw")],
                ("Revenue", round(rv, 0), "millions"),
                f"Growth fades 10% per year from {cagr:.1%} base CAGR"),
            "ebitda": _m("Projected EBITDA", "Revenue x EBITDA Margin", [
                ("Revenue", round(rv, 0), "millions"), ("EBITDA Margin", round(em, 4), "pct")],
                ("EBITDA", round(eb, 0), "millions")),
            "ufcf": _m("Unlevered FCF", "NOPAT + D&A + CapEx + NWC Change", [
                ("NOPAT", round(nopat, 0), "millions"), ("D&A", round(dn, 0), "millions"),
                ("CapEx", round(cx, 0), "millions"), ("NWC Change", round(nwc, 0), "millions")],
                ("UFCF", round(uf, 0), "millions")),
        }
        projs.append({"year":yr0+i,"revenue":round(rv,0),"rev_growth":round(gf,4),
            "ebitda":round(eb,0),"ebitda_margin":round(em,4),"dna":round(dn,0),
            "ebit":round(ebit,0),"taxes":round(tx,0),"nopat":round(nopat,0),
            "capex":round(cx,0),"nwc_change":round(nwc,0),"ufcf":round(uf,0),"_math":proj_math})
    df = [(1+w)**i for i in range(1,6)]
    pv_f = sum(pr["ufcf"]/d for pr,d in zip(projs,df)); cash = isd[-1]["revenue"]*0.05
    # Back-solve terminal growth via bisection
    te = tp*sh if sh else p["mc"]; tev = te+max(nd,0)-cash; req = tev-pv_f
    lu = projs[-1]["ufcf"]; d5 = df[-1]; tg = 0.025
    if lu > 0 and req > 0:
        lo, hi = 0.005, w-0.005
        for _ in range(50):
            mid = (lo+hi)/2
            if (lu*(1+mid))/(w-mid)/d5 < req: lo = mid
            else: hi = mid
        tg = max(0.005, min(round((lo+hi)/2, 5), w-0.005))
    elif lu <= 0: tg = 0.015
    tv = lu*(1+tg)/(w-tg) if w>tg else lu*(1+tg)/0.005
    pvt = tv/d5; iev = pv_f+pvt; ieq = iev-max(nd,0)+cash
    ip = max(ieq/sh if sh else 0, price*0.15)
    # WACC build math
    wacc_math = {
        "cost_of_equity": _m("Cost of Equity (CAPM)", "Risk-Free Rate + Beta x ERP", [
            ("Risk-Free Rate", rf, "pct"), ("Beta", round(b, 2), "raw"), ("Equity Risk Premium", erp, "pct")],
            ("Cost of Equity", round(coe, 4), "pct")),
        "after_tax_cost_of_debt": _m("After-Tax Cost of Debt", "Pre-Tax CoD x (1 - Tax Rate)", [
            ("Pre-Tax Cost of Debt", pcod, "pct"), ("Tax Rate", round(p["tax"], 4), "pct")],
            ("After-Tax CoD", round(acod, 4), "pct")),
        "wacc": _m("WACC", "CoE x Equity Weight + CoD x Debt Weight", [
            ("Cost of Equity", round(coe, 4), "pct"), ("Equity Weight", round(ew, 4), "pct"),
            ("After-Tax CoD", round(acod, 4), "pct"), ("Debt Weight", round(dw, 4), "pct")],
            ("WACC", round(cw, 4), "pct")),
    }
    # Output math
    output_math = {
        "pv_fcfs": _m("PV of Cash Flows", "Sum of UFCF[i] / (1+WACC)^i", [
            ("WACC", round(w, 4), "pct")] + [
            (f"UFCF Year {pr['year']}", pr["ufcf"], "millions") for pr in projs],
            ("PV of FCFs", round(pv_f, 0), "millions")),
        "pv_terminal": _m("PV of Terminal Value", "Terminal Value / (1+WACC)^5", [
            ("Last UFCF", lu, "millions"), ("Terminal Growth", round(tg, 4), "pct"),
            ("WACC", round(w, 4), "pct"), ("Terminal Value", round(tv, 0), "millions"),
            ("Discount Factor (5yr)", round(d5, 4), "raw")],
            ("PV of Terminal", round(pvt, 0), "millions"),
            "Gordon Growth: UFCF x (1+g) / (WACC-g)"),
        "implied_ev": _m("Implied Enterprise Value", "PV of FCFs + PV of Terminal", [
            ("PV of FCFs", round(pv_f, 0), "millions"), ("PV of Terminal", round(pvt, 0), "millions")],
            ("Implied EV", round(iev, 0), "millions")),
        "implied_price": _m("Implied Price / Share", "(Implied EV - Net Debt + Cash) / Shares", [
            ("Implied EV", round(iev, 0), "millions"), ("Net Debt", round(nd, 0), "millions"),
            ("Cash", round(cash, 0), "millions"), ("Shares", round(sh, 0), "raw")],
            ("Implied Price", round(ip, 2), "usd")),
        "upside_pct": _m("Upside / (Downside)", "(Implied Price / Current Price) - 1", [
            ("Implied Price", round(ip, 2), "usd"), ("Current Price", price, "usd")],
            ("Upside", round(ip / price - 1 if price else 0, 4), "pct")),
    }
    # Sensitivity math template
    sens_math = {
        "template": _m("Sensitivity Cell", "(UFCF x (1+g)) / (WACC-g) / (1+WACC)^5 + PV FCFs - Net Debt + Cash) / Shares", [
            ("Last UFCF (Year 5)", lu, "millions"), ("PV of FCFs", round(pv_f, 0), "millions"),
            ("Net Debt", round(nd, 0), "millions"), ("Cash", round(cash, 0), "millions"),
            ("Shares", round(sh, 0), "raw")],
            ("Implied Price", 0, "usd"),
            "WACC and Terminal Growth vary per cell"),
    }
    # Sensitivity 5x5
    wv = [round(w+d,4) for d in [-.01,-.005,0,.005,.01]]
    gv = [round(tg+d,4) for d in [-.005,-.0025,0,.0025,.005]]
    wa = np.array(wv).reshape(5,1); ga = np.array(gv).reshape(1,5)
    tvm = (lu*(1+ga))/np.maximum(wa-ga, 0.005)
    pvm = tvm/(1+wa)**5; evm = pv_f+pvm
    ipm = np.maximum((evm-max(nd,0)+cash)/sh if sh else evm, price*0.05)
    return {"assumptions":{"projection_years":5,"revenue_cagr":round(cagr,4),
        "terminal_ebitda_margin":round(em_term+em_boost,4),"tax_rate":round(p["tax"],4),
        "dna_pct_rev":round(p["dna"],4),"capex_pct_rev":round(p["capex"],4),
        "nwc_pct_delta_rev":0.01,"wacc":round(w,4),"terminal_growth":round(tg,4),
        "terminal_method":"perpetuity_growth"},
    "wacc_build":{"risk_free_rate":rf,"erp":erp,"beta":round(b,2),"cost_of_equity":round(coe,4),
        "pretax_cost_of_debt":pcod,"after_tax_cost_of_debt":round(acod,4),
        "debt_weight":round(dw,4),"equity_weight":round(ew,4),"wacc":round(cw,4),
        "_math":wacc_math},
    "projected_ufcf":projs,
    "output":{"pv_fcfs":round(pv_f,0),"pv_terminal":round(pvt,0),
        "terminal_pct_of_total":round(pvt/iev if iev else 0,4),
        "implied_ev":round(iev,0),"net_debt":round(nd,0),"cash":round(cash,0),
        "implied_equity_value":round(ieq,0),"shares":round(sh,0),
        "implied_price":round(ip,2),"current_price":0,"upside_pct":0,
        "_math":output_math},
    "sensitivity":{"wacc_values":wv,"growth_values":gv,"matrix":np.round(ipm,2).tolist(),
        "_math":sens_math}}


def _build_comps(p: dict, a: _Anchors, sh: float, sig: dict, isd: List[dict]) -> dict:
    sec = sig.get("sector",""); sk = _SMAP.get(sec, sec.lower().replace(" ","_"))
    pool = [t for t in _SECTOR_TICKERS.get(sk,[]) if t != sig["ticker"]]
    n = min(random.choice([3,4]), len(pool)) if pool else 3
    chosen = random.sample(pool, n) if len(pool) >= n else pool
    jit = lambda b, s=0.2: b*random.gauss(1.0, s)
    rv, eb, ni = isd[-1]["revenue"], isd[-1]["ebitda"], isd[-1]["net_income"]
    ev_s = p["mc"]+(a.net_debt or p["mc"]*0.15)
    evr_s = ev_s/rv if rv else 0; eve_s = ev_s/eb if eb else 0
    pe_s = p["mc"]/ni if ni and ni>0 else 0; rg = p["rg"]
    peers = []
    for tk in chosen:
        prg = jit(rg if rg>0 else 0.05, 0.3); pem = jit(isd[-1]["ebitda_margin"], 0.2)
        pnm = jit(isd[-1]["net_margin"], 0.25); pro = jit(p["roic"], 0.2)
        pevr = jit(evr_s if evr_s>0 else 2.0, 0.2)
        peve = jit(eve_s if eve_s>0 else 12.0, 0.2)
        ppe = jit(pe_s if pe_s>0 else 18.0, 0.2)
        peg = ppe/(prg*100) if prg>0 else 0
        peers.append({"name":_NAMES.get(tk, tk),"ticker":tk,"ev":round(jit(ev_s,.25),0),
            "ev_revenue":round(pevr,2),"ev_ebitda":round(peve,2),"pe_fwd":round(ppe,2),
            "peg":round(peg,2),"rev_growth":round(prg,4),"ebitda_margin":round(pem,4),
            "net_margin":round(pnm,4),"roic":round(pro,4)})
    def _med(k):
        v = sorted(x[k] for x in peers if x[k]); return round(v[len(v)//2],4) if v else 0
    med = {k:_med(k) for k in ["ev_revenue","ev_ebitda","pe_fwd","peg","rev_growth","ebitda_margin","net_margin","roic"]}
    med.update({"name":"Peer Median","ticker":"","ev":round(sum(x["ev"] for x in peers)/len(peers),0) if peers else 0})
    subj = {"name":sig.get("short_name",""),"ticker":sig["ticker"],"ev":round(ev_s,0),
        "ev_revenue":round(evr_s,2),"ev_ebitda":round(eve_s,2),"pe_fwd":round(pe_s,2),
        "peg":round(pe_s/(rg*100) if rg>0 else 0,2),"rev_growth":round(rg,4),
        "ebitda_margin":round(isd[-1]["ebitda_margin"],4),
        "net_margin":round(isd[-1]["net_margin"],4),"roic":round(p["roic"],4)}
    prem = {"ev_revenue":round(evr_s/med["ev_revenue"]-1,4) if med["ev_revenue"] else 0,
        "ev_ebitda":round(eve_s/med["ev_ebitda"]-1,4) if med["ev_ebitda"] else 0,
        "pe":round(pe_s/med["pe_fwd"]-1,4) if med["pe_fwd"] else 0,
        "roic":round(p["roic"]/med["roic"]-1,4) if med["roic"] else 0}
    impl = []
    if med["ev_revenue"] and rv:
        ie = med["ev_revenue"]*rv
        impl.append({"method":"EV/Revenue","peer_median":med["ev_revenue"],"subject_metric":round(rv,0),
            "implied_ev":round(ie,0),"implied_price":round((ie-(a.net_debt or 0))/sh if sh else 0,2)})
    if med["ev_ebitda"] and eb and eb>0:
        ie = med["ev_ebitda"]*eb
        impl.append({"method":"EV/EBITDA","peer_median":med["ev_ebitda"],"subject_metric":round(eb,0),
            "implied_ev":round(ie,0),"implied_price":round((ie-(a.net_debt or 0))/sh if sh else 0,2)})
    return {"peers":peers,"peer_median":med,"subject":subj,"premium_discount":prem,"implied_valuation":impl}


def _moat(sec: str, pio: Optional[int], roic: float, wacc: float) -> dict:
    sp = roic - wacc
    if sp > 0.12 and (pio or 0) >= 7: r, d = "Wide", "Exceptional returns on capital with durable competitive position."
    elif sp > 0.06: r, d = "Narrow", "Above-average returns suggesting some competitive advantage."
    else: r, d = "None", "Returns do not indicate a sustainable competitive advantage."
    tams = {"Energy":"$800B global","Technology":"$500B addressable","Healthcare":"$600B global",
        "Industrials":"$1.2T global","Financials":"$400B alternatives","Materials":"$350B domestic",
        "Consumer Discretionary":"$300B specialty retail","Consumer Staples":"$200B US packaged",
        "Utilities":"$150B US distributed"}
    tam_vals = {"Energy":800e9,"Technology":500e9,"Healthcare":600e9,
        "Industrials":1.2e12,"Financials":400e9,"Materials":350e9,
        "Consumer Discretionary":300e9,"Consumer Staples":200e9,
        "Utilities":150e9}
    return {"rating":r,"description":d,"tam":tam_vals.get(sec,200e9),"market_share":round(random.uniform(0.005,0.03),4)}


def _build_ddm(anchors: _Anchors, sector: str, p: dict, price: float) -> dict:
    """Build Gordon Growth Model / Dividend Discount Model section."""
    div_yield = anchors.dividend_yield or p["div"] or 0.025
    d0 = price * div_yield

    # Growth rate assumptions
    roe = anchors.roic or 0.12  # approximate ROE from ROIC
    eps_est = max(price * 0.05, 0.01)  # rough EPS fallback
    payout = min(div_yield * price / eps_est, 0.95)
    sustainable_g = min(roe * (1 - payout), 0.08)
    sector_terminal = {"Energy": 0.015, "Consumer Staples": 0.02, "Technology": 0.035,
                       "Healthcare": 0.03, "Industrials": 0.02, "Financials": 0.025,
                       "Materials": 0.015, "Consumer Discretionary": 0.025,
                       "Utilities": 0.015}.get(sector, 0.02)
    g = min(sustainable_g, sector_terminal * 1.5)

    beta = anchors.beta or p["beta"] or 1.0
    rf = 0.04
    erp = 0.06
    r = rf + beta * erp

    # Single-stage DDM
    d1 = d0 * (1 + g)
    if r > g:
        intrinsic = d1 / (r - g)
    else:
        intrinsic = price  # fallback

    # Sensitivity table: rows = required return (r), cols = growth rate (g)
    r_range = [r - 0.02, r - 0.01, r, r + 0.01, r + 0.02]
    g_range = [g - 0.01, g - 0.005, g, g + 0.005, g + 0.01]
    sensitivity = []
    for ri in r_range:
        row_vals = []
        for gi in g_range:
            if ri > gi and ri > 0:
                val = d1 * (1 + gi - g) / (ri - gi)
                row_vals.append(round(val, 2))
            else:
                row_vals.append(None)
        sensitivity.append({"r": round(ri, 4), "values": row_vals})

    return {
        "model": "Gordon Growth (DDM)",
        "assumptions": {
            "current_dividend": round(d0, 2),
            "next_year_dividend": round(d1, 2),
            "sustainable_growth": round(sustainable_g, 4),
            "growth_rate_used": round(g, 4),
            "required_return": round(r, 4),
            "cost_of_equity_method": "CAPM",
        },
        "output": {
            "intrinsic_value": round(intrinsic, 2),
            "current_price": round(price, 2),
            "upside": round((intrinsic / price - 1), 4),
            "margin_of_safety": round((intrinsic - price) / max(intrinsic, 0.01), 4),
        },
        "sensitivity": {
            "row_label": "Required Return",
            "col_label": "Dividend Growth Rate",
            "g_values": [round(gi, 4) for gi in g_range],
            "rows": sensitivity,
        },
    }


def _build_ceo_info(signal: dict) -> dict:
    """Build CEO information section from signal data."""
    ceo_data = signal.get("ceo_info") or {}
    return {
        "ceo_changed_recently": ceo_data.get("ceo_changed_recently"),
        "change_date": ceo_data.get("change_date"),
        "filing_url": ceo_data.get("filing_url"),
        "has_data": ceo_data.get("has_data", False),
        "note": "CEO changed within past 2 years — monitor management transition" if ceo_data.get("ceo_changed_recently") else "No recent CEO change detected",
    }


def _build_compensation(signal: dict) -> dict:
    """Build executive compensation structure section."""
    comp = signal.get("compensation") or {}
    if not comp.get("has_data"):
        return {"has_data": False, "note": "Compensation data unavailable"}

    return {
        "has_data": True,
        "equity_heavy": comp.get("equity_heavy"),
        "equity_pct": comp.get("equity_pct"),
        "cash_pct": comp.get("cash_pct"),
        "total_ceo_compensation": comp.get("total_ceo_compensation"),
        "latest_proxy_date": comp.get("latest_proxy_date"),
        "filing_url": comp.get("filing_url"),
        "alignment_note": "Equity-heavy compensation suggests management-shareholder alignment" if comp.get("equity_heavy") else "Cash-heavy compensation — review incentive alignment",
    }


def _build_scoring_breakdown(signal: dict) -> dict:
    """Build scoring model breakdown section from signal data."""
    bd = signal.get("scoring_breakdown")
    if not bd:
        return {"has_data": False}

    calc_details = bd.get("calculation_details", {})
    dim_details = calc_details.get("dimensions", {}) if calc_details else {}
    safety_gates = calc_details.get("safety_gates", []) if calc_details else []

    # Order components by contribution descending
    components = []
    for key in ["piotroski", "cash_flow_quality", "roic_spread", "balance_sheet",
                "dcf_upside", "income_health", "growth_momentum", "margin_trajectory",
                "blindspot", "price_momentum", "low_volatility"]:
        entry = bd.get(key)
        if entry:
            comp = {
                "key": key,
                "label": entry["label"],
                "score": entry["score"],
                "weight": entry["weight"],
                "contribution": entry["contribution"],
            }
            # Attach detailed calculation inputs if available
            if key in dim_details:
                comp["calc"] = dim_details[key]
            components.append(comp)
    components.sort(key=lambda c: c["contribution"], reverse=True)

    return {
        "has_data": True,
        "components": components,
        "composite_total": bd.get("composite_total"),
        "rank": bd.get("rank"),
        "safety_gates": safety_gates,
    }


def _build_roi(signal: dict, anchors: _Anchors, p: dict, price: float) -> dict:
    """Build ROI analysis section."""
    roi = signal.get("roi_analysis") or {}
    if not roi:
        target = signal.get("take_profit", price * 1.08)
        div_yield = anchors.dividend_yield or p["div"] or 0
        beta = anchors.beta or p["beta"] or 1.0
        if price > 0:
            cap_gain = (target - price) / price
            total = cap_gain + div_yield
            roi = {
                "total_roi_pct": round(total, 4),
                "capital_gain_pct": round(cap_gain, 4),
                "income_return_pct": round(div_yield, 4),
                "risk_adjusted_roi": round(total / max(beta, 0.3), 4),
            }
    return roi


def _default_rsi(action: str, confidence: float) -> float:
    """Generate a plausible RSI(14) value based on action and confidence."""
    if action == "BUY":
        # Value stocks being recommended: RSI typically 35-55
        return round(35 + (1 - confidence) * 20 + random.uniform(-5, 5), 1)
    elif action == "SELL":
        # Overbought stocks being sold: RSI typically 60-80
        return round(60 + confidence * 20 + random.uniform(-5, 5), 1)
    else:
        return round(45 + random.uniform(-10, 10), 1)


def _build_capital_efficiency(isd: list[dict], cfd: list[dict], p: dict) -> dict:
    """Build capital efficiency metrics from income statement and cash flow data."""
    latest_is = isd[-1]
    latest_cf = cfd[-1]
    rev = latest_is["revenue"]
    rd = latest_is["rd_expense"]
    capex_abs = abs(latest_cf["capex"]) if latest_cf["capex"] else 0

    rd_intensity = round(rd / rev, 4) if rev else 0
    capex_intensity = round(capex_abs / rev, 4) if rev else 0
    rd_to_capex = round(rd / capex_abs, 2) if capex_abs else 0
    rg = latest_is.get("yoy_growth", p["rg"])
    rd_roi = round(rg / rd_intensity, 2) if rd_intensity else 0

    math = {
        "rd_intensity": _m("R&D Intensity", "R&D Expense / Revenue", [
            ("R&D Expense", round(rd, 0), "millions"), ("Revenue", round(rev, 0), "millions")],
            ("R&D Intensity", rd_intensity, "pct")),
        "capex_intensity": _m("CapEx Intensity", "CapEx / Revenue", [
            ("CapEx", round(capex_abs, 0), "millions"), ("Revenue", round(rev, 0), "millions")],
            ("CapEx Intensity", capex_intensity, "pct")),
        "rd_to_capex": _m("R&D / CapEx", "R&D Expense / CapEx", [
            ("R&D Expense", round(rd, 0), "millions"), ("CapEx", round(capex_abs, 0), "millions")],
            ("R&D/CapEx", rd_to_capex, "x")),
        "rd_roi_proxy": _m("R&D ROI Proxy", "Revenue Growth / R&D Intensity", [
            ("Revenue Growth", round(rg, 4), "pct"), ("R&D Intensity", rd_intensity, "pct")],
            ("R&D ROI Proxy", rd_roi, "x"),
            "Higher = more revenue growth per unit of R&D spend"),
    }
    return {
        "rd_intensity": rd_intensity,
        "capex_intensity": capex_intensity,
        "rd_to_capex": rd_to_capex,
        "rd_roi_proxy": rd_roi,
        "rd_expense": round(rd, 0),
        "capex": round(capex_abs, 0),
        "revenue": round(rev, 0),
        "_math": math,
    }


def generate_mock_report(signal: dict) -> dict:
    """Generate a comprehensive, internally-consistent mock equity research report."""
    random.seed(hash(signal["ticker"]))
    anc = _parse_anchors(signal)
    sec = signal.get("sector", "Industrials")
    price = signal.get("entry_price", 50.0)
    act = signal.get("action", "BUY")
    conf = signal.get("confidence", 0.75)
    p = _resolve(anc, sec)
    mc = p["mc"]; sh = mc/price if price else mc/50
    rat = "Strong Buy" if act=="BUY" and conf>=.85 else "Buy" if act=="BUY" and conf>=.70 else \
          "Strong Sell" if act=="SELL" and conf>=.85 else "Sell" if act=="SELL" and conf>=.70 else "Hold"
    isd = _build_is(p, anc, sh); bsd = _build_bs(p, anc, sh, isd)
    cfd = _build_cf(p, anc, sh, isd); dcf = _build_dcf(p, anc, sh, isd, act, price)
    comps = _build_comps(p, anc, sh, signal, isd)
    dp = dcf["output"]["implied_price"]
    # Override mock DCF with real screener intrinsic value when available
    real_iv = signal.get("intrinsic_value_per_share") or signal.get("take_profit")
    if real_iv and real_iv > 0 and real_iv != price:
        dp = real_iv
        dcf["output"]["implied_price"] = round(dp, 2)
    dcf["output"]["current_price"] = price
    real_upside = signal.get("real_dcf_upside_pct")
    if real_upside is not None:
        dcf["output"]["upside_pct"] = round(real_upside, 4)
    else:
        dcf["output"]["upside_pct"] = round(dp/price-1, 4) if price else 0
    cp = [iv["implied_price"] for iv in comps["implied_valuation"] if iv["implied_price"] > 0]
    ca = sum(cp)/len(cp) if cp else dp
    tt = signal.get("take_profit", price*1.08)
    bl = dp*0.50 + ca*0.30 + tt*0.20; up = bl/price-1 if price else 0
    nd = anc.net_debt or mc*0.15; ev = mc+nd
    # Catalysts
    tpts = signal.get("technical",{}).get("points",[])[:2]
    mpts = signal.get("macro",{}).get("points",[])[:2]
    _cat_dates = ["Next 7 days", "Next 30 days", "Next quarter", "Next 6 months"]
    cats = [{"date":_cat_dates[i] if i < len(_cat_dates) else "Next 6 months",
             "event":pt,"impact":"Positive" if act=="BUY" else "Negative"}
            for i, pt in enumerate(tpts+mpts)]
    # Risks
    _sev_cycle = ["High", "Medium", "Low"]
    _prob_cycle = ["High", "Moderate", "Low"]
    raw_risks = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)', signal.get("risk_context","")) if len(s.strip()) > 10]
    rsks = [{"factor": s, "severity": _sev_cycle[i % 3], "probability": _prob_cycle[i % 3],
             "detail": f"{s}. This could materially impact {'revenue growth' if i % 3 == 0 else 'margin trajectory' if i % 3 == 1 else 'investor sentiment'} over the next 12 months."}
            for i, s in enumerate(raw_risks)]
    if not rsks:
        rsks = [{"factor": "General market risk", "severity": "High", "probability": "High",
                 "detail": "Broad market downturn could impact share price and compress valuation multiples."}]
    rsks.append({"factor": "Broader market correction", "severity": "Medium", "probability": "Moderate",
                 "detail": "Macroeconomic downturn or risk-off sentiment could compress valuations across the sector."})
    rsks.append({"factor": "Liquidity risk", "severity": "Low", "probability": "Low",
                 "detail": "Small-mid cap names may face wider bid-ask spreads during periods of market stress."})
    mo = _moat(sec, anc.piotroski, p["roic"], p["wacc"])
    i0, b0 = isd[-1], bsd[-1]
    te, ta = b0["total_equity"], b0["total_assets"]
    rv, cg, inv, recv, ap_ = i0["revenue"], i0["cogs"], b0["inventory"], b0["receivables"], b0["accounts_payable"]
    nde = round(nd / i0["ebitda"] if i0["ebitda"] else 0, 2)
    dte = round((b0["long_term_debt"] + b0["short_term_debt"]) / te if te else 0, 2)
    icov = round(i0["operating_income"] / i0["interest_expense"] if i0["interest_expense"] else 0, 2)
    cr = round(b0["total_current_assets"] / b0["total_current_liabilities"] if b0["total_current_liabilities"] else 0, 2)
    qr = round((b0["cash"] + b0["receivables"]) / b0["total_current_liabilities"] if b0["total_current_liabilities"] else 0, 2)
    cap_math = {
        "net_debt": _m("Net Debt", "Long-Term Debt + Short-Term Debt - Cash", [
            ("LT Debt", b0["long_term_debt"], "millions"), ("ST Debt", b0["short_term_debt"], "millions"),
            ("Cash", b0["cash"], "millions")],
            ("Net Debt", round(nd, 0), "millions")),
        "net_debt_ebitda": _m("Net Debt / EBITDA", "Net Debt / EBITDA", [
            ("Net Debt", round(nd, 0), "millions"), ("EBITDA", i0["ebitda"], "millions")],
            ("ND/EBITDA", nde, "x")),
        "debt_to_equity": _m("Debt / Equity", "(LT Debt + ST Debt) / Total Equity", [
            ("Total Debt", b0["long_term_debt"] + b0["short_term_debt"], "millions"),
            ("Total Equity", te, "millions")],
            ("D/E", dte, "x")),
        "interest_coverage": _m("Interest Coverage", "Operating Income / Interest Expense", [
            ("Operating Income", i0["operating_income"], "millions"),
            ("Interest Expense", i0["interest_expense"], "millions")],
            ("Interest Coverage", icov, "x")),
        "current_ratio": _m("Current Ratio", "Current Assets / Current Liabilities", [
            ("Current Assets", b0["total_current_assets"], "millions"),
            ("Current Liabilities", b0["total_current_liabilities"], "millions")],
            ("Current Ratio", cr, "x")),
        "quick_ratio": _m("Quick Ratio", "(Cash + Receivables) / Current Liabilities", [
            ("Cash", b0["cash"], "millions"), ("Receivables", b0["receivables"], "millions"),
            ("Current Liabilities", b0["total_current_liabilities"], "millions")],
            ("Quick Ratio", qr, "x")),
    }
    cap = {"net_debt":round(nd,0),"net_debt_ebitda":nde,"debt_to_equity":dte,
        "interest_coverage":icov,"current_ratio":cr,"quick_ratio":qr,"_math":cap_math}
    _roe = round(i0["net_income"] / te if te else 0, 4)
    _roa = round(i0["net_income"] / ta if ta else 0, 4)
    _at = round(rv / ta if ta else 0, 4)
    _it = round(cg / inv if inv else 0, 2)
    _dso = round(recv / rv * 365 if rv else 0, 1)
    _dpo = round(ap_ / cg * 365 if cg else 0, 1)
    _dio = round(inv / cg * 365 if cg else 0, 1)
    _ccc = round((_dso + _dio - _dpo) if rv and cg else 0, 1)
    prof_math = {
        "roe": _m("Return on Equity", "Net Income / Total Equity", [
            ("Net Income", i0["net_income"], "millions"), ("Total Equity", te, "millions")],
            ("ROE", _roe, "pct")),
        "roa": _m("Return on Assets", "Net Income / Total Assets", [
            ("Net Income", i0["net_income"], "millions"), ("Total Assets", ta, "millions")],
            ("ROA", _roa, "pct")),
        "roic": _m("Return on Invested Capital", "Source: anchor or WACC + spread", [
            ("WACC", p["wacc"], "pct"), ("ROIC-WACC Spread", round(p["roic"] - p["wacc"], 4), "pct")],
            ("ROIC", round(p["roic"], 4), "pct"),
            "ROIC from signal anchor or derived from Piotroski score"),
        "asset_turnover": _m("Asset Turnover", "Revenue / Total Assets", [
            ("Revenue", rv, "millions"), ("Total Assets", ta, "millions")],
            ("Asset Turnover", _at, "x")),
        "inventory_turnover": _m("Inventory Turnover", "COGS / Inventory", [
            ("COGS", cg, "millions"), ("Inventory", inv, "millions")],
            ("Inventory Turnover", _it, "x")),
        "dso": _m("Days Sales Outstanding", "Receivables / Revenue x 365", [
            ("Receivables", recv, "millions"), ("Revenue", rv, "millions")],
            ("DSO", _dso, "raw")),
        "dpo": _m("Days Payable Outstanding", "Accounts Payable / COGS x 365", [
            ("Accounts Payable", ap_, "millions"), ("COGS", cg, "millions")],
            ("DPO", _dpo, "raw")),
        "cash_conversion_cycle": _m("Cash Conversion Cycle", "DSO + DIO - DPO", [
            ("DSO", _dso, "raw"), ("DIO", _dio, "raw"), ("DPO", _dpo, "raw")],
            ("CCC", _ccc, "raw"), "Lower is better — measures days to convert inventory to cash"),
    }
    prof = {"roe":_roe,"roa":_roa,"roic":round(p["roic"],4),"asset_turnover":_at,
        "inventory_turnover":_it,"dso":_dso,"dpo":_dpo,"cash_conversion_cycle":_ccc,
        "_math":prof_math}
    # ── Section summaries ──────────────────────────────────────
    _fmt_m = lambda v: f"${v/1e6:.0f}M" if abs(v) < 1e9 else f"${v/1e9:.1f}B"
    snap_math = {
        "enterprise_value": _m("Enterprise Value", "Market Cap + Net Debt", [
            ("Market Cap", round(mc, 0), "millions"), ("Net Debt", round(nd, 0), "millions")],
            ("Enterprise Value", round(ev, 0), "millions")),
        "shares_outstanding": _m("Shares Outstanding", "Market Cap / Share Price", [
            ("Market Cap", round(mc, 0), "millions"), ("Share Price", price, "usd")],
            ("Shares", round(sh, 0), "raw")),
    }
    snap = {"market_cap":round(mc,0),"enterprise_value":round(ev,0),
        "shares_outstanding":round(sh,0),
        "range_52w_low":round(price*random.uniform(.65,.85),2),
        "range_52w_high":round(price*random.uniform(1.10,1.40),2),
        "avg_volume_3m":round(sh*random.uniform(.008,.02),0),
        "dividend_yield":round(p["div"],4),"beta":round(p["beta"],2),
        "short_interest_pct":round(anc.short_interest or random.uniform(.02,.06),4),
        "rsi_14":round(anc.rsi or _default_rsi(act, conf), 1),
        "_math":snap_math}
    iv_prices = [iv["implied_price"] for iv in comps["implied_valuation"] if iv["implied_price"] > 0]
    iv_avg = sum(iv_prices)/len(iv_prices) if iv_prices else ca
    iv_gap = round((iv_avg/price - 1)*100, 1) if price else 0
    high_risk_ct = sum(1 for r in rsks if r["severity"] == "High")
    roic_spread = round((p["roic"] - p["wacc"])*10000)
    # Pre-compute new sections for summaries
    _ddm = _build_ddm(anc, sec, p, price) if (signal.get("category") == "dividend" or (anc.dividend_yield or p["div"] or 0) > 0.015) else None
    _ceo = _build_ceo_info(signal)
    _comp_sec = _build_compensation(signal)
    _roi = _build_roi(signal, anc, p, price)
    _scoring = _build_scoring_breakdown(signal)
    _capeff = _build_capital_efficiency(isd, cfd, p)
    summaries = {
        "thesis": f"{rat}. {up:+.1%} upside to ${bl:.2f}. EV/Rev {p['ev_rev']:.1f}x vs {comps['peer_median']['ev_revenue']:.1f}x peer median.",
        "snapshot": f"{_fmt_m(mc)} cap, {p['beta']:.2f} beta, {snap['short_interest_pct']:.1%} SI. 52w range ${snap['range_52w_low']:.0f}\u2013${snap['range_52w_high']:.0f}.",
        "income": f"Revenue {_fmt_m(i0['revenue'])} growing {i0.get('yoy_growth',0):.1%}. GM {i0['gross_margin']:.1%}, OM {i0['operating_margin']:.1%}. EPS ${i0['diluted_eps']:.2f}.",
        "balance": f"BV ${b0['book_value_per_share']:.2f}/sh. Current ratio {cap['current_ratio']:.1f}x. Net debt {_fmt_m(nd)} ({cap['net_debt_ebitda']:.1f}x EBITDA).",
        "capital": f"Net debt/EBITDA {cap['net_debt_ebitda']:.1f}x. Interest coverage {cap['interest_coverage']:.1f}x. D/E {cap['debt_to_equity']:.2f}x. Quick {cap['quick_ratio']:.1f}x.",
        "cashflow": f"FCF {_fmt_m(cfd[-1]['fcf'])} ({cfd[-1]['fcf_margin']:.1%} margin, {cfd[-1]['fcf_yield']:.1%} yield). CFO {_fmt_m(cfd[-1]['cfo'])}.",
        "dcf": f"Implied ${dcf['output']['implied_price']:.2f}/sh ({dcf['output']['upside_pct']:+.1%}). WACC {dcf['assumptions']['wacc']:.1%}, TGR {dcf['assumptions']['terminal_growth']:.1%}. Terminal = {dcf['output']['terminal_pct_of_total']:.0%} of EV.",
        "comps": f"EV/Rev {comps['subject']['ev_revenue']:.1f}x vs {comps['peer_median']['ev_revenue']:.1f}x peers ({comps['premium_discount']['ev_revenue']:+.1f}%). EV/EBITDA {comps['subject']['ev_ebitda']:.1f}x vs {comps['peer_median']['ev_ebitda']:.1f}x.",
        "impliedval": f"Comps imply ${iv_avg:.2f}/sh ({iv_gap:+.1f}% vs current). Average of {len(iv_prices)} methods.",
        "profitability": f"ROIC {prof['roic']:.1%} vs {p['wacc']:.1%} WACC ({roic_spread:+,}bps spread). ROE {prof['roe']:.1%}. DSO {prof['dso']:.0f} days, CCC {prof['cash_conversion_cycle']:.0f} days.",
        "catalysts": f"{len(cats)} catalyst{'s' if len(cats)!=1 else ''} identified. Nearest: {cats[0]['event'][:80] if cats else 'None'} ({cats[0]['date'] if cats else 'N/A'}).",
        "moat": f"{mo['rating']} moat. {mo['market_share']:.1%} share of {_fmt_m(mo['tam'])} TAM.",
        "risks": f"{len(rsks)} risk factors. {high_risk_ct} rated high severity. Primary: {rsks[0]['factor'][:60]}.",
        "viewchangers": f"Bull: revenue above {abs(p['rg'])*100+5:.0f}%, margin expansion. Bear: growth below {max(abs(p['rg'])*100-8,0):.0f}%, guidance cut.",
        "pricetarget": f"Blended ${bl:.2f}: DCF ${dp:.2f} (50%), comps ${ca:.2f} (30%), technical ${tt:.2f} (20%).",
        "verdict": f"{rat} at {conf:.0%} confidence. ${bl:.2f} target = {up:+.1%} upside.",
        "ddm": f"DDM intrinsic ${_ddm['output']['intrinsic_value']:.2f}, {_ddm['output']['upside']:.0%} upside using {_ddm['assumptions']['growth_rate_used']:.1%} growth." if _ddm else None,
        "ceo": _ceo["note"] if _ceo else None,
        "compensation": _comp_sec.get("alignment_note") if _comp_sec.get("has_data") else "Compensation data unavailable",
        "roi": f"Total ROI {_roi['total_roi_pct']:.1%} ({_roi['capital_gain_pct']:.1%} capital + {_roi['income_return_pct']:.1%} income). Risk-adjusted: {_roi['risk_adjusted_roi']:.1%}." if _roi.get("total_roi_pct") is not None else None,
        "scoring": f"Composite {_scoring['composite_total']:.3f}, rank #{_scoring['rank']}. Top driver: {_scoring['components'][0]['label']} ({_scoring['components'][0]['contribution']:.3f})." if _scoring.get("has_data") and _scoring.get("components") else None,
        "capeff": f"R&D intensity {_capeff['rd_intensity']:.1%}, CapEx intensity {_capeff['capex_intensity']:.1%}. R&D/CapEx {_capeff['rd_to_capex']:.1f}x.",
    }
    return {
        "header":{"ticker":signal["ticker"],"name":signal.get("short_name",""),"sector":sec,
            "industry":sec,"exchange":"NASDAQ","date":datetime.now().strftime("%Y-%m-%d"),
            "rating":rat,"price_target":round(dp,2),"current_price":price,"upside_pct":round(real_upside, 4) if real_upside is not None else (round(dp/price-1,4) if price else 0)},
        "thesis":{"text":f"{rat} with ${dp:.2f} price target ({dp/price-1:+.1%} upside). "
                         f"{signal.get('ml_insight','')} "
                         f"Trading at {p['ev_rev']:.1f}x EV/Revenue vs "
                         f"{comps['peer_median']['ev_revenue']:.1f}x peer median suggests "
                         f"{'meaningful discount to' if p['ev_rev'] < comps['peer_median']['ev_revenue'] else 'premium over'} comparable companies."},
        "snapshot":snap,
        "income_statement":isd, "balance_sheet":bsd,
        "capital_structure":cap,
        "cash_flow":cfd, "dcf":dcf, "comps":comps,
        "profitability":prof,
        "catalysts":cats, "moat":mo, "risks":rsks,
        "view_changers":{"bullish":[
                f"Revenue growth accelerates above {abs(p['rg'])*100+5:.0f}%",
                "Margin expansion exceeds consensus expectations",
                "Insider buying increases or major institutional accumulation"],
            "bearish":[f"Revenue growth decelerates below {max(abs(p['rg'])*100-8,0):.0f}%",
                "Key customer concentration risk materializes",
                "Management credibility deteriorates or guidance is cut"]},
        "price_target":{"dcf_weight":0.50,"dcf_value":round(dp,2),"comps_weight":0.30,
            "comps_value":round(ca,2),"technical_weight":0.20,"technical_value":round(tt,2),
            "blended":round(bl,2),
            "_math":{"blended":_m("Blended Price Target","DCF x 50% + Comps x 30% + Technical x 20%",[
                ("DCF Implied",round(dp,2),"usd"),("DCF Weight",0.50,"pct"),
                ("Comps Implied",round(ca,2),"usd"),("Comps Weight",0.30,"pct"),
                ("Technical",round(tt,2),"usd"),("Technical Weight",0.20,"pct")],
                ("Blended Target",round(bl,2),"usd"))}},
        "verdict":{"rating":rat,"price_target":round(bl,2),"confidence":round(conf,4),
            "summary":f"{rat} with {bl:.2f} price target ({up:+.1%} upside). {signal.get('ml_insight','')}"},
        "capital_efficiency": _capeff,
        "ddm": _ddm,
        "ceo_info": _ceo,
        "compensation": _comp_sec,
        "roi_analysis": _roi,
        "scoring_breakdown": _scoring,
        "section_summaries":summaries}
