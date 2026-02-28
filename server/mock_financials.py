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
    "Utilities":    (.30,.10,.05, 2.0,.22,.06,.15,.03, 0.9),
}
_SD_KEYS = ("gm","om","nm","ev_rev","tax","dna","capex","div","beta")
_DFLT = (.40,.12,.07, 2.5,.22,.05,.05,.01, 1.0)
_SMAP = {"Energy":"energy","Technology":"technology","Consumer Discretionary":"consumer_discretionary",
    "Healthcare":"healthcare","Industrials":"industrials","Financials":"financials",
    "Materials":"materials","Consumer Staples":"consumer_staples","Utilities":"utilities",
    "Communication Services":"communication_services","Real Estate":"real_estate"}


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


def _dollar_val(m) -> float:
    return float(m.group(1)) * (1e9 if m.group(2).upper() == "B" else 1e6)


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
        rows.append({"year":f"FY{yr0+off}","revenue":round(rev,0),
            "yoy_growth":round(p["rg"] if off>-2 else p["rg"]*0.8,4),
            "cogs":round(cogs,0),"gross_profit":round(gp,0),"gross_margin":round(gm,4),
            "rd_expense":round(rd,0),"sga_expense":round(sga,0),
            "operating_income":round(op,0),"operating_margin":round(op/rev if rev else 0,4),
            "interest_expense":round(ie,0),"pretax_income":round(pt,0),
            "tax_expense":round(tx,0),"tax_rate":round(p["tax"],4),
            "net_income":round(ni,0),"net_margin":round(ni/rev if rev else 0,4),
            "diluted_eps":round(ni/sh if sh else 0,2),
            "ebitda":round(ebitda,0),"ebitda_margin":round(ebitda/rev if rev else 0,4),
            "sbc":round(sbc,0)})
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
        rows.append({"year":inc["year"],"cash":round(cash,0),"receivables":round(recv,0),
            "inventory":round(inv,0),"total_current_assets":round(tca,0),
            "ppe_net":round(ppe,0),"goodwill":round(gw,0),"total_assets":round(ta,0),
            "accounts_payable":round(ap,0),"short_term_debt":round(std,0),
            "total_current_liabilities":round(tcl,0),"long_term_debt":round(ltd,0),
            "total_liabilities":round(tl,0),"total_equity":round(eq,0),
            "book_value_per_share":round(eq/sh if sh else 0,2)})
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
        rows.append({"year":inc["year"],"net_income":round(ni,0),"dna":round(dna,0),
            "sbc":round(sbc,0),"working_capital_change":round(wc,0),"cfo":cfo,"capex":cap,
            "acquisitions":0,"cfi":cap,"debt_change":0,"buybacks":round(bb,0),
            "dividends":round(div,0),"cff":round(cff,0),"fcf":fcf,
            "fcf_margin":round(fcf/rev if rev else 0,4),
            "fcf_yield":round(fcf/p["mc"] if p["mc"] else 0,4),
            "fcf_per_share":round(fcf/sh if sh else 0,2)})
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
        projs.append({"year":yr0+i,"revenue":round(rv,0),"rev_growth":round(gf,4),
            "ebitda":round(eb,0),"ebitda_margin":round(em,4),"dna":round(dn,0),
            "ebit":round(ebit,0),"taxes":round(tx,0),"nopat":round(nopat,0),
            "capex":round(cx,0),"nwc_change":round(nwc,0),"ufcf":round(uf,0)})
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
        "debt_weight":round(dw,4),"equity_weight":round(ew,4),"wacc":round(cw,4)},
    "projected_ufcf":projs,
    "output":{"pv_fcfs":round(pv_f,0),"pv_terminal":round(pvt,0),
        "terminal_pct_of_total":round(pvt/iev if iev else 0,4),
        "implied_ev":round(iev,0),"net_debt":round(nd,0),"cash":round(cash,0),
        "implied_equity_value":round(ieq,0),"shares":round(sh,0),
        "implied_price":round(ip,2),"current_price":0,"upside_pct":0},
    "sensitivity":{"wacc_values":wv,"growth_values":gv,"matrix":np.round(ipm,2).tolist()}}


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
    dcf["output"]["current_price"] = price
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
    raw_risks = [s.strip() for s in re.split(r'[.]', signal.get("risk_context","")) if len(s.strip()) > 10]
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
    return {
        "header":{"ticker":signal["ticker"],"name":signal.get("short_name",""),"sector":sec,
            "industry":sec,"exchange":"NASDAQ","date":datetime.now().strftime("%Y-%m-%d"),
            "rating":rat,"price_target":round(bl,2),"current_price":price,"upside_pct":round(up,4)},
        "thesis":{"text":f"{rat} with ${bl:.2f} price target ({up:+.1%} upside). "
                         f"{signal.get('ml_insight','')} "
                         f"Trading at {p['ev_rev']:.1f}x EV/Revenue vs "
                         f"{comps['peer_median']['ev_revenue']:.1f}x peer median suggests "
                         f"{'meaningful discount to' if p['ev_rev'] < comps['peer_median']['ev_revenue'] else 'premium over'} comparable companies."},
        "snapshot":{"market_cap":round(mc,0),"enterprise_value":round(ev,0),
            "shares_outstanding":round(sh,0),
            "range_52w_low":round(price*random.uniform(.65,.85),2),
            "range_52w_high":round(price*random.uniform(1.10,1.40),2),
            "avg_volume_3m":round(sh*random.uniform(.008,.02),0),
            "dividend_yield":round(p["div"],4),"beta":round(p["beta"],2),
            "short_interest_pct":round(anc.short_interest or random.uniform(.02,.06),4)},
        "income_statement":isd, "balance_sheet":bsd,
        "capital_structure":{"net_debt":round(nd,0),
            "net_debt_ebitda":round(nd/i0["ebitda"] if i0["ebitda"] else 0,2),
            "debt_to_equity":round((b0["long_term_debt"]+b0["short_term_debt"])/te if te else 0,2),
            "interest_coverage":round(i0["operating_income"]/i0["interest_expense"] if i0["interest_expense"] else 0,2),
            "current_ratio":round(b0["total_current_assets"]/b0["total_current_liabilities"] if b0["total_current_liabilities"] else 0,2),
            "quick_ratio":round((b0["cash"]+b0["receivables"])/b0["total_current_liabilities"] if b0["total_current_liabilities"] else 0,2)},
        "cash_flow":cfd, "dcf":dcf, "comps":comps,
        "profitability":{"roe":round(i0["net_income"]/te if te else 0,4),
            "roa":round(i0["net_income"]/ta if ta else 0,4),"roic":round(p["roic"],4),
            "asset_turnover":round(rv/ta if ta else 0,4),
            "inventory_turnover":round(cg/inv if inv else 0,2),
            "dso":round(recv/rv*365 if rv else 0,1),
            "dpo":round(ap_/cg*365 if cg else 0,1),
            "cash_conversion_cycle":round((recv/rv*365+inv/cg*365-ap_/cg*365) if rv and cg else 0,1)},
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
            "blended":round(bl,2)},
        "verdict":{"rating":rat,"price_target":round(bl,2),"confidence":round(conf,4),
            "summary":f"{rat} with {bl:.2f} price target ({up:+.1%} upside). {signal.get('ml_insight','')}"}}
