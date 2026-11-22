"""
Market constellation: every liquid stock as a point of light, clustered by theme.

Where a theme sits means something: x = the theme ETF's RS rating (right = stronger), y = its RS change this week
(up = gaining). Each stock orbits its theme; size and brightness = RS rating; gold glow = in a SETUP; rings = an
episodic pivot, insider buying, Code 33. A side log lists today's events. Built by the RS Engine each morning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

F_SETUP, F_LEADER, F_EP, F_INSIDER, F_C33, F_SCAN, F_TT, F_A50, F_RUN = 1, 2, 4, 8, 16, 32, 64, 128, 256


def _n(x, d=0):
    try:
        return None if x is None or pd.isna(x) else round(float(x), d) if d else int(round(float(x)))
    except Exception:
        return None


def build(stk: pd.DataFrame, payload: dict, settings: dict, scans: dict | None = None) -> dict:
    themes = payload.get("themes") or []
    tnames = [t["theme"] for t in themes]
    tix = {t: i for i, t in enumerate(tnames)}
    X = payload.get("extras") or {}
    scans = scans or {}
    ep = {e["symbol"]: e for e in (scans.get("ep") or []) if (e.get("days_ago") or 0) <= 2}
    ins = {x["ticker"]: x for x in (X.get("insiders") or [])}
    c33 = {x["t"] for x in (X.get("code33") or [])}
    scan_rows = {r["symbol"]: r for r in (scans.get("rows") or []) if not r.get("biotech")}   # the final list, as on Scans
    leaders = {r["symbol"] for r in (payload.get("leaders") or [])}

    liq = (stk.price >= settings.get("min_price", 5)) & (stk.dollar_vol >= settings.get("min_dollar_volume", 1e7) / 2)
    S = stk[liq & stk.rs_rating.notna()]
    nodes = []
    for t, r in S.iterrows():
        th = r.theme if r.theme in tix else None
        if th is None:                                           # fall back to any of its themes that has an ETF here
            th = next((x for x in (r.all_themes or []) if x in tix), None)
        if th is None:
            continue
        f = 0
        f |= F_SETUP if bool(r.get("setup_all")) else 0
        f |= F_LEADER if t in leaders else 0
        f |= F_EP if t in ep else 0
        f |= F_INSIDER if t in ins else 0
        f |= F_C33 if t in c33 else 0
        f |= F_SCAN if t in scan_rows else 0
        f |= F_TT if bool(r.get("trend_template")) else 0
        f |= F_A50 if bool(r.get("above_50")) else 0
        f |= F_RUN if bool(r.get("runner")) else 0
        p = (scan_rows.get(t) or {}).get("persist") or {}
        nodes.append([t, tix[th], _n(r.rs_rating), _n(r.chg_1d, 1), f, _n(r.get("rs_chg_1w")),
                      _n(r.dollar_vol / 1e6), p.get("n") or 0, str(r.get("name") or "")[:40], r.get("exchange") or ""])

    T = [{"n": t["theme"], "etf": t["symbol"], "rs": t.get("rs_rating"), "d": t.get("rs_chg_1w") or 0,
          "style": t.get("style"), "top": bool(t.get("top"))} for t in themes]
    mco = {x["theme"]: x for x in (X.get("theme_mco") or [])}
    for t in T:
        m = mco.get(t["n"])
        if m:
            t["mco"], t["xu"], t["xd"] = m["mco"], m["cross_up"], m["cross_down"]

    # today's events, most important first
    ev = []
    rg = payload.get("regime") or {}
    env = payload.get("environment") or {}
    if rg.get("label"):
        ev.append({"k": "REGIME", "t": rg["label"], "tone": rg.get("tone", "warn")})
    if env.get("verdict"):
        ev.append({"k": "BREAKOUTS", "t": f"{env['verdict']} · {env['setups']} setups ({env['setup_rank']}th pct)"
                   + (f" · {env['hold_rate']:.0f}% holding" if env.get("hold_rate") is not None else ""), "tone": env.get("tone", "warn")})
    try:
        st = json.loads((ROOT / "docs" / "data" / "stretch.json").read_text())
        for sym, e in (st.get("index") or {}).items():
            n_os, n_ob = len(e.get("oversold", [])), len(e.get("overbought", []))
            if n_os >= 2:
                ev.append({"k": "STRETCH", "t": f"{sym} {n_os} oversold signals · {e['now']['ret5']:+.1f}% in 5 days", "tone": "bull", "sym": sym})
            elif n_ob >= 2:
                ev.append({"k": "STRETCH", "t": f"{sym} {n_ob} overbought signals", "tone": "bear", "sym": sym})
    except Exception:
        pass
    for s, e in sorted(ep.items(), key=lambda kv: -(kv[1].get("gap") or 0))[:12]:
        ev.append({"k": "EP", "t": f"{s} gap {e.get('gap', 0):+.0f}% · {e.get('rvol') or 0:.1f}× vol"
                   + (" · earnings" if e.get("earnings_gap") else ""), "sym": s, "tone": "bull"})
    for x in (X.get("insiders") or [])[:10]:
        usd = x["usd"]
        ev.append({"k": "INSIDER", "t": f"{x['ticker']} {x['insiders']} insiders · ${usd / 1e6:.1f}M" if usd >= 1e6 else
                   f"{x['ticker']} {x['insiders']} insiders · ${usd / 1e3:.0f}k", "sym": x["ticker"], "tone": "info"})
    for m in (X.get("theme_mco") or []):
        if m.get("cross_up"):
            ev.append({"k": "BREADTH", "t": f"{m['theme']} McClellan crossed above 0 ({m['mco']:+d})", "sym": m["etf"], "tone": "bull"})
        elif m.get("cross_down"):
            ev.append({"k": "BREADTH", "t": f"{m['theme']} McClellan crossed below 0 ({m['mco']:+d})", "sym": m["etf"], "tone": "bear"})
    for c in (X.get("clusters") or [])[:5]:
        ev.append({"k": "CLUSTER", "t": f"{c['industry']} · {c['n']} names ({', '.join(n['t'] for n in c['names'][:4])})", "tone": "info"})
    if c33:
        ev.append({"k": "CODE 33", "t": f"{len(c33)} names: " + ", ".join(sorted(c33)[:8]), "tone": "info"})

    counts = {"stocks": len(nodes), "setups": sum(1 for n in nodes if n[4] & F_SETUP), "runners": sum(1 for n in nodes if n[4] & F_RUN),
              "leaders": sum(1 for n in nodes if n[4] & F_LEADER), "ep": sum(1 for n in nodes if n[4] & F_EP),
              "insiders": sum(1 for n in nodes if n[4] & F_INSIDER), "c33": sum(1 for n in nodes if n[4] & F_C33),
              "scan": sum(1 for n in nodes if n[4] & F_SCAN)}
    return {"asof": payload.get("asof"), "themes": T, "nodes": nodes, "events": ev, "counts": counts,
            "regime": rg.get("label"), "tone": rg.get("tone")}


def write_page(data: dict, docs: Path):
    tpl = (ROOT / "templates" / "constellation.html").read_text()
    out = docs / "constellation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), allow_nan=False)
                                                .replace("</", "<\\/")))
