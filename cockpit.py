"""
Cockpit page (docs/cockpit/): one screen that pulls the engine together.

  1. The $100k book — top-theme setups, current rules — with its equity curve vs RSP
  2. The R lattice — every trade the book took, by exit date and result in R, plus the distribution
  3. The theme map — leading themes with their leaders orbiting, early turners coming in from the edge
  4. Market pulse — regime, gauges, breadth and what is on today's list

Everything comes from payloads the engine has already built; nothing is downloaded here.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _tail(xs, n):
    return list(xs[-n:]) if xs else []


def build(dash: dict, tracker: dict | None, scans: dict | None) -> dict:
    out = {"asof": dash.get("asof"), "generated": dash.get("generated"), "benchmark": dash.get("benchmark", "RSP")}

    # 1 · the book
    if tracker:
        pf = tracker["portfolio"]["top"]
        out["book"] = {k: pf.get(k) for k in ("start", "end", "ret", "cagr", "max_dd", "trades", "open", "win",
                                               "avg_R", "exposure", "from", "years")}
        out["book"]["curve"] = [[d, e] for d, e, *_ in pf.get("curve", [])]
        out["book"]["bench"] = tracker["portfolio"].get("bench", [])
        out["book"]["bench_ret"] = tracker["portfolio"].get("bench_ret")
        out["trades"] = pf.get("closed", [])           # [entry, exit, symbol, R]
        out["open"] = pf.get("open_pos", [])           # [entry, symbol, R so far]
        live = tracker.get("live", {}).get("rows", [])
        out["live"] = {"waiting": [r["symbol"] for r in live if r.get("status") == "waiting"][:12],
                       "running": [[r["symbol"], r.get("R")] for r in live if r.get("status") == "running"][:12]}

    # 10-year view, if the backtest has been run
    bt = ROOT / "data" / "backtest10.json"
    if bt.exists():
        try:
            b = json.loads(bt.read_text())
            t = b.get("top", {})
            out["decade"] = {"from": b.get("from"), "to": b.get("to"), "end": t.get("end"), "cagr": t.get("cagr"),
                             "dd": t.get("max_dd"), "avg_R": t.get("avg_R"), "win": t.get("win"),
                             "trades": t.get("trades"), "bench_cagr": b.get("bench_cagr"),
                             "closed": t.get("closed", []),
                             "curve": [[d, e] for d, e, *_ in t.get("curve", [])][::2],
                             "bench": (b.get("bench") or [])[::2]}
        except Exception:
            pass

    # 2 · theme map
    top = [t for t in dash.get("themes", []) if t.get("top")]
    out["themes"] = [{"theme": t["theme"], "etf": t["symbol"], "rs": t["rs_rating"], "chg": t.get("rs_chg_1w"),
                      "style": t.get("style")} for t in top]
    out["leaders"] = [{"s": l["symbol"], "n": l.get("name"), "t": l["theme"], "rs": l["rs_rating"],
                       "c": l.get("rs_chg_1w"), "m": l.get("chg_1m"), "setup": bool(l.get("setup")),
                       "pd": l.get("pivot_dist"), "x": l.get("exchange"), "new": bool(l.get("new"))}
                      for l in dash.get("leaders", [])]
    out["turners"] = [{"theme": g["theme"], "etf": g["etf"], "rs": g.get("rs"), "chg": g.get("rs_chg_1w"),
                       "breadth": g.get("breadth"), "q": g.get("quadrant"),
                       "stocks": [{"s": s["symbol"], "rs": s.get("rs_rating"), "x": s.get("exchange"),
                                   "setup": bool(s.get("setup"))} for s in g.get("stocks", [])[:5]]}
                      for g in dash.get("turners", [])]

    # 3 · pulse
    r = dash.get("regime", {})
    out["regime"] = {k: r.get(k) for k in ("label", "tone", "verdict", "risk_on", "gauges_n", "breadth", "breadth_chg")}
    out["gauges"] = [{"label": g["label"], "on": g.get("risk_on"), "chg": g.get("chg_1m"),
                      "s": _tail(g.get("series"), 63), "ma": _tail(g.get("ma"), 63)} for g in dash.get("gauges", [])]
    out["funnel"] = [{"label": f["label"], "n": f["n"]} for f in dash.get("funnel", [])]
    if scans:
        tops = set(scans.get("top_themes") or [])
        out["ep_today"] = [e["symbol"] for e in scans.get("ep", []) if e["days_ago"] == 0
                           and (e.get("vol_m") or 0) * (e.get("price") or 0) >= 20
                           and (e["neglected"] or (e.get("rs_rating") or 0) >= 80 or e.get("theme") in tops)][:8]
        out["parabolic"] = [p["symbol"] for p in scans.get("parabolic", [])][:6]
        out["weekly"] = len([x for x in scans.get("rows", []) if not x.get("biotech")])
    out["setups"] = [l["s"] for l in out["leaders"] if l["setup"]]
    return out


def write_page(data: dict, docs: Path, clean):
    tpl = (ROOT / "templates" / "cockpit.html").read_text()
    blob = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (docs / "cockpit").mkdir(parents=True, exist_ok=True)
    (docs / "cockpit" / "index.html").write_text(tpl.replace("/*__DATA__*/null", blob))
