"""
Setup tracker: did the RS SETUPs (and weekly-scan names) actually work?

Two sources, measured the same way:
  • Live log   — every stock the engine flagged (RS-leader SETUP, Weekly scan final list) from the day
                 tracking started, appended to data/history/setup_log.csv. Exact signals, including the theme filter.
  • Backtest   — the same SETUP shape rebuilt from 2 years of prices (RS ≥ min, above the 50-day, ADR, liquidity,
                 near pivot, tight, 0–4 ATR above the 50-day). No top-theme filter (theme history isn't stored),
                 so it is a slightly wider net than the live list. Gives a year of results on day one.

How a trade is scored (the way you'd trade it):
  entry  = first break above the pivot (20-day high on the signal day) within 10 sessions, at max(open, pivot)
  stop   = lowest low of the 10 sessions before the signal (the tight area)
  exit   = stop if hit, otherwise the close 20 sessions after entry
  result in R (multiples of the risk entry − stop); MFE = best R reached. No breakout in 10 days = "no trigger".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "data" / "history" / "setup_log.csv"
HOLD, TRIGGER_WIN, GAP = 20, 10, 20          # sessions held · days allowed to trigger · min days between signals


def score(i, o, h, l, c):
    """Score one signal at bar i on numpy arrays of one stock."""
    n = len(c)
    piv = np.nanmax(h[max(0, i - 19):i + 1])
    stop = np.nanmin(l[max(0, i - 9):i + 1])
    out = {"pivot": piv, "stop": stop, "status": "waiting", "R": None, "mfe": None, "entry": None, "days": None}
    for j in range(i + 1, min(n, i + 1 + TRIGGER_WIN)):
        if h[j] > piv:
            entry = max(o[j], piv) if np.isfinite(o[j]) else piv
            risk = entry - stop
            if not (risk > 0):
                out["status"] = "bad risk"; return out
            out["entry"] = entry
            hi, k_end = entry, min(n - 1, j + HOLD)
            for k in range(j, k_end + 1):
                if l[k] <= stop:                                       # stop hit (gap through it → the open)
                    ex = stop if k == j else min(o[k], stop)
                    out.update(status="stopped", R=(ex - entry) / risk, mfe=(max(hi, h[k]) - entry) / risk, days=k - j)
                    return out
                hi = max(hi, h[k])
            done = j + HOLD <= n - 1
            out.update(status="closed" if done else "running", R=(c[k_end] - entry) / risk,
                       mfe=(hi - entry) / risk, days=k_end - j)
            return out
    if i + TRIGGER_WIN <= n - 1:
        out["status"] = "no trigger"
    return out


RULES = {
    "fixed20": "Fixed: stop or close after 20 days",
    "q10": "Sell ⅓ on day 3 · stop to breakeven · trail the rest on the 10-day",
    "p2r20": "Sell ⅓ at +2R · stop to breakeven · trail the rest on the 20-day",
}
MAX_HOLD = 120


def manage(i, o, h, l, c, sma, rule):
    """Your selling rules. Same entry and initial stop as score(); exits at the close when price closes below the
    trailing average. Returns total R (weighted for the partial sale), status, exit bar."""
    n = len(c)
    piv = np.nanmax(h[max(0, i - 19):i + 1]); stop = np.nanmin(l[max(0, i - 9):i + 1])
    for j in range(i + 1, min(n, i + 1 + TRIGGER_WIN)):
        if h[j] <= piv:
            continue
        entry = max(o[j], piv) if np.isfinite(o[j]) else piv
        risk = entry - stop
        if not (risk > 0):
            return None
        part, banked, st = 0.0, 0.0, stop                       # part = fraction already sold, banked = R from it
        for k in range(j, min(n, j + MAX_HOLD)):
            if l[k] <= st:                                      # stop (initial or breakeven)
                ex = st if k == j else min(o[k], st)
                return {"R": banked + (1 - part) * (ex - entry) / risk, "status": "stopped" if part == 0 else "closed", "exit": k}
            if part == 0:
                if rule == "q10" and k == j + 3 and c[k] > entry:
                    part, banked, st = 1 / 3, (c[k] - entry) / risk / 3, max(st, entry)
                elif rule == "p2r20" and h[k] >= entry + 2 * risk:
                    part, banked, st = 1 / 3, 2 / 3, max(st, entry)
            if part > 0 and k > j + 3 and c[k] < sma[k]:       # trail: close below the average
                return {"R": banked + (1 - part) * (c[k] - entry) / risk, "status": "closed", "exit": k}
        k = min(n, j + MAX_HOLD) - 1
        done = j + MAX_HOLD <= n
        return {"R": banked + (1 - part) * (c[k] - entry) / risk, "status": "closed" if done else "running", "exit": k}
    return None


LOSS_TESTS = [
    ("Current: stop at 10-day low", {}),
    ("Stop under the prior day's low", {"stop": "prev"}),
    ("Stop 1 ATR below entry", {"stop": "atr"}),
    ("Prior-day-low stop + don't chase", {"stop": "prev", "nochase": 0.02}),
    ("Exit if it closes back below the pivot (days 0–2)", {"fail": True}),
    ("Time stop: out on day 5 if not above entry", {"time": 5}),
    ("Prior-day-low stop + time stop day 5", {"stop": "prev", "time": 5}),
    ("Don't chase: skip if it opens > 2% over the pivot", {"nochase": 0.02}),
    ("Pivot fail + time stop + no chase", {"fail": True, "time": 5, "nochase": 0.02}),

]


def manage_v(i, o, h, l, c, sma, a, opt):
    """Day-3 rule (⅓ on day 3, breakeven, 10-day trail) with loss-control variants."""
    n = len(c)
    piv = np.nanmax(h[max(0, i - 19):i + 1]); base_stop = np.nanmin(l[max(0, i - 9):i + 1])
    for j in range(i + 1, min(n, i + 1 + TRIGGER_WIN)):
        if h[j] <= piv:
            continue
        if opt.get("nochase") and o[j] > piv * (1 + opt["nochase"]):
            return None                                           # gapped too far over the pivot: skip
        entry = max(o[j], piv) if np.isfinite(o[j]) else piv
        # stops known at the moment of entry only (no peeking at the breakout day's own low)
        stop = {"prev": l[j - 1], "atr": entry - a[i]}.get(opt.get("stop"), base_stop)
        stop = max(stop, base_stop) if np.isfinite(stop) else base_stop
        risk = entry - stop
        if not (risk > 0):
            return None
        if opt.get("maxrisk") and (risk / entry > 0.08 or risk > 1.5 * a[i]):
            return None
        part, banked, st = 0.0, 0.0, stop
        for k in range(j, min(n, j + MAX_HOLD)):
            if l[k] <= st:                                    # on the entry day this assumes the low came after entry
                ex = st if k == j else min(o[k], st)
                return {"R": banked + (1 - part) * (ex - entry) / risk, "status": "stopped" if part == 0 else "closed",
                        "exit": k, "risk_pct": risk / entry * 100}
            if part == 0 and opt.get("fail") and k <= j + 2 and c[k] < piv:
                return {"R": (c[k] - entry) / risk, "status": "stopped", "exit": k, "risk_pct": risk / entry * 100}
            if part == 0 and opt.get("time") and k == j + opt["time"] and c[k] <= entry:
                return {"R": (c[k] - entry) / risk, "status": "stopped", "exit": k, "risk_pct": risk / entry * 100}
            if part == 0 and k == j + 3 and c[k] > entry:
                part, banked, st = 1 / 3, (c[k] - entry) / risk / 3, max(st, entry)
            if part > 0 and k > j + 3 and c[k] < sma[k]:
                return {"R": banked + (1 - part) * (c[k] - entry) / risk, "status": "closed", "exit": k,
                        "risk_pct": risk / entry * 100}
        k = min(n, j + MAX_HOLD) - 1
        return {"R": banked + (1 - part) * (c[k] - entry) / risk, "status": "closed" if j + MAX_HOLD <= n else "running",
                "exit": k, "risk_pct": risk / entry * 100}
    return None


def curve(trades, idx, cap=None):
    """Cumulative R by exit date. cap = max new trades per signal day (highest RS first) — closer to a real book."""
    if cap:
        by_day = {}
        for t in trades:
            by_day.setdefault(t["i"], []).append(t)
        trades = [t for d in by_day.values() for t in sorted(d, key=lambda x: -(x["rs"] or 0))[:cap]]
    done = sorted([t for t in trades if t["status"] != "running"], key=lambda t: t["exit"])
    pts, tot = {}, 0.0
    for t in done:
        tot += t["R"]
        pts[idx[t["exit"]].strftime("%Y-%m-%d")] = round(tot, 2)
    R = np.array([t["R"] for t in done]) if done else np.array([])
    months = {}
    for t in done:
        m = idx[t["exit"]].strftime("%Y-%m")
        months[m] = round(months.get(m, 0) + t["R"], 2)
    return {"points": list(pts.items()), "total": round(tot, 1), "trades": len(done),
            "open": sum(t["status"] == "running" for t in trades),
            "open_R": round(sum(t["R"] for t in trades if t["status"] == "running"), 1),
            "avg": round(float(R.mean()), 2) if len(R) else None,
            "win": round(100 * float((R > 0).mean())) if len(R) else None,
            "avg_win": round(float(R[R > 0].mean()), 2) if (R > 0).any() else None,
            "avg_loss": round(float(R[R <= 0].mean()), 2) if (R <= 0).any() else None,
            "best": round(float(R.max()), 1) if len(R) else None,
            "big": int((R >= 5).sum()), "big_R": round(float(R[R >= 5].sum()), 1),
            "months": sorted(months.items())}


GATES = [
    ("Baseline: every RS setup", lambda t: True),
    ("Market: RSP above its 50-day", lambda t: t["mkt_up"]),
    ("Market: ≥ 4 of 7 gauges risk-on", lambda t: (t["riskon"] or 0) >= 4),
    ("Market: breadth > 50%", lambda t: (t["breadth"] or 0) > 50),
    ("Theme: in a top-8 theme", lambda t: t["top"] is True),
    ("Stock: ADR ≥ 4%", lambda t: t["adr"] >= 4),
    ("Stock: RS ≥ 90", lambda t: (t["rs"] or 0) >= 90),
    ("Top theme + risk-on", lambda t: t["top"] is True and (t["riskon"] or 0) >= 4),
    ("Top theme + ADR ≥ 4%", lambda t: t["top"] is True and t["adr"] >= 4),
    ("Top theme + ADR ≥ 4% + risk-on", lambda t: t["top"] is True and t["adr"] >= 4 and (t["riskon"] or 0) >= 4),
    ("Top theme + ADR ≥ 4% + RSP > 50-day", lambda t: t["top"] is True and t["adr"] >= 4 and t["mkt_up"]),
]


def gate_test(trades, idx, ok=True):
    """Apply each gate, then take at most 3 new trades a day (highest RS), and score with the R curve."""
    out = []
    years = max(0.5, (idx[-1] - idx[max(0, len(idx) - 400)]).days / 365.25)
    for name, f in GATES:
        sub = [t for t in trades if f(t)]
        c = curve(sub, idx, cap=3)
        tot = c["total"] + c["open_R"]
        # max drawdown of the closed-trade R curve
        dd, peak = 0.0, 0.0
        for _, v in c["points"]:
            peak = max(peak, v); dd = min(dd, v - peak)
        n = c["trades"] + c["open"]
        out.append({"gate": name, "trades": n, "per_year": round(n / years), "R": round(tot, 1),
                    "R_trade": round(tot / n, 3) if n else None, "win": c["win"], "big": c["big"],
                    "dd": round(dd, 1), "R_year": round(tot / years, 1)})
    return out


def stats(rows):
    t = [r for r in rows if r["status"] in ("stopped", "closed")]
    trig = [r for r in rows if r["status"] in ("stopped", "closed", "running")]
    done_or_failed = [r for r in rows if r["status"] in ("stopped", "closed", "no trigger")]
    if not t:
        return {"n": len(rows), "trades": 0}
    R = np.array([r["R"] for r in t])
    return {"n": len(rows), "trades": len(t), "running": sum(r["status"] == "running" for r in rows),
            "trigger_rate": round(100 * len(trig) / max(1, len(done_or_failed) + sum(r["status"] == "running" for r in rows))),
            "win_rate": round(100 * float((R > 0).mean())), "avg_R": round(float(R.mean()), 2),
            "avg_win": round(float(R[R > 0].mean()), 2) if (R > 0).any() else None,
            "avg_loss": round(float(R[R <= 0].mean()), 2) if (R <= 0).any() else None,
            "hit_2R": round(100 * float(np.mean([r["mfe"] >= 2 for r in t]))),
            "stopped": round(100 * float(np.mean([r["status"] == "stopped" for r in t]))),
            "ret20_avg": round(float(np.nanmean([r["ret20"] for r in rows if r.get("ret20") is not None])), 1)
            if any(r.get("ret20") is not None for r in rows) else None}


def run_tracker(ctx: dict) -> dict:
    S = ctx["S"]
    o, h, l, c, v = (ctx[k][S] for k in ("open", "high", "low", "close", "vol"))
    rating, bench, atr = ctx["rating"][S], ctx["bench"], ctx["atr"]
    st = ctx["settings"]
    idx = c.index
    asof = idx[-1]

    # ── backtest: rebuild the SETUP flag for every day ─────────────
    sma50 = c.rolling(50).mean()
    a = atr(h, l, c)
    adr = (h / l - 1).rolling(20).mean() * 100
    pivot_dist = (c / h.rolling(20).max() - 1) * 100
    tight = ((h.rolling(10).max() - l.rolling(10).min()) / c * 100) / adr
    ext = (c - sma50) / a
    dv = (c * v).rolling(20).mean()
    sig = ((pivot_dist >= -6) & (tight <= 2.5) & (ext >= 0) & (ext <= 4) & (rating >= st["min_rs_rating"]) & (c > sma50)
           & (adr >= st.get("min_adr", 2.5)) & (c >= st["min_price"]) & (dv >= st["min_dollar_volume"]))
    start = max(130, len(idx) - 400)                      # ~1.5 years once the 50-day and RS have warmed up
    S10, S20 = c.rolling(10).mean().values, c.rolling(20).mean().values
    Av = atr(h, l, c).values
    lossv = {name: [] for name, _ in LOSS_TESTS}
    RO = ctx["riskon"].reindex(idx).values if ctx.get("riskon") is not None else None
    BR = ctx["breadth"].reindex(idx).values if ctx.get("breadth") is not None else None
    IT = ctx["in_top"].reindex(idx) if ctx.get("in_top") is not None else None
    ADRv = ((h / l - 1).rolling(20).mean() * 100).values
    BUP = (bench > bench.rolling(50).mean()).reindex(idx).values

    def attrs(i, jx, t):
        it = IT[t].iat[i] if IT is not None and t in IT.columns else None
        return {"adr": float(ADRv[i, jx]), "mkt_up": bool(BUP[i]), "riskon": None if RO is None else float(RO[i]),
                "breadth": None if BR is None else float(BR[i]), "top": None if it is None or pd.isna(it) else bool(it)}
    managed = {r: [] for r in RULES}
    bt = []
    O, H, L, C = o.values, h.values, l.values, c.values
    Bv = bench.reindex(idx).values
    b_up = (bench > bench.rolling(50).mean()).reindex(idx).values          # market proxy: RSP above its 50-day
    ADR = adr.values
    sg = sig.values
    for jx, t in enumerate(S):
        col = np.where(sg[start:, jx])[0] + start
        last = -999
        for i in col:
            if i - last < GAP:
                continue
            last = i
            r = score(i, O[:, jx], H[:, jx], L[:, jx], C[:, jx])
            for rule in ("q10", "p2r20"):
                m = manage(i, O[:, jx], H[:, jx], L[:, jx], C[:, jx], (S10 if rule == "q10" else S20)[:, jx], rule)
                if m:
                    managed[rule].append({"i": i, "rs": _i(rating.iat[i, jx]), "sym": t, **m, **attrs(i, jx, t)})
            at_ = attrs(i, jx, t)
            if at_["top"]:                                   # loss tests run on the recommended set: top-theme setups
                for name, opt in LOSS_TESTS:
                    m = manage_v(i, O[:, jx], H[:, jx], L[:, jx], C[:, jx], S10[:, jx], Av[:, jx], opt)
                    if m:
                        lossv[name].append({"i": i, "rs": _i(rating.iat[i, jx]), "sym": t, **m})
            if r["status"] in ("stopped", "closed", "running") and r["R"] is not None:
                managed["fixed20"].append({"i": i, "rs": _i(rating.iat[i, jx]), "sym": t, "R": r["R"],
                                           "status": "running" if r["status"] == "running" else "closed",
                                           "exit": min(len(idx) - 1, i + 1 + (r["days"] or 0)), **attrs(i, jx, t)})
            ret20 = (C[i + 20, jx] / C[i, jx] - 1) * 100 if i + 20 < len(idx) else None
            b20 = (Bv[i + 20] / Bv[i] - 1) * 100 if i + 20 < len(idx) else None
            bt.append({"symbol": t, "date": idx[i].strftime("%Y-%m-%d"), "rs": _i(rating.iat[i, jx]),
                       "tight": _f(tight.iat[i, jx]), "adr": _f(ADR[i, jx]), "mkt_up": 1 if b_up[i] else 0,
                       "ret20": _f(ret20), "bench20": _f(b20), **_clean(r)})
    # baseline: same stocks and filters, any shape (entered at the close) — does the SETUP shape add anything?
    base = ((rating >= st["min_rs_rating"]) & (c > sma50) & (adr >= st.get("min_adr", 2.5)) & (c >= st["min_price"])
            & (dv >= st["min_dollar_volume"])).values[start:-20]
    fwd = (c.shift(-20) / c - 1).values[start:-20] * 100
    base_ret20 = round(float(np.nanmean(np.where(base, fwd, np.nan))), 1) if base.any() else None
    bench_ret20 = round(float(np.nanmean((bench.shift(-20) / bench - 1).reindex(idx).values[start:-20] * 100)), 1)

    # ── live log ───────────────────────────────────────────────────
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=["date", "symbol", "source", "price", "rs", "theme",
                                                                      "lead_theme", "cs", "regime"])
    d = asof.strftime("%Y-%m-%d")
    new = []
    for src, items in (("RS setup", ctx["setups"]), ("Weekly scan", ctx["scan_rows"])):
        for it in items:
            t = it["symbol"]
            prev = log[(log.symbol == t) & (log.source == src)]
            if len(prev) and (asof - pd.Timestamp(prev.date.max())).days < 28:
                continue                                     # already logged this setup cluster
            if t not in c.columns:
                continue
            new.append({"date": d, "symbol": t, "source": src, "price": round(float(c[t].iloc[-1]), 2),
                        "rs": it.get("rs_rating"), "theme": it.get("theme"), "lead_theme": bool(it.get("lead_theme", True)),
                        "cs": it.get("cs"), "regime": ctx["regime"]})
    if new and not ctx.get("demo"):
        log = pd.concat([log[log.date != d] if False else log, pd.DataFrame(new)], ignore_index=True)
        log = log.drop_duplicates(["date", "symbol", "source"], keep="last")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(LOG, index=False)
    live = []
    pos = {t: k for k, t in enumerate(S)}
    for r in log.itertuples():
        if r.symbol not in pos or pd.Timestamp(r.date) not in idx:
            continue
        i, jx = idx.get_loc(pd.Timestamp(r.date)), pos[r.symbol]
        sc = score(i, O[:, jx], H[:, jx], L[:, jx], C[:, jx])
        mq = manage(i, O[:, jx], H[:, jx], L[:, jx], C[:, jx], S10[:, jx], "q10")
        now = C[-1, jx]
        live.append({"symbol": r.symbol, "date": r.date, "source": r.source, "price": r.price, "rs": _i(r.rs),
                     "theme": r.theme if isinstance(r.theme, str) else "", "cs": r.cs if isinstance(r.cs, str) else None,
                     "regime": r.regime, "since": _f((now / r.price - 1) * 100), "exchange": ctx["exchange"].get(r.symbol, ""),
                     "spark": [round(float(x), 2) for x in C[i:, jx][:45]], **_clean(sc),
                     "R_q10": _f(mq["R"], 2) if mq else None, "q10_status": mq["status"] if mq else None})
    live.sort(key=lambda x: x["date"], reverse=True)

    # ── breakdowns of the backtest ─────────────────────────────────
    def bucket(rows, key, edges, labels):
        out = []
        for lo, hi, lb in zip(edges[:-1], edges[1:], labels):
            sub = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
            out.append({"label": lb, **stats(sub)})
        return out
    by_rs = bucket(bt, "rs", [0, 90, 95, 101], ["RS 80–89", "RS 90–94", "RS 95–99"])
    by_tight = bucket(bt, "tight", [0, 1.5, 2.0, 2.6], ["Tight < 1.5 ADR", "1.5–2.0", "2.0–2.5"])
    by_adr = bucket(bt, "adr", [0, 4, 6, 100], ["ADR < 4%", "ADR 4–6%", "ADR ≥ 6%"])
    by_mkt = bucket(bt, "mkt_up", [0, 1, 2], ["RSP below 50-day", "RSP above 50-day"])
    months = sorted({r["date"][:7] for r in bt})
    by_month = [{"label": m, **stats([r for r in bt if r["date"][:7] == m])} for m in months]
    gates = {"q10": gate_test(managed["q10"], idx), "fixed20": gate_test(managed["fixed20"], idx)}
    loss_tests = []
    for name, _ in LOSS_TESTS:
        tr = lossv[name]
        cv = curve(tr, idx, cap=3)
        n_ = cv["trades"] + cv["open"]
        tot = cv["total"] + cv["open_R"]
        kept = {t["i"]: t for t in tr}
        loss_tests.append({"test": name, "trades": n_, "R": round(tot, 1), "R_trade": round(tot / n_, 3) if n_ else None,
                           "win": cv["win"], "avg_win": cv["avg_win"], "avg_loss": cv["avg_loss"], "big": cv["big"],
                           "risk_pct": round(float(np.median([t["risk_pct"] for t in tr])), 1) if tr else None})
    curves = {rule: {"label": RULES[rule], "all": curve(tr, idx), "cap3": curve(tr, idx, cap=3)}
              for rule, tr in managed.items()}
    top = sorted(managed["q10"], key=lambda t: -t["R"])[:12]
    curves["top_q10"] = [{"symbol": t["sym"], "date": idx[t["i"]].strftime("%Y-%m-%d"), "R": round(t["R"], 1),
                          "exit": idx[t["exit"]].strftime("%Y-%m-%d"), "status": t["status"]} for t in top]
    return {"asof": ctx["asof_label"], "generated": ctx["generated"], "curves": curves, "gates": gates, "loss_tests": loss_tests,
            "backtest": {"summary": stats(bt), "by_rs": by_rs, "by_tight": by_tight, "by_adr": by_adr, "by_mkt": by_mkt, "by_month": by_month,
                         "from": idx[start].strftime("%d %b %Y"), "base_ret20": base_ret20, "bench_ret20": bench_ret20,
                         "recent": sorted([r for r in bt if r["status"] in ("running", "waiting")], key=lambda r: r["date"], reverse=True)[:40]},
            "live": {"rows": live, "rs": stats([r for r in live if r["source"] == "RS setup"]),
                     "scan": stats([r for r in live if r["source"] == "Weekly scan"])}}


def _f(x, n=1):
    try:
        x = float(x)
        return None if not np.isfinite(x) else round(x, n)
    except (TypeError, ValueError):
        return None


def _i(x):
    x = _f(x, 0)
    return None if x is None else int(x)


def _clean(r):
    return {"status": r["status"], "R": _f(r["R"], 2), "mfe": _f(r["mfe"], 2), "entry": _f(r["entry"], 2),
            "stop": _f(r["stop"], 2), "pivot": _f(r["pivot"], 2), "days": r["days"]}


def write_page(payload, docs: Path, clean):
    tpl = (ROOT / "templates" / "tracker.html").read_text()
    css = (ROOT / "templates" / "dashboard.html").read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    data = json.dumps(clean(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (docs / "tracker").mkdir(parents=True, exist_ok=True)
    (docs / "tracker" / "index.html").write_text(tpl.replace("/*__SHARED_CSS__*/", css).replace("/*__DATA__*/null", data))
