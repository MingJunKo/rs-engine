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
    start = max(60, len(idx) - 260)
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
        now = C[-1, jx]
        live.append({"symbol": r.symbol, "date": r.date, "source": r.source, "price": r.price, "rs": _i(r.rs),
                     "theme": r.theme if isinstance(r.theme, str) else "", "cs": r.cs if isinstance(r.cs, str) else None,
                     "regime": r.regime, "since": _f((now / r.price - 1) * 100), "exchange": ctx["exchange"].get(r.symbol, ""),
                     "spark": [round(float(x), 2) for x in C[i:, jx][:45]], **_clean(sc)})
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
    return {"asof": ctx["asof_label"], "generated": ctx["generated"],
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
