"""
Which market / macro environments make breakouts more likely to become 100%+ winners?

Events: every breakout in the model book (data/history/modelbook_events.csv.gz), complete ones only
(126-day outcome known). For each condition, as it stood the evening BEFORE the breakout (no look-ahead:
weekly values use the last completed week, monthly macro data is lagged a month), compare the share of
breakouts that became winners when the condition was ON vs OFF.

Robustness: per-year consistency (share of years where ON beat OFF), and the lift with 2020 removed.
Sources: Yahoo (SPY, QQQ, IWM, RSP, VIX, VIX3M), FRED public CSVs (no key), the engine's breadth and
breakout-environment history.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rs_engine as E   # noqa: E402

log = lambda *a: print(*a, flush=True)
ev = pd.read_csv(ROOT / "data" / "history" / "modelbook_events.csv.gz", parse_dates=["date"])
ev = ev[ev.complete == True].copy()
log(f"{len(ev):,} complete breakouts, {int(ev.winner.sum())} winners, {ev.date.min().date()} → {ev.date.max().date()}")

# ── market data ─────────────────────────────────────────────────────
px = E.download_yahoo(["SPY", "QQQ", "IWM", "RSP", "^VIX", "^VIX3M"], period="16y", want=("Close",))["Close"]
px = px.ffill()
idx = px.index


def weekly_macd_green(s: pd.Series) -> pd.Series:
    w = s.resample("W-FRI").last().dropna()
    m = w.ewm(span=12, adjust=False).mean() - w.ewm(span=26, adjust=False).mean()
    hist = m - m.ewm(span=9, adjust=False).mean()
    g = (hist > 0).astype(float)
    g.index = g.index + pd.Timedelta(days=3)                 # known from the following Monday
    return g.reindex(idx, method="ffill")


def above(s, n):
    return (s > s.rolling(n).mean()).astype(float)


spy, qqq, iwm, rsp, vix = px["SPY"], px["QQQ"], px["IWM"], px["RSP"], px["^VIX"]
C = {}
C["SPY weekly MACD green"] = weekly_macd_green(spy)
C["QQQ weekly MACD green"] = weekly_macd_green(qqq)
C["IWM weekly MACD green (small caps)"] = weekly_macd_green(iwm)
C["SPY above its 50-day"] = above(spy, 50)
C["SPY uptrend (above 50 & 50 > 200)"] = ((spy > spy.rolling(50).mean()) & (spy.rolling(50).mean() > spy.rolling(200).mean())).astype(float)
C["IWM above its 200-day"] = above(iwm, 200)
C["Small caps leading (IWM/SPY up over 3 months)"] = ((iwm / spy).pct_change(63) > 0).astype(float)
C["Equal-weight leading (RSP/SPY up over 3 months)"] = ((rsp / spy).pct_change(63) > 0).astype(float)
C["VIX below 20"] = (vix < 20).astype(float)
C["VIX below its 50-day average"] = (vix < vix.rolling(50).mean()).astype(float)
if "^VIX3M" in px:
    C["VIX term structure calm (VIX < VIX3M)"] = (px["^VIX"] < px["^VIX3M"]).astype(float).where(px["^VIX3M"].notna())

# engine breadth + breakout environment
try:
    b = pd.read_csv(ROOT / "data" / "history" / "breadth_long.csv", index_col=0, parse_dates=True).reindex(idx, method="ffill")
    C["Breadth: >50% of stocks above their 40-day (T2108)"] = (b.t2108 > 50).astype(float).where(b.t2108.notna())
    C["Breadth: McClellan oscillator above 0"] = (b.mco > 0).astype(float).where(b.mco.notna())
    C["Breadth: more 4% up-days than down-days (5-day)"] = (b.ratio5 > 1).astype(float).where(b.ratio5.notna())
except Exception as e:
    log("breadth skipped:", e)
try:
    en = pd.read_csv(ROOT / "data" / "history" / "environment.csv", index_col=0, parse_dates=True).reindex(idx, method="ffill")
    C["Recent breakouts holding above typical"] = (en.hold_rate > en.hold_rate.median()).astype(float).where(en.hold_rate.notna())
    C["Setup count above its median"] = (en.setup_pct > en.setup_pct.median()).astype(float).where(en.setup_pct.notna())
except Exception as e:
    log("environment history skipped:", e)

# ── macro (FRED) ────────────────────────────────────────────────────
def fred(sid, lag_days=0):
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["date", "v"]
        s = pd.Series(pd.to_numeric(d.v, errors="coerce").values, index=pd.to_datetime(d.date)).dropna()
        s.index = s.index + pd.Timedelta(days=lag_days)       # publication lag: when you'd actually know it
        log(f"  FRED {sid}: {len(s)} obs to {s.index[-1].date()}")
        return s.reindex(s.index.union(idx)).ffill().reindex(idx)
    except Exception as e:
        log(f"  FRED {sid} failed: {e}")
        return None


ff = fred("DFF", 1)
if ff is not None:
    ch = ff - ff.shift(126)
    C["Fed cutting (funds rate down ≥ 0.25 in 6 months)"] = (ch <= -0.25).astype(float)
    C["Fed hiking (funds rate up ≥ 0.25 in 6 months)"] = (ch >= 0.25).astype(float)
    C["Fed on hold (6-month change under 0.25)"] = (ch.abs() < 0.25).astype(float)
    C["Fed funds rate below 2%"] = (ff < 2).astype(float)
y10 = fred("DGS10", 1)
if y10 is not None:
    C["10-year yield falling (3 months)"] = ((y10 - y10.shift(63)) < 0).astype(float)
crv = fred("T10Y2Y", 1)
if crv is not None:
    C["Yield curve not inverted (10y > 2y)"] = (crv > 0).astype(float)
hy = fred("BAMLH0A0HYM2", 1)
if hy is not None:
    C["Credit spreads tightening (high-yield, 3 months)"] = ((hy - hy.shift(63)) < 0).astype(float)
    C["Credit spreads below their 1-year median"] = (hy < hy.rolling(252).median()).astype(float)
bs = fred("WALCL", 7)
if bs is not None:
    C["Fed balance sheet growing (13 weeks)"] = ((bs - bs.shift(63)) > 0).astype(float)
m2 = fred("M2SL", 45)
if m2 is not None:
    C["M2 money supply growing > 5% a year"] = ((m2 / m2.shift(252) - 1) * 100 > 5).astype(float)
nf = fred("NFCI", 7)
if nf is not None:
    C["Financial conditions loose (Chicago Fed NFCI < 0)"] = (nf < 0).astype(float)
    C["Financial conditions easing (NFCI falling, 3 months)"] = ((nf - nf.shift(63)) < 0).astype(float)
usd = fred("DTWEXBGS", 1)
if usd is not None:
    C["US dollar falling (3 months)"] = ((usd - usd.shift(63)) < 0).astype(float)
cpi = fred("CPIAUCSL", 45)
if cpi is not None:
    yoy = (cpi / cpi.shift(252) - 1) * 100
    C["Inflation falling (CPI YoY lower than 3 months ago)"] = (yoy < yoy.shift(63)).astype(float)

# ── score every condition ───────────────────────────────────────────
prev = lambda s: s.shift(1)                                  # as known the evening before the breakout
ev = ev.set_index("date")
rows = []
for name, s in C.items():
    on = prev(s).reindex(ev.index)
    m = on.notna()
    e = ev[m].assign(on=on[m].astype(bool).values)
    if e.on.sum() < 300 or (~e.on).sum() < 300:
        continue
    w_on, w_off = e[e.on].winner.mean(), e[~e.on].winner.mean()
    yrs = []
    for y, g in e.groupby(e.index.year):
        a_, b_ = g[g.on], g[~g.on]
        if len(a_) >= 100 and len(b_) >= 100:
            yrs.append(a_.winner.mean() > b_.winner.mean())
    x = e[e.index.year != 2020]
    lift_ex = (x[x.on].winner.mean() / x[~x.on].winner.mean()) if x[~x.on].winner.mean() > 0 else None
    rows.append({"condition": name, "events_on": int(e.on.sum()), "share_on": round(100 * e.on.mean()),
                 "win_on": round(100 * w_on, 2), "win_off": round(100 * w_off, 2),
                 "lift": round(w_on / w_off, 2) if w_off > 0 else None,
                 "lift_ex2020": None if lift_ex is None else round(lift_ex, 2),
                 "ret20_on": round(float(e[e.on].ret_20.mean()), 2), "ret20_off": round(float(e[~e.on].ret_20.mean()), 2),
                 "years_tested": len(yrs), "years_on_better": int(sum(yrs)),
                 "since": str(e.index.min().date())})
R = pd.DataFrame(rows).sort_values("lift", ascending=False)

# ── checklist score: robust conditions only ─────────────────────────
robust = R[(R.lift >= 1.25) & (R.lift_ex2020 >= 1.15) & (R.years_tested >= 5) & (R.years_on_better >= 0.65 * R.years_tested)]
picked = robust.condition.tolist()
score = sum(prev(C[k]).fillna(0) for k in picked) if picked else None
by_score = []
if picked:
    sc = score.reindex(ev.index)
    for k in range(len(picked) + 1):
        g = ev[sc == k]
        if len(g) >= 200:
            by_score.append({"score": k, "of": len(picked), "events": int(len(g)), "win_pct": round(100 * g.winner.mean(), 2),
                             "ret20": round(float(g.ret_20.mean()), 2), "share_of_days": round(100 * float((score == k).mean()), 1)})
now = {k: (None if pd.isna(C[k].iloc[-1]) else bool(C[k].iloc[-1])) for k in C}
out = {"events": int(len(ev)), "winners": int(ev.winner.sum()), "base_win_pct": round(100 * ev.winner.mean(), 2),
       "period": f"{ev.index.min():%b %Y} – {ev.index.max():%b %Y}", "conditions": R.to_dict("records"),
       "checklist": picked, "by_score": by_score, "today": now, "today_score": None if not picked else int(score.iloc[-1]),
       "asof": str(idx[-1].date())}
(ROOT / "research" / "environment_study.json").write_text(json.dumps(out, indent=1, default=str))
print("RESULT_JSON_START"); print(json.dumps(out, indent=1, default=str)); print("RESULT_JSON_END")
