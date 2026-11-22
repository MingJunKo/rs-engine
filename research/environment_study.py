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
px = E.download_yahoo(["SPY", "QQQ", "IWM", "RSP", "^VIX", "^VIX3M", "^IRX", "^TNX", "HYG", "IEF", "DX-Y.NYB"],
                      period="16y", want=("Close",))["Close"]
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

# where the market is in its cycle
hi = spy.rolling(252, min_periods=120).max()
dd = (spy / hi - 1) * 100
C["SPY in a correction (≥ 10% below its high)"] = (dd <= -10).astype(float)
C["SPY within 3% of its 52-week high"] = (dd >= -3).astype(float)
C["Recovering: was ≥ 10% off the high in the last 3 months, now above the 50-day"] = (
    (dd.rolling(63).min() <= -10) & (spy > spy.rolling(50).mean())).astype(float)
C["VIX above 20"] = (vix >= 20).astype(float)
C["VIX above 25"] = (vix >= 25).astype(float)
# macro, from market prices (FRED blocks GitHub's servers)
irx, tnx = px.get("^IRX"), px.get("^TNX")
if irx is not None:
    ch = irx - irx.shift(126)
    C["Rates falling: 3-month T-bill down ≥ 0.25 in 6 months (Fed cutting)"] = (ch <= -0.25).astype(float).where(irx.notna())
    C["Rates rising: 3-month T-bill up ≥ 0.25 in 6 months (Fed hiking)"] = (ch >= 0.25).astype(float).where(irx.notna())
    C["Short rates near zero (T-bill < 1%)"] = (irx < 1).astype(float).where(irx.notna())
if tnx is not None:
    C["10-year yield falling (3 months)"] = ((tnx - tnx.shift(63)) < 0).astype(float).where(tnx.notna())
    if irx is not None:
        C["Yield curve not inverted (10-year > 3-month)"] = ((tnx - irx) > 0).astype(float).where(tnx.notna() & irx.notna())
if "HYG" in px and "IEF" in px:
    C["Credit improving (HYG/IEF up over 3 months)"] = ((px["HYG"] / px["IEF"]).pct_change(63) > 0).astype(float)
if "DX-Y.NYB" in px:
    C["US dollar falling (3 months)"] = (px["DX-Y.NYB"].pct_change(63) < 0).astype(float).where(px["DX-Y.NYB"].notna())

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
inv = R[R.lift < 1].copy()                                   # "NOT x" is the useful side
inv["condition"] = "NOT: " + inv.condition
inv["events_on"] = len(ev) - inv.events_on
inv["share_on"] = 100 - inv.share_on
inv[["win_on", "win_off"]] = inv[["win_off", "win_on"]].values
inv[["ret20_on", "ret20_off"]] = inv[["ret20_off", "ret20_on"]].values
inv["lift"] = (inv.win_on / inv.win_off).round(2)
inv["lift_ex2020"] = (1 / inv.lift_ex2020).round(2)
inv["years_on_better"] = inv.years_tested - inv.years_on_better
for k in [r_["condition"] for r_ in rows if r_["lift"] is not None and r_["lift"] < 1]:
    C["NOT: " + k] = 1 - C[k]
R = pd.concat([R[R.lift >= 1], inv]).sort_values("lift", ascending=False)

# ── checklist score: robust conditions only ─────────────────────────
robust = R[(R.lift >= 1.25) & (R.lift_ex2020 >= 1.15) & (R.years_tested >= 5) & (R.years_on_better >= 0.65 * R.years_tested)]
picked = [k for k in ["VIX above 20", "NOT: Rates rising: 3-month T-bill up ≥ 0.25 in 6 months (Fed hiking)",
                      "NOT: SPY within 3% of its 52-week high", "Setup count above its median",
                      "Small caps leading (IWM/SPY up over 3 months)"] if k in C]   # distinct factors, one each
score = sum(prev(C[k]) for k in picked) if picked else None          # NaN where any factor is unknown
by_score = []
if picked:
    sc = score.reindex(ev.index)
    for k in range(len(picked) + 1):
        g = ev[sc == k]
        if len(g) >= 200:
            by_score.append({"score": k, "of": len(picked), "events": int(len(g)), "win_pct": round(100 * g.winner.mean(), 2),
                             "ret20": round(float(g.ret_20.mean()), 2), "share_of_days": round(100 * float((score == k).mean()), 1)})
now = {k: (None if pd.isna(C[k].iloc[-1]) else bool(C[k].iloc[-1])) for k in C}
# leave-one-out: how much each factor adds on top of the others
loo = {}
if picked:
    for k in picked:
        sc2 = sum(prev(C[x]) for x in picked if x != k).reindex(ev.index)
        g = ev[(sc2 >= len(picked) - 2)]
        f_on = prev(C[k]).reindex(g.index) == 1
        loo[k] = {"with": round(100 * g[f_on].winner.mean(), 2), "without": round(100 * g[~f_on].winner.mean(), 2)}
    # per-year check of the composite: high score (≥ 3) vs low (≤ 1)
    scy = score.reindex(ev.index).values                       # aligned by position (dates repeat across events)
    yrs_ = ev.index.year.values
    py = []
    for y in sorted(set(yrs_)):
        m_ = yrs_ == y
        hi_, lo_ = ev[m_ & (scy >= 3)], ev[m_ & (scy <= 1)]
        if len(hi_) >= 100 and len(lo_) >= 100:
            py.append({"year": int(y), "hi": round(100 * hi_.winner.mean(), 2), "lo": round(100 * lo_.winner.mean(), 2)})
out = {"events": int(len(ev)), "winners": int(ev.winner.sum()), "base_win_pct": round(100 * ev.winner.mean(), 2),
       "period": f"{ev.index.min():%b %Y} – {ev.index.max():%b %Y}", "conditions": R.to_dict("records"),
       "checklist": picked, "by_score": by_score, "leave_one_out": loo if picked else {}, "by_year_score": py if picked else [], "today": now, "today_score": None if not picked or pd.isna(score.iloc[-1]) else int(score.iloc[-1]),
       "today_inputs": {"vix": round(float(vix.iloc[-1]), 1), "spy_off_high": round(float(dd.iloc[-1]), 1),
                        "tbill_6m_change": None if irx is None else round(float(irx.iloc[-1] - irx.iloc[-127]), 2)},
       "asof": str(idx[-1].date())}
(ROOT / "research" / "environment_study.json").write_text(json.dumps(out, indent=1, default=str))
print("RESULT_JSON_START"); print(json.dumps(out, indent=1, default=str)); print("RESULT_JSON_END")
