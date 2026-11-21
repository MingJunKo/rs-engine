"""Monthly contributions; compare deployment rules vs plain DCA. SPY & QQQ, dividend-adjusted, cash earns 2.5% (CPF OA).
Execution: DCA / bear-month buys at the month's first open; 50-MA buys as a limit at the MA when the day's low touches it;
σ buys at the NEXT day's open (signal is known only after the close)."""
import json, math, sys
import numpy as np, pandas as pd, yfinance as yf

CASH = 0.025
DCF = (1 + CASH) ** (1 / 252)

def load(s):
    d = yf.download(s, period="max", auto_adjust=True, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d.dropna()

def episodes(mask, gap=10):
    out, last = np.zeros(len(mask), bool), None
    for i, m in enumerate(mask):
        if m and (last is None or i - last > gap):
            out[i] = True
        if m:
            last = i
    return out

def prep(d):
    c, o, lo = d.Close, d.Open, d.Low
    r = c.pct_change()
    z5 = c.pct_change(5) / (r.rolling(60).std().shift(5) * math.sqrt(5))
    z1 = r / r.rolling(20).std().shift(1)
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    ms = np.r_[False, d.index.month[1:] != d.index.month[:-1]]
    return dict(c=c.values, o=o.values, lo=lo.values, ma50=ma50.shift(1).values,          # yesterday's MA = today's limit level
                bull=(c > ma200).shift(1, fill_value=False).values,                        # regime known at yesterday's close
                bull2=(ma50 > ma200).shift(1, fill_value=False).values,
                z3=np.r_[False, episodes((z5 <= -3).values)[:-1]], z4=np.r_[False, episodes((z5 <= -4).values)[:-1]],
                z3any=np.r_[False, episodes(((z5 <= -3) | (z1 <= -3)).values)[:-1]],
                ms=ms, idx=d.index)

def sim(P, s, e, base="dca", reserve=0.0, cap=12, r3=0.5, r4=1.0, main_on_sig=False, sig="z3", regime="bull"):
    c, o, lo, ma = P["c"], P["o"], P["lo"], P["ma50"]
    cash = res = sh = 0.0; flows = []
    for i in range(s, e):
        cash *= DCF; res *= DCF
        if P["ms"][i]:
            flows.append(i)
            put_res = reserve if res < cap * reserve * 1.0 + 1e-9 else 0.0       # reserve stops growing at `cap` months' worth
            res += put_res; cash += 1 - put_res
        # main bucket
        if base == "dca":
            if P["ms"][i] and cash > 0:
                sh += cash / o[i]; cash = 0
        elif base == "mj":
            bull = P[regime][i]
            if not bull:
                if P["ms"][i] and cash > 0:
                    sh += cash / o[i]; cash = 0
            elif cash > 0 and not np.isnan(ma[i]) and lo[i] <= ma[i]:
                px = min(o[i], ma[i]); sh += cash / px; cash = 0
        # σ events (bought at today's open = day after the signal)
        hit4, hit3 = P["z4"][i], P[sig][i]
        if hit3 or hit4:
            if main_on_sig and cash > 0:
                sh += cash / o[i]; cash = 0
            if res > 0:
                amt = res * (r4 if hit4 else r3); sh += amt / o[i]; res -= amt
    final = sh * c[e - 1] + cash + res
    return final, flows

def xirr(P, flows, final, e):
    t = np.array([(P["idx"][f] - P["idx"][flows[0]]).days / 365.25 for f in flows]); T = (P["idx"][e - 1] - P["idx"][flows[0]]).days / 365.25
    lo_, hi_ = -0.5, 1.0
    for _ in range(80):
        m = (lo_ + hi_) / 2
        v = np.sum((1 + m) ** (T - t)) - final
        lo_, hi_ = (m, hi_) if v < 0 else (lo_, m)
    return m

STRATS = {
    "DCA monthly": dict(base="dca"),
    "MJ: 50MA in bull / monthly in bear": dict(base="mj"),
    "MJ, bull = 50MA>200MA": dict(base="mj", regime="bull2"),
    "MJ + σ≤−3 also releases waiting cash": dict(base="mj", main_on_sig=True),
    "MJ + 10% reserve (½ at σ≤−3, all at σ≤−4)": dict(base="mj", reserve=0.10),
    "MJ + 20% reserve (½ at σ≤−3, all at σ≤−4)": dict(base="mj", reserve=0.20),
    "MJ + 30% reserve (½ at σ≤−3, all at σ≤−4)": dict(base="mj", reserve=0.30),
    "MJ + 20% reserve, σ≤−4 only": dict(base="mj", reserve=0.20, r3=0.0),
    "MJ + 20% reserve, all at σ≤−3": dict(base="mj", reserve=0.20, r3=1.0),
    "MJ + 20% reserve, cap 24 months": dict(base="mj", reserve=0.20, cap=24),
    "MJ + σ release + 20% reserve": dict(base="mj", reserve=0.20, main_on_sig=True),
    "MJ + 20% reserve, daily or 5d σ≤−3": dict(base="mj", reserve=0.20, sig="z3any"),
    "DCA + 20% reserve (½ at σ≤−3, all at σ≤−4)": dict(base="dca", reserve=0.20),
}

OUT = {}
for sym in ("SPY", "QQQ"):
    P = prep(load(sym)); n = len(P["c"]); s0 = 260
    full = {}
    fd, ff = sim(P, s0, n, **STRATS["DCA monthly"])
    for k, kw in STRATS.items():
        f, fl = sim(P, s0, n, **kw)
        half = (s0 + n) // 2
        a, _ = sim(P, s0, half, **kw); b, _ = sim(P, half, n, **kw)
        a0, _ = sim(P, s0, half); b0, _ = sim(P, half, n)
        full[k] = {"vs_dca_pct": round(100 * (f / fd - 1), 1), "irr": round(100 * xirr(P, fl, f, n), 2),
                   "first_half_vs_dca": round(100 * (a / a0 - 1), 1), "second_half_vs_dca": round(100 * (b / b0 - 1), 1)}
    # rolling 10-year windows, monthly starts
    starts = [i for i in range(s0, n - 2520) if P["ms"][i]][::1]
    roll = {k: [] for k in STRATS}
    for s in starts:
        e = s + 2520
        d0, _ = sim(P, s, e)
        for k, kw in STRATS.items():
            if k == "DCA monthly":
                continue
            f, _ = sim(P, s, e, **kw); roll[k].append(f / d0 - 1)
    for k, v in roll.items():
        if v:
            v = np.array(v)
            full[k].update({"roll10_beat_pct": round(100 * float(np.mean(v > 0))), "roll10_median": round(100 * float(np.median(v)), 1),
                            "roll10_worst": round(100 * float(v.min()), 1), "roll10_best": round(100 * float(v.max()), 1)})
    OUT[sym] = {"since": str(P["idx"][s0].date()), "windows": len(starts), "results": full}
    print(sym, "done", file=sys.stderr)
print(json.dumps(OUT, indent=1))
