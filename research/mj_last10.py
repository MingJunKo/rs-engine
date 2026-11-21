import json, sys
sys.path.insert(0, "research")
import numpy as np, pandas as pd
import beat_dca as B
V = {
 "MJ, bull = 50>200 cross (limit fill at MA, cash 2.5%)": dict(base="mj", regime="bull2"),
 "MJ, bull = 50>200, cash earns 0%": dict(base="mj", regime="bull2", cash_rate=0.0),
 "MJ, bull = 50>200, buy at that day's close": dict(base="mj", regime="bull2", fill="close"),
 "MJ, bull = price>200MA": dict(base="mj"),
 "DCA + pull forward 3m/6m at σ≤−3/−4": dict(base="dca", borrow3=3, borrow4=6),
 "MJ(50>200) + pull forward 3m/6m at σ": dict(base="mj", regime="bull2", borrow3=3, borrow4=6),
}
out = {}
for sym in ("SPY", "QQQ"):
    P = B.prep(B.load(sym)); n = len(P["c"]); idx = P["idx"]
    r = {}
    for yrs in (5, 10, 15, 20):
        s = int(np.searchsorted(idx, idx[-1] - pd.DateOffset(years=yrs)))
        d0, _ = B.sim(P, s, n); d00, _ = B.sim(P, s, n, cash_rate=0.0)
        r[f"last {yrs}y"] = {k: round(100 * (B.sim(P, s, n, **kw)[0] / (d00 if kw.get("cash_rate") == 0.0 else d0) - 1), 2) for k, kw in V.items()}
    # calendar-year start sensitivity for last-10y windows starting each month of 2014-2017
    sens = []
    for s in [i for i in range(n) if P["ms"][i] and pd.Timestamp("2014-01-01") <= idx[i] <= pd.Timestamp("2017-12-31")]:
        d0, _ = B.sim(P, s, n); f, _ = B.sim(P, s, n, base="mj", regime="bull2"); sens.append(round(100 * (f / d0 - 1), 2))
    r["MJ(50>200) vs DCA, start any month 2014-2017 → today"] = {"min": min(sens), "median": float(np.median(sens)), "max": max(sens)}
    # time in bull regime and how long cash waits
    bull = P["bull2"][-2520:]
    r["share of last 10y in bull (50>200)"] = round(100 * float(bull.mean()))
    out[sym] = r
print(json.dumps(out, indent=1))
