"""Lump sum vs DCA vs 50-day MA vs σ-dip deployment, SPY & QQQ (dividend-adjusted). Cash earns the CPF OA 2.5%."""
import json, math, sys
import numpy as np, pandas as pd, yfinance as yf

CASH = 0.025
OUT = {}

def load(s):
    d = yf.download(s, period="max", auto_adjust=True, progress=False)
    return d["Close"].squeeze().dropna()

def z5(c):
    r = c.pct_change()
    return c.pct_change(5) / (r.rolling(60).std().shift(5) * math.sqrt(5))

def episodes(mask, gap=10):
    idx, last = [], None
    for i, m in enumerate(mask.values):
        if m and (last is None or i - last > gap):
            idx.append(i)
        if m:
            last = i
    return idx

def ma_cross_events(c):
    ma = c.rolling(50).mean()
    below = (c < ma)
    ev = below & ~below.shift(1, fill_value=True)          # first close below the 50-day after being above
    return [i for i, v in enumerate(ev.values) if v and not np.isnan(ma.iloc[i])]

def rarity(c, z):
    yrs = (c.index[-1] - c.index[60]).days / 365.25
    res = {}
    for th in (2.5, 3, 3.5, 4):
        ep = episodes(z <= -th)
        res[f"<=-{th}"] = {"episodes": len(ep), "per_year": round(len(ep) / yrs, 2), "every_years": round(yrs / max(len(ep), 1), 1),
                           "dates": [c.index[i].strftime("%b %Y") for i in ep][-8:]}
    return res

def fwd_stats(c, ev, hs=(20, 60, 250)):
    out = {}
    for h in hs:
        r = [c.iloc[i + h] / c.iloc[i] - 1 for i in ev if i + h < len(c)]
        out[h] = {"n": len(r), "avg": round(100 * np.mean(r), 1) if r else None, "up": round(100 * np.mean(np.array(r) > 0)) if r else None,
                  "worst": round(100 * min(r), 1) if r else None}
    return out

# ---------- A) lump sum of 400k, rolling start months, value after H years
def run_lump(c, events: dict, H=3):
    days = c.index
    month_starts = [i for i in range(1, len(c)) if days[i].month != days[i - 1].month]
    dcf = (1 + CASH) ** (1 / 252)
    res = {k: [] for k in ["Lump sum now", "DCA 12 months", "DCA 24 months", "Buy 50-day MA cross (25% each)",
                           "σ ≤ −3 dips (33% each)", "σ ≤ −4 dips (50% each)", "DCA 12m + σ≤−3 boost"]}
    deployed = {k: [] for k in res}
    for s in month_starts:
        e = s + 252 * H
        if e >= len(c) or s < 300:
            continue
        ms = [m for m in month_starts if s <= m < e]
        for name in res:
            cash, sh = 1.0, 0.0
            buys = {}
            if name == "Lump sum now":
                buys[s] = 1.0
            elif name.startswith("DCA 12 months"):
                for m in ms[:12]: buys[m] = 1 / 12
            elif name.startswith("DCA 24"):
                for m in ms[:24]: buys[m] = 1 / 24
            for i in range(s, e):
                cash *= dcf
                amt = 0.0
                if name in ("Lump sum now", "DCA 12 months", "DCA 24 months") and i in buys:
                    amt = min(cash, buys[i] * (1.0 * (1 + CASH) ** ((i - s) / 252)))
                elif name.startswith("Buy 50") and i in events["ma"]:
                    amt = cash * 0.25 if cash > 0.26 else cash
                elif name.startswith("σ ≤ −3") and i in events["z3"]:
                    amt = cash * 0.33 if cash > 0.34 else cash
                elif name.startswith("σ ≤ −4") and i in events["z4"]:
                    amt = cash * 0.5 if cash > 0.51 else cash
                elif name.startswith("DCA 12m +"):
                    if i in ms[:12]:
                        amt += min(cash, 1 / 12)
                    if i in events["z3"]:
                        amt += (cash - amt) / 3
                if amt > 0:
                    sh += amt / c.iloc[i]; cash -= amt
            res[name].append(sh * c.iloc[e] + cash)
            deployed[name].append(1 - cash / (1 + CASH) ** H)
    lump = np.array(res["Lump sum now"])
    out = {}
    for k, v in res.items():
        v = np.array(v)
        out[k] = {"median_multiple": round(float(np.median(v)), 3), "avg_multiple": round(float(np.mean(v)), 3),
                  "worst": round(float(v.min()), 3), "beats_lump_pct": round(100 * float(np.mean(v > lump))) if k != "Lump sum now" else None,
                  "avg_deployed_pct": round(100 * float(np.mean(deployed[k])))}
    out["_windows"] = len(lump)
    return out

# ---------- B) monthly contribution of 1 over the whole history; deploy rule decides when the cash is invested
def run_monthly(c, events: dict):
    days = c.index
    start = 300
    month_starts = set(i for i in range(start, len(c)) if days[i].month != days[i - 1].month)
    dcf = (1 + CASH) ** (1 / 252)
    out = {}
    for name in ["DCA monthly", "50-day MA cross (all cash)", "σ ≤ −3 dips (all cash)", "σ ≤ −4 dips (all cash)", "Never invest (CPF OA 2.5%)"]:
        cash = sh = contrib = 0.0; cash_days = 0
        for i in range(start, len(c)):
            cash *= dcf
            if i in month_starts:
                cash += 1; contrib += 1
            go = (name == "DCA monthly" and i in month_starts) or (name.startswith("50") and i in events["ma"]) or \
                 (name.startswith("σ ≤ −3") and i in events["z3"]) or (name.startswith("σ ≤ −4") and i in events["z4"])
            if go and cash > 0:
                sh += cash / c.iloc[i]; cash = 0
            cash_days += cash > 0.5
        final = sh * c.iloc[-1] + cash
        out[name] = {"final_per_100_contributed": round(100 * final / contrib, 1), "cash_left_pct": round(100 * cash / final, 1),
                     "contributed_months": int(contrib)}
    return out

for sym in ("SPY", "QQQ"):
    c = load(sym)
    z = z5(c)
    ev = {"ma": set(ma_cross_events(c)), "z3": set(episodes(z <= -3)), "z4": set(episodes(z <= -4))}
    OUT[sym] = {"since": c.index[0].strftime("%Y-%m-%d"), "last": c.index[-1].strftime("%Y-%m-%d"),
                "z5_now": round(float(z.iloc[-1]), 2), "rarity_5d": rarity(c, z),
                "events": {k: len(v) for k, v in ev.items()},
                "fwd_after_z3": fwd_stats(c, sorted(ev["z3"])), "fwd_after_z4": fwd_stats(c, sorted(ev["z4"])),
                "fwd_after_ma": fwd_stats(c, sorted(ev["ma"])),
                "fwd_all_days": fwd_stats(c, list(range(300, len(c)))),
                "lump_3y": run_lump(c, ev, 3), "lump_5y": run_lump(c, ev, 5), "monthly": run_monthly(c, ev),
                "z4_dates": [c.index[i].strftime("%Y-%m-%d") for i in sorted(ev["z4"])],
                "z3_dates": [c.index[i].strftime("%Y-%m-%d") for i in sorted(ev["z3"])]}
    print(sym, "done", file=sys.stderr)
json.dump(OUT, open("research/deploy_backtest.json", "w"), indent=1)
print(json.dumps(OUT, indent=1)[:200000])
