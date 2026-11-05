"""
MJ's weekly screening process (after Jeff Sun), rebuilt on the RS Engine's data, plus Episodic Pivot and
Parabolic Short scans. Called from rs_engine.run() with the day's price frames; writes docs/scans/index.html.

Weekly process (mirrors the TradingView screens in "MJ Weekly Process of screening"):
  WL1  Daily Tightness Swing         screen 1
  WL2  Strong Move > $10B            screens 2, 3, 4, 5, 7
  WL3  Post-Earnings Continuation    screen 6 (any day in the last 5 sessions, not only today)
  WL4  Fundamentals                  screen 8
  WL5  Strong Move (< $10B)          screens 9, 10, 11, 12, 13
  WL12 Institutional transactions    screen 14 (the FinViz screen)
  → each watchlist through the two 5-EMA tightness screens (price within 5% below / above the 5-EMA)
  + the Weekly RS scan               → WL6 Weekly Scan Result → price > $3, biotech set aside → final list

TradingView fields are matched as closely as the free data allows; see FIELD NOTES at the bottom.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


# ════════════════════════════════════════════════════════════════════
#  per-stock fields (latest bar), all vectorised over the price frames
# ════════════════════════════════════════════════════════════════════
def fields(o, h, l, c, v, names: pd.DataFrame, F: dict, inst: pd.DataFrame, yoy) -> pd.DataFrame:
    T = list(c.columns)
    d = pd.DataFrame(index=T)
    last = c.iloc[-1]
    d["price"] = last
    d["vol"] = v.iloc[-1]
    d["avg_vol60"] = v.rolling(60, min_periods=40).mean().iloc[-1]
    d["avg_vol50"] = v.rolling(50, min_periods=40).mean().shift(1).iloc[-1]          # excludes today
    d["dvol20"] = (c * v).rolling(20).mean().iloc[-1]
    low52 = l.rolling(252, min_periods=120).min().iloc[-1]
    d["above_low"] = (last / low52 - 1) * 100
    ema5 = c.ewm(span=5, adjust=False).mean().iloc[-1]
    ema10 = c.ewm(span=10, adjust=False).mean()
    d["ema5_gap"] = (last / ema5 - 1) * 100                                            # + = price above the 5-EMA
    sma = {n: c.rolling(n).mean().iloc[-1] for n in (10, 20, 50)}
    d["sma10_gap"] = (last / sma[10] - 1) * 100
    d["sma10_gt_20"] = sma[10] > sma[20]
    d["ema10_gt_sma20"] = ema10.iloc[-1] > sma[20]
    d["above_sma20"] = last > sma[20]
    d["above_sma50"] = last > sma[50]
    for k, n in (("perf_1w", 5), ("perf_1m", 21), ("perf_3m", 63), ("perf_6m", 126)):
        d[k] = (last / c.iloc[-1 - n] - 1) * 100
    rng = (h / l - 1) * 100
    d["volat_1m"] = rng.rolling(21).mean().iloc[-1]                                     # TradingView "Volatility, 1M"
    d["volat_1w"] = rng.rolling(5).mean().iloc[-1]                                      # FinViz "Volatility week"
    d["adr"] = rng.rolling(20).mean().iloc[-1]
    d["above_low20"] = last >= 1.05 * l.rolling(20).min().iloc[-1]
    d["above_low50"] = last >= 1.05 * l.rolling(50).min().iloc[-1]
    gap = o / c.shift(1) - 1
    d["gap_today"] = gap.iloc[-1] * 100
    # post-earnings continuation: a gap > 5% on ≥ 2× the 60-day volume within the last 5 sessions
    av60 = v.rolling(60, min_periods=40).mean().shift(1)
    pe = (gap > 0.05) & (v >= 2 * av60)
    d["pe_gap_recent"] = pe.tail(5).any()
    d["pe_gap_days"] = [int(np.argmax(pe[t].tail(5).values[::-1])) if pe[t].tail(5).any() else None for t in T]

    nm = names.drop_duplicates("ticker").set_index("ticker") if len(names) else pd.DataFrame()
    mc = pd.to_numeric(nm["mcap"], errors="coerce") if "mcap" in nm else pd.Series(dtype=float)
    d["mcap"] = [mc.get(t, np.nan) for t in T]
    fb = pd.Series({t: (F.get(t, {}).get("mcap_b") or np.nan) * 1e9 for t in T})
    d["mcap"] = d.mcap.where(d.mcap > 0, fb)
    shares = d.mcap / d.price                                                           # float fallback (upper bound)
    fl = pd.Series({t: (F.get(t, {}).get("float_m") or np.nan) * 1e6 for t in T})
    d["float"] = fl.where(fl > 0, shares)
    d["float_est"] = ~(fl > 0)
    d["industry"] = [nm.industry.get(t, "") if len(nm) else "" for t in T]
    d["ipo_year"] = [str(nm["ipo_year"].get(t, "")) if "ipo_year" in nm else "" for t in T]

    # fundamentals: quarterly EPS / revenue growth (latest quarter vs a year earlier), FCF & sales TTM growth
    def last_yoy(rows):
        y = yoy(rows or [])
        return y[-1][2] if y and y[-1][2] is not None else np.nan
    d["eps_q_yoy"] = [last_yoy(F.get(t, {}).get("eps_q")) for t in T]
    d["rev_q_yoy"] = [last_yoy(F.get(t, {}).get("sales_q")) for t in T]

    def ann(rows):
        rows = [x for x in (rows or []) if x[1] is not None]
        if len(rows) < 2 or rows[-2][1] in (0, None):
            return np.nan
        a, b = rows[-2][1], rows[-1][1]
        return (b / a - 1) * 100 if a > 0 else (999.0 if b > 0 else np.nan)
    d["fcf_yoy"] = [ann(F.get(t, {}).get("fcf_a")) for t in T]
    d["sales_ttm_yoy"] = [ann(F.get(t, {}).get("sales_a")) for t in T]
    d["short_pct"] = [F.get(t, {}).get("short_pct") for t in T]
    d["last_eps_date"] = [((F.get(t, {}).get("eps_q") or [[None]])[-1][0]) for t in T]

    # institutional transactions: 13F shares held, latest quarter vs the one before
    if len(inst):
        p = sorted(inst.period.unique())[-2:]
        piv = inst[inst.period.isin(p)].pivot_table(index="ticker", columns="period", values="shares")
        chg = (piv[p[-1]] / piv[p[0]] - 1) * 100 if len(p) == 2 else pd.Series(dtype=float)
        d["inst_chg"] = [chg.get(t, np.nan) for t in T]
    else:
        d["inst_chg"] = np.nan
    return d


# ════════════════════════════════════════════════════════════════════
#  MJ's screens
# ════════════════════════════════════════════════════════════════════
def btw(s, lo, hi):
    return (s >= lo) & (s <= hi)


def conditions(d: pd.DataFrame) -> dict[str, list[tuple[str, pd.Series]]]:
    """Each screen as labelled conditions, so a stock can show exactly which filter it missed."""
    B10 = 10e9
    C = lambda lbl, ser: (lbl, ser.fillna(False).astype(bool))
    cap10 = [C("Mkt cap > $10B", d.mcap > B10), C("Float < 150M sh", d.float < 150e6),
             C("0–10% above SMA10", btw(d.sma10_gap, 0, 10)), C("≥ 100% above 52W low", d.above_low >= 100),
             C("Avg vol 60D > 300K", d.avg_vol60 > 300e3), C("Vol > 100K", d.vol > 100e3)]
    small = [C("Mkt cap $300M–10B", btw(d.mcap, 300e6, B10)), C("≥ 100% above 52W low", d.above_low >= 100),
             C("Avg vol 60D > 300K", d.avg_vol60 > 300e3), C("Float < 50M sh", d.float < 50e6),
             C("0–20% above SMA10", btw(d.sma10_gap, 0, 20)), C("Vol > 100K", d.vol > 100e3)]
    v3 = C("Volatility 1M > 3%", d.volat_1m > 3)
    eps = C("EPS Q YoY > 25%", d.eps_q_yoy > 25)
    rev = C("Rev Q YoY > 25%", d.rev_q_yoy > 25)
    return {
        "s1": [C("Mkt cap > $300M", d.mcap > 300e6), C("≥ 50% above 52W low", d.above_low >= 50),
               C("Avg vol 60D > 300K", d.avg_vol60 > 300e3), C("Vol > 100K", d.vol > 100e3), C("Float < 50M sh", d.float < 50e6),
               C("0–3% above 5-EMA", btw(d.ema5_gap, 0, 3)), C("SMA10 > SMA20", d.sma10_gt_20),
               C("1W −5% to +5%", btw(d.perf_1w, -5, 5)), C("Volatility 1M > 3.5%", d.volat_1m > 3.5)],
        "s2": cap10 + [v3, C("6M > +50%", d.perf_6m > 50)],
        "s3": cap10 + [v3, C("3M > +70%", d.perf_3m > 70)],
        "s4": cap10 + [C("1M > +30%", d.perf_1m > 30)],
        "s5": cap10 + [C("1W > +20%", d.perf_1w > 20)],
        "s6": [C("Mkt cap > $50M", d.mcap > 50e6), C("Above SMA20", d.above_sma20),
               C("Gap > 5% on 2× vol (5d)", d.pe_gap_recent), C("Float < 150M sh", d.float < 150e6),
               C("Avg vol 60D > 250K", d.avg_vol60 > 250e3)],
        "s7": [C("Mkt cap > $10B", d.mcap > B10), eps, rev, C("Avg vol 60D > 500K", d.avg_vol60 > 500e3),
               C("Float < 300M sh", d.float < 300e6), C("0–3% above SMA10", btw(d.sma10_gap, 0, 3)),
               C("Volatility 1M > 2%", d.volat_1m > 2), C("≥ 70% above 52W low", d.above_low >= 70)],
        "s8": [C("Mkt cap > $300M", d.mcap > 300e6), eps, rev, C("Avg vol 60D > 300K", d.avg_vol60 > 300e3),
               C("Float < 100M sh", d.float < 100e6), C("FCF growth > 25%", d.fcf_yoy > 25), C("≥ 70% above 52W low", d.above_low >= 70)],
        "s9": [C("Mkt cap $300M–10B", btw(d.mcap, 300e6, B10)), eps, rev, C("≥ 70% above 52W low", d.above_low >= 70),
               C("Avg vol 60D > 500K", d.avg_vol60 > 500e3), C("Float < 50M sh", d.float < 50e6),
               C("0–10% above SMA10", btw(d.sma10_gap, 0, 10)), v3, C("Above SMA50", d.above_sma50)],
        "s10": small + [C("6M > +100%", d.perf_6m > 100), v3],
        "s11": small + [C("3M > +70%", d.perf_3m > 70), v3],
        "s12": small + [C("1M > +30%", d.perf_1m > 30)],
        "s13": small + [C("1W > +20%", d.perf_1w > 20)],
        "s14": [C("Mkt cap > $2B", d.mcap > 2e9), C("Sales Q/Q > 25%", d.rev_q_yoy > 25), C("Sales TTM > 25%", d.sales_ttm_yoy > 25),
                C("13F shares rising", d.inst_chg > 0), C("Volatility week > 4%", d.volat_1w > 4),
                C("≥ 5% above 20D low", d.above_low20), C("≥ 5% above 50D low", d.above_low50),
                C("Avg vol > 2M", d.avg_vol60 > 2e6), C("Vol > 1M", d.vol > 1e6)],
        "s27": [C("Mkt cap > $150M", d.mcap > 150e6), rev, C("EMA10 > SMA20", d.ema10_gt_sma20), C("ADR > 3.5%", d.adr > 3.5),
                C("Vol > 500K", d.vol > 500e3), C("1W −5% to +5%", btw(d.perf_1w, -5, 5)), C("Avg vol 60D > 1M", d.avg_vol60 > 1e6),
                C("Above SMA50", d.above_sma50)],
    }


def screens(d: pd.DataFrame, conds=None) -> dict[str, pd.Series]:
    conds = conds or conditions(d)
    return {k: pd.concat([c for _, c in v], axis=1).all(axis=1) for k, v in conds.items()}


def why(d, conds, member, tb, ta, bio, final):
    """Per stock: the screen it came closest to (fewest misses) and what it missed, plus the tightness check."""
    keys = list(conds)
    miss = {k: pd.concat([~c for _, c in conds[k]], axis=1) for k in keys}
    nmiss = pd.DataFrame({k: m.sum(axis=1) for k, m in miss.items()})
    labels = {k: [l for l, _ in conds[k]] for k in keys}
    tcommon = [("SMA10 > SMA20", d.sma10_gt_20), ("1W < 5%", d.perf_1w < 5), ("Volatility 1M > 3.5%", d.volat_1m > 3.5),
               ("Within 5% of the 5-EMA", d.ema5_gap.abs() <= 5)]
    out = {}
    for t in d.index:
        best = nmiss.loc[t].sort_values(kind="stable")
        k = best.index[0]
        rec = {"best": k, "miss": [labels[k][i] for i, m in enumerate(miss[k].loc[t].values) if m]}
        if member[t]:
            rec["tight_miss"] = [l for l, ser in tcommon if not bool(ser.fillna(False).get(t))]
        rec["final"] = bool(final.get(t)); rec["bio"] = bool(bio.get(t))
        out[t] = rec
    return out


WATCHLISTS = [("WL1", "Daily Tightness Swing", ["s1"]),
              ("WL2", "Strong Move > $10B", ["s2", "s3", "s4", "s5", "s7"]),
              ("WL3", "Post-Earnings Continuation Base", ["s6"]),
              ("WL4", "Fundamentals", ["s8"]),
              ("WL5", "Strong Move < $10B", ["s9", "s10", "s11", "s12", "s13"]),
              ("WL12", "Institutional Accumulation (FinViz)", ["s14"])]
SCREEN_NAMES = {"s1": "Daily tightness", "s2": "Momentum 6M +50% >10B", "s3": "Momentum 3M +70% >10B",
                "s4": "Momentum 1M +30% >10B", "s5": "Momentum 1W +20% >10B", "s6": "Post-earnings gap",
                "s7": "Strongest >10B (EPS & Rev +25%)", "s8": "Fundamentals (EPS, Rev, FCF +25%)",
                "s9": "Strongest <10B (EPS & Rev +25%)", "s10": "Momentum 6M +100% <10B",
                "s11": "Momentum 3M +70% <10B", "s12": "Momentum 1M +30% <10B", "s13": "Momentum 1W +20% <10B",
                "s14": "Institutional accumulation", "s27": "Weekly RS scan"}
FUND_SCREENS = {"s7", "s8", "s9", "s14"}                    # screens with earnings / sales / ownership filters
TECH_SCREENS = {"s1", "s2", "s3", "s4", "s5", "s6", "s10", "s11", "s12", "s13"}


def tight(d):
    """Your two watchlist scans: 5-EMA above price by 0–5% (pullback) or below price by 0–5% (riding it)."""
    common = d.sma10_gt_20 & (d.perf_1w < 5) & (d.volat_1m > 3.5)
    below = btw(-d.ema5_gap, 0, 5) & common          # "EMA 5 above Price by 0–5%"
    above = btw(d.ema5_gap, 0, 5) & common           # "EMA 5 below Price by 0–5%"
    return below.fillna(False), above.fillna(False)


def is_biotech(d, stk):
    th = stk.reindex(d.index).theme.fillna("")
    return d.industry.str.contains("Biotechnology", case=False, na=False) & ~th.eq("Pharma") | th.eq("Biotech")


# ════════════════════════════════════════════════════════════════════
#  Episodic Pivots & Parabolic Shorts
# ════════════════════════════════════════════════════════════════════
def ep_scan(o, h, l, c, v, d, lookback=10):
    """Pradeep Bonde's EP: a big gap up on huge volume, driven by news, ideally in a neglected stock.
    Gap ≥ 8% (open vs prior close), volume ≥ 3× the 50-day average, $vol ≥ $10M, price ≥ $3,
    closes in the upper half of the day's range. Neglect = up < 20% over the 3 months before the gap."""
    gap = (o / c.shift(1) - 1) * 100
    av50 = v.rolling(50, min_periods=30).mean().shift(1)
    rvol = v / av50
    upper = (c - l) / (h - l).replace(0, np.nan) >= 0.5
    hit = (gap >= 8) & (rvol >= 3) & (c * v >= 10e6) & (c >= 3) & upper
    out = []
    n = len(c)
    for t in c.columns:
        days = np.where(hit[t].values[-lookback:])[0]
        if not len(days):
            continue
        i = n - lookback + days[-1]                                        # most recent EP day
        pre = c[t].iloc[i - 1] / c[t].iloc[max(0, i - 64)] - 1 if i > 64 else np.nan
        gap_low = l[t].iloc[i]
        out.append({"symbol": t, "date": c.index[i].strftime("%Y-%m-%d"), "days_ago": n - 1 - i,
                    "gap": round(float(gap[t].iloc[i]), 1), "rvol": round(float(rvol[t].iloc[i]), 1),
                    "day_chg": round(float((c[t].iloc[i] / c[t].iloc[i - 1] - 1) * 100), 1),
                    "vol_m": round(float(v[t].iloc[i]) / 1e6, 1), "nine_m": bool(v[t].iloc[i] >= 9e6),
                    "pre_3m": None if pd.isna(pre) else round(float(pre * 100)),
                    "neglected": bool(pd.notna(pre) and pre < 0.20),
                    "holding": bool(c[t].iloc[-1] >= gap_low), "since": round(float((c[t].iloc[-1] / c[t].iloc[i] - 1) * 100), 1),
                    "price": round(float(c[t].iloc[-1]), 2)})
    return sorted(out, key=lambda x: (x["days_ago"], -x["gap"]))


def parabolic_scan(o, h, l, c, v, d):
    """Parabolic shorts (Qullamaggie): a stock that went vertical — ≥ 50% in ≤ 5 days (≥ 25% for > $10B),
    3+ up days in a row, ≥ 25% above its 10-EMA, liquid. The trade is the first crack (first red day /
    opening-range low break), never the top-picking; listed as a watchlist."""
    ema10 = c.ewm(span=10, adjust=False).mean()
    run5 = (c / c.shift(5) - 1) * 100
    up = (c > c.shift(1)).astype(int)
    out = []
    for t in c.columns:
        big = (d.at[t, "mcap"] or 0) > 10e9 if t in d.index else False
        thr = 25 if big else 50
        r5 = run5[t].iloc[-1]; r5p = run5[t].iloc[-2]
        ext = (c[t].iloc[-1] / ema10[t].iloc[-1] - 1) * 100
        if not ((r5 >= thr or r5p >= thr) and ext >= 25 and c[t].iloc[-1] >= 5 and (c[t] * v[t]).iloc[-1] >= 20e6):
            continue
        red = c[t].iloc[-1] < o[t].iloc[-1] and c[t].iloc[-1] < c[t].iloc[-2]
        u = up[t].values[::-1][1 if red else 0:]                         # the run's up days (before a red day)
        streak_ = int(np.argmin(u)) if (u == 0).any() else len(u)
        out.append({"symbol": t, "run_5d": round(float(max(r5, r5p)), 1), "ext_ema10": round(float(ext), 1),
                    "up_days": streak_, "first_red": bool(red and run5[t].iloc[-2] >= thr),
                    "vol_x": round(float(v[t].iloc[-1] / v[t].rolling(50).mean().iloc[-2]), 1),
                    "price": round(float(c[t].iloc[-1]), 2), "large": bool(big)})
    return sorted(out, key=lambda x: -x["run_5d"])


def catalysts(tickers, limit=4):
    """Recent headlines for EP / parabolic names — Yahoo news first, Nasdaq as a fallback."""
    import requests
    out = {}
    for t in tickers:
        items = []
        try:
            import yfinance as yf
            for n in (yf.Ticker(t.replace(".", "-")).news or [])[:limit]:
                ct = n.get("content", n)
                title = ct.get("title")
                url = ((ct.get("canonicalUrl") or {}).get("url") if isinstance(ct.get("canonicalUrl"), dict) else None) \
                    or ct.get("link") or ((ct.get("clickThroughUrl") or {}).get("url") if isinstance(ct.get("clickThroughUrl"), dict) else None)
                src = (ct.get("provider") or {}).get("displayName") if isinstance(ct.get("provider"), dict) else ct.get("publisher")
                when = ct.get("pubDate") or ct.get("providerPublishTime")
                if title:
                    items.append({"title": title, "url": url, "src": src, "when": str(when)[:10]})
        except Exception:
            pass
        if not items:
            try:
                r = requests.get("https://api.nasdaq.com/api/news/topic/articlebysymbol",
                                 params={"q": f"{t}|stocks", "offset": 0, "limit": limit},
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=15)
                for n in (((r.json() or {}).get("data") or {}).get("rows") or [])[:limit]:
                    items.append({"title": n.get("title"), "url": "https://www.nasdaq.com" + (n.get("url") or ""),
                                  "src": n.get("publisher"), "when": n.get("created")})
            except Exception:
                pass
        out[t] = items
        time.sleep(0.3)
    return out


# ════════════════════════════════════════════════════════════════════
#  run + page
# ════════════════════════════════════════════════════════════════════
def run_scans(ctx: dict) -> dict:
    o, h, l, c, v = ctx["open"], ctx["high"], ctx["low"], ctx["close"], ctx["vol"]
    stk, leaders, top_themes, turners = ctx["stk"], ctx["leaders"], ctx["top_themes"], ctx["turners"]
    S_ = [t for t in stk.index if t in c.columns]
    o, h, l, c, v = (x[S_] for x in (o, h, l, c, v))
    d = fields(o, h, l, c, v, ctx["names"], ctx["F"], ctx["inst"], ctx["yoy"])
    conds = conditions(d)
    S = screens(d, conds)
    tb, ta = tight(d)
    bio = is_biotech(d, stk)
    turn_th = {g["theme"] for g in turners}

    wls, in_wl = [], pd.Series(False, index=d.index)
    member = {t: [] for t in d.index}
    for code, name, scr in WATCHLISTS:
        m = pd.concat([S[k] for k in scr], axis=1).any(axis=1)
        in_wl |= m
        for t in d.index[m]:
            member[t].append(code)
        wls.append({"code": code, "name": name, "n": int(m.sum()),
                    "screens": [{"key": k, "name": SCREEN_NAMES[k], "n": int(S[k].sum())} for k in scr],
                    "tickers": sorted(d.index[m], key=lambda t: -(stk.at[t, "rs_rating"] if pd.notna(stk.at[t, "rs_rating"]) else 0))})
    wl6 = (in_wl & (tb | ta)) | S["s27"]
    final = wl6 & (d.price > 3)

    def row(t):
        r, s = stk.loc[t], d.loc[t]
        scr = [k for k in S if S[k].get(t)]
        fund, tech = bool(set(scr) & FUND_SCREENS), bool(set(scr) & TECH_SCREENS)
        rs_ = r.rs_rating
        return {"symbol": t, "name": r["name"], "exchange": r.exchange, "theme": r.theme, "themes": r.all_themes,
                "industry": r.industry, "style": r["style"], "rs_rating": None if pd.isna(rs_) else int(rs_),
                "rs_chg_1w": None if pd.isna(r.rs_chg_1w) else int(r.rs_chg_1w),
                "price": round(float(s.price), 2), "perf_1w": _r(s.perf_1w), "perf_1m": _r(s.perf_1m), "perf_3m": _r(s.perf_3m),
                "adr": _r(s.adr), "ema5_gap": _r(s.ema5_gap), "dollar_vol": _r(s.dvol20 / 1e6),
                "mcap_b": _r(s.mcap / 1e9, 2), "float_m": _r(s.float / 1e6), "float_est": bool(s.float_est),
                "eps_q": _r(s.eps_q_yoy, 0), "rev_q": _r(s.rev_q_yoy, 0), "short_pct": s.short_pct,
                "wls": member[t] + (["RS"] if S["s27"].get(t) else []), "screens": scr,
                "tight": "below" if tb.get(t) else "above" if ta.get(t) else None,
                "lead_theme": bool(set(r.all_themes) & set(top_themes)), "leader": t in leaders.index,
                "turner": bool(set(r.all_themes) & turn_th), "fund": fund, "tech": tech, "both": fund and tech,
                "cs": r.get("cs"), "funds": None if pd.isna(r.get("funds")) else int(r.funds),
                "funds_chg": _r(r.get("funds_chg")), "setup": bool(r.get("setup_all")),
                **_er(t, ctx),
                "biotech": bool(bio.get(t)), "spark_px": [round(float(x), 2) for x in c[t].tail(22)]}

    rows = [row(t) for t in d.index[final]]
    try:                                                    # "why isn't X on the list?" lookup, loaded on demand
        W = why(d, conds, member, tb, ta, bio, final)
        (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
        (ROOT / "docs" / "data" / "scan_why.json").write_text(json.dumps(W, separators=(",", ":")))
    except Exception as e:
        print("  scan lookup skipped:", e)
    rows.sort(key=lambda x: (-(x["lead_theme"] + x["both"] + x["leader"]), -(x["rs_rating"] or 0)))

    eps = ep_scan(o, h, l, c, v, d)
    para = parabolic_scan(o, h, l, c, v, d)
    news = catalysts([e["symbol"] for e in eps if e["days_ago"] <= 2][:25] + [p["symbol"] for p in para][:8]) \
        if ctx.get("live") else {}
    for e in eps + para:
        t = e["symbol"]
        r = stk.loc[t]
        e.update({"name": r["name"], "exchange": r.exchange, "theme": r.theme, "rs_rating": None if pd.isna(r.rs_rating) else int(r.rs_rating),
                  "earnings_gap": bool((e.get("date") and d.at[t, "last_eps_date"] and
                                        abs((pd.Timestamp(e["date"]) - pd.Timestamp(d.at[t, "last_eps_date"])).days) <= 3)
                                       or any(re.search(r"\bearnings\b|\bQ[1-4]\b|quarter|results|guidance|revenue", n.get("title") or "", re.I)
                                              for n in news.get(t, [])[:3])),
                  "news": news.get(t, []), "cs": r.get("cs"), "mcap_b": _r(d.at[t, "mcap"] / 1e9, 2),
                  "short_pct": d.at[t, "short_pct"], "spark_px": [round(float(x), 2) for x in c[t].tail(30)]})

    cover = {"mcap": int(d.mcap.notna().sum()), "float_real": int((~d.float_est & d.float.notna()).sum()),
             "fund": int(d.eps_q_yoy.notna().sum()), "inst": int(d.inst_chg.notna().sum()), "n": len(d)}
    return {"asof": ctx["asof"], "generated": ctx["generated"], "watchlists": wls,
            "funnel": {"base": int(in_wl.sum()), "tight": int((in_wl & (tb | ta)).sum()), "rs_scan": int(S["s27"].sum()),
                       "wl6": int(wl6.sum()), "final": len(rows), "biotech": sum(r["biotech"] for r in rows)},
            "rows": rows, "ep": eps, "parabolic": para, "coverage": cover, "top_themes": top_themes,
            "premarket": _premarket()}


def _er(t, ctx):
    """Next earnings date from the Nasdaq calendar the engine already fetched."""
    d = (ctx.get("ecal") or {}).get(t)
    if not d:
        return {"earnings": None, "er_days": None}
    d = pd.Timestamp(d)
    return {"earnings": d.strftime("%d %b"), "er_days": int((d.normalize() - ctx["asof_ts"].normalize()).days)}


def _premarket():
    p = ROOT / "docs" / "data" / "premarket.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except ValueError:
        return None


def _r(v, n=1):
    try:
        v = float(v)
        return None if not np.isfinite(v) else (int(round(v)) if n == 0 else round(v, n))
    except (TypeError, ValueError):
        return None


def write_page(payload: dict, docs: Path, clean):
    tpl = (ROOT / "templates" / "scans.html").read_text()
    dash = (ROOT / "templates" / "dashboard.html").read_text()            # one source of truth for the look
    css = dash.split("<style>", 1)[1].split("</style>", 1)[0]
    tpl = tpl.replace("/*__SHARED_CSS__*/", css)
    data = json.dumps(clean(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (docs / "scans").mkdir(parents=True, exist_ok=True)
    (docs / "scans" / "index.html").write_text(tpl.replace("/*__DATA__*/null", data))


def tradingview_sections(payload: dict) -> list[str]:
    ex = {r["symbol"]: r["exchange"] for r in payload["rows"]}
    sym = lambda t, e=None: f"{e}:{t}" if e else t
    lines = []
    keep = [r for r in payload["rows"] if not r["biotech"]]
    if keep:
        lines += ["###Weekly scan result", ",".join(sym(r["symbol"], r["exchange"]) for r in keep)]
    eps = [e for e in payload["ep"] if e["days_ago"] <= 5]
    if eps:
        lines += ["###Episodic pivots", ",".join(sym(e["symbol"], e.get("exchange")) for e in eps)]
    if payload["parabolic"]:
        lines += ["###Parabolic short watch", ",".join(sym(p["symbol"], p.get("exchange")) for p in payload["parabolic"])]
    return lines


# FIELD NOTES — how TradingView / FinViz fields map to the engine's free data
#   Volatility 1M / ADR      average daily (high/low − 1) over 21 / 20 sessions
#   Float (shares)           Yahoo floatShares where the fundamentals job has it, else shares outstanding
#                            (market cap ÷ price) — an upper bound, so a few low-float names can be missed
#   Market cap               Nasdaq listing data, Yahoo as a fallback
#   EPS / revenue growth     latest reported quarter vs the same quarter a year earlier (Yahoo); TradingView
#                            uses diluted GAAP EPS, Yahoo the reported (usually adjusted) figure
#   FCF growth TTM           latest fiscal year vs the year before (Yahoo cash-flow statement)
#   Institutional trans.     13F shares held, latest quarter vs the previous one (FinViz uses 3-month changes)
#   Perf 1W/1M/3M/6M         5 / 21 / 63 / 126 sessions
