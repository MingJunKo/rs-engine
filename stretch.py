#!/usr/bin/env python3
"""
Stretch monitor — how unusual is the market's move, and what usually happens next?

For SPY and QQQ it tracks independent "stretch" signals in four families:
  price      1-day move in σ (vs the last 20 days), 5-day move in σ, distance from the 20-day average in σ,
             ATRs from the 50-day MA, RSI(2), losing/winning streak
  volatility VIX ≥ 30, VIX spike (≥ 1.4× its 20-day average), VIX term structure inverted (VIX ≥ VIX3M)
  breadth    (RS Engine universe) % above the 40-day MA (Stockbee's T2108 idea), % of stocks down ≥ 4% today,
             5-day ratio of 4% up vs 4% down days, McClellan-style oscillator on the universe's advances/declines
For every signal it looks up history: how often it happens (episodes, de-duplicated within 10 sessions),
the forward 5/10/20/60-day SPY or QQQ return after it, how often that was positive, and when it last happened.
Price and VIX history goes back to 1993 (SPY), 1999 (QQQ) and 2007 (VIX3M); breadth history covers the engine's
~2 years of universe data, so its statistics are thinner.

    python stretch.py --demo                 # synthetic data, no network
    python stretch.py --ticker NVDA          # one stock: how unusual is today's move? (Telegram + stdout)
    (the daily run is called from rs_engine.py with the universe prices for the breadth signals)
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "docs" / "data" / "stretch.json"
STATE = ROOT / "data" / "history" / "stretch_state.json"
HORIZONS = (5, 10, 20, 60)
GAP = 10                                     # sessions between separate episodes


# ───────────────────────── helpers
def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def streak(c: pd.Series) -> pd.Series:
    """+n after n up closes in a row, −n after n down closes."""
    s = np.sign(c.diff().fillna(0)).astype(int).values
    out, run = np.zeros(len(s), int), 0
    for i, v in enumerate(s):
        run = (run + v if np.sign(run) == v else v) if v else 0
        out[i] = run
    return pd.Series(out, index=c.index)


def episodes(mask: pd.Series, gap: int = GAP) -> list:
    """First day of each run of True, ignoring re-fires within `gap` sessions."""
    idx = np.flatnonzero(mask.fillna(False).values)
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last > gap:
            keep.append(i)
        last = i if i - last > gap else last
    return keep


def freq_text(n: int, years: float) -> str:
    if n == 0:
        return "first time in the data"
    every = years / n
    if every >= 1.5:
        return f"about once every {every:.0f} years" if every >= 3 else f"about once every {every:.1f} years"
    per = n / years
    return f"about {per:.0f}× a year" if per >= 1.5 else "about once a year"


def fwd_stats(price: pd.Series, starts: list, today_i: int) -> dict:
    """Forward returns after each episode start (excluding ones too recent to measure)."""
    out = {}
    v = price.values
    for h in HORIZONS:
        rets = [(v[i + h] / v[i] - 1) * 100 for i in starts if i + h < len(v) and i != today_i]
        if rets:
            a = np.array(rets)
            out[h] = {"n": len(a), "avg": round(float(a.mean()), 1), "med": round(float(np.median(a)), 1),
                      "up": round(float((a > 0).mean() * 100)), "worst": round(float(a.min()), 1),
                      "best": round(float(a.max()), 1)}
    return out


def baseline(price: pd.Series) -> dict:
    out = {}
    for h in HORIZONS:
        r = (price.shift(-h) / price - 1).dropna() * 100
        out[h] = {"avg": round(float(r.mean()), 1), "up": round(float((r > 0).mean() * 100))}
    return out


# ───────────────────────── signal library
def price_features(c: pd.Series, h: pd.Series, l: pd.Series) -> pd.DataFrame:
    r = c.pct_change()
    sd20 = r.rolling(20).std().shift(1)
    sd60 = r.rolling(60).std().shift(5)
    f = pd.DataFrame(index=c.index)
    f["ret1"] = r * 100
    f["z1"] = r / sd20
    f["ret5"] = c.pct_change(5) * 100
    f["z5"] = c.pct_change(5) / (sd60 * math.sqrt(5))
    f["band"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["ext50"] = (c - c.rolling(50).mean()) / atr(h, l, c)
    f["rsi2"] = rsi(c, 2)
    f["streak"] = streak(c)
    f["off_high"] = (c / c.rolling(252, min_periods=20).max() - 1) * 100
    return f


# side = −1 for oversold (snap-back up), +1 for stretched up (pullback risk)
PRICE_SIGNALS = [
    # key, family, oversold test, overbought test, label(feature row) -> str
    ("z1", "price", lambda f: f.z1 <= -2.5, lambda f: f.z1 >= 2.5,
     lambda f: f"{f.ret1:+.1f}% day = {f.z1:+.1f}σ"),
    ("z5", "price", lambda f: f.z5 <= -2.5, lambda f: f.z5 >= 2.5,
     lambda f: f"{f.ret5:+.1f}% in 5 days = {f.z5:+.1f}σ"),
    ("band", "price", lambda f: f.band <= -2.5, lambda f: f.band >= 2.5,
     lambda f: f"{f.band:+.1f}σ from the 20-day average"),
    ("ext50", "price", lambda f: f.ext50 <= -3, lambda f: f.ext50 >= 4,
     lambda f: f"{f.ext50:+.1f} ATRs from the 50-day MA"),
    ("rsi2", "price", lambda f: f.rsi2 <= 5, lambda f: f.rsi2 >= 95,
     lambda f: f"RSI(2) {f.rsi2:.0f}"),
    ("streak", "price", lambda f: f.streak <= -5, lambda f: f.streak >= 7,
     lambda f: f"{abs(int(f.streak))} {'down' if f.streak < 0 else 'up'} days in a row"),
]
NAMES = {"z1": "σ day", "z5": "5-day σ move", "band": "Stretch from 20-day avg", "ext50": "ATRs from 50-day MA",
         "rsi2": "RSI(2)", "streak": "Streak", "vix30": "VIX ≥ 30", "vixspike": "VIX spike",
         "vixinv": "VIX curve inverted", "vixlow": "VIX complacency",
         "t2108": "% above 40-day MA", "down4": "Stocks down ≥4% today", "ratio5": "5-day 4% ratio",
         "mco": "McClellan (universe)"}


def vix_features(vix: pd.Series, vix3m: pd.Series | None) -> pd.DataFrame:
    f = pd.DataFrame(index=vix.index)
    f["vix"] = vix
    f["spike"] = vix / vix.rolling(20).mean()
    f["term"] = vix / vix3m.reindex(vix.index) if vix3m is not None else np.nan
    return f


VIX_SIGNALS = [
    ("vix30", "volatility", lambda f: f.vix >= 30, None, lambda f: f"VIX {f.vix:.1f}"),
    ("vixspike", "volatility", lambda f: f.spike >= 1.4, None, lambda f: f"VIX {f.spike:.2f}× its 20-day average"),
    ("vixinv", "volatility", lambda f: f.term >= 1.0, None, lambda f: f"VIX/VIX3M {f.term:.2f} (inverted)"),
    ("vixlow", "volatility", None, lambda f: f.vix <= 12, lambda f: f"VIX {f.vix:.1f}"),
]


def breadth_features(close: pd.DataFrame) -> pd.DataFrame:
    r = close.pct_change()
    valid = r.notna().sum(axis=1).replace(0, np.nan)
    f = pd.DataFrame(index=close.index)
    f["t2108"] = (close > close.rolling(40).mean()).sum(axis=1) / close.rolling(40).mean().notna().sum(axis=1) * 100
    up4, dn4 = (r >= 0.04).sum(axis=1), (r <= -0.04).sum(axis=1)
    f["down4"] = dn4 / valid * 100
    f["up4"] = up4 / valid * 100
    f["ratio5"] = up4.rolling(5).sum() / dn4.rolling(5).sum().replace(0, np.nan)
    adv, dec = (r > 0).sum(axis=1), (r < 0).sum(axis=1)
    rana = (adv - dec) / (adv + dec).replace(0, np.nan) * 1000
    f["mco"] = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    f["zbt"] = (adv / (adv + dec).replace(0, np.nan)).ewm(span=10, adjust=False).mean()   # Zweig: 10-day EMA of adv/(adv+dec)
    f.loc[valid < 200] = np.nan                   # too few stocks that day to trust
    return f


BREADTH_SIGNALS = [
    ("t2108", "breadth", lambda f: f.t2108 <= 20, lambda f: f.t2108 >= 80, lambda f: f"{f.t2108:.0f}% of stocks above their 40-day MA"),
    ("down4", "breadth", lambda f: f.down4 >= 5, None, lambda f: f"{f.down4:.1f}% of stocks fell ≥4% today"),
    ("ratio5", "breadth", lambda f: f.ratio5 <= 0.5, lambda f: f.ratio5 >= 2.5, lambda f: f"5-day 4% up/down ratio {f.ratio5:.2f}"),
    ("mco", "breadth", lambda f: f.mco <= -60, lambda f: f.mco >= 60, lambda f: f"McClellan-style oscillator {f.mco:+.0f}"),
]


# confirmation (the bounce is real): breadth thrusts — bullish side only
THRUST_SIGNALS = [
    ("zbt", "breadth", None, lambda f: (f.zbt >= 0.615) & (f.zbt.rolling(10).min() <= 0.40),
     lambda f: f"Zweig breadth thrust: 10-day adv ratio {f.zbt:.2f} (from ≤ 0.40 within 10 days)"),
    ("up4", "breadth", None, lambda f: f.up4 >= 5, lambda f: f"{f.up4:.1f}% of stocks rose ≥4% today (breadth thrust day)"),
]
NAMES.update({"zbt": "Zweig breadth thrust", "up4": "4% breadth thrust", "ftd": "Follow-through day"})


def ftd_track(d: pd.DataFrame) -> tuple[dict, list, list]:
    """O'Neil rally attempt / follow-through day. Returns today's state, FTD day indices, and for each FTD whether it
    failed (closed below the rally low within 25 sessions).
      correction  close below the 50-day MA or ≥ 8% under the 63-day high
      day 1       first up close right after a new 20-day low; the rally low is the low of that dip
      FTD         day 4–25 of the attempt, up ≥ 1.25% on higher volume than the day before
      reset       an undercut of the rally low kills the attempt (or the FTD)"""
    c, h, l = d.Close.values, d.High.values, d.Low.values
    v = d.Volume.values if "Volume" in d else np.ones(len(c))
    sma50 = pd.Series(c).rolling(50).mean().values
    hi63 = pd.Series(h).rolling(63, min_periods=1).max().values
    lo20 = pd.Series(l).rolling(20, min_periods=1).min().values
    state, day, rl, fi = "none", 0, None, None
    ftds, fails = [], []
    for i in range(1, len(c)):
        corr = (not np.isnan(sma50[i]) and c[i] < sma50[i]) or c[i] <= hi63[i] * 0.92
        if state == "attempt":
            day += 1
            if l[i] < rl:
                state = "none"
            elif 4 <= day <= 25 and c[i] >= c[i - 1] * 1.0125 and v[i] > v[i - 1]:
                state, fi = "confirmed", i
                ftds.append(i)
            elif day > 25:
                state = "none"
        elif state == "confirmed":
            if c[i] < rl:
                state = "none"
                if fi is not None and i - fi <= 25:
                    fails.append(fi)
            elif i - fi > 40:
                state = "none"
        if state == "none" and corr and c[i] > c[i - 1] and min(l[i - 1], l[i]) <= lo20[i]:
            state, day, rl = "attempt", 1, min(l[i - 1], l[i])
    n = len(c) - 1
    cur = {"state": state, "day": day if state == "attempt" else None, "rally_low": None if rl is None else round(float(rl), 2),
           "ftd_date": d.index[fi].strftime("%d %b %Y") if state == "confirmed" and fi is not None else None,
           "ftd_today": bool(ftds and ftds[-1] == n),
           "in_correction": bool((not np.isnan(sma50[n]) and c[n] < sma50[n]) or c[n] <= hi63[n] * 0.92)}
    return cur, ftds, fails


def ftd_entry(d: pd.DataFrame, years: float) -> dict:
    cur, ftds, fails = ftd_track(d)
    c, i = d.Close, len(d) - 1
    past = [k for k in ftds if k < i - 25]
    fail_rate = round(len([k for k in fails if k in past]) / len(past) * 100) if past else None
    cur.update({"episodes": len(ftds), "freq": freq_text(len(ftds), years), "fwd": fwd_stats(c, ftds, i),
                "fail_rate": fail_rate,
                "last": None if not past else {"date": d.index[past[-1]].strftime("%d %b %Y"),
                                               "fwd20": round((c.iloc[past[-1] + 20] / c.iloc[past[-1]] - 1) * 100, 1)}})
    if cur["state"] == "attempt":
        cur["label"] = f"Rally attempt day {cur['day']} (rally low {cur['rally_low']}) — a follow-through day can come from day 4"
    elif cur["state"] == "confirmed":
        cur["label"] = ("FOLLOW-THROUGH DAY today" if cur["ftd_today"] else f"Follow-through day on {cur['ftd_date']}") + \
                       f" — holds while above the rally low {cur['rally_low']}"
    else:
        cur["label"] = "In a correction — waiting for day 1 of a rally attempt" if cur["in_correction"] else "Uptrend — no correction to confirm"
    return cur


def evaluate(feat: pd.DataFrame, lib: list, price: pd.Series, today_i: int, years: float, side: int) -> list:
    """Signals active today on one side, each with its history."""
    out = []
    row = feat.iloc[today_i]
    for key, fam, lo, hi, label in lib:
        test = lo if side < 0 else hi
        if test is None:
            continue
        mask = test(feat).fillna(False)
        if not bool(mask.iloc[today_i]):
            continue
        eps = episodes(mask)
        prev = [i for i in eps if i < today_i - GAP]
        last = prev[-1] if prev else None
        out.append({"key": key, "family": fam, "name": NAMES[key], "label": label(row),
                    "episodes": len(eps), "freq": freq_text(len(eps), years),
                    "fwd": fwd_stats(price, eps, today_i),
                    "last": None if last is None else {
                        "date": feat.index[last].strftime("%d %b %Y"),
                        "fwd20": None if last + 20 >= len(price) else round((price.iloc[last + 20] / price.iloc[last] - 1) * 100, 1)}})
    return out


def rank_text(z: pd.Series, today_i: int) -> str | None:
    """'Biggest down day (in σ) since …' for an unusual 1-day move."""
    v = z.iloc[today_i]
    if pd.isna(v) or abs(v) < 2:
        return None
    past = z.iloc[:today_i]
    hit = past[past <= v] if v < 0 else past[past >= v]
    if hit.empty:
        return f"the biggest {'down' if v < 0 else 'up'} day in σ terms in the data"
    return f"the biggest {'down' if v < 0 else 'up'} day in σ terms since {hit.index[-1].strftime('%d %b %Y')}"


def composite(feats: dict, price: pd.Series, today_i: int, years: float, side: int) -> dict:
    """How many price + volatility signals are on today, and what happened historically when that many were."""
    masks = []
    for feat, lib in feats.values():
        for key, fam, lo, hi, _ in lib:
            t = lo if side < 0 else hi
            if t is not None:
                masks.append(t(feat).reindex(price.index).fillna(False).astype(int))
    cnt = pd.concat(masks, axis=1).sum(axis=1)
    k = int(cnt.iloc[today_i])
    if k < 2:
        return {"k": k}
    m = cnt >= k
    eps = episodes(m)
    return {"k": k, "episodes": len(eps), "freq": freq_text(len(eps), years), "fwd": fwd_stats(price, eps, today_i),
            "dates": list(dict.fromkeys(price.index[i].strftime("%b %Y") for i in eps if i < today_i - GAP))[-6:]}


# ───────────────────────── data
def download(tickers, period="max"):
    """One ticker at a time with retries — parallel yfinance downloads can trip its cache lock."""
    import time
    import yfinance as yf
    out = {}
    for t in tickers:
        for attempt in range(3):
            try:
                df = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close"])
                if len(df):
                    out[t] = df
                    break
            except Exception as e:
                print(f"  {t}: {e}")
            time.sleep(2 + attempt * 3)
    return out


def demo_data():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2005-01-03", "2026-11-19")
    def walk(mu, sd, start):
        r = rng.normal(mu, sd, len(idx))
        r[-1] = -4.5 * sd                              # make "today" a big down day
        c = start * np.exp(np.cumsum(r))
        return pd.DataFrame({"Close": c, "High": c * (1 + abs(rng.normal(0, sd / 2, len(idx)))),
                             "Low": c * (1 - abs(rng.normal(0, sd / 2, len(idx)))),
                             "Volume": rng.integers(5e7, 1e8, len(idx))}, index=idx)
    spy, qqq = walk(0.0003, 0.011, 100), walk(0.0004, 0.014, 50)
    vix = pd.DataFrame({"Close": np.clip(18 + np.cumsum(rng.normal(0, 0.8, len(idx))) * 0.1 + rng.normal(0, 2, len(idx)), 9, 80)}, index=idx)
    vix.iloc[-1, 0] = 33
    v3 = pd.DataFrame({"Close": vix.Close.rolling(60, min_periods=1).mean() + 1.5}, index=idx)
    return {"SPY": spy, "QQQ": qqq, "^VIX": vix, "^VIX3M": v3}


# ───────────────────────── the daily market read
def movers(close: pd.DataFrame, vol: pd.DataFrame | None, min_z: float = 4.0, min_dvol: float = 20e6, top: int = 12) -> list:
    """Universe stocks whose move today was ≥ min_z σ of their own recent volatility, liquid ones only."""
    r = close.pct_change()
    z = r / r.rolling(20).std().shift(1)
    zt, rt = z.iloc[-1], r.iloc[-1] * 100
    if vol is not None:
        dv = (close * vol).rolling(20).mean().iloc[-2]
        zt = zt[dv.reindex(zt.index).fillna(0) >= min_dvol]
    zt = zt.dropna()
    zt = zt[zt.abs() >= min_z]
    zt = zt.reindex(zt.abs().sort_values(ascending=False).index)[:top]
    out = []
    for t, zv in zt.items():
        rk = rank_text(z[t].dropna(), len(z[t].dropna()) - 1)
        out.append({"symbol": t, "ret": round(float(rt[t]), 1), "z": round(float(zv), 1),
                    "price": round(float(close[t].iloc[-1]), 2), "rank": rk})
    return out


def market_read(data: dict, universe_close: pd.DataFrame | None = None, universe_vol: pd.DataFrame | None = None) -> dict:
    vix = data.get("^VIX", pd.DataFrame()).get("Close")
    v3 = data.get("^VIX3M", pd.DataFrame()).get("Close")
    bf = breadth_features(universe_close) if universe_close is not None and universe_close.shape[1] >= 200 else None
    res = {"asof": None, "index": {}, "breadth_now": None}
    for sym in ("SPY", "QQQ"):
        d = data.get(sym)
        if d is None or len(d) < 300:
            continue
        c, h, l = d.Close, d.High, d.Low
        pf = price_features(c, h, l)
        i = len(c) - 1
        years = (c.index[-1] - c.index[0]).days / 365.25
        vf = vix_features(vix.reindex(c.index).ffill(), v3) if vix is not None else None
        entry = {"close": round(float(c.iloc[-1]), 2), "years": round(years, 1), "since": c.index[0].strftime("%Y"),
                 "now": {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in pf.iloc[i].items()},
                 "rank": rank_text(pf.z1, i), "base": baseline(c)}
        if vf is not None:
            entry["now"].update({"vix": round(float(vf.vix.iloc[i]), 1),
                                 "vix_spike": None if pd.isna(vf.spike.iloc[i]) else round(float(vf.spike.iloc[i]), 2),
                                 "vix_term": None if pd.isna(vf.term.iloc[i]) else round(float(vf.term.iloc[i]), 2)})
        for side, tag in ((-1, "oversold"), (1, "overbought")):
            sig = evaluate(pf, PRICE_SIGNALS, c, i, years, side)
            if vf is not None:
                sig += evaluate(vf, VIX_SIGNALS, c, i, years, side)
            if bf is not None:
                b = bf.reindex(c.index)
                b_i = b.index.get_loc(b.dropna(how="all").index[-1]) if b.dropna(how="all").size else None
                if b_i == i:
                    by = (b.dropna(how="all").index[-1] - b.dropna(how="all").index[0]).days / 365.25
                    sig += evaluate(b, BREADTH_SIGNALS, c, i, by, side)
            entry[tag] = sig
            entry[tag + "_combo"] = composite({"p": (pf, PRICE_SIGNALS), **({"v": (vf, VIX_SIGNALS)} if vf is not None else {})},
                                              c, i, years, side)
        entry["ftd"] = ftd_entry(d, years)
        conf = []
        if bf is not None:
            b2 = bf.reindex(c.index)
            if b2.dropna(how="all").size and b2.index.get_loc(b2.dropna(how="all").index[-1]) == i:
                by = (b2.dropna(how="all").index[-1] - b2.dropna(how="all").index[0]).days / 365.25
                conf = evaluate(b2, THRUST_SIGNALS, c, i, by, 1)
        entry["confirm"] = conf
        res["index"][sym] = entry
        res["asof"] = c.index[-1].strftime("%a %d %b %Y")
    if bf is not None and bf.dropna(how="all").size:
        last = bf.dropna(how="all").iloc[-1]
        res["breadth_now"] = {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in last.items()}
        res["breadth_hist"] = {k: [None if pd.isna(x) else round(float(x), 1) for x in bf[k].tail(120)] for k in ("t2108", "mco")}
    if universe_close is not None and universe_close.shape[1] >= 50:
        try:
            res["movers"] = movers(universe_close, universe_vol)
        except Exception as e:
            print("movers failed:", e)
    return res


def families(sig: list) -> set:
    return {s["family"] for s in sig}


def score(entry: dict, tag="oversold") -> tuple[int, int]:
    s = entry.get(tag, [])
    return len(s), len(families(s))


# ───────────────────────── Telegram
def _fwd_line(fwd: dict, h=20) -> str:
    x = fwd.get(h) or fwd.get(10) or fwd.get(5)
    if not x:
        return ""
    hh = h if fwd.get(h) else (10 if fwd.get(10) else 5)
    return f"next {hh}d: avg {x['avg']:+.1f}%, up {x['up']}% of {x['n']}"


def build_message(res: dict, new_keys: set, weekly: bool) -> str | None:
    e = lambda s: html.escape(str(s))
    lines = []
    for sym, ent in res["index"].items():
        n, fams = score(ent)
        nu, famu = score(ent, "overbought")
        conf_on = bool(ent.get("confirm")) or (ent.get("ftd") or {}).get("ftd_today")
        if not (n or nu or conf_on) and not weekly:
            continue
        head = f"<b>{sym}</b> {ent['close']}  ·  {ent['now']['ret1']:+.1f}% today ({ent['now']['z1']:+.1f}σ)"
        lines.append(head)
        if ent.get("rank"):
            lines.append(f"   <i>{e(ent['rank'])}</i>")
        for s in ent.get("oversold", []):
            star = "🆕 " if s["key"] in new_keys else ""
            fl = _fwd_line(s['fwd'])
            lines.append(f"   🟢 {star}{e(s['label'])} — {e(s['freq'])}" + (f"; {e(fl)}" if fl else ""))
            if s.get("last"):
                lt = s["last"]
                lines.append(f"      last time {lt['date']}" + (f" → {lt['fwd20']:+.1f}% 20d later" if lt['fwd20'] is not None else ""))
        for s in ent.get("overbought", []):
            fl = _fwd_line(s['fwd'])
            lines.append(f"   🔴 {e(s['label'])} — {e(s['freq'])}" + (f"; {e(fl)}" if fl else ""))
        cb = ent.get("oversold_combo", {})
        if cb.get("k", 0) >= 2 and cb.get("fwd"):
            lines.append(f"   ➜ <b>{cb['k']} price/VIX signals at once</b>: {e(cb['freq'])}; {e(_fwd_line(cb['fwd']))}"
                         + (f" (e.g. {', '.join(cb['dates'][-3:])})" if cb.get("dates") else ""))
        f_ = ent.get("ftd") or {}
        if f_.get("state") in ("attempt", "confirmed"):
            extra = ""
            if f_.get("state") == "confirmed" and f_.get("fwd", {}).get(20):
                x = f_["fwd"][20]
                extra = f"; past FTDs: next 20d avg {x['avg']:+.1f}%, up {x['up']}%" + \
                        (f", {f_['fail_rate']}% failed" if f_.get("fail_rate") is not None else "")
            lines.append(f"   🟦 {e(f_['label'])}{e(extra)}")
        for s in ent.get("confirm", []):
            fl = _fwd_line(s['fwd'])
            lines.append(f"   🟦 {e(s['label'])} — {e(s['freq'])}" + (f"; {e(fl)}" if fl else ""))
        if weekly and not (n or nu):
            nw = ent["now"]
            lines.append(f"   calm: {nw['band']:+.1f}σ from 20d avg · {nw['ext50']:+.1f} ATR from 50d · RSI(2) {nw['rsi2']:.0f}"
                         + (f" · VIX {nw['vix']}" if nw.get("vix") else ""))
        lines.append("")
    if not lines:
        return None
    b = res.get("breadth_now")
    if b:
        lines.append(f"Breadth (RS universe): {b['t2108']:.0f}% above 40d MA · {b['down4']:.1f}% down ≥4% today · "
                     f"5d ratio {b['ratio5'] if b['ratio5'] is not None else '—'} · McClellan {b['mco']:+.0f}")
    mv = res.get("movers") or []
    if mv:
        lines.append("Unusual stock moves (≥4σ, $20M+/day): " + ", ".join(f"{m['symbol']} {m['ret']:+.0f}% ({m['z']:+.1f}σ)" for m in mv[:6]))
    title = "Weekly stretch check" if weekly and not any(score(x)[0] for x in res["index"].values()) else "Mean-reversion watch"
    hdr = [f"<b>{title} — {e(res['asof'])}</b>",
           "<i>🟢 stretched down (snap-back odds) · 🔴 stretched up. Stats = SPY/QQQ after past episodes.</i>", ""]
    return "\n".join(hdr + lines)


def notify(res: dict, settings: dict, force_weekly: bool = False):
    """Alert when the oversold side has signals from ≥ 2 families, or a new signal fires; weekly summary on Saturdays."""
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    keys_now = sorted({f"{sym}:{s['key']}" for sym, ent in res["index"].items() for s in ent.get("oversold", [])})
    new_keys = {k.split(":")[1] for k in keys_now if k not in set(prev.get("keys", []))}
    strongest = max((score(ent) for ent in res["index"].values()), default=(0, 0))
    weekly = force_weekly or dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).weekday() == 5
    cluster = strongest[1] >= 2                        # oversold signals from at least two families
    confirm_now = any((ent.get("ftd") or {}).get("ftd_today") or ent.get("confirm") for ent in res["index"].values())
    fire = (cluster and (bool(new_keys) or not prev.get("cluster"))) or confirm_now or \
           any(k in new_keys for k in ("z1", "vix30", "vixinv", "t2108", "down4"))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"date": res["asof"], "keys": keys_now, "cluster": cluster}))
    if not (fire or weekly):
        print("stretch: nothing new to send")
        return
    msg = build_message(res, new_keys, weekly and not fire)
    if not msg:
        return
    url = (settings.get("dashboard_url") or "").rstrip("/")
    if url:
        msg += f'\n<a href="{url}/stretch/">Open the stretch page</a>'
    from notify import tg_send
    topic = "stretch" if (settings.get("telegram_topics") or {}).get("stretch") else "alerts"
    tg_send("sendMessage", {"text": msg[:4000], "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, topic)


# ───────────────────────── single stock
def stock_check(ticker: str, data: dict | None = None) -> str:
    t = ticker.upper().strip()
    if data is None:
        data = download([t, "SPY"])
    d = data.get(t)
    if d is None or len(d) < 120:
        return f"<b>{html.escape(t)}</b>: not enough price history on Yahoo."
    c, h, l = d.Close, d.High, d.Low
    pf = price_features(c, h, l)
    i, years = len(c) - 1, (c.index[-1] - c.index[0]).days / 365.25
    nw = pf.iloc[i]
    e = lambda s: html.escape(str(s))
    lines = [f"<b>{e(t)} — how unusual is this move?</b>  ({c.index[-1].strftime('%a %d %b %Y')}, {years:.0f} yrs of data)",
             f"Close {c.iloc[-1]:.2f}  ·  {nw.ret1:+.1f}% today = <b>{nw.z1:+.1f}σ</b> vs its last 20 days",
             f"5 days {nw.ret5:+.1f}% ({nw.z5:+.1f}σ)  ·  {nw.band:+.1f}σ from 20d avg  ·  {nw.ext50:+.1f} ATR from 50d MA  ·  "
             f"RSI(2) {nw.rsi2:.0f}  ·  {nw.off_high:+.0f}% from 52w high"]
    rk = rank_text(pf.z1, i)
    if rk:
        lines.append(f"<i>{e(rk)}</i>")
    # how often a day at least this big happened, and what followed
    z = nw.z1
    if not pd.isna(z) and abs(z) >= 1.5:
        thr = math.floor(abs(z) * 2) / 2                    # e.g. 3.4σ → "≥ 3σ days"
        mask = pf.z1 <= -thr if z < 0 else pf.z1 >= thr
        eps = episodes(mask, gap=5)
        fw = fwd_stats(c, eps, i)
        lines.append(f"\n{'Down' if z < 0 else 'Up'} days of {thr:.1f}σ or more: {len(eps)} since {c.index[0]:%Y} — {freq_text(len(eps), years)}")
        for hz in (5, 20, 60):
            if fw.get(hz):
                x = fw[hz]
                lines.append(f"   next {hz}d: avg {x['avg']:+.1f}% · median {x['med']:+.1f}% · up {x['up']}% of {x['n']} "
                             f"(worst {x['worst']:+.0f}%, best {x['best']:+.0f}%)")
        prev = [k for k in eps if k < i - 5]
        if prev:
            p = prev[-1]
            after = (c.iloc[min(p + 20, i)] / c.iloc[p] - 1) * 100
            lines.append(f"   last time: {c.index[p]:%d %b %Y} ({pf.ret1.iloc[p]:+.1f}%) → {after:+.1f}% over the next 20 days")
    for side, tag in ((-1, "Stretched down"), (1, "Stretched up")):
        sig = evaluate(pf, PRICE_SIGNALS, c, i, years, side)
        if sig:
            lines.append(f"\n{tag}:")
            for s in sig:
                fl = _fwd_line(s['fwd'])
                lines.append(f"   • {e(s['label'])} — {e(s['freq'])}" + (f"; {e(fl)}" if fl else ""))
    spy = data.get("SPY")
    if spy is not None:
        rs = spy.Close.pct_change().reindex(c.index)
        beta = c.pct_change().tail(250).cov(rs.tail(250)) / rs.tail(250).var()
        exp = beta * rs.iloc[-1] * 100
        lines.append(f"\nVs market: SPY {rs.iloc[-1] * 100:+.1f}% today, β {beta:.1f} → expected {exp:+.1f}%, "
                     f"so {nw.ret1 - exp:+.1f}% was stock-specific")
    lines.append("<i>σ = today's move divided by the stock's own daily volatility over the last 20 sessions.</i>")
    return "\n".join(lines)


# ───────────────────────── page
def write_page(res: dict, docs: Path):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, separators=(",", ":"), default=str))
    tpl = (ROOT / "templates" / "stretch.html").read_text()
    dash = (ROOT / "templates" / "dashboard.html").read_text()            # same look as the rest of the site
    tpl = tpl.replace("/*__SHARED_CSS__*/", dash.split("<style>", 1)[1].split("</style>", 1)[0])
    (docs / "stretch").mkdir(parents=True, exist_ok=True)
    (docs / "stretch" / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(res, default=str).replace("</", "<\\/")))


def run_daily(universe_close: pd.DataFrame | None, settings: dict, docs: Path, demo: bool = False, notify_on: bool = True,
              universe_vol: pd.DataFrame | None = None):
    data = demo_data() if demo else download(["SPY", "QQQ", "^VIX", "^VIX3M"])
    res = market_read(data, universe_close, universe_vol)
    res["generated"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%d %b %Y %H:%M SGT")
    write_page(res, docs)
    if notify_on and not demo:
        notify(res, settings)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ticker", default="")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--weekly", action="store_true", help="send the market read now")
    a = ap.parse_args()
    settings = json.loads((ROOT / "config" / "settings.json").read_text())
    sys.path.insert(0, str(ROOT))
    if a.ticker:
        for t in [x for x in a.ticker.replace(",", " ").split() if x]:
            msg = stock_check(t, demo_data() if a.demo else None) if not a.demo else stock_check("SPY", demo_data())
            print(msg)
            if not a.no_telegram and not a.demo:
                from notify import tg_send
                topic = "stretch" if (settings.get("telegram_topics") or {}).get("stretch") else "alerts"
                tg_send("sendMessage", {"text": msg[:4000], "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, topic)
    else:
        data = demo_data() if a.demo else download(["SPY", "QQQ", "^VIX", "^VIX3M"])
        res = market_read(data, None)
        res["generated"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%d %b %Y %H:%M SGT")
        write_page(res, ROOT / "docs")
        print(build_message(res, set(), True))
        for sym, ent in res["index"].items():
            print(sym, "FTD:", ent["ftd"]["label"], "| FTDs in history:", ent["ftd"]["episodes"], "| failed %:", ent["ftd"].get("fail_rate"))
        if a.weekly and not a.no_telegram and not a.demo:
            notify(res, settings, force_weekly=True)
