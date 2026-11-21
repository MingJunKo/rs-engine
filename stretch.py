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
import os
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "docs" / "data" / "stretch.json"
STATE = ROOT / "data" / "history" / "stretch_state.json"
LONG_BREADTH = ROOT / "data" / "history" / "breadth_long.csv"     # written by backtest10.py (10 years of the universe)
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
         "mco": "McClellan (universe)", "nymo": "NYMO proxy", "namo": "NAMO proxy"}


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


def mcclellan(close: pd.DataFrame) -> pd.Series:
    r = close.pct_change()
    adv, dec = (r > 0).sum(axis=1), (r < 0).sum(axis=1)
    rana = (adv - dec) / (adv + dec).replace(0, np.nan) * 1000
    out = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    out[r.notna().sum(axis=1) < 100] = np.nan
    return out


def breadth_all(close: pd.DataFrame, exchange: dict | None) -> pd.DataFrame:
    """Universe breadth plus NYSE-only and Nasdaq-only McClellan oscillators (proxies for $NYMO / $NAMO)."""
    f = breadth_features(close)
    if exchange:
        ny = [t for t in close.columns if str(exchange.get(t, "")).upper().startswith("NYSE")]
        nq = [t for t in close.columns if str(exchange.get(t, "")).upper().startswith("NASDAQ")]
        f["nymo"] = mcclellan(close[ny]) if len(ny) >= 100 else np.nan
        f["namo"] = mcclellan(close[nq]) if len(nq) >= 100 else np.nan
    return f


def with_long_history(recent: pd.DataFrame) -> pd.DataFrame:
    """Prepend the 10-year breadth history saved by backtest10.py, so breadth signals get real sample sizes."""
    if not LONG_BREADTH.exists() or recent is None or len(recent) < 80:
        return recent
    try:
        lg = pd.read_csv(LONG_BREADTH, index_col=0, parse_dates=True)
        cut = recent.index[60]                                  # let the recent EMAs warm up first
        return pd.concat([lg[lg.index < cut], recent[recent.index >= cut]]).sort_index()
    except Exception as e:
        print("long breadth history unreadable:", e)
        return recent


BREADTH_SIGNALS = [
    ("t2108", "breadth", lambda f: f.t2108 <= 20, lambda f: f.t2108 >= 80, lambda f: f"{f.t2108:.0f}% of stocks above their 40-day MA"),
    ("down4", "breadth", lambda f: f.down4 >= 5, None, lambda f: f"{f.down4:.1f}% of stocks fell ≥4% today"),
    ("ratio5", "breadth", lambda f: f.ratio5 <= 0.5, lambda f: f.ratio5 >= 2.5, lambda f: f"5-day 4% up/down ratio {f.ratio5:.2f}"),
    ("mco", "breadth", lambda f: f.mco <= -60, lambda f: f.mco >= 60, lambda f: f"McClellan-style oscillator {f.mco:+.0f}"),
    ("nymo", "breadth", lambda f: f.nymo <= -60, lambda f: f.nymo >= 60, lambda f: f"NYMO (NYSE stocks) {f.nymo:+.0f}"),
    ("namo", "breadth", lambda f: f.namo <= -60, lambda f: f.namo >= 60, lambda f: f"NAMO (Nasdaq stocks) {f.namo:+.0f}"),
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
        if test is None or (key in ("nymo", "namo", "zbt", "up4") and key not in feat):
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


def magnitude(c: pd.Series, i: int, years: float, day_gap: int = 5) -> list:
    """Plain-% questions: 'a −6.7% day — how often, and then what?' and '−15% from the high in 18 sessions — then what?'"""
    out = []
    r = c.pct_change() * 100
    rt = r.iloc[i]
    if not pd.isna(rt) and (rt <= -2.5 or rt >= 3):
        mask = r <= rt if rt < 0 else r >= rt
        eps = episodes(mask, gap=day_gap)
        prev = [k for k in eps if k < i - day_gap]
        out.append({"key": "bigday", "label": f"{rt:+.1f}% day", "what": f"days {'down' if rt < 0 else 'up'} {abs(rt):.1f}% or more",
                    "episodes": len(eps), "freq": freq_text(len(eps), years), "fwd": fwd_stats(c, eps, i),
                    "dates": list(dict.fromkeys(c.index[k].strftime("%b %Y") for k in prev))[-6:],
                    "last": None if not prev else {"date": c.index[prev[-1]].strftime("%d %b %Y"),
                                                   "move": round(float(r.iloc[prev[-1]]), 1),
                                                   "fwd20": None if prev[-1] + 20 > i else round((c.iloc[prev[-1] + 20] / c.iloc[prev[-1]] - 1) * 100, 1)}})
    hi = c.rolling(63, min_periods=1).max()
    dd = (c / hi - 1) * 100
    since = c.rolling(63, min_periods=1).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    ddn, sn = dd.iloc[i], int(since.iloc[i])
    if ddn <= -7:
        window = max(int(sn * 1.5), sn + 5)
        mask = (dd <= ddn) & (since <= window)
        eps = episodes(mask, gap=20)
        prev = [k for k in eps if k < i - 20]
        out.append({"key": "drawdown", "label": f"{ddn:.1f}% from the high {sn} sessions ago",
                    "what": f"declines of {abs(ddn):.0f}%+ within ~{window} sessions of a 3-month high",
                    "episodes": len(eps), "freq": freq_text(len(eps), years), "fwd": fwd_stats(c, eps, i),
                    "dates": list(dict.fromkeys(c.index[k].strftime("%b %Y") for k in prev))[-6:],
                    "last": None if not prev else {"date": c.index[prev[-1]].strftime("%d %b %Y"),
                                                   "fwd20": None if prev[-1] + 20 > i else round((c.iloc[prev[-1] + 20] / c.iloc[prev[-1]] - 1) * 100, 1)}})
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


def market_read(data: dict, universe_close: pd.DataFrame | None = None, universe_vol: pd.DataFrame | None = None,
                exchange: dict | None = None) -> dict:
    vix = data.get("^VIX", pd.DataFrame()).get("Close")
    v3 = data.get("^VIX3M", pd.DataFrame()).get("Close")
    bf = breadth_all(universe_close, exchange) if universe_close is not None and universe_close.shape[1] >= 200 else None
    bf = with_long_history(bf)
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
        r_ = c.pct_change()
        past = r_.iloc[1:i]
        if len(past) > 250 and not pd.isna(r_.iloc[i]):
            entry["now"]["z1_long"] = round(float(r_.iloc[i] / past.std()), 1)       # vs its whole-history volatility
            entry["now"]["z1_1y"] = round(float(r_.iloc[i] / r_.iloc[i - 252:i].std()), 1)
            entry["now"]["worse"] = int((past <= r_.iloc[i]).sum() if r_.iloc[i] < 0 else (past >= r_.iloc[i]).sum())
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
        entry["magnitude"] = magnitude(c, i, years)
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
        res["breadth_hist"] = {k: [None if pd.isna(x) else round(float(x), 1) for x in bf[k].tail(120)]
                               for k in ("t2108", "mco", "nymo", "namo") if k in bf}
        res["breadth_since"] = bf.dropna(how="all").index[0].strftime("%b %Y")
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


def _h(fwd: dict, h: int):
    return (fwd or {}).get(h) or (fwd or {}).get(str(h))


def _per_year(sig: dict, years: float) -> float:
    return sig.get("episodes", 0) / max(years, 1)


def _short(label: str) -> str:
    return label.replace(" from the 50-day MA", " from 50d MA").replace(" from the 20-day average", " from 20d avg") \
                .replace(" its 20-day average", " its 20d avg").replace(" of stocks fell ≥4% today", " of stocks −4%+ today")


def build_message(res: dict, new_keys: set, weekly: bool) -> str | None:
    """Digest format: verdict first, then per index how big / how rare / what usually follows, breadth once, caution."""
    e = lambda s: html.escape(str(s))
    idx = res["index"]
    total = {sym: len(ent.get("oversold", [])) for sym, ent in idx.items()}
    up_tot = {sym: len(ent.get("overbought", [])) for sym, ent in idx.items()}
    any_conf = any(bool(ent.get("confirm")) or (ent.get("ftd") or {}).get("ftd_today") for ent in idx.values())
    if not weekly and not any(total.values()) and not any(up_tot.values()) and not any_conf:
        return None
    mx = max(total.values(), default=0)
    verdict = ("🟢🟢🟢 Extreme oversold" if mx >= 9 else "🟢🟢 Deeply oversold" if mx >= 5 else
               "🟢 Oversold" if mx >= 2 else "🔴 Stretched up" if max(up_tot.values(), default=0) >= 2 else "⚪ Calm")
    L = [f"<b>📉 Mean-reversion watch — {e(res['asof'])}</b>",
         f"<b>{verdict}</b> · " + " · ".join(f"{s} {n} signals" for s, n in total.items()), ""]
    for sym, ent in idx.items():
        nw, yrs = ent["now"], ent.get("years") or 20
        os_ = [x for x in ent.get("oversold", []) if x["family"] != "breadth"]
        ob = ent.get("overbought", [])
        if not (os_ or ob or weekly or ent.get("magnitude")):
            continue
        L.append(f"<b>{sym} {ent['close']}</b>  {nw['ret1']:+.1f}% today · {nw['ret5']:+.1f}% 5d · {nw['off_high']:+.0f}% from high")
        # how big
        if nw.get("z1_long") is not None and abs(nw["ret1"]) >= 1.5:
            sz = f"{nw['z1']:+.1f}σ vs last 20d · <b>{nw['z1_long']:+.1f}σ vs its {yrs:.0f}-yr norm</b>"
            if nw.get("worse") is not None:
                sz += f" · {'worse' if nw['ret1'] < 0 else 'bigger'} days since {ent['since']}: {nw['worse']}"
            L.append(f"  📐 {sz}")
        # how rare (the single rarest thing) + what usually follows
        cb = ent.get("oversold_combo", {})
        if cb.get("k", 0) >= 2 and cb.get("fwd"):
            x = _h(cb["fwd"], 20) or _h(cb["fwd"], 10)
            small = " (small sample)" if x and x["n"] < 10 else ""
            L.append(f"  🎯 {cb['k']} price/VIX signals at once — {e(cb['freq'])}")
            if x:
                L.append(f"      after those: next 20d avg <b>{x['avg']:+.1f}%</b>, up {x['up']}% of {x['n']}{small}"
                         + (f" · e.g. {', '.join(cb['dates'][-2:])}" if cb.get("dates") else ""))
        for m in ent.get("magnitude", [])[:1]:
            x = _h(m["fwd"], 20)
            L.append(f"  📏 {e(m['label'])}: {m['episodes']}× since {ent['since']}"
                     + (f" → next 20d avg {x['avg']:+.1f}%, up {x['up']}%" if x else ""))
        # three rarest individual signals, one line
        top = sorted(os_, key=lambda x: _per_year(x, yrs))[:3]
        if top:
            L.append("  🔎 " + " · ".join(("🆕" if x["key"] in new_keys else "") + e(_short(x["label"])) for x in top))
        if ob:
            L.append("  🔴 " + " · ".join(e(_short(x["label"])) for x in ob[:3]))
        f_ = ent.get("ftd") or {}
        if f_.get("state") in ("attempt", "confirmed") or ent.get("confirm"):
            bits = [f_["label"]] if f_.get("state") in ("attempt", "confirmed") else []
            bits += [x["label"] for x in ent.get("confirm", [])]
            L.append("  🟦 " + e(" · ".join(bits)))
        if weekly and not os_ and not ob:
            L.append(f"  calm: {nw['band']:+.1f}σ from 20d avg · {nw['ext50']:+.1f} ATR from 50d · RSI(2) {nw['rsi2']:.0f}")
        L.append("")
    # breadth once (it is the same market for both)
    b = res.get("breadth_now")
    ref = idx.get("SPY") or next(iter(idx.values()), {})
    bsig = [x for x in ref.get("oversold", []) if x["family"] == "breadth"]
    if b:
        mo = " · ".join(f"{k.upper()} {b[k]:+.0f}" for k in ("nymo", "namo") if b.get(k) is not None)
        L.append(f"<b>Breadth</b> {len(bsig)} signals: {b['down4']:.0f}% of stocks −4%+ · {b['t2108']:.0f}% above 40d MA"
                 + (f" · {mo}" if mo else ""))
    # caution: recent episodes that kept falling (oversold can get more oversold in a downtrend)
    bad = {}
    for ent in idx.values():
        for x in ent.get("oversold", []):
            lt = x.get("last") or {}
            if lt.get("fwd20") is not None and lt["fwd20"] < -3:
                bad[lt["date"]] = min(bad.get(lt["date"], 0), lt["fwd20"])
    recent = sorted(bad.items(), key=lambda kv: pd.Timestamp(kv[0]))[-2:]
    if recent:
        L.append("⚠️ <i>Recent washouts kept falling: " + ", ".join(f"{d} → {v:+.0f}% 20d later" for d, v in recent)
                 + ". Snap-backs fail more often in a downtrend — wait for a reversal day / FTD.</i>")
    mv = res.get("movers") or []
    if mv:
        L.append("Unusual moves: " + ", ".join(f"{m['symbol']} {m['ret']:+.0f}%" for m in mv[:6]))
    return "\n".join(L).rstrip()


def notify(res: dict, settings: dict, force_weekly: bool = False):
    """Alert when the oversold side has signals from ≥ 2 families, or a new signal fires; weekly summary on Saturdays."""
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    keys_now = sorted({f"{sym}:{s['key']}" for sym, ent in res["index"].items()
                       for s in ent.get("oversold", []) + ent.get("magnitude", [])})
    new_keys = {k.split(":")[1] for k in keys_now if k not in set(prev.get("keys", []))}
    strongest = max((score(ent) for ent in res["index"].values()), default=(0, 0))
    weekly = force_weekly or dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).weekday() == 5
    cluster = strongest[1] >= 2                        # oversold signals from at least two families
    confirm_now = any((ent.get("ftd") or {}).get("ftd_today") or ent.get("confirm") for ent in res["index"].values())
    fire = (cluster and (bool(new_keys) or not prev.get("cluster"))) or confirm_now or \
           any(k in new_keys for k in ("z1", "vix30", "vixinv", "t2108", "down4", "nymo", "namo", "bigday", "drawdown"))
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
    tg_send("sendMessage", {"text": msg if len(msg) <= 4096 else msg[:4000].rsplit("\n", 1)[0], "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, topic)


# ───────────────────────── single stock
def stock_check(ticker: str, data: dict | None = None) -> str:
    """ticker, or ticker@YYYY-MM-DD to look at a past day (and see what actually happened after it)."""
    t, _, on = ticker.upper().strip().partition("@")
    if data is None:
        data = download([t, "SPY"])
    full = data.get(t)
    if full is None or len(full) < 120:
        return f"<b>{html.escape(t)}</b>: not enough price history on Yahoo."
    d = full
    if on:
        d = full.loc[:pd.Timestamp(on)]
        data = {**data, "SPY": data["SPY"].loc[:pd.Timestamp(on)]} if data.get("SPY") is not None else data
        if len(d) < 120:
            return f"<b>{html.escape(t)}</b>: no history before {html.escape(on)}."
    c, h, l = d.Close, d.High, d.Low
    pf = price_features(c, h, l)
    i, years = len(c) - 1, (c.index[-1] - c.index[0]).days / 365.25
    nw = pf.iloc[i]
    e = lambda s: html.escape(str(s))
    lines = [f"<b>{e(t)} — how unusual is this move?</b>  ({c.index[-1].strftime('%a %d %b %Y')}, {years:.0f} yrs of data)",
             f"Close {c.iloc[-1]:.2f}  ·  {nw.ret1:+.1f}% on the day = <b>{nw.z1:+.1f}σ</b> vs its prior 20 days",
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
    for m in magnitude(c, i, years):
        fw = m["fwd"]
        lines.append(f"\n📏 {e(m['label'])}: {e(m['what'])} — {m['episodes']} since {c.index[0]:%Y}, {e(m['freq'])}")
        for hz in (5, 20, 60):
            if fw.get(hz):
                x = fw[hz]
                lines.append(f"   next {hz}d: avg {x['avg']:+.1f}% · median {x['med']:+.1f}% · up {x['up']}% of {x['n']} "
                             f"(worst {x['worst']:+.0f}%, best {x['best']:+.0f}%)")
        if m.get("dates"):
            lines.append(f"   e.g. {', '.join(m['dates'])}")
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
        lines.append(f"\nVs market: SPY {rs.iloc[-1] * 100:+.1f}% that day, β {beta:.1f} → expected {exp:+.1f}%, "
                     f"so {nw.ret1 - exp:+.1f}% was stock-specific")
    if on and len(full) > len(d):
        after = full.Close.iloc[len(d) - 1:]
        n_ = len(after) - 1
        pts = [(k, (after.iloc[k] / after.iloc[0] - 1) * 100) for k in (5, 20, 60) if k <= n_]
        lines.append("\n<b>What actually happened next:</b> " + " · ".join(f"{k}d {v:+.1f}%" for k, v in pts)
                     + f" · today ({n_} sessions later) {(after.iloc[-1] / after.iloc[0] - 1) * 100:+.1f}%")
    lines.append("<i>σ = the day's move divided by the stock's own daily volatility over the prior 20 sessions.</i>")
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


# ───────────────────────── personal deployment plan (Telegram only; amounts live in the DEPLOY_PLAN secret)
def deploy_events(spy: pd.DataFrame, plan: dict) -> tuple[list, float]:
    """Replay the plan from its start date. Rules: a third of remaining cash on each new SPY 5-day σ ≤ −3 episode or each
    further `drop_step`% fall below the start price; backstop = remaining ÷ months left on the first session on/after the
    `day`th of each month. Returns (events [(date, reason, amount)], remaining)."""
    c = spy.Close
    z5 = price_features(c, spy.High, spy.Low).z5
    start = pd.Timestamp(plan["start"])
    ref = float(c.loc[:start].iloc[-1])
    rem = float(plan["total"]) - float(plan["initial"])
    months, step, day = int(plan.get("months", 12)), float(plan.get("drop_step", 5)), int(plan.get("day", 25))
    last_sig, level, done_months, ev = c.index.get_loc(c.loc[:start].index[-1]), 0, 0, []
    for i in range(last_sig + 1, len(c)):
        d, px = c.index[i], float(c.iloc[i])
        if rem < 1:
            break
        why = []
        if z5.iloc[i] <= -3 and i - last_sig > 10:
            why.append(f"SPY 5-day move {z5.iloc[i]:+.1f}σ")
        if z5.iloc[i] <= -3:
            last_sig = i
        lv = int(max(0, (1 - px / ref) * 100) // step)
        if lv > level:
            why.append(f"SPY {100 * (px / ref - 1):+.1f}% from the start price {ref:.2f}")
            level = lv
        if why:
            amt = rem / 3; rem -= amt; ev.append((d, " + ".join(why), amt))
        mk = (d.year - start.year) * 12 + d.month - start.month
        if mk >= 1 and d.day >= day and done_months < min(mk, months) and rem >= 1:
            left = months - done_months
            amt = rem / left; rem -= amt; done_months += 1
            ev.append((d, f"monthly backstop {done_months}/{months}", amt))
    return ev, rem


def deploy_alert(data: dict, settings: dict):
    raw = os.environ.get("DEPLOY_PLAN", "").strip()
    spy = data.get("SPY")
    if not raw or spy is None:
        return
    try:
        plan = json.loads(raw)
    except Exception as e:
        print("DEPLOY_PLAN is not valid JSON:", e); return
    ev, rem = deploy_events(spy, plan)
    today = spy.index[-1]
    now = [x for x in ev if x[0] == today]
    print(f"deploy plan: {len(ev)} events since start, {len(now)} today")
    if not now:
        return
    invested = float(plan["total"]) - rem
    lines = [f"<b>💰 Deployment plan — buy today ({today:%a %d %b})</b>"]
    for _, why, amt in now:
        lines.append(f"• <b>${amt:,.0f}</b> — {html.escape(why)}")
    lines += [f"Plan: ${invested:,.0f} of ${float(plan['total']):,.0f} invested · ${rem:,.0f} left",
              "<i>S&P 500 unit trust via CPFIS — place the order today (fills at today's NAV). "
              "Amounts assume every earlier step was followed.</i>"]
    from notify import tg_send
    topic = "stretch" if (settings.get("telegram_topics") or {}).get("stretch") else "alerts"
    tg_send("sendMessage", {"text": "\n".join(lines), "parse_mode": "HTML"}, settings, topic)


def run_daily(universe_close: pd.DataFrame | None, settings: dict, docs: Path, demo: bool = False, notify_on: bool = True,
              universe_vol: pd.DataFrame | None = None, exchange: dict | None = None):
    data = demo_data() if demo else download(["SPY", "QQQ", "^VIX", "^VIX3M"])
    res = market_read(data, universe_close, universe_vol, exchange)
    res["generated"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%d %b %Y %H:%M SGT")
    write_page(res, docs)
    if notify_on and not demo:
        notify(res, settings)
        try:
            deploy_alert(data, settings)
        except Exception as e:
            print("deploy plan failed:", e)
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
                tg_send("sendMessage", {"text": msg if len(msg) <= 4096 else msg[:4000].rsplit("\n", 1)[0], "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, topic)
    else:
        data = demo_data() if a.demo else download(["SPY", "QQQ", "^VIX", "^VIX3M"])
        res = market_read(data, None)
        res["generated"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%d %b %Y %H:%M SGT")
        if not a.demo:                                   # never publish synthetic data
            write_page(res, ROOT / "docs")
        print(build_message(res, set(), True))
        for sym, ent in res["index"].items():
            print(sym, "FTD:", ent["ftd"]["label"], "| FTDs in history:", ent["ftd"]["episodes"], "| failed %:", ent["ftd"].get("fail_rate"))
        if a.weekly and not a.no_telegram and not a.demo:
            notify(res, settings, force_weekly=True)
