#!/usr/bin/env python3
"""
MJ — RS Engine
Sector, group and stock relative strength, market regime and rotation.

Run:  python rs_engine.py            (real data from Yahoo Finance)
      python rs_engine.py --demo     (simulated prices, for testing offline)
      python rs_engine.py --notify   (also send the Telegram summary)

Outputs:
  docs/index.html          the dashboard (served by GitHub Pages)
  docs/data/*.csv          tables for Google Sheets (IMPORTDATA)
  docs/data/tradingview_leaders.txt
  data/history/*.csv       daily snapshots
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG_DIR = ROOT / "config"
DOCS = ROOT / "docs"
DATA_OUT = DOCS / "data"
HIST = ROOT / "data" / "history"
SGT = dt.timezone(dt.timedelta(hours=8))

SETTINGS = json.loads((CFG_DIR / "settings.json").read_text())
BENCH = SETTINGS.get("benchmark", "RSP")
W3M, W1M, W1W = 0.50, 0.35, 0.15          # same weighting as the sheet
STYLE_ORDER = ["growth", "cyclical", "defensive"]


def log(*a):
    print(*a, flush=True)


# ════════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════════
def yf_symbol(t: str) -> str:
    return t.replace(".", "-")


def download_yahoo(tickers: list[str], period: str = "2y",
                   want=("Close", "High", "Low", "Volume")) -> dict[str, pd.DataFrame]:
    """Batch-download daily OHLCV. Returns wide frames keyed by field."""
    import yfinance as yf

    fields = {f: {} for f in want}
    chunk = 120
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        df = None
        for attempt in range(4):
            try:
                df = yf.download([yf_symbol(t) for t in part], period=period, interval="1d",
                                 auto_adjust=True, group_by="ticker", threads=True,
                                 progress=False)
                break
            except Exception as e:                        # rate limits etc.
                log(f"  chunk {i // chunk + 1}: retry {attempt + 1} ({e})")
                time.sleep(10 * (attempt + 1))
        if df is None or df.empty:
            continue
        for t in part:
            s = yf_symbol(t)
            try:
                sub = df[s] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            for f in fields:
                if f in sub:
                    fields[f][t] = sub[f]
        log(f"  downloaded {min(i + chunk, len(tickers))}/{len(tickers)}")
        time.sleep(1.5)

    out = {f: pd.DataFrame(v).sort_index() for f, v in fields.items()}
    close = out["Close"]
    good = close.columns[close.notna().sum() > 30]           # need some history
    for f in out:
        out[f] = out[f].reindex(columns=good)
    missing = sorted(set(tickers) - set(good))
    if missing:
        log(f"  no data for {len(missing)} tickers (non-US or delisted): {', '.join(missing[:40])}"
            + (" …" if len(missing) > 40 else ""))
    return out


def demo_prices(tickers: list[str], styles: dict[str, str], seed: int = 7) -> dict[str, pd.DataFrame]:
    """Simulated prices with a style rotation, so the dashboard can be tested offline."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=510)
    n = len(dates)
    mkt = rng.normal(0.0004, 0.009, n)
    t = np.arange(n)
    late = t > n - 70                                    # last ~3 months: defensive rotation
    drift = {
        "growth":    np.where(late, -0.0012, 0.0006),
        "cyclical":  np.where(late,  0.0003, 0.0002),
        "defensive": np.where(late,  0.0010, -0.0001),
        "neutral":   np.zeros(n),
    }
    style_f = {k: v + rng.normal(0, 0.004, n) for k, v in drift.items()}
    close, high, low, vol = {}, {}, {}, {}
    for tk in tickers:
        st = styles.get(tk, "neutral")
        beta = rng.uniform(0.7, 1.5)
        idio = rng.normal(rng.normal(0, 0.0008), rng.uniform(0.008, 0.03), n)
        r = beta * mkt + style_f.get(st, style_f["neutral"]) + idio
        if tk == BENCH:
            r = mkt
        c = 50 * np.exp(np.cumsum(r)) * rng.uniform(0.3, 6)
        rngd = np.abs(rng.normal(0, np.std(r) * 0.8, n))
        close[tk], high[tk], low[tk] = c, c * (1 + rngd), c * (1 - rngd)
        vol[tk] = rng.lognormal(np.log(rng.uniform(2e5, 2e7)), 0.35, n)
    mk = lambda d: pd.DataFrame(d, index=dates)
    out = {"Close": mk(close), "High": mk(high), "Low": mk(low), "Volume": mk(vol)}
    young = [t for t in ("NCLD", "LYTE", "CHAT", "DRAM") if t in tickers] + list(rng.choice(tickers, 25, replace=False))
    for f in out:                                   # recent listings and missing days, like real data
        for i, tk in enumerate(young):
            out[f].loc[out[f].index[:-(40 + 15 * i)], tk] = np.nan
        out[f].iloc[-3, 5:9] = np.nan
    return out


def fetch_names() -> pd.DataFrame:
    """Names, exchange, sector, industry for US listings (NASDAQ screener). Cached weekly."""
    cache = CFG_DIR / "names.csv"
    if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 86400:
        return pd.read_csv(cache)
    try:
        import requests
        rows = []
        for ex, tv in (("nasdaq", "NASDAQ"), ("nyse", "NYSE"), ("amex", "AMEX")):
            r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                             params={"tableonly": "true", "download": "true", "exchange": ex},
                             headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                             timeout=30)
            for d in r.json()["data"]["rows"]:
                rows.append({"ticker": d["symbol"].strip().replace("/", "."), "name": d["name"],
                             "exchange": tv, "sector": d.get("sector", ""),
                             "industry": d.get("industry", "")})
        df = pd.DataFrame(rows).drop_duplicates("ticker")
        df["name"] = (df["name"].str.replace(r"\s+(Common Stock|Class [A-C] Common Stock|"
                                             r"Ordinary Shares|American Depositary Shares).*$", "",
                                             regex=True).str.strip())
        df.to_csv(cache, index=False)
        log(f"  names: {len(df)} US listings cached")
        return df
    except Exception as e:
        log(f"  names: lookup failed ({e}); using cache/tickers")
        return pd.read_csv(cache) if cache.exists() else pd.DataFrame(
            columns=["ticker", "name", "exchange", "sector", "industry"])


# ════════════════════════════════════════════════════════════════════
#  INDICATORS  (all wide frames: rows = dates, cols = tickers)
# ════════════════════════════════════════════════════════════════════
def atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return pd.DataFrame(tr, index=close.index, columns=close.columns).ewm(alpha=1 / n, adjust=False).mean()


def xs_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile (0–1) per date."""
    return df.rank(axis=1, pct=True)


def rs_rating(close, high, low, bench_close, bench_atr):
    """
    Volatility-adjusted RS (Jeff Sun's VARS idea), blended over 3 horizons, ranked 1–99.
      daily move in ATR units = Δclose / ATR(14)
      VARS_n = Σ_n stock ATR-moves − Σ_n benchmark ATR-moves
    """
    a = atr(high, low, close)
    vadj = close.diff() / a.shift(1)
    bvadj = bench_close.diff() / bench_atr.shift(1)
    vars_ = {}
    for n in (5, 21, 63):
        vars_[n] = vadj.rolling(n).sum().sub(bvadj.rolling(n).sum(), axis=0)
    blend = W3M * xs_pct(vars_[63]) + W1M * xs_pct(vars_[21]) + W1W * xs_pct(vars_[5])
    rating = (xs_pct(blend) * 98 + 1).round()
    return rating, vars_, a


def weekly_rrg(close: pd.Series, bench: pd.Series, n: int = 10, tail: int = 6):
    """Relative Rotation Graph coordinates (weekly). 100/100 = in line with benchmark."""
    wk = pd.concat([close, bench], axis=1).resample("W-FRI").last().dropna()
    rs = (100 * wk.iloc[:, 0] / wk.iloc[:, 1]).ewm(span=3, adjust=False).mean()
    ratio = 100 + (rs - rs.rolling(n).mean()) / rs.rolling(n).std()
    mom = 100 + (ratio - ratio.rolling(n).mean()) / ratio.rolling(n).std()
    pts = pd.concat([ratio, mom], axis=1).dropna().tail(tail)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pts.clip(94, 106).values]


def quadrant(pt):
    if not pt:
        return ""
    x, y = pt
    return ("Leading" if y >= 100 else "Weakening") if x >= 100 else ("Improving" if y >= 100 else "Lagging")


def earnings_calendar(asof, days: int = 60) -> dict:
    """Upcoming US earnings dates for every listed stock, from Nasdaq's public calendar (one request per
    weekday). Cached for the day in data/history/earnings_cal.json; past dates are dropped."""
    path = HIST / "earnings_cal.json"
    cal = {}
    if path.exists():
        try:
            cal = json.loads(path.read_text())
        except ValueError:
            cal = {}
    today = asof.normalize()
    if cal.get("_fetched") == dt.date.today().isoformat():
        return {k: v for k, v in cal.items() if not k.startswith("_")}
    import requests
    got, ok_days = {}, 0
    for d in pd.bdate_range(today, today + pd.Timedelta(days=days)):
        try:
            r = requests.get("https://api.nasdaq.com/api/calendar/earnings", params={"date": d.strftime("%Y-%m-%d")},
                             headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=20)
            rows = ((r.json() or {}).get("data") or {}).get("rows") or []
            ok_days += 1
            for row in rows:
                sym = str(row.get("symbol", "")).strip().upper().replace("/", ".")
                if sym and sym not in got:
                    got[sym] = d.strftime("%Y-%m-%d")
        except Exception:
            continue
        time.sleep(0.25)
    if ok_days >= 10:                                 # keep the old calendar if Nasdaq didn't answer
        keep = {k: v for k, v in cal.items() if not k.startswith("_") and k not in got and v >= today.strftime("%Y-%m-%d")}
        cal = {**keep, **got, "_fetched": dt.date.today().isoformat()}
        HIST.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cal, separators=(",", ":")))
        log(f"  earnings calendar: {len(got)} upcoming reports over {ok_days} days")
    else:
        log("  earnings calendar: Nasdaq unavailable, using the cached calendar")
    return {k: v for k, v in cal.items() if not k.startswith("_") and v >= today.strftime("%Y-%m-%d")}


def next_earnings(ticker: str, asof, cal: dict | None = None) -> tuple:
    """Next earnings date: Nasdaq calendar first, Yahoo as a fallback. Returns ('12 Nov', days_away) or (None, nan)."""
    if cal and ticker in cal:
        d = pd.Timestamp(cal[ticker])
        return d.strftime("%d %b"), (d.normalize() - asof.normalize()).days
    try:
        import yfinance as yf
        c = yf.Ticker(yf_symbol(ticker)).calendar or {}
        dates = [pd.Timestamp(d) for d in c.get("Earnings Date", []) if d is not None]
        future = [d for d in dates if d.normalize() >= asof.normalize()]
        if future:
            d = min(future)
            return d.strftime("%d %b"), (d.normalize() - asof.normalize()).days
    except Exception:
        pass
    return None, np.nan


def _clean_json(o, path="payload", bad=None):
    """Replace NaN/inf (and numpy scalars) with JSON-safe values; logs where they were found."""
    top = bad is None
    bad = [] if top else bad
    if isinstance(o, dict):
        o = {k: _clean_json(v, f"{path}.{k}", bad) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        o = [_clean_json(v, f"{path}[]", bad) for v in o]
    elif isinstance(o, (np.bool_,)):
        o = bool(o)
    elif isinstance(o, (np.integer,)):
        o = int(o)
    elif isinstance(o, (float, np.floating)):
        o = float(o)
        if not np.isfinite(o):
            bad.append(path)
            o = None
    if top and bad:
        from collections import Counter
        log("  cleaned non-finite values: " + ", ".join(f"{k} x{n}" for k, n in Counter(bad).most_common(8)))
    return o


def desk_notes(regime, risk_on, n_gauges, breadth, style_chg, L, dropped):
    """Plain-English observations generated by rules from this run's numbers (no AI needed)."""
    notes, n = [], len(L)
    if not n:
        return [{"tag": "Leaders", "text": "No stocks pass the leader filters today — stay in cash and let the list rebuild."}]
    best = max(style_chg, key=style_chg.get)
    sizing = {"Risk-On": "conditions support full position size",
              "Mixed": "trade smaller and take only A+ setups",
              "Defensive Rotation": "money is hiding in defensives — trade smaller, fewer positions",
              "Risk-Off": "no new longs; protect capital"}.get(regime, "trade selectively")
    notes.append({"tag": "Regime", "text": f"{regime}: {risk_on} of {n_gauges} risk-appetite gauges risk-on, {best} leading "
                  f"({style_chg[best]:+.1f}% vs benchmark over 1M), breadth {breadth:.0f}%. Sizing: {sizing}."})

    tc = L.theme.value_counts()
    top3 = tc.head(3)
    share = top3.sum() / n * 100
    conc = "concentrated" if share >= 60 else "spread out"
    notes.append({"tag": "Themes", "text": f"Leadership is {conc}: " + ", ".join(f"{k} {v}" for k, v in top3.items())
                  + f" = {top3.sum()} of {n} leaders ({share:.0f}%)."})

    ext3, ext4 = (L.atr_ext > 3).sum(), (L.atr_ext > 4).sum()
    far = L.atr_ext.idxmax()
    if ext3 / n >= 0.5:
        notes.append({"tag": "Extension", "text": f"Most leaders are extended: {ext3} of {n} are more than 3 ATRs above the 50-DMA "
                      f"({ext4} above 4); {far} is at {L.atr_ext.max():.1f}. Wait for pullbacks to the 10/21-day — don't chase."})
    else:
        notes.append({"tag": "Extension", "text": f"{ext3} of {n} leaders are more than 3 ATRs above the 50-DMA "
                      f"({ext4} above 4) — most are still within reach of their moving averages."})

    soon = L[L.er_days <= 7].sort_values("rs_rating", ascending=False)
    missing = int(L.earnings.isna().sum())
    if len(soon):
        lead = "Peak earnings season. " if len(soon) / n >= 0.3 else ""
        notes.append({"tag": "Earnings", "text": f"{lead}{len(soon)} of {n} leaders report within 7 days, including "
                      + ", ".join(soon.index[:7]) + ". Don't open new positions into the print."
                      + (f" {missing} have no date on Yahoo — check before buying." if missing else "")})

    st = L[L.setup & ~(L.er_days <= 7)].sort_values("rs_rating", ascending=False)
    if len(st):
        notes.append({"tag": "Setups", "text": f"{len(st)} leader{'s' if len(st) > 1 else ''} in a buyable shape "
                      "(near pivot, tight, not extended, no earnings this week): " + ", ".join(st.index[:10]) + "."})
    else:
        notes.append({"tag": "Setups", "text": "No leader is in a buyable setup right now. Build the Focus list from the "
                      "extended names and wait for them to tighten."})

    new = list(L.index[L.new])
    if new or dropped:
        notes.append({"tag": "Changes", "text": (f"New this week: {', '.join(new[:12])}{' …' if len(new) > 12 else ''}. " if new else "")
                      + (f"Dropped off: {', '.join(dropped[:12])}{' …' if len(dropped) > 12 else ''}." if dropped else "")})
    return notes


def pct(a, b):
    return (a / b - 1) * 100


def _growth(now, prev):
    """% growth that stays meaningful when the base is tiny or negative."""
    if now is None or prev is None:
        return None
    if prev <= 0:
        return 999.0 if now > 0 else None          # turned profitable → treated as a pass, shown as "turn"
    return (now / prev - 1) * 100


def _yoy_series(rows, max_gap=45):
    """rows = [[date, value, ...], ...] oldest→newest. YoY % for each row that has a row ~1 year earlier."""
    pts = [(pd.Timestamp(r[0]), r[1]) for r in rows if r[1] is not None]
    out = []
    for d, v in pts:
        prev = [pv for pd_, pv in pts if abs((d - pd_).days - 365) <= max_gap]
        if prev:
            out.append((d, v, _growth(v, prev[0])))
    return out


def canslim(stk: pd.DataFrame, market_ok: bool) -> dict:
    """Adds cs / cs_n / funds / funds_chg columns to stk and returns per-ticker detail for the drawer.

    Each letter is 1 (pass), 0 (fail) or '-' (no data):
      C  latest quarter EPS ≥ +25% YoY                 A  3-year annual EPS growth ≥ 25%/yr
      N  within 15% of the 52-week high                S  up-volume ÷ down-volume (50d) ≥ 1.2
      L  RS rating ≥ 80                                I  more 13F holders than the quarter before
      M  at least half the risk-appetite gauges are risk-on
    """
    fpath, ipath = ROOT / "data" / "fundamentals.json", ROOT / "data" / "institutions.csv"
    F = json.loads(fpath.read_text()) if fpath.exists() else {}
    inst = pd.read_csv(ipath) if ipath.exists() else pd.DataFrame(columns=["ticker", "period", "holders", "shares"])
    periods = sorted(inst.period.unique())[-8:]
    hold = {t: g.set_index("period").holders.reindex(periods).tolist() for t, g in inst.groupby("ticker")} if len(inst) else {}

    det, letters, n_ok, funds, fchg = {}, [], [], [], []
    for t, r in stk.iterrows():
        f = F.get(t, {})
        d = {}
        # C — current quarterly EPS, plus sales and acceleration
        ey = _yoy_series(f.get("eps_q") or [])
        C = None
        if ey:
            C = ey[-1][2] is not None and ey[-1][2] >= 25
            d["eps"] = [[x[0].strftime("%b %y"), round(x[2]) if x[2] is not None else None] for x in ey[-4:]]
            q = (f.get("eps_q") or [])[-1]
            if q[1] is not None and q[2] not in (None, 0):
                d["surprise"] = round((q[1] - q[2]) / abs(q[2]) * 100)
        sy = _yoy_series(f.get("sales_q") or [])
        if sy:
            d["sales"] = [[x[0].strftime("%b %y"), round(x[2]) if x[2] is not None else None] for x in sy[-3:]]
        # A — annual EPS growth over up to 3 years
        A = None
        ea = [x for x in (f.get("eps_a") or []) if x[1] is not None][-4:]
        if len(ea) >= 3:
            a0, a1, yrs = ea[0][1], ea[-1][1], len(ea) - 1
            if a0 > 0 and a1 > 0:
                cagr = ((a1 / a0) ** (1 / yrs) - 1) * 100
                d["eps_cagr"] = round(cagr)
                A = cagr >= 25 and all(ea[i + 1][1] >= ea[i][1] * 0.95 for i in range(yrs))
            else:
                A = False
            d["eps_a"] = ea
        # N, S, L — from prices
        N = None if pd.isna(r.off_high) else bool(r.off_high >= -15)
        S_ = None if pd.isna(r.ud_ratio) else bool(r.ud_ratio >= 1.2)
        L = None if pd.isna(r.rs_rating) else bool(r.rs_rating >= 80)
        # I — 13F holders trend
        I, fn, fc = None, np.nan, np.nan
        h = [x for x in hold.get(t, []) if pd.notna(x)]
        if len(h) >= 2:
            fn, fc = h[-1], (h[-1] / h[-2] - 1) * 100 if h[-2] else np.nan
            I = bool(h[-1] > h[-2])
            d["funds"] = [None if pd.isna(x) else int(x) for x in hold[t]]
        elif len(h) == 1:
            fn = h[0]
        M = bool(market_ok)
        vals = [C, A, N, S_, L, I, M]
        letters.append("".join("-" if v is None else "1" if v else "0" for v in vals))
        n_ok.append(sum(v is True for v in vals))
        funds.append(fn); fchg.append(fc)
        for k in ("fpe", "ps", "float_m", "mcap_b", "inst_pct"):
            if f.get(k) is not None:
                d[k] = f[k]
        if not pd.isna(r.ud_ratio):
            d["ud"] = round(float(r.ud_ratio), 2)
        if f.get("updated"):
            d["upd"] = f["updated"]
        det[t] = d
    stk["cs"], stk["cs_n"], stk["funds"], stk["funds_chg"] = letters, n_ok, funds, fchg

    # valuation context: median forward P/E and P/S of the stock's theme peers (profitable, sane values)
    vals = pd.DataFrame({"theme": stk.theme,
                         "fpe": [F.get(t, {}).get("fpe") for t in stk.index],
                         "ps": [F.get(t, {}).get("ps") for t in stk.index]}, index=stk.index)
    vals["fpe"] = pd.to_numeric(vals.fpe, errors="coerce").where(lambda x: (x > 0) & (x < 400))
    vals["ps"] = pd.to_numeric(vals.ps, errors="coerce").where(lambda x: (x > 0) & (x < 200))
    med = vals.groupby("theme").agg(fpe=("fpe", "median"), ps=("ps", "median"), n=("ps", "count"))
    for t, th in stk.theme.items():
        if th in med.index and med.at[th, "n"] >= 4:
            det[t]["peer"] = [th, round(float(med.at[th, "fpe"]), 1) if pd.notna(med.at[th, "fpe"]) else None,
                              round(float(med.at[th, "ps"]), 1) if pd.notna(med.at[th, "ps"]) else None, int(med.at[th, "n"])]
    det = {t: d for t, d in det.items() if d}
    det["_meta"] = {"periods": [pd.Timestamp(p).strftime("%b %y") for p in periods],
                    "fund_stocks": len(F), "inst_stocks": len(hold), "market_ok": bool(market_ok)}
    log(f"  CANSLIM: fundamentals for {len(F)} stocks, 13F for {len(hold)} ({periods[-1] if periods else 'none'})")
    return det


# ════════════════════════════════════════════════════════════════════
#  ENGINE
# ════════════════════════════════════════════════════════════════════
def broad_universe(known: set, names: pd.DataFrame, demo: bool) -> list[str]:
    """Every US common stock that isn't already in an ETF's holdings and is liquid enough to trade.
    Re-screened weekly (3 months of prices for ~4,000 names), cached in config/broad.csv."""
    if not SETTINGS.get("broad_universe", True):
        return []
    cache = CFG_DIR / "broad.csv"
    fresh = cache.exists() and time.time() - cache.stat().st_mtime < 6 * 86400
    if demo or fresh or not len(names):
        return sorted(set(pd.read_csv(cache).ticker) - known) if cache.exists() else []
    n = names.fillna("")
    bad = n.name.str.contains(r"Warrant|\bUnits?\b|\bRights?\b|Preferred|Notes due|Debenture|Depositary Shares? Each|%|"
                              r"\bETF\b|\bETN\b|\bFund\b|\bTrust\b", case=False, regex=True) | n.industry.eq("Blank Checks")
    cand = sorted(set(n.ticker[n.ticker.str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?") & ~bad]) - known)
    log(f"  broad universe: screening {len(cand)} US common stocks outside your ETF holdings")
    px = download_yahoo(cand, period="3mo", want=("Close", "Volume"))
    c, v = px["Close"], px["Volume"]
    dv = (c * v).tail(20).mean()
    keep = c.columns[(c.iloc[-1] >= SETTINGS["min_price"] * 0.8) & (dv >= SETTINGS["min_dollar_volume"] * 0.5)]
    pd.DataFrame({"ticker": sorted(keep)}).to_csv(cache, index=False)
    log(f"  broad universe: {len(keep)} liquid names added")
    return sorted(keep)


def run(demo: bool, notify: bool):
    etfs = pd.read_csv(CFG_DIR / "etfs.csv")
    uni = pd.read_csv(CFG_DIR / "universe.csv")
    uni["ticker"] = uni["ticker"].str.strip().str.upper()
    etf_set = set(etfs.symbol)
    names = pd.DataFrame(columns=["ticker", "name", "exchange", "sector", "industry"]) if demo else fetch_names()
    if demo and (CFG_DIR / "names.csv").exists():
        names = pd.read_csv(CFG_DIR / "names.csv")
    extras = broad_universe(set(uni.ticker) | etf_set, names, demo)
    stocks = sorted((set(uni.ticker) - etf_set) | set(extras))
    meta = etfs.set_index("symbol")

    # style for each stock: SPDR sector membership first, else first member ETF
    spdr = set(etfs.loc[etfs.category == "SPDR Sector", "symbol"])
    stock_style = {}
    for t, g in uni.groupby("ticker"):
        m = [e for e in g.etf if e in spdr] or list(g.etf)
        stock_style[t] = meta.loc[m[0], "style"] if m[0] in meta.index else "neutral"

    all_tickers = list(etfs.symbol) + stocks
    log(f"Universe: {len(etfs)} ETFs, {len(stocks)} stocks ({len(extras)} outside ETF holdings)")
    if demo:
        styles = {**meta["style"].to_dict(), **stock_style}
        px = demo_prices(all_tickers, styles)
    else:
        px = download_yahoo(all_tickers)

    close, high, low, vol = (px[k].ffill(limit=3) for k in ("Close", "High", "Low", "Volume"))
    if BENCH not in close:
        sys.exit(f"Benchmark {BENCH} has no data — aborting.")
    close = close.loc[close[BENCH].notna()]
    if not demo:                                   # run during the US session? drop today's partial bar
        from zoneinfo import ZoneInfo
        now_et = dt.datetime.now(ZoneInfo("America/New_York"))
        if close.index[-1].date() == now_et.date() and now_et.time() < dt.time(16, 15):
            close = close.iloc[:-1]
            log("  US market still open — using the previous close")
    high, low, vol = high.reindex(close.index), low.reindex(close.index), vol.reindex(close.index)
    asof = close.index[-1]
    log(f"Data as of {asof.date()}, {close.shape[1]} tickers with prices")

    b_close = close[BENCH]
    b_atr = atr(high[[BENCH]], low[[BENCH]], close[[BENCH]])[BENCH]

    E = [s for s in etfs.symbol if s in close]
    S = [s for s in stocks if s in close]

    # ── common per-ticker metrics ───────────────────────────────────
    def base_metrics(cols):
        c, h, l, v = close[cols], high[cols], low[cols], vol[cols]
        rating, vars_, a = rs_rating(c, h, l, b_close, b_atr)
        ret = lambda n: pct(c.iloc[-1], c.iloc[-1 - n])
        bret = lambda n: pct(b_close.iloc[-1], b_close.iloc[-1 - n])
        sma50, sma150, sma200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
        hi52, lo52 = c.rolling(252, min_periods=120).max(), c.rolling(252, min_periods=120).min()
        excess = c.pct_change().sub(b_close.pct_change(), axis=0)
        df = pd.DataFrame(index=cols)
        df["price"] = c.iloc[-1]
        df["chg_1d"] = ret(1)
        df["chg_1w"] = ret(5)
        df["chg_1m"] = ret(21)
        df["chg_3m"] = ret(63)
        df["chg_6m"] = ret(126)
        df["rs_rating"] = rating.iloc[-1]
        df["rs_rating_1w"] = rating.iloc[-6]
        df["rs_chg_1w"] = df.rs_rating - df.rs_rating_1w
        df["rs_chg_3d"] = df.rs_rating - rating.iloc[-4]
        rl = c.div(b_close, axis=0)
        df["rs_line_hi"] = rl.iloc[-1] >= rl.tail(63).max() * 0.995                # RS line at a 3-month high
        df["above_20"] = c.iloc[-1] > c.rolling(20).mean().iloc[-1]
        df["rs_score"] = W3M * (df.chg_3m - bret(63)) + W1M * (df.chg_1m - bret(21)) + W1W * (df.chg_1w - bret(5))
        df["rs_win_1m"] = (excess.tail(20) > 0).sum() / 20 * 100
        df["rs_1m"] = pct(c.iloc[-1] / b_close.iloc[-1], c.iloc[-22] / b_close.iloc[-22])
        df["off_high"] = pct(c.iloc[-1], hi52.iloc[-1])
        df["above_50"] = c.iloc[-1] > sma50.iloc[-1]
        df["adr"] = ((h / l - 1).rolling(20).mean().iloc[-1]) * 100
        df["dollar_vol"] = (c * v).rolling(20).mean().iloc[-1]
        df["rvol"] = v.iloc[-1] / v.rolling(50).mean().shift(1).iloc[-1]
        df["atr_ext"] = (c.iloc[-1] - sma50.iloc[-1]) / a.iloc[-1]
        hi20 = h.rolling(20).max()
        df["pivot_dist"] = pct(c.iloc[-1], hi20.iloc[-1])                       # % below the 20-day high
        rng10 = (h.rolling(10).max() - l.rolling(10).min()).iloc[-1] / c.iloc[-1] * 100
        df["tightness"] = rng10 / df["adr"]                                      # 10-day range in ADRs
        df["vol_dryup"] = v.rolling(10).mean().iloc[-1] / v.rolling(50).mean().iloc[-1]
        d1 = c.diff().tail(50)                                                   # up/down volume, 50 days
        df["ud_ratio"] = v.tail(50).where(d1 > 0).sum() / v.tail(50).where(d1 < 0).sum().replace(0, np.nan)
        df["trend_template"] = (
            (c.iloc[-1] > sma150.iloc[-1]) & (c.iloc[-1] > sma200.iloc[-1]) &
            (sma150.iloc[-1] > sma200.iloc[-1]) & (sma200.iloc[-1] > sma200.iloc[-22]) &
            (sma50.iloc[-1] > sma150.iloc[-1]) & (c.iloc[-1] > sma50.iloc[-1]) &
            (c.iloc[-1] >= 1.3 * lo52.iloc[-1]) & (c.iloc[-1] >= 0.75 * hi52.iloc[-1]) &
            (rating.iloc[-1] >= 70))
        # sparkline series (last 22 sessions)
        df["spark_px"] = [[round(float(x), 4) for x in c[t].tail(22)] for t in cols]
        df["spark_rs"] = [[round(float(x), 6) for x in (c[t] / b_close).tail(22)] for t in cols]
        return df, rating

    # ── ETFs ────────────────────────────────────────────────────────
    etf_df, etf_rating = base_metrics(E)
    etf_df = etf_df.join(meta[["name", "category", "theme", "style"]])
    etf_df["rrg"] = [weekly_rrg(close[t], b_close) for t in E]
    etf_df["quadrant"] = [quadrant(p[-1] if p else None) for p in etf_df.rrg]

    # breadth inside each ETF that has holdings: % of members above 50-DMA
    st_c = close[S]
    above50 = st_c > st_c.rolling(50).mean()
    a50_now = above50.iloc[-1]
    members = uni[uni.ticker.isin(S)].groupby("etf").ticker.apply(list)
    etf_df["breadth_50"] = [round(a50_now[members[e]].mean() * 100) if e in members else np.nan for e in E]
    a50_5d = above50.iloc[-6]
    etf_df["breadth_chg_1w"] = [round((a50_now[members[e]].mean() - a50_5d[members[e]].mean()) * 100) if e in members else np.nan
                                for e in E]

    # rank within category, now and 1 week ago
    for col, src in (("rank", "rs_rating"), ("rank_1w", "rs_rating_1w")):
        etf_df[col] = etf_df.groupby("category")[src].rank(ascending=False, method="first")
    etf_df["rank_chg"] = etf_df.rank_1w - etf_df["rank"]

    # ── Themes (Groups + SPDR sectors), best ETF per theme ─────────
    th = etf_df[etf_df.category.isin(["Group", "SPDR Sector"]) & etf_df.rs_rating.notna()]
    theme_tbl = (th.sort_values("rs_rating", ascending=False).groupby("theme").head(1)
                 .sort_values("rs_rating", ascending=False))
    # only themes with stocks in the universe can feed the leaders list
    themes_with_stocks = set(meta.loc[meta.index.isin(set(uni[uni.ticker.isin(S)].etf)), "theme"])
    top_themes = [x for x in theme_tbl.theme if x in themes_with_stocks][:SETTINGS["top_themes"]]
    theme_score = theme_tbl.set_index("theme").rs_rating.to_dict()
    etf_theme = meta.theme.to_dict()

    # ── Stocks ──────────────────────────────────────────────────────
    stk, stk_rating = base_metrics(S)
    mem = uni[uni.ticker.isin(S)].groupby("ticker").etf.apply(list)
    stk["etfs"] = [mem.get(t, []) for t in stk.index]

    def primary(etf_list):
        themes = sorted({etf_theme.get(e) for e in etf_list if etf_theme.get(e) in theme_score},
                        key=lambda x: -theme_score[x])
        return themes[0] if themes else ""
    stk["theme"] = stk.etfs.apply(primary)
    stk["all_themes"] = stk.etfs.apply(lambda l: sorted({etf_theme.get(e, e) for e in l}))
    stk["style"] = [stock_style.get(t, "neutral") for t in stk.index]
    nm = names.set_index("ticker") if len(names) else pd.DataFrame()
    stk["name"] = [nm.name.get(t, t) if len(nm) else t for t in stk.index]
    stk["industry"] = [nm.industry.get(t, "") if len(nm) else "" for t in stk.index]
    stk["exchange"] = [nm.exchange.get(t, "") if len(nm) else "" for t in stk.index]
    stk["sector"] = [nm.sector.get(t, "") if len(nm) and "sector" in nm else "" for t in stk.index]

    # stocks found outside ETF holdings get a theme from their industry: the theme most of that industry's
    # ETF-held stocks belong to (industry-group themes preferred; sector theme as a fallback)
    stk["broad"] = stk.etfs.str.len() == 0
    if stk.broad.any():
        from collections import Counter
        cat_of = meta.category.to_dict()
        def vote(key):                     # industry → industry-group theme, only when most of the industry agrees
            out = {}
            for k, g in stk[~stk.broad & stk[key].astype(bool)].groupby(key):
                cg = Counter()
                for l in g.etfs:
                    gth = {etf_theme[e] for e in l if cat_of.get(e) == "Group" and e in etf_theme}
                    for x in gth: cg[x] += 1 / len(gth)
                if cg and cg.most_common(1)[0][1] >= max(2, 0.3 * len(g)):
                    out[k] = cg.most_common(1)[0][0]
            return out
        ind_th = vote("industry")
        SEC = {"Technology": "Technology", "Health Care": "Health Care", "Finance": "Financials",
               "Consumer Discretionary": "Consumer Discretionary", "Industrials": "Industrials", "Real Estate": "Real Estate",
               "Energy": "Energy", "Utilities": "Utilities", "Basic Materials": "Materials",
               "Telecommunications": "Communication Services", "Consumer Staples": "Consumer Staples"}
        new_th = {t: [x for x in (ind_th.get(stk.at[t, "industry"]), SEC.get(stk.at[t, "sector"])) if x]
                  for t in stk.index[stk.broad]}
        stk["all_themes"] = [sorted(new_th[t]) if t in new_th else l for t, l in stk.all_themes.items()]
        stk["theme"] = [(sorted(new_th[t], key=lambda x: -theme_score.get(x, 0))[0] if new_th[t] else "")
                        if t in new_th else th for t, th in stk.theme.items()]
        th_style = theme_tbl.set_index("theme")["style"].to_dict()
        stk.loc[stk.broad, "style"] = stk.loc[stk.broad, "theme"].map(th_style).fillna("neutral")
        log(f"  broad stocks: {int(stk.broad.sum())}, of which {sum(bool(v) for v in new_th.values())} mapped to a theme")

    in_top = stk.all_themes.apply(lambda l: bool(set(l) & set(top_themes)))
    elig = ((stk.price >= SETTINGS["min_price"]) & (stk.dollar_vol >= SETTINGS["min_dollar_volume"])
            & (stk.rs_rating >= SETTINGS["min_rs_rating"]))
    if SETTINGS.get("require_above_50dma", True):
        elig &= stk.above_50
    elig &= stk.adr >= SETTINGS.get("min_adr", 2.5)      # drops pinned / takeover / ultra-low-vol names
    leaders = stk[elig & in_top].copy()

    # ── Selection funnel: how many stocks survive each filter (last 40 sessions) ──
    n_hist = 40
    c_s, h_s, l_s, v_s = close[S], high[S], low[S], vol[S]
    st_liq = (c_s >= SETTINGS["min_price"]) & ((c_s * v_s).rolling(20).mean() >= SETTINGS["min_dollar_volume"])
    st_a50 = st_liq & (c_s > c_s.rolling(50).mean())
    st_rs = st_a50 & (stk_rating >= SETTINGS["min_rs_rating"])
    st_adr = st_rs & ((h_s / l_s - 1).rolling(20).mean() * 100 >= SETTINGS.get("min_adr", 2.5))
    funnel_hist = {k: [int(x) for x in m.tail(n_hist).sum(axis=1)] for k, m in
                   (("universe", c_s.notna()), ("liquid", st_liq), ("above50", st_a50), ("rs", st_rs), ("adr", st_adr))}
    funnel_now = {"universe": int(stk.price.notna().sum()),
                  "liquid": int(((stk.price >= SETTINGS["min_price"]) & (stk.dollar_vol >= SETTINGS["min_dollar_volume"])).sum())}
    funnel_now["above50"] = int(((stk.price >= SETTINGS["min_price"]) & (stk.dollar_vol >= SETTINGS["min_dollar_volume"]) & stk.above_50).sum())
    funnel_now["rs"] = int((((stk.price >= SETTINGS["min_price"]) & (stk.dollar_vol >= SETTINGS["min_dollar_volume"]) & stk.above_50)
                            & (stk.rs_rating >= SETTINGS["min_rs_rating"])).sum())
    funnel_now["adr"] = int(elig.sum())
    funnel_now["theme"] = int((elig & in_top).sum())
    stk["setup_all"] = ((stk.pivot_dist >= -6) & (stk.tightness <= 2.5) & (stk.atr_ext.between(0, 4)))
    leaders["theme"] = leaders.all_themes.apply(
        lambda l: sorted([x for x in l if x in top_themes], key=lambda x: top_themes.index(x))[0])
    leaders = leaders.sort_values(["rs_rating", "rs_score"], ascending=False).head(SETTINGS["max_leaders"])
    leaders["near_high"] = leaders.off_high >= -15
    leaders["extended"] = leaders.atr_ext > 4
    # Setup = buyable shape: near a pivot, tight, not extended (a screen, not a signal)
    leaders["setup"] = ((leaders.pivot_dist >= -6) & (leaders.tightness <= 2.5)
                        & (leaders.atr_ext.between(0, 4)))
    leaders["earnings"], leaders["er_days"] = None, np.nan
    if not demo and len(leaders):
        try:
            ecal = earnings_calendar(asof)
            er = [next_earnings(t, asof, ecal) for t in leaders.index]
            leaders["earnings"] = [e[0] for e in er]
            leaders["er_days"] = [e[1] for e in er]
        except Exception as e:
            log(f"  earnings lookup skipped ({e})")
    else:
        rng_ = np.random.default_rng(3)
        leaders["er_days"] = rng_.integers(1, 80, len(leaders))
        leaders["earnings"] = [(asof + pd.Timedelta(days=int(d))).strftime("%d %b") for d in leaders.er_days]
    leaders["er_soon"] = leaders.er_days <= 7

    funnel_now["setups"] = int(leaders.setup.sum())            # same count as the table's "Setups only" filter
    setups_er = int((leaders.setup & leaders.er_soon).sum())
    HIST.mkdir(parents=True, exist_ok=True)
    fpath_ = HIST / "funnel.csv"
    frow = pd.DataFrame([{"date": asof.strftime("%Y-%m-%d"), **funnel_now}])
    flog = pd.read_csv(fpath_) if fpath_.exists() else pd.DataFrame(columns=frow.columns)
    flog = pd.concat([flog[flog.date != frow.date[0]], frow]).sort_values("date").tail(400)
    flog.to_csv(fpath_, index=False)
    for k in ("theme", "setups"):                 # these depend on the day's top themes: from the daily log
        funnel_hist[k] = [int(x) for x in flog[k].dropna().tail(n_hist)]
    for k in ("universe", "liquid", "above50", "rs", "adr"):
        funnel_hist[k][-1] = funnel_now[k]
    funnel = [{"key": k, "label": lbl, "n": funnel_now[k], "hist": funnel_hist[k],
               "chg_1w": (funnel_now[k] - funnel_hist[k][-6]) if len(funnel_hist[k]) >= 6 else None,
               "note": (f"{setups_er} report ≤ 7d" if k == "setups" and setups_er else None)}
              for k, lbl in (("universe", "Universe"), ("liquid", "Liquid"), ("above50", "Above 50-day"),
                             ("rs", f"RS ≥ {SETTINGS['min_rs_rating']}"), ("adr", f"ADR ≥ {SETTINGS.get('min_adr', 2.5)}%"),
                             ("theme", "In a top theme"), ("setups", "Setups"))]

    # ── Early turners: groups outside the top themes whose RS is turning up fast ──
    # Watchlist only. A group qualifies when its RS jumped (1 week or 3 days), it is back above its
    # 20-day, and the move is confirmed by rising breadth or an Improving/Leading rotation quadrant.
    T = SETTINGS.get("turners", {})
    tg_ = etf_df[etf_df.category.isin(["Group", "SPDR Sector"]) & etf_df.index.isin(members.index)
                 & ~etf_df.theme.isin(top_themes) & etf_df.rs_rating.notna()].copy()
    jump = (tg_.rs_chg_1w >= T.get("min_rs_jump_1w", 12)) | (tg_.rs_chg_3d >= T.get("min_rs_jump_3d", 8))
    confirm = (tg_.breadth_chg_1w >= T.get("min_breadth_rise", 10)) | tg_.quadrant.isin(["Improving", "Leading"])
    tg_ = tg_[jump & confirm & tg_.above_20 & (tg_.rs_rating >= T.get("min_group_rs", 40))]
    tg_["turn_score"] = tg_.rs_chg_1w.clip(lower=0) + tg_.rs_chg_3d.clip(lower=0) + tg_.breadth_chg_1w.fillna(0).clip(lower=0) / 2
    tg_ = tg_.sort_values("turn_score", ascending=False).groupby("theme").head(1)
    turners, seen_ = [], []
    for e_, g_ in tg_.iterrows():
        mem_ = [t for t in members.get(e_, []) if t in stk.index]
        mem_ += [t for t in stk.index[stk.broad] if g_.theme in stk.at[t, "all_themes"]]
        # skip near-duplicate groups (e.g. IYR vs XLRE): mostly the same stocks as one already listed
        if any(len(set(mem_) & s_) > 0.5 * min(len(mem_), len(s_)) for s_ in seen_ if s_ and mem_):
            continue
        if len(turners) >= T.get("max_groups", 5):
            break
        seen_.append(set(mem_))
        cand = stk.loc[mem_]
        cand = cand[(cand.price >= SETTINGS["min_price"]) & (cand.dollar_vol >= SETTINGS["min_dollar_volume"])
                    & cand.above_50 & (cand.rs_rating >= T.get("min_stock_rs", 70)) & (cand.rs_chg_1w > 0)]
        cand = cand.assign(_k=cand.rs_line_hi.astype(int)).sort_values(["_k", "rs_rating"], ascending=False)
        turners.append({"etf": e_, "theme": g_.theme, "name": g_["name"], "style": g_["style"],
                        "rs": g_.rs_rating, "rs_chg_1w": g_.rs_chg_1w, "rs_chg_3d": g_.rs_chg_3d,
                        "breadth": g_.breadth_50, "breadth_chg": g_.breadth_chg_1w, "quadrant": g_.quadrant,
                        "stocks": list(cand.index[:T.get("max_stocks", 5)])})

    # NEW vs the snapshot from about a week ago
    HIST.mkdir(parents=True, exist_ok=True)
    lh_path = HIST / "leaders.csv"
    prev = set()
    lh = pd.DataFrame(columns=["ticker", "theme", "rs_rating", "rs_score", "price", "date"])
    if lh_path.exists():
        lh = pd.read_csv(lh_path, parse_dates=["date"])
        old = lh[lh.date <= asof - pd.Timedelta(days=6)]
        if len(old):
            prev = set(old[old.date == old.date.max()].ticker)
    leaders["new"] = [bool(prev) and t not in prev for t in leaders.index]
    dropped = sorted(prev - set(leaders.index)) if prev else []

    # ── Leader history: last N weeks of the daily log + today ──────
    hist_weeks = int(SETTINGS.get("history_weeks", 8))
    today_rows = leaders[["theme", "rs_rating", "price"]].assign(date=asof.normalize()).rename_axis("ticker").reset_index()
    lh_all = pd.concat([lh[lh.date != asof.normalize()], today_rows]) if lh_path.exists() else today_rows
    lh_all = lh_all.dropna(subset=["ticker"])
    first_seen = lh_all.groupby("ticker").date.min()
    first_px = lh_all.sort_values("date").groupby("ticker").price.first()
    recent = lh_all[lh_all.date > asof - pd.Timedelta(weeks=hist_weeks)].copy()
    recent["wk"] = recent.date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    week_ends = sorted(recent.wk.unique())[-hist_weeks:]
    lastday = recent.sort_values("date").groupby(["ticker", "wk"]).tail(1)   # each week's last snapshot
    history = []
    for t, g in recent.groupby("ticker"):
        wk_rs = {w: r for w, r in zip(lastday[lastday.ticker == t].wk, lastday[lastday.ticker == t].rs_rating)}
        px_now = close[t].iloc[-1] if t in close else np.nan
        history.append({"symbol": t, "theme": g.sort_values("date").theme.iloc[-1],
                        "exchange": (names.set_index("ticker").exchange.get(t) if len(names) else None),
                        "first_seen": first_seen[t].strftime("%d %b"), "first_seen_iso": first_seen[t].strftime("%Y-%m-%d"),
                        "days_on": int(g.date.nunique()), "on_now": t in leaders.index,
                        "ret_since": pct(px_now, first_px[t]) if first_px.get(t) else np.nan,
                        "weeks": [wk_rs.get(w) for w in week_ends]})
    history.sort(key=lambda h: (-h["on_now"], -h["days_on"]))
    hist_week_labels = [pd.Timestamp(w).strftime("%d %b") for w in week_ends]

    # ── Market regime: risk-appetite gauges & style rotation ───────
    def basket(cols):
        cols = [c for c in cols if c in close]
        return (close[cols] / close[cols].iloc[0]).mean(axis=1)

    ew = etf_df[etf_df.category == "EW Sector"]
    style_basket = {s: basket(list(ew[ew["style"] == s].index)) for s in STYLE_ORDER}

    def ratio(num, den, label, note):
        n = style_basket[num] if num in style_basket else close.get(num)
        d = style_basket[den] if den in style_basket else close.get(den)
        if n is None or d is None:
            return None
        r = (n / d).dropna()
        ma = r.rolling(50).mean()
        s = r.tail(126)
        return {"label": label, "note": note,
                "series": [round(float(x / s.iloc[0] * 100), 2) for x in s],
                "ma": [round(float(x / s.iloc[0] * 100), 2) if pd.notna(x) else None for x in ma.tail(126)],
                "above_ma": bool(r.iloc[-1] > ma.iloc[-1]),
                "chg_1m": round(float(pct(r.iloc[-1], r.iloc[-22])), 2),
                "risk_on": bool(r.iloc[-1] > ma.iloc[-1] and r.iloc[-1] > r.iloc[-22])}

    gauges = [g for g in [
        ratio("growth", "defensive", "Growth vs Defensive", "EW tech/discretionary/comms vs staples/utilities/health/REITs"),
        ratio("cyclical", "defensive", "Cyclical vs Defensive", "EW financials/industrials/materials/energy vs defensives"),
        ratio("XLY", "XLP", "Discretionary vs Staples", "XLY / XLP — classic risk appetite"),
        ratio("RSP", "SPY", "Equal vs Cap Weight", "RSP / SPY — is breadth broadening?"),
        ratio("IWM", "SPY", "Small vs Large", "IWM / SPY — appetite for small caps"),
        ratio("SPMO", "RSP", "Momentum vs Market", "SPMO / RSP — momentum leadership"),
        ratio("HYG", "IEF", "Credit Appetite", "HYG / IEF — junk bonds vs Treasuries"),
    ] if g]
    risk_on_n = sum(g["risk_on"] for g in gauges)

    rel = {s: (style_basket[s] / (b_close / b_close.iloc[0])).tail(126) for s in STYLE_ORDER}
    style_lines = {s: [round(float(x / v.iloc[0] * 100), 2) for x in v] for s, v in rel.items()}
    style_chg = {s: round(float(pct(v.iloc[-1], v.iloc[-22])), 2) for s, v in rel.items()}
    dates126 = [d.strftime("%d %b") for d in close.index[-126:]]

    # participation: % of universe stocks above 50-DMA, overall and by style
    part = {"all": [round(float(x) * 100, 1) for x in above50.mean(axis=1).tail(126)]}
    for s in STYLE_ORDER:
        cols = [t for t in S if stock_style.get(t) == s]
        if cols:
            part[s] = [round(float(x) * 100, 1) for x in above50[cols].mean(axis=1).tail(126)]

    lead_style = max(style_chg, key=style_chg.get)
    breadth_now, breadth_1m = part["all"][-1], part["all"][-22]
    if risk_on_n >= 5:
        regime, tone = "Risk-On", "bull"
    elif risk_on_n <= 1 and breadth_now < 40:
        regime, tone = "Risk-Off", "bear"
    elif lead_style == "defensive":
        regime, tone = "Defensive Rotation", "warn"
    else:
        regime, tone = "Mixed", "warn"
    verdict = (f"{risk_on_n} of {len(gauges)} risk-appetite gauges are risk-on. "
               f"Over the past month {lead_style} stocks led ({style_chg[lead_style]:+.1f}% vs {BENCH}); "
               f"growth {style_chg['growth']:+.1f}%, cyclical {style_chg['cyclical']:+.1f}%, defensive {style_chg['defensive']:+.1f}%. "
               f"Universe breadth: {breadth_now:.0f}% above 50-DMA ({round(breadth_now - breadth_1m) + 0:+d} pts in a month).")
    if lead_style == "defensive" and breadth_now > breadth_1m:
        verdict += " Breadth is improving, but the participation is coming from defensives — rotation, not broad risk appetite."

    # ── rotation heatmap: weekly RS rating within sector sets ──────
    def heat(cat):
        cols = [c for c in etf_df[etf_df.category == cat].index]
        r, _, _ = rs_rating(close[cols], high[cols], low[cols], b_close, b_atr)
        w = r.resample("W-FRI").last().tail(26)
        order = etf_df.loc[cols].sort_values("rs_rating", ascending=False).index
        return {"weeks": [d.strftime("%d %b") for d in w.index],
                "rows": [{"symbol": t, "name": meta.loc[t, "name"], "style": meta.loc[t, "style"],
                          "values": [None if pd.isna(x) else int(x) for x in w[t]]} for t in order]}
    heatmaps = {"SPDR Sector": heat("SPDR Sector"), "EW Sector": heat("EW Sector")}
    tg = theme_tbl.head(15).index.tolist()
    if len(tg) >= 3:
        r, _, _ = rs_rating(close[tg], high[tg], low[tg], b_close, b_atr)
        w = r.resample("W-FRI").last().tail(26)
        heatmaps["Top Themes"] = {"weeks": [d.strftime("%d %b") for d in w.index],
                                  "rows": [{"symbol": t, "name": meta.loc[t, "theme"], "style": meta.loc[t, "style"],
                                            "values": [None if pd.isna(x) else int(x) for x in w[t]]} for t in tg]}

    # ── CANSLIM + fund ownership + valuation context ───────────────
    cs_detail = canslim(stk, risk_on_n >= len(gauges) / 2)
    for c_ in ("cs", "cs_n", "funds", "funds_chg"):
        leaders[c_] = stk.loc[leaders.index, c_]

    # ════════════════════════════════════════════════════════════════
    #  OUTPUTS
    # ════════════════════════════════════════════════════════════════
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    def rnd(v, d=1):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if not np.isfinite(v) else (int(round(v)) if d == 0 else round(v, d))

    def clean_list(xs):
        return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in xs]

    def etf_rec(t, r):
        return {"symbol": t, "name": r["name"], "category": r.category, "theme": r.theme, "style": r["style"],
                "rs_rating": rnd(r.rs_rating, 0), "rs_chg_1w": rnd(r.rs_chg_1w, 0), "rank": rnd(r["rank"], 0),
                "rank_chg": rnd(r.rank_chg, 0), "rs_win_1m": rnd(r.rs_win_1m, 0), "rs_1m": rnd(r.rs_1m),
                "chg_1d": rnd(r.chg_1d), "chg_1w": rnd(r.chg_1w), "chg_1m": rnd(r.chg_1m), "chg_3m": rnd(r.chg_3m),
                "off_high": rnd(r.off_high), "above_50": bool(r.above_50), "breadth_50": rnd(r.breadth_50, 0),
                "quadrant": r.quadrant, "rrg": r.rrg, "spark_px": clean_list(r.spark_px),
                "spark_rs": clean_list(r.spark_rs), "breadth_chg_1w": rnd(r.breadth_chg_1w, 0)}

    def stk_rec(t, r):
        return {"symbol": t, "name": r["name"], "theme": r.theme, "themes": r.all_themes, "industry": r.industry,
                "exchange": r.exchange, "style": r["style"], "rs_rating": rnd(r.rs_rating, 0),
                "rs_chg_1w": rnd(r.rs_chg_1w, 0), "rs_win_1m": rnd(r.rs_win_1m, 0), "rs_score": rnd(r.rs_score),
                "chg_1w": rnd(r.chg_1w), "chg_1m": rnd(r.chg_1m), "chg_3m": rnd(r.chg_3m), "off_high": rnd(r.off_high),
                "adr": rnd(r.adr), "dollar_vol": rnd(r.dollar_vol / 1e6, 1), "rvol": rnd(r.rvol, 2),
                "atr_ext": rnd(r.atr_ext), "trend_template": bool(r.trend_template),
                "near_high": bool(r.near_high), "extended": bool(r.extended), "new": bool(r.new),
                "setup": bool(r.setup), "pivot_dist": rnd(r.pivot_dist), "tightness": rnd(r.tightness),
                "vol_dryup": rnd(r.vol_dryup, 2), "earnings": r.earnings, "er_days": rnd(r.er_days, 0),
                "er_soon": bool(r.er_soon), "broad": bool(r.broad), "cs": r.cs, "cs_n": int(r.cs_n),
                "funds": rnd(r.funds, 0), "funds_chg": rnd(r.funds_chg),
                "price": rnd(r.price, 2), "spark_px": clean_list(r.spark_px)}

    def turn_rec(t):
        r = stk.loc[t]
        return {"symbol": t, "name": r["name"], "exchange": r.exchange, "rs_rating": rnd(r.rs_rating, 0),
                "rs_chg_1w": rnd(r.rs_chg_1w, 0), "rs_win_1m": rnd(r.rs_win_1m, 0), "chg_1m": rnd(r.chg_1m),
                "off_high": rnd(r.off_high), "adr": rnd(r.adr), "atr_ext": rnd(r.atr_ext), "pivot_dist": rnd(r.pivot_dist),
                "tightness": rnd(r.tightness), "price": rnd(r.price, 2), "rs_line_hi": bool(r.rs_line_hi),
                "dollar_vol": rnd(r.dollar_vol / 1e6, 1), "broad": bool(r.broad),
                "rs_first": bool(r.rs_line_hi and r.off_high < -5), "setup": bool(r.setup_all),
                "leader": t in leaders.index, "cs": r.cs, "funds": rnd(r.funds, 0), "funds_chg": rnd(r.funds_chg),
                "spark_px": clean_list(r.spark_px)}

    turners_out = [{**{k: (rnd(v, 0) if k in ("rs", "rs_chg_1w", "rs_chg_3d", "breadth", "breadth_chg") else v)
                       for k, v in g.items() if k != "stocks"},
                    "stocks": [turn_rec(t) for t in g["stocks"]]} for g in turners]

    etf_sorted = etf_df.sort_values(["category", "rs_rating"], ascending=[True, False])
    payload = {
        "asof": asof.strftime("%a %d %b %Y"),
        "generated": dt.datetime.now(SGT).strftime("%d %b %Y %H:%M SGT"),
        "demo": demo,
        "benchmark": BENCH,
        "settings": {**{k: SETTINGS[k] for k in ("min_dollar_volume", "min_price", "min_rs_rating", "top_themes")},
                     "min_adr": SETTINGS.get("min_adr", 2.5)},
        "regime": {"label": regime, "tone": tone, "verdict": verdict, "risk_on": risk_on_n, "gauges_n": len(gauges),
                   "breadth": breadth_now, "breadth_chg": round(breadth_now - breadth_1m, 1) + 0.0},
        "gauges": gauges,
        "style": {"dates": dates126, "lines": style_lines, "chg": style_chg, "participation": part},
        "etfs": [etf_rec(t, r) for t, r in etf_sorted.iterrows() if r.category != "Macro"],
        "themes": [{"theme": r.theme, "symbol": t, "rs_rating": rnd(r.rs_rating, 0), "style": r["style"],
                    "rs_chg_1w": rnd(r.rs_chg_1w, 0), "top": r.theme in top_themes,
                    "has_stocks": r.theme in themes_with_stocks}
                   for t, r in theme_tbl.iterrows()],
        "leaders": [stk_rec(t, r) for t, r in leaders.iterrows()],
        "turners": turners_out,
        # every stock in the universe, compact, so any ETF row can expand to its holdings
        "stocks": {t: [r["name"], r.exchange, rnd(r.rs_rating, 0), rnd(r.rs_chg_1w, 0), rnd(r.chg_1m), rnd(r.off_high),
                       rnd(r.adr), rnd(r.atr_ext), int(bool(r.above_50)), int(bool(r.trend_template)),
                       int(bool(r.setup_all)), int(t in leaders.index), rnd(r.rs_win_1m, 0),
                       r.cs, rnd(r.funds, 0), rnd(r.funds_chg)]
                   for t, r in stk.iterrows()},
        "members": {e: sorted(set(g)) for e, g in uni[uni.ticker.isin(S)].groupby("etf").ticker},
        "weights": ({e: {r.ticker: float(r.weight) for r in g.itertuples() if pd.notna(r.weight)}
                     for e, g in uni[uni.ticker.isin(S)].groupby("etf")} if "weight" in uni.columns else {}),
        "notes": desk_notes(regime, risk_on_n, len(gauges), breadth_now, style_chg, leaders, dropped),
        "history": {"weeks": hist_week_labels, "rows": history, "dropped": dropped,
                    "log_start": first_seen.min().strftime("%d %b %Y") if len(first_seen) else None},
        "heatmaps": heatmaps,
        "universe": {"etfs": len(E), "stocks": len(S)},
        "funnel": funnel,
        "fund_meta": cs_detail.pop("_meta", {}),
    }

    tpl = (ROOT / "templates" / "dashboard.html").read_text()
    payload = _clean_json(payload)
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), allow_nan=False)
                       .replace("</", "<\\/"))
    (DOCS / "index.html").write_text(html)
    (DOCS / ".nojekyll").write_text("")
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "fund.json").write_text(json.dumps(_clean_json(cs_detail), separators=(",", ":"), allow_nan=False))

    # CSVs for Google Sheets
    cols_e = ["category", "theme", "name", "rs_rating", "rs_chg_1w", "rank", "rank_chg", "rs_win_1m", "rs_1m",
              "chg_1d", "chg_1w", "chg_1m", "chg_3m", "off_high", "breadth_50", "quadrant", "style"]
    for df_ in (etf_sorted, leaders, stk):
        for c_ in ("rs_rating", "rs_chg_1w"):
            df_[c_] = df_[c_].round().astype("Int64")
    etf_sorted[cols_e].round(2).rename_axis("symbol").to_csv(DATA_OUT / "sectors.csv")
    cols_s = ["name", "theme", "rs_rating", "rs_chg_1w", "rs_win_1m", "rs_score", "price", "chg_1w", "chg_1m",
              "chg_3m", "off_high", "adr", "dollar_vol", "rvol", "atr_ext", "pivot_dist", "tightness", "vol_dryup",
              "setup", "earnings", "trend_template", "near_high",
              "extended", "new", "industry"]
    leaders[cols_s].round(2).rename_axis("ticker").to_csv(DATA_OUT / "leaders.csv")
    allc = ["name", "theme", "rs_rating", "rs_chg_1w", "rs_win_1m", "rs_score", "price", "chg_1w", "chg_1m",
            "chg_3m", "off_high", "adr", "dollar_vol", "atr_ext", "trend_template", "above_50", "cs", "funds", "funds_chg"]
    out_all = stk[allc].copy()
    out_all["etfs"] = stk.etfs.apply(lambda l: " ".join(l))
    out_all.sort_values("rs_rating", ascending=False).round(2).rename_axis("ticker").to_csv(DATA_OUT / "stocks.csv")

    # TradingView import file, one section per theme
    lines = []
    for th_name, g in leaders.groupby("theme", sort=False):
        lines.append(f"###{th_name}")
        lines.append(",".join(f"{r.exchange}:{t}" if r.exchange else t for t, r in g.iterrows()))
    for g in turners:
        if g["stocks"]:
            lines.append(f"###Turning: {g['theme']}")
            lines.append(",".join(f"{stk.at[t, 'exchange']}:{t}" if stk.at[t, "exchange"] else t for t in g["stocks"]))
    (DATA_OUT / "tradingview_leaders.txt").write_text("\n".join(lines) + "\n")

    # history snapshots (one row-set per day)
    d = asof.strftime("%Y-%m-%d")
    snap_e = etf_df[["category", "rs_rating", "rank", "rs_score"]].assign(date=d).rename_axis("symbol").reset_index()
    snap_cols = ["theme", "rs_rating", "rs_score", "price", "atr_ext", "pivot_dist", "tightness", "adr",
                 "setup", "extended", "er_soon", "trend_template", "new"]
    snap_l = (leaders[snap_cols].round(2).assign(date=d, regime=regime, bench_price=round(float(b_close.iloc[-1]), 2))
              .rename_axis("ticker").reset_index())
    for path, snap in ((HIST / "etf_ratings.csv", snap_e), (lh_path, snap_l)):
        if path.exists():
            old = pd.read_csv(path)
            snap = pd.concat([old[old.date != d], snap])
        snap.to_csv(path, index=False)

    log(f"Regime: {regime} | top themes: {', '.join(top_themes)} | leaders: {len(leaders)}")
    log(f"Wrote {DOCS / 'index.html'}")

    # ── Alerts: what changed since the previous run ────────────────
    state_path = HIST / "state.json"
    setups_now = sorted(leaders.index[leaders.setup & ~leaders.er_soon])
    old_state = {}
    if state_path.exists():
        try:
            old_state = json.loads(state_path.read_text())
        except ValueError:
            old_state = {}
    if old_state.get("date") == d:                 # re-run of the same day: compare with the day before
        old_state = old_state.get("prev") or {}
    alert = []
    if old_state:
        if old_state.get("regime") and old_state["regime"] != regime:
            alert.append(f"Regime changed: {old_state['regime']} → {regime} ({risk_on_n}/{len(gauges)} gauges risk-on)")
        fresh = [t for t in setups_now if t not in set(old_state.get("setups", []))]
        if fresh:
            alert.append("New setups: " + ", ".join(
                f"{t} (RS {int(leaders.at[t, 'rs_rating'])}, {leaders.at[t, 'pivot_dist']:+.1f}% from pivot)" for t in fresh[:8]))
        turn_old = set(old_state.get("turners", []))
        fresh_t = [g for g in turners if g["etf"] not in turn_old] if "turners" in old_state else []
        if fresh_t:
            alert.append("Early turner" + ("s" if len(fresh_t) > 1 else "") + ": " + "; ".join(
                f"{g['theme']} ({g['etf']}, RS {int(g['rs'])}, {int(g['rs_chg_1w']):+d} in a week"
                + (f", breadth {int(g['breadth'])}% {int(g['breadth_chg']):+d}" if pd.notna(g['breadth_chg']) else "") + ")"
                + (" → watch " + ", ".join(g["stocks"][:4]) if g["stocks"] else "") for g in fresh_t[:4]))
    new_state = {"date": d, "regime": regime, "setups": setups_now, "turners": [g["etf"] for g in turners]}
    new_state["prev"] = (old_state if old_state.get("date") != d else None) or None
    new_state["prev"] = {k: v for k, v in (new_state["prev"] or {}).items() if k != "prev"} or None
    state_path.write_text(json.dumps(new_state, indent=1))
    if alert:
        log("  alert: " + " | ".join(alert))

    # ── Telegram ────────────────────────────────────────────────────
    is_saturday = dt.datetime.now(SGT).weekday() == 5
    if notify or is_saturday or SETTINGS.get("telegram_daily"):
        from notify import send_summary
        send_summary(payload, DATA_OUT / "tradingview_leaders.txt", SETTINGS)
    elif alert and not demo and SETTINGS.get("telegram_alerts", True):
        from notify import send_alert
        send_alert(payload, alert, SETTINGS)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use simulated prices (offline test)")
    ap.add_argument("--notify", action="store_true", help="send the Telegram summary now")
    args = ap.parse_args()
    run(demo=args.demo, notify=args.notify or os.environ.get("FORCE_NOTIFY", "").lower() == "true")
