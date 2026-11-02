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


def download_yahoo(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Batch-download daily OHLCV. Returns wide frames keyed by field."""
    import yfinance as yf

    fields = {"Close": {}, "High": {}, "Low": {}, "Volume": {}}
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


def next_earnings(ticker: str, asof) -> tuple:
    """Next earnings date from Yahoo (best effort). Returns ('12 Nov', days_away) or (None, nan)."""
    try:
        import yfinance as yf
        cal = yf.Ticker(yf_symbol(ticker)).calendar or {}
        dates = [pd.Timestamp(d) for d in cal.get("Earnings Date", []) if d is not None]
        future = [d for d in dates if d.normalize() >= asof.normalize()]
        if future:
            d = min(future)
            return d.strftime("%d %b"), (d.normalize() - asof.normalize()).days
    except Exception:
        pass
    return None, np.nan


def pct(a, b):
    return (a / b - 1) * 100


# ════════════════════════════════════════════════════════════════════
#  ENGINE
# ════════════════════════════════════════════════════════════════════
def run(demo: bool, notify: bool):
    etfs = pd.read_csv(CFG_DIR / "etfs.csv")
    uni = pd.read_csv(CFG_DIR / "universe.csv")
    uni["ticker"] = uni["ticker"].str.strip().str.upper()
    etf_set = set(etfs.symbol)
    stocks = sorted(set(uni.ticker) - etf_set)
    meta = etfs.set_index("symbol")

    # style for each stock: SPDR sector membership first, else first member ETF
    spdr = set(etfs.loc[etfs.category == "SPDR Sector", "symbol"])
    stock_style = {}
    for t, g in uni.groupby("ticker"):
        m = [e for e in g.etf if e in spdr] or list(g.etf)
        stock_style[t] = meta.loc[m[0], "style"] if m[0] in meta.index else "neutral"

    all_tickers = list(etfs.symbol) + stocks
    log(f"Universe: {len(etfs)} ETFs, {len(stocks)} stocks")
    if demo:
        styles = {**meta["style"].to_dict(), **stock_style}
        px = demo_prices(all_tickers, styles)
        names = pd.DataFrame(columns=["ticker", "name", "exchange", "sector", "industry"])
    else:
        px = download_yahoo(all_tickers)
        names = fetch_names()

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

    in_top = stk.all_themes.apply(lambda l: bool(set(l) & set(top_themes)))
    elig = ((stk.price >= SETTINGS["min_price"]) & (stk.dollar_vol >= SETTINGS["min_dollar_volume"])
            & (stk.rs_rating >= SETTINGS["min_rs_rating"]))
    if SETTINGS.get("require_above_50dma", True):
        elig &= stk.above_50
    elig &= stk.adr >= SETTINGS.get("min_adr", 2.5)      # drops pinned / takeover / ultra-low-vol names
    leaders = stk[elig & in_top].copy()
    leaders["theme"] = leaders.all_themes.apply(
        lambda l: sorted([x for x in l if x in top_themes], key=lambda x: top_themes.index(x))[0])
    leaders = leaders.sort_values(["rs_rating", "rs_score"], ascending=False).head(SETTINGS["max_leaders"])
    leaders["near_high"] = leaders.off_high >= -15
    leaders["extended"] = leaders.atr_ext > 4
    # Setup = buyable shape: near a pivot, tight, not extended (a screen, not a signal)
    leaders["setup"] = ((leaders.pivot_dist >= -6) & (leaders.tightness <= 2.5)
                        & (leaders.atr_ext.between(0, 4)))
    leaders["earnings"], leaders["er_days"] = None, np.nan
    if not demo:
        leaders["earnings"], leaders["er_days"] = zip(*[next_earnings(t, asof) for t in leaders.index])
    else:
        rng_ = np.random.default_rng(3)
        leaders["er_days"] = rng_.integers(1, 80, len(leaders))
        leaders["earnings"] = [(asof + pd.Timedelta(days=int(d))).strftime("%d %b") for d in leaders.er_days]
    leaders["er_soon"] = leaders.er_days <= 7

    # NEW vs the snapshot from about a week ago
    HIST.mkdir(parents=True, exist_ok=True)
    lh_path = HIST / "leaders.csv"
    prev = set()
    if lh_path.exists():
        lh = pd.read_csv(lh_path, parse_dates=["date"])
        old = lh[lh.date <= asof - pd.Timedelta(days=6)]
        if len(old):
            prev = set(old[old.date == old.date.max()].ticker)
    leaders["new"] = [bool(prev) and t not in prev for t in leaders.index]

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

    # ════════════════════════════════════════════════════════════════
    #  OUTPUTS
    # ════════════════════════════════════════════════════════════════
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    def rnd(v, d=1):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(v) else (int(round(v)) if d == 0 else round(v, d))

    def clean_list(xs):
        return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in xs]

    def etf_rec(t, r):
        return {"symbol": t, "name": r["name"], "category": r.category, "theme": r.theme, "style": r["style"],
                "rs_rating": rnd(r.rs_rating, 0), "rs_chg_1w": rnd(r.rs_chg_1w, 0), "rank": rnd(r["rank"], 0),
                "rank_chg": rnd(r.rank_chg, 0), "rs_win_1m": rnd(r.rs_win_1m, 0), "rs_1m": rnd(r.rs_1m),
                "chg_1d": rnd(r.chg_1d), "chg_1w": rnd(r.chg_1w), "chg_1m": rnd(r.chg_1m), "chg_3m": rnd(r.chg_3m),
                "off_high": rnd(r.off_high), "above_50": bool(r.above_50), "breadth_50": rnd(r.breadth_50, 0),
                "quadrant": r.quadrant, "rrg": r.rrg, "spark_px": clean_list(r.spark_px),
                "spark_rs": clean_list(r.spark_rs)}

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
                "er_soon": bool(r.er_soon),
                "price": rnd(r.price, 2), "spark_px": clean_list(r.spark_px)}

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
        "heatmaps": heatmaps,
        "universe": {"etfs": len(E), "stocks": len(S)},
    }

    tpl = (ROOT / "templates" / "dashboard.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), allow_nan=False)
                       .replace("</", "<\\/"))
    (DOCS / "index.html").write_text(html)
    (DOCS / ".nojekyll").write_text("")

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
            "chg_3m", "off_high", "adr", "dollar_vol", "atr_ext", "trend_template", "above_50"]
    out_all = stk[allc].copy()
    out_all["etfs"] = stk.etfs.apply(lambda l: " ".join(l))
    out_all.sort_values("rs_rating", ascending=False).round(2).rename_axis("ticker").to_csv(DATA_OUT / "stocks.csv")

    # TradingView import file, one section per theme
    lines = []
    for th_name, g in leaders.groupby("theme", sort=False):
        lines.append(f"###{th_name}")
        lines.append(",".join(f"{r.exchange}:{t}" if r.exchange else t for t, r in g.iterrows()))
    (DATA_OUT / "tradingview_leaders.txt").write_text("\n".join(lines) + "\n")

    # history snapshots (one row-set per day)
    d = asof.strftime("%Y-%m-%d")
    snap_e = etf_df[["category", "rs_rating", "rank", "rs_score"]].assign(date=d).rename_axis("symbol").reset_index()
    snap_l = leaders[["theme", "rs_rating", "rs_score", "price"]].assign(date=d).rename_axis("ticker").reset_index()
    for path, snap in ((HIST / "etf_ratings.csv", snap_e), (lh_path, snap_l)):
        if path.exists():
            old = pd.read_csv(path)
            snap = pd.concat([old[old.date != d], snap])
        snap.to_csv(path, index=False)

    log(f"Regime: {regime} | top themes: {', '.join(top_themes)} | leaders: {len(leaders)}")
    log(f"Wrote {DOCS / 'index.html'}")

    # ── Telegram ────────────────────────────────────────────────────
    is_saturday = dt.datetime.now(SGT).weekday() == 5
    if notify or is_saturday or SETTINGS.get("telegram_daily"):
        from notify import send_summary
        send_summary(payload, DATA_OUT / "tradingview_leaders.txt", SETTINGS)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use simulated prices (offline test)")
    ap.add_argument("--notify", action="store_true", help="send the Telegram summary now")
    args = ap.parse_args()
    run(demo=args.demo, notify=args.notify or os.environ.get("FORCE_NOTIFY", "").lower() == "true")
