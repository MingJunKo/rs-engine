"""
Model Book — every US stock that doubled after a base breakout, and what it looked like beforehand.

    python modelbook.py           # real data (runs on GitHub; ~10–15 min)
    python modelbook.py --demo    # simulated prices with planted winners (offline test)

Method
  Breakout event  close above the highest close of the prior 63 sessions, after at least 15 sessions
                  below that high (i.e. out of a base of 3+ weeks); price ≥ $5 and 50-day $ volume ≥ floor.
  Winner          the close went on to gain ≥ min_gain % within horizon_days (default 100% in 126 sessions).
  Features        measured on the day before the breakout (setup) and on the breakout day (volume, gap).
  Base rates      every complete breakout event is kept, so each feature gets a hit rate vs the base rate:
                  "of breakouts with RS ≥ 90, x% doubled vs y% of all breakouts".
Caveat: Yahoo has no delisted stocks, so failures are under-counted and hit rates are flattering.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil

import numpy as np
import pandas as pd

import rs_engine as E

MB = E.SETTINGS.get("modelbook", {})
PERIOD = MB.get("period", "5y")
MIN_GAIN = MB.get("min_gain", 100) / 100
HORIZON = int(MB.get("horizon_days", 126))
MIN_PRICE = MB.get("min_price", 5)
MIN_DVOL = MB.get("min_dollar_volume", 5_000_000)
CHART_MAX = int(MB.get("chart_max", 500))
BASE_LOOK, MIN_BASE = 63, 15
OUT = E.DOCS / "modelbook"
log = E.log


# ── universe: all US common stocks (+ your ETF universe) ───────────────
def universe() -> tuple[list[str], pd.DataFrame]:
    names = pd.read_csv(E.CFG_DIR / "names.csv") if (E.CFG_DIR / "names.csv").exists() else \
        pd.DataFrame(columns=["ticker", "name", "exchange", "sector", "industry"])
    names = names.fillna("")
    bad = names.name.str.contains(r"Warrant|\bUnits?\b|\bRights?\b|Preferred|Notes due|Debenture|Depositary Shares? Each|%",
                                  case=False, regex=True) | names.industry.eq("Blank Checks")
    ok = names.ticker.str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?") & ~bad
    tick = set(names.ticker[ok])
    uni = pd.read_csv(E.CFG_DIR / "universe.csv")
    etfs = set(pd.read_csv(E.CFG_DIR / "etfs.csv").symbol)
    tick |= set(uni.ticker.str.strip().str.upper()) - etfs
    return sorted(tick) + ["SPY"], names


def theme_map() -> dict:
    uni = pd.read_csv(E.CFG_DIR / "universe.csv")
    etfs = pd.read_csv(E.CFG_DIR / "etfs.csv").set_index("symbol")
    out = {}
    for t, g in uni.groupby(uni.ticker.str.upper()):
        th = [etfs.at[e, "theme"] for e in g.etf if e in etfs.index and etfs.at[e, "category"] in ("Group", "SPDR Sector")]
        if th:
            out[t] = th[0]
    return out


# ── simulated prices with planted base→breakout→run patterns ───────────
def demo_prices(tickers, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1260)
    n = len(dates)
    mkt = rng.normal(0.0004, 0.010, n)
    O, H, L, C, V = {}, {}, {}, {}, {}
    planted = set(rng.choice([t for t in tickers if t != "SPY"], min(90, len(tickers) // 8), replace=False))
    for tk in tickers:
        sig = rng.uniform(0.012, 0.035)
        r = rng.uniform(0.6, 1.4) * mkt + rng.normal(0.0002, sig, n)
        vol = rng.lognormal(np.log(rng.uniform(3e5, 6e6)), 0.3, n)
        if tk == "SPY":
            r, sig = mkt, 0.008
        if tk in planted:
            i = int(rng.integers(300, n - 200))
            base = int(rng.integers(20, 60))
            r[i - base:i] = rng.normal(0, sig * 0.45, base)          # quiet base
            vol[i - base:i] *= 0.6                                    # volume dries up
            run = int(rng.integers(50, 120))
            r[i:i + run] += rng.uniform(1.2, 3.0) / run * rng.uniform(0.8, 1.2, run)  # +120–300% run
            r[i] += rng.uniform(0.04, 0.15)
            vol[i:i + 5] *= rng.uniform(2, 5)
            r[i + run:i + run + 40] -= 0.006                          # give some back
        c = np.exp(np.cumsum(r)) * rng.uniform(8, 120)
        rngd = np.abs(rng.normal(0, sig * 0.7, n))
        o = c / np.exp(r) * np.exp(rng.normal(0, sig * 0.3, n))
        C[tk], O[tk] = c, o
        H[tk], L[tk] = np.maximum(c, o) * (1 + rngd), np.minimum(c, o) * (1 - rngd)
        V[tk] = vol
    mk = lambda d: pd.DataFrame(d, index=dates)
    return {"Open": mk(O), "High": mk(H), "Low": mk(L), "Close": mk(C), "Volume": mk(V)}


def run(demo: bool):
    tickers, names = universe()
    if demo:
        names = pd.DataFrame({"ticker": tickers, "name": tickers, "exchange": "", "sector": "Demo", "industry": ""})
        px = demo_prices(tickers[:900] + ["SPY"])
    else:
        log(f"Model book: downloading {len(tickers)} tickers ({PERIOD})")
        px = E.download_yahoo(tickers, period=PERIOD, want=("Open", "High", "Low", "Close", "Volume"))
    c = px["Close"].astype("float32")
    c = c.loc[c["SPY"].notna()]
    o, h, l, v = (px[k].reindex(c.index).astype("float32") for k in ("Open", "High", "Low", "Volume"))
    spy = c["SPY"]
    stocks = [t for t in c.columns if t != "SPY"]
    c, o, h, l, v = (x[stocks] for x in (c, o, h, l, v))
    T = len(c)
    log(f"  {len(stocks)} stocks × {T} sessions ({c.index[0].date()} → {c.index[-1].date()})")

    # ── indicators ──────────────────────────────────────────────────
    sma50, sma150, sma200 = (c.rolling(n, min_periods=n).mean() for n in (50, 150, 200))
    hi252 = c.rolling(252, min_periods=200).max()
    lo252 = c.rolling(252, min_periods=200).min()
    r = lambda n: c / c.shift(n) - 1
    score = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)          # IBD-style RS
    rs = score.rank(axis=1, pct=True) * 98 + 1
    rsl = c.div(spy, axis=0)
    rsl_nh = rsl >= rsl.rolling(252, min_periods=200).max()
    adr = ((h / l - 1) * 100).rolling(20).mean()
    vol50 = v.rolling(50).mean()
    dvol50 = (c * v).rolling(50).mean()
    tt = ((c > sma150) & (c > sma200) & (sma150 > sma200) & (sma200 > sma200.shift(21)) &
          (sma50 > sma150) & (c > sma50) & (c >= lo252 * 1.3) & (c >= hi252 * 0.75) & (rs >= 70))
    prior_max = c.shift(1).rolling(BASE_LOOK, min_periods=BASE_LOOK).max()
    recent_max = c.shift(1).rolling(MIN_BASE, min_periods=MIN_BASE).max()
    bo = ((c > prior_max) & (recent_max < prior_max) & (c.shift(1) >= MIN_PRICE) &
          (dvol50.shift(1) >= MIN_DVOL) & sma50.shift(1).notna())

    fwd = c[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1)
    gain = fwd / c - 1
    complete = pd.Series(np.arange(T) <= T - 1 - HORIZON, index=c.index)
    spy50, spy200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    mkt_up = (spy > spy50) & (spy50 > spy200)

    p1 = lambda f: f.shift(1)                                   # value on the day before the breakout
    feats = {
        "rs": rs, "rs_prev": p1(rs), "tt": p1(tt), "rsl_nh": rsl_nh,
        "rs_lead": (rsl_nh & (c < hi252 * 0.98)).astype("float32").shift(1).rolling(10).max(),
        "base_depth": (h.shift(1).rolling(BASE_LOOK).max() / l.shift(1).rolling(BASE_LOOK).min() - 1) * 100,
        "off_high": p1(c / hi252 - 1) * 100, "above_low": p1(c / lo252 - 1) * 100,
        "prior_6m": p1(r(126)) * 100, "adr": p1(adr),
        "tight": p1((h.rolling(10).max() - l.rolling(10).min()) / c * 100 / adr),
        "vol_dry": p1(v.rolling(10).mean() / vol50), "bo_vol": v / vol50.shift(1),
        "gap": (o / c.shift(1) - 1) * 100, "price": c, "dvol": dvol50.shift(1) / 1e6,
        "gain": gain * 100, "ret_20": (c.shift(-20) / c - 1) * 100,
    }
    ev = bo.stack()
    ev = ev[ev].index                                            # (date, ticker)
    E_ = pd.DataFrame(index=ev)
    for k, f in feats.items():
        E_[k] = f.stack(future_stack=True).reindex(ev).astype("float64").values
    E_["complete"] = complete.reindex(ev.get_level_values(0)).values
    E_["mkt_up"] = mkt_up.reindex(ev.get_level_values(0)).values
    E_["winner"] = E_.gain >= MIN_GAIN * 100
    E_["ep"] = (E_.gap >= 10) & (E_.bo_vol >= 3)
    E_ = E_.reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    E_.columns = ["date", "ticker"] + list(E_.columns[2:])
    log(f"  breakout events: {len(E_):,} ({int(E_.complete.sum()):,} with a full {HORIZON}-day outcome), "
        f"winners: {int(E_.winner.sum()):,}")

    # ── winners: one entry per run (first breakout), full run measured ──
    pos = {d: i for i, d in enumerate(c.index)}
    col = {t: j for j, t in enumerate(c.columns)}
    C, Hh, Ll = c.values, h.values, l.values
    rows = []
    for t, g in E_[E_.winner].sort_values("date").groupby("ticker"):
        last_peak = -1
        j = col[t]
        for _, e in g.iterrows():
            b = pos[e.date]
            if b <= last_peak:
                continue
            cb = C[b, j]
            runmax, pk, k = cb, b, b + 1
            while k < T and k <= b + 378:
                if C[k, j] > runmax:
                    runmax, pk = C[k, j], k
                elif C[k, j] < runmax * 0.75:
                    break
                k += 1
            last_peak = pk
            base_seg = C[max(0, b - BASE_LOOK):b, j]
            hi_i = int(np.nanargmax(base_seg)) if len(base_seg) else 0
            lows_to_peak = Ll[b + 1:pk + 1, j]
            first20 = Ll[b + 1:min(T, b + 21), j]
            hit20 = np.where(C[b + 1:pk + 1, j] >= cb * 1.2)[0]
            rows.append({**e.to_dict(), "b": b, "pk": pk, "base_start": max(0, b - BASE_LOOK) + hi_i,
                         "base_len": len(base_seg) - hi_i, "run_gain": (C[pk, j] / cb - 1) * 100,
                         "run_days": pk - b, "heat": (np.nanmin(lows_to_peak) / cb - 1) * 100 if len(lows_to_peak) else 0,
                         "shake20": (np.nanmin(first20) / cb - 1) * 100 if len(first20) else 0,
                         "days_to_20": int(hit20[0]) + 1 if len(hit20) else None, "still_running": k >= T})
    W = pd.DataFrame(rows)
    log(f"  winning runs (deduplicated): {len(W):,}")

    # ── base rates & lift (complete events only) ────────────────────
    A = E_[E_.complete]
    base = A.winner.mean() if len(A) else np.nan
    conds = [
        ("RS rating ≥ 90 at breakout", A.rs >= 90), ("RS rating 80–89", A.rs.between(80, 89.99)), ("RS rating < 70", A.rs < 70),
        ("RS line at a new high on breakout day", A.rsl_nh == 1), ("RS line hit a new high before price (10d)", A.rs_lead == 1),
        ("Passed Minervini trend template", A.tt == 1),
        ("Base depth ≤ 25%", A.base_depth <= 25), ("Base depth 25–40%", A.base_depth.between(25, 40)), ("Base depth > 40%", A.base_depth > 40),
        ("Tight: 10-day range ≤ 2.5 ADR", A.tight <= 2.5), ("Volume dried up (10d/50d < 0.8)", A.vol_dry < 0.8),
        ("Breakout volume ≥ 2× average", A.bo_vol >= 2), ("Breakout volume ≥ 3× average", A.bo_vol >= 3),
        ("Episodic pivot: gap ≥ 10% on ≥ 3× volume", A.ep),
        ("ADR ≥ 4%", A.adr >= 4), ("ADR < 2.5%", A.adr < 2.5),
        ("Within 15% of 52-week high before breakout", A.off_high >= -15), ("Prior 6-month gain ≥ 30%", A.prior_6m >= 30),
        ("Price under $20", A.price < 20), ("Market in uptrend (SPY > 50 > 200 DMA)", A.mkt_up == True),
        ("Market NOT in uptrend", A.mkt_up == False),
    ]
    lift = []
    for label, m in conds:
        m = m.fillna(False).astype(bool)
        n_ = int(m.sum())
        if n_ < 20:
            continue
        hit = A.winner[m].mean()
        lift.append({"label": label, "n": n_, "hit": hit * 100, "lift": hit / base if base else None,
                     "share": A.winner[m].sum() / max(1, A.winner.sum()) * 100})
    combo = (A.rs >= 80) & (A.tight <= 2.5) & (A.off_high >= -15) & (A.mkt_up == True)
    combo2 = combo & (A.bo_vol >= 2)
    for label, m in (("Your SETUP: RS ≥ 80 + tight + near high + market uptrend", combo),
                     ("…plus breakout volume ≥ 2×", combo2)):
        m = m.fillna(False).astype(bool)
        if m.sum() >= 10:
            hit = A.winner[m].mean()
            lift.append({"label": label, "n": int(m.sum()), "hit": hit * 100, "lift": hit / base if base else None,
                         "share": A.winner[m].sum() / max(1, A.winner.sum()) * 100, "combo": True})

    cmp_cols = [("rs", "RS rating at breakout", 0), ("base_depth", "Base depth %", 0), ("base_len", "Base length (days)", 0),
                ("tight", "Tightness (ADRs)", 1), ("vol_dry", "Volume dry-up (10d/50d)", 2), ("bo_vol", "Breakout volume ×", 1),
                ("adr", "ADR %", 1), ("off_high", "% from 52-week high", 0), ("prior_6m", "Prior 6-month gain %", 0),
                ("price", "Price $", 0)]
    compare = []
    for k, label, d in cmp_cols:
        wv = W[k].dropna() if k in W else pd.Series(dtype=float)
        av = A[k].dropna() if k in A else pd.Series(dtype=float)
        if len(wv):
            compare.append({"label": label, "d": d, "win_med": float(wv.median()), "win_q1": float(wv.quantile(.25)),
                            "win_q3": float(wv.quantile(.75)), "all_med": float(av.median()) if len(av) else None})
    run_stats = {}
    if len(W):
        run_stats = {"median_gain": float(W.run_gain.median()), "median_days": float(W.run_days.median()),
                     "median_heat": float(W.heat.median()), "median_shake20": float(W.shake20.median()),
                     "pct_heat_gt_15": float((W.heat < -15).mean() * 100),
                     "median_days_to_20": float(W.days_to_20.dropna().median()) if W.days_to_20.notna().any() else None}

    # ── tables for the page ─────────────────────────────────────────
    nm = names.drop_duplicates("ticker").set_index("ticker")
    th = theme_map()
    W = W.sort_values("run_gain", ascending=False).reset_index(drop=True)
    W["id"] = [f"{t}_{d:%Y%m%d}" for t, d in zip(W.ticker, W.date)]
    yr = W.date.dt.year if len(W) else pd.Series(dtype=int)
    by_year = [{"year": int(y), "n": int(n_)} for y, n_ in yr.value_counts().sort_index().items()]
    W["sector"] = [nm.sector.get(t, "") or "–" for t in W.ticker] if len(W) else []
    by_sector = [{"sector": s, "n": int(n_)} for s, n_ in W.sector.value_counts().head(14).items()] if len(W) else []

    rnd = lambda x, d=1: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else (int(round(x)) if d == 0 else round(float(x), d))
    book = []
    for _, w in W.iterrows():
        book.append({"id": w.id, "symbol": w.ticker, "name": nm.name.get(w.ticker, ""), "exchange": nm.exchange.get(w.ticker, ""),
                     "sector": w.sector, "industry": nm.industry.get(w.ticker, ""), "theme": th.get(w.ticker, ""),
                     "date": w.date.strftime("%Y-%m-%d"), "year": int(w.date.year),
                     "rs": rnd(w.rs, 0), "tt": bool(w.tt == 1), "rsl_nh": bool(w.rsl_nh == 1), "rs_lead": bool(w.rs_lead == 1),
                     "base_depth": rnd(w.base_depth, 0), "base_len": int(w.base_len), "tight": rnd(w.tight),
                     "vol_dry": rnd(w.vol_dry, 2), "bo_vol": rnd(w.bo_vol), "gap": rnd(w.gap), "ep": bool(w.ep),
                     "adr": rnd(w.adr), "off_high": rnd(w.off_high, 0), "prior_6m": rnd(w.prior_6m, 0),
                     "price": rnd(w.price, 2), "dvol": rnd(w.dvol), "mkt_up": bool(w.mkt_up),
                     "run_gain": rnd(w.run_gain, 0), "run_days": int(w.run_days), "heat": rnd(w.heat),
                     "shake20": rnd(w.shake20), "days_to_20": None if pd.isna(w.days_to_20) else int(w.days_to_20),
                     "running": bool(w.still_running), "chart": _ < CHART_MAX})

    # ── chart files for the study panel ─────────────────────────────
    cdir = OUT / "c"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    O_, V_, S_ = o.values, v.values, spy.values
    ep_ = np.asarray((c.index - pd.Timestamp("1970-01-01")).days, dtype=int)     # days since 1970 (unit-safe)
    for _, w in W.head(CHART_MAX).iterrows():
        j, b, pk = col[w.ticker], int(w.b), int(w.pk)
        s0, s1 = max(0, b - 160), min(T, pk + 40)
        seg = slice(s0, s1)
        cl = C[seg, j]
        ok = ~np.isnan(cl)
        r2 = lambda a: [None if not np.isfinite(x) else round(float(x), 2 if x < 1000 else 0) for x in a]
        data = {"t": ep_[seg][ok].tolist(), "o": r2(O_[seg, j][ok]), "h": r2(Hh[seg, j][ok]), "l": r2(Ll[seg, j][ok]),
                "c": r2(cl[ok]), "v": [int(x) // 100 if np.isfinite(x) else 0 for x in V_[seg, j][ok]],
                "rs": r2((cl / S_[seg])[ok] / np.nanmax((cl / S_[seg])[ok]) * 100),
                "m": {"base": int(ep_[int(w.base_start)]), "bo": int(ep_[b]), "peak": int(ep_[pk])}}
        (cdir / f"{w.id}.json").write_text(json.dumps(data, separators=(",", ":")))

    payload = {
        "generated": dt.datetime.now(E.SGT).strftime("%d %b %Y %H:%M SGT"), "demo": demo,
        "span": f"{c.index[0]:%b %Y} – {c.index[-1]:%b %Y}", "n_stocks": len(stocks),
        "rules": {"min_gain": MIN_GAIN * 100, "horizon": HORIZON, "min_price": MIN_PRICE, "min_dvol": MIN_DVOL / 1e6},
        "n_events": int(len(E_)), "n_complete": int(len(A)), "n_winners": int(len(W)), "base_rate": None if pd.isna(base) else base * 100,
        "lift": lift, "compare": compare, "run": run_stats, "by_year": by_year, "by_sector": by_sector, "book": book,
    }
    payload = E._clean_json(payload)
    tpl = (E.ROOT / "templates" / "modelbook.html").read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), allow_nan=False)
                                                .replace("</", "<\\/")))
    W.drop(columns=["b", "pk", "base_start"]).assign(date=W.date.dt.strftime("%Y-%m-%d")).round(2).to_csv(E.DATA_OUT / "modelbook.csv", index=False)
    log(f"Wrote {OUT / 'index.html'} — {len(W)} winners, base rate {payload['base_rate'] or 0:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    run(ap.parse_args().demo)
