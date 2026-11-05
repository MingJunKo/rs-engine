#!/usr/bin/env python3
"""
10-year backtest of the RS-setup strategy on a $100,000 account (run on demand from GitHub Actions).

Strategy (the one the tracker settled on):
  universe   today's RS Engine universe (ETF holdings + liquid US stocks) — see the survivorship note
  signal     RS ≥ 80 (vs RSP, volatility-adjusted), above the 50-day, ADR ≥ 2.5%, price ≥ $5, $vol ≥ $10M,
             within 6% of the 20-day high, 10-day range ≤ 2.5 ADR, 0–4 ATR above the 50-day
  theme gate stock belongs to one of the 8 strongest themes that day (theme ETFs' RS, where they existed)
  entry      break of the 20-day high within 10 days; skip if it opens > 2% above it
  stop       under the prior day's low (never below the 10-day low)
  selling    ⅓ on day 3 if in profit, stop to breakeven, rest trailed on a close below the 10-day average
  book       0.5% of equity risked per trade, max 8 positions, max 25% in one name, highest RS first, compounding

    python backtest10.py            # writes docs/backtest/index.html and data/backtest10.json
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rs_engine as E          # noqa: E402
import tracker as T            # noqa: E402

GAP = 20


def log(*a):
    print(*a, flush=True)


def main(years="10y"):
    st = pd.read_csv(ROOT / "docs" / "data" / "stocks.csv")
    etfs = pd.read_csv(E.CFG_DIR / "etfs.csv")
    S = sorted(set(st.ticker))
    themes_of = {r.ticker: ({E_t for E_t in str(r.etfs).split() if E_t and E_t != "nan"}, r.theme) for r in st.itertuples()}
    meta = etfs.set_index("symbol")
    grp = [e for e in etfs.symbol if meta.at[e, "category"] in ("Group", "SPDR Sector")]
    log(f"{len(S)} stocks, {len(grp)} theme ETFs — downloading {years} of prices")
    if years == "demo":                                     # offline test with simulated prices
        sty = {**meta["style"].to_dict(), **{t: "growth" for t in S}}
        px = E.demo_prices(S + grp + [E.BENCH], sty)
        px["Open"] = px["Close"].shift(1) * (1 + np.random.default_rng(1).normal(0, .004, px["Close"].shape))
    else:
        px = E.download_yahoo(S + grp + [E.BENCH], period=years, want=("Open", "Close", "High", "Low", "Volume"))
    close = px["Close"].ffill(limit=3)
    close = close.loc[close[E.BENCH].notna()]
    o, h, l, v = (px[k].reindex(close.index) for k in ("Open", "High", "Low", "Volume"))
    h, l, v = h.ffill(limit=3), l.ffill(limit=3), v.fillna(0)
    idx = close.index
    b_close = close[E.BENCH]
    b_atr = E.atr(h[[E.BENCH]], l[[E.BENCH]], close[[E.BENCH]])[E.BENCH]
    S = [t for t in S if t in close.columns]
    G = [e for e in grp if e in close.columns]
    log(f"{len(idx)} sessions from {idx[0].date()} · {len(S)} stocks · {len(G)} theme ETFs with data")

    rating, _, _ = E.rs_rating(close[S], h[S], l[S], b_close, b_atr)
    erating, _, _ = E.rs_rating(close[G], h[G], l[G], b_close, b_atr)
    # top-8 themes each day, from the ETFs that existed then
    th = {e: meta.at[e, "theme"] for e in G}
    best = erating.T.groupby(erating.columns.map(th)).max().T
    topset = [set(r.dropna().nlargest(8).index) if r.notna().sum() >= 10 else None for _, r in best.iterrows()]
    etf_theme = meta.theme.to_dict()
    stock_themes = {t: {etf_theme.get(e) for e in themes_of[t][0]} | {themes_of[t][1]} for t in S}

    sig = T.setup_signals(close[S], h[S], l[S], v[S], rating, atr=E.atr).values
    O, H, L, C = o[S].values, h[S].values, l[S].values, close[S].values
    S10 = close[S].rolling(10).mean().values
    book, start = [], 130
    for jx, t in enumerate(S):
        last = -999
        for i in np.where(sig[start:, jx])[0] + start:
            if i - last < GAP:
                continue
            last = i
            p = T.path(i, O[:, jx], H[:, jx], L[:, jx], C[:, jx], S10[:, jx])
            if p:
                ts = topset[i]
                book.append({**p, "jx": jx, "sym": t, "rs": int(rating.iat[i, jx]) if pd.notna(rating.iat[i, jx]) else None,
                             "top": None if ts is None else bool(stock_themes[t] & ts)})
    log(f"{len(book)} triggered setups")

    top = T.portfolio([b for b in book if b["top"]], idx, C)
    allp = T.portfolio(book, idx, C)
    f0 = idx.get_loc(pd.Timestamp(top["from"]))
    bser = b_close.iloc[f0:]
    spy = close["SPY"].iloc[f0:] if "SPY" in close else None

    def yearly(curve):
        s = pd.Series({pd.Timestamp(d): e for d, e, _ in curve})
        y = s.groupby(s.index.year).last()
        prev = pd.concat([pd.Series({y.index[0] - 1: s.iloc[0]}), y.iloc[:-1]])
        return {int(k): round((y[k] / prev.iloc[i] - 1) * 100, 1) for i, k in enumerate(y.index)}

    def bench_curve(ser):
        return [(d.strftime("%Y-%m-%d"), round(100_000 * x / ser.iloc[0])) for d, x in ser.items()][::max(1, len(ser) // 600)]

    rb = bench_curve(bser)
    out = {"generated": dt.datetime.now(E.SGT).strftime("%d %b %Y %H:%M SGT"), "from": top["from"], "to": idx[-1].strftime("%d %b %Y"),
           "top": top, "all": allp, "bench": rb, "bench_name": E.BENCH,
           "bench_ret": round((bser.iloc[-1] / bser.iloc[0] - 1) * 100, 1),
           "bench_cagr": round(((bser.iloc[-1] / bser.iloc[0]) ** (1 / top["years"]) - 1) * 100, 1),
           "bench_dd": round(float(((bser / bser.cummax()) - 1).min() * 100), 1),
           "years_top": yearly(top["curve"]), "years_all": yearly(allp["curve"]),
           "years_bench": yearly([(d, e, 0) for d, e in rb]),
           "n_signals": len(book), "n_stocks": len(S), "n_etfs": len(G)}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "backtest10.json").write_text(json.dumps(E._clean_json(out), separators=(",", ":")))
    tpl = (ROOT / "templates" / "backtest.html").read_text()
    css = (ROOT / "templates" / "dashboard.html").read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    (E.DOCS / "backtest").mkdir(parents=True, exist_ok=True)
    (E.DOCS / "backtest" / "index.html").write_text(
        tpl.replace("/*__SHARED_CSS__*/", css).replace("/*__DATA__*/null", json.dumps(E._clean_json(out), separators=(",", ":"))))
    log(f"Top-theme book: $100,000 → ${top['end']:,} ({top['cagr']}%/yr, max DD {top['max_dd']}%) · "
        f"all setups: ${allp['end']:,} ({allp['cagr']}%/yr, DD {allp['max_dd']}%) · {E.BENCH} {out['bench_cagr']}%/yr")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10y")
