#!/usr/bin/env python3
"""
Fundamentals for the CANSLIM letters and the valuation line (Yahoo Finance, free).

    python fundamentals.py            # refresh stale entries (strongest RS first), cap per run
    python fundamentals.py --max 300  # smaller run
    python fundamentals.py NVDA PLTR  # just these

Per stock it keeps:
    eps_q      last ~8 reported quarterly EPS (date, actual, estimate)   → C, acceleration, surprise
    sales_q    last ~5 quarterly revenues                                → sales growth
    eps_a      last ~4 annual diluted EPS                                → A (3-year growth)
    fpe, ps, float_m, mcap_b, inst_pct                                   → valuation & supply

Refresh policy: RS ≥ 70 every 7 days, everything else every 21 days, and anything whose last
quarter is older than ~100 days (a new report should be out). Writes data/fundamentals.json.
"""
import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "fundamentals.json"
STOCKS = ROOT / "docs" / "data" / "stocks.csv"


def log(*a):
    print(*a, flush=True)


def num(v):
    try:
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def r(v, d=2):
    v = num(v)
    return None if v is None else round(v, d)


def fetch(sym):
    import yfinance as yf
    t = yf.Ticker(sym.replace(".", "-"))
    rec = {"updated": dt.date.today().isoformat()}

    # quarterly EPS, as reported (same basis analysts use, like IBD/MarketSurge)
    try:
        ed = t.get_earnings_dates(limit=16)
        if ed is not None and len(ed):
            ed = ed.dropna(subset=["Reported EPS"]).sort_index()
            ed = ed[~ed.index.normalize().duplicated(keep="last")].tail(9)
            rec["eps_q"] = [[d.strftime("%Y-%m-%d"), r(x.get("Reported EPS"), 3), r(x.get("EPS Estimate"), 3)]
                            for d, x in ed.iterrows()]
    except Exception as e:
        rec["err_eps"] = str(e)[:80]

    try:
        q = t.quarterly_income_stmt
        if q is not None and len(q.columns):
            q = q[sorted(q.columns)]
            if "Total Revenue" in q.index:
                rec["sales_q"] = [[c.strftime("%Y-%m-%d"), r(v, 0)] for c, v in q.loc["Total Revenue"].items() if num(v)]
            if not rec.get("eps_q"):                          # fallback: diluted EPS from the statements
                row = next((k for k in ("Diluted EPS", "Basic EPS") if k in q.index), None)
                if row:
                    rec["eps_q"] = [[c.strftime("%Y-%m-%d"), r(v, 3), None] for c, v in q.loc[row].items() if num(v) is not None]
    except Exception as e:
        rec["err_q"] = str(e)[:80]

    try:
        a = t.income_stmt
        if a is not None and len(a.columns):
            a = a[sorted(a.columns)]
            row = next((k for k in ("Diluted EPS", "Basic EPS") if k in a.index), None)
            if row:
                rec["eps_a"] = [[c.strftime("%Y"), r(v, 3)] for c, v in a.loc[row].items() if num(v) is not None]
            if "Total Revenue" in a.index:
                rec["sales_a"] = [[c.strftime("%Y"), r(v, 0)] for c, v in a.loc["Total Revenue"].items() if num(v)]
    except Exception as e:
        rec["err_a"] = str(e)[:80]

    try:
        i = t.info or {}
        rec.update({
            "fpe": r(i.get("forwardPE"), 1), "tpe": r(i.get("trailingPE"), 1),
            "ps": r(i.get("priceToSalesTrailing12Months"), 1),
            "float_m": r((num(i.get("floatShares")) or 0) / 1e6, 1) or None,
            "shares_m": r((num(i.get("sharesOutstanding")) or 0) / 1e6, 1) or None,
            "mcap_b": r((num(i.get("marketCap")) or 0) / 1e9, 2) or None,
            "inst_pct": r((num(i.get("heldPercentInstitutions")) or 0) * 100, 1) or None,
            "eps_fwd_g": r((num(i.get("earningsGrowth")) or 0) * 100, 1) if i.get("earningsGrowth") is not None else None,
            "rev_g": r((num(i.get("revenueGrowth")) or 0) * 100, 1) if i.get("revenueGrowth") is not None else None,
            "industry": i.get("industry"), "sector": i.get("sector"),
        })
    except Exception as e:
        rec["err_info"] = str(e)[:80]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--max", type=int, default=650)
    ap.add_argument("--minutes", type=float, default=75)
    a = ap.parse_args()

    db = json.loads(OUT.read_text()) if OUT.exists() else {}
    today = dt.date.today()

    if a.tickers:
        todo = [t.upper() for t in a.tickers]
    else:
        if not STOCKS.exists():
            sys.exit("docs/data/stocks.csv not found — run the RS engine first.")
        st = pd.read_csv(STOCKS).sort_values("rs_rating", ascending=False)
        live = set(st.ticker)
        for t in [t for t in db if t not in live]:            # drop names that left the universe
            db.pop(t)

        def stale(t, rs):
            e = db.get(t)
            if not e:
                return True
            age = (today - dt.date.fromisoformat(e["updated"])).days
            last_q = (e.get("eps_q") or [[None]])[-1][0]
            if last_q and (today - dt.date.fromisoformat(last_q)).days > 100 and age >= 3:
                return True                                    # a new quarter should be reported by now
            return age >= (7 if (rs or 0) >= 70 else 21)
        todo = [t for t, rs in zip(st.ticker, st.rs_rating) if stale(t, rs)]
        log(f"{len(st)} stocks · {len(todo)} stale · refreshing up to {a.max}")
        todo = todo[:a.max]

    t0, ok = time.time(), 0
    for k, t in enumerate(todo, 1):
        if time.time() - t0 > a.minutes * 60:
            log(f"  time budget reached after {k-1}")
            break
        try:
            rec = fetch(t)
            if rec.get("eps_q") or rec.get("fpe") or rec.get("mcap_b"):
                db[t] = rec; ok += 1
        except Exception as e:
            log(f"  {t}: {e}")
        if k % 50 == 0:
            log(f"  {k}/{len(todo)} · {ok} ok · {time.time()-t0:.0f}s")
            OUT.write_text(json.dumps(db, separators=(",", ":")))   # checkpoint
        time.sleep(0.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(db, separators=(",", ":")))
    log(f"Wrote {OUT}: {len(db)} stocks ({ok} refreshed this run)")


if __name__ == "__main__":
    main()
