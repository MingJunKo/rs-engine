#!/usr/bin/env python3
"""
Refresh ETF holdings in config/universe.csv straight from the fund issuers.

  python refresh_universe.py            # refresh every ETF in config/etfs.csv (Group + SPDR Sector)
  python refresh_universe.py XLK XSD    # just these

Sources, tried in order:
  1. State Street SPDR  — full holdings (daily xlsx)
  2. Invesco            — full holdings (csv)
  3. Yahoo Finance      — top 10 holdings only (fallback; adds, never removes)

Safety: an ETF's holdings are only replaced when the issuer file returns a plausible list
(>= 8 tickers and >= 50% of what we had). Anything else is left exactly as it was.
A summary of changes is printed and written to data/universe_changes.md.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
UNI = ROOT / "config" / "universe.csv"
ETFS = ROOT / "config" / "etfs.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TICK = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

SPDR = {"XLK", "XLY", "XLC", "XLF", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE", "XSD", "XOP", "XME",
        "XHB", "KRE", "KBE", "KIE", "KCE", "XAR", "XBI", "XRT", "XHE", "XPH", "XHS", "XTN", "XES", "XSW"}
INVESCO = {"PEJ", "PPA", "PBE", "PHO", "CGW", "PBJ", "RSPT", "RSPD", "RSPC", "RSPF", "RSPN", "RSPM",
           "RSPG", "RSPH", "RSPS", "RSPU", "RSPR", "SPMO"}
SKIP = {"GLD", "USO", "IBIT", "ETH", "TLT", "HYG", "IEF"}          # commodities / crypto / bonds


def clean(ts) -> list[str]:
    out = []
    for t in ts:
        t = str(t).strip().upper().replace(" ", ".").replace("/", ".")
        if TICK.match(t) and t not in out:
            out.append(t)
    return out


def from_spdr(sym: str) -> list[str]:
    url = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/"
           f"holdings-daily-us-en-{sym.lower()}.xlsx")
    r = requests.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), header=None)
    hdr = raw.index[raw.apply(lambda row: row.astype(str).str.strip().eq("Ticker").any(), axis=1)][0]
    df = pd.read_excel(io.BytesIO(r.content), header=hdr)
    return clean(df["Ticker"].dropna())


def from_invesco(sym: str) -> list[str]:
    url = ("https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
           f"?audienceType=Investor&action=download&ticker={sym}")
    r = requests.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    col = next(c for c in df.columns if "ticker" in c.lower())
    return clean(df[col].dropna())


def from_yahoo(sym: str) -> list[str]:
    import yfinance as yf
    th = yf.Ticker(sym).funds_data.top_holdings
    return clean(th.index)


def main(only: list[str]):
    etfs = pd.read_csv(ETFS)
    uni = pd.read_csv(UNI)
    targets = only or list(etfs.loc[etfs.category.isin(["Group", "SPDR Sector"]), "symbol"])
    targets = [t for t in targets if t not in SKIP]
    report = []
    for sym in targets:
        old = sorted(uni.loc[uni.etf == sym, "ticker"])
        new, src = [], ""
        for name, fn in (("SPDR", from_spdr), ("Invesco", from_invesco), ("Yahoo top-10", from_yahoo)):
            if name == "SPDR" and sym not in SPDR:
                continue
            if name == "Invesco" and sym not in INVESCO:
                continue
            try:
                new, src = fn(sym), name
                if new:
                    break
            except Exception as e:
                print(f"  {sym}: {name} failed ({type(e).__name__})")
        if not new:
            report.append(f"- {sym}: no data, kept {len(old)} holdings")
            continue
        if src == "Yahoo top-10":                       # partial list → add only
            merged = sorted(set(old) | set(new))
        elif len(new) >= 8 and len(new) >= 0.5 * len(old):
            merged = sorted(new)
        else:
            report.append(f"- {sym}: {src} returned {len(new)} (suspicious), kept {len(old)}")
            continue
        added, removed = sorted(set(merged) - set(old)), sorted(set(old) - set(merged))
        if added or removed:
            report.append(f"- {sym} ({src}): +{len(added)} −{len(removed)}"
                          + (f" | added {', '.join(added[:15])}" if added else "")
                          + (f" | removed {', '.join(removed[:15])}" if removed else ""))
        uni = pd.concat([uni[uni.etf != sym], pd.DataFrame({"ticker": merged, "etf": sym})])
        print(f"  {sym}: {len(merged)} holdings via {src}")

    uni = uni.drop_duplicates().sort_values(["etf", "ticker"])
    uni.to_csv(UNI, index=False)
    out = ROOT / "data" / "universe_changes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "# Universe refresh\n\n" + ("\n".join(report) if report else "No changes.") + \
           f"\n\nTotal: {uni.ticker.nunique()} unique stocks across {uni.etf.nunique()} ETFs.\n"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]])
