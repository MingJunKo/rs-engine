#!/usr/bin/env python3
"""
Open-market insider purchases (SEC Form 4, transaction code "P") for the RS Engine universe.

Reads EDGAR's daily form index for each business day not yet processed, keeps the Form 4 filings whose issuer is in
the universe, opens each filing and records every open-market purchase. Results go to data/history/insider_buys.csv
(last 90 days). extras.insider_clusters() turns them into "2+ insiders bought in 7 days" alerts.

    python insiders.py              # catch up (first run backfills 7 business days)
    python insiders.py --days 3     # backfill a specific number of days

Needs the SEC_USER_AGENT secret ("Name email") — the SEC asks every automated client to identify itself.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "history" / "insider_buys.csv"
STATE = ROOT / "data" / "history" / "insider_state.json"
TICKERS_CACHE = ROOT / "config" / "sec_tickers.json"
MAX_FILINGS, MAX_MINUTES = 3000, int(os.environ.get("INSIDER_MINUTES", "6"))


def log(*a):
    print(*a, flush=True)


def session():
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        log("SEC_USER_AGENT not set — skipping insider scan.")
        sys.exit(0)
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    return s


def get(s, url, tries=3):
    for k in range(tries):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429):
                time.sleep(2 + 3 * k)
            elif r.status_code == 404:
                return None
        except requests.RequestException:
            time.sleep(1 + k)
    return None


def cik_map(s) -> dict:
    """SEC CIK → ticker (cached a week)."""
    if TICKERS_CACHE.exists() and time.time() - TICKERS_CACHE.stat().st_mtime < 7 * 86400:
        return json.loads(TICKERS_CACHE.read_text())
    r = get(s, "https://www.sec.gov/files/company_tickers.json")
    if r is None:
        return json.loads(TICKERS_CACHE.read_text()) if TICKERS_CACHE.exists() else {}
    m = {str(int(v["cik_str"])): v["ticker"].upper().replace("-", ".") for v in r.json().values()}
    TICKERS_CACHE.write_text(json.dumps(m))
    return m


def day_index(s, d: pd.Timestamp) -> list:
    q = (d.month - 1) // 3 + 1
    r = get(s, f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx")
    if r is None:
        return []
    rows = []
    for line in r.text.splitlines():
        if not line.startswith("4 "):
            continue
        m = re.match(r"^(\S+)\s+(.+?)\s+(\d+)\s+(\d{8})\s+(\S+)\s*$", line)
        if m and m.group(1) == "4":
            rows.append((m.group(3), m.group(5)))
    return rows


def TAG(x: str, t: str):
    """Text of <t>…</t> or <t><value>…</value>…</t> (Form 4 wraps most numbers in <value> and adds footnote ids)."""
    m = re.search(rf"<{t}>\s*(?:<value>\s*([^<]*?)\s*</value>|([^<]*))", x, re.S)
    if not m:
        return None
    v = m.group(1) if m.group(1) is not None else m.group(2)
    return v.strip() if v is not None else None


def parse(txt: str) -> list:
    if "<ownershipDocument>" not in txt:
        return []
    doc = txt[txt.index("<ownershipDocument>"):]
    sym = (TAG(doc, "issuerTradingSymbol") or "").upper().replace("-", ".")
    owner = TAG(doc, "rptOwnerName") or ""
    title = TAG(doc, "officerTitle") or ("Director" if (TAG(doc, "isDirector") or "").strip() in ("1", "true") else
                                         "10% owner" if (TAG(doc, "isTenPercentOwner") or "").strip() in ("1", "true") else "")
    out = []
    for tx in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", doc, re.S):
        if (TAG(tx, "transactionCode") or "").strip() != "P":
            continue
        try:
            sh = float(TAG(tx, "transactionShares") or 0)
            px = float(TAG(tx, "transactionPricePerShare") or 0)
        except ValueError:
            continue
        if sh <= 0 or px <= 0:
            continue
        out.append({"ticker": sym, "owner": owner.strip(), "title": title.strip(), "trade_date": (TAG(tx, "transactionDate") or "")[:10],
                    "shares": sh, "price": px, "value": round(sh * px)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    s = session()
    universe = set(pd.read_csv(ROOT / "docs" / "data" / "stocks.csv").ticker)
    cmap = cik_map(s)
    want = {c for c, t in cmap.items() if t in universe}
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    last = pd.Timestamp(st["last_day"]) if st.get("last_day") and not a.days else today - pd.offsets.BDay(a.days or 7)
    # EDGAR publishes a day's index late that evening, so read up to yesterday (New York)
    days = [d for d in pd.bdate_range(last + pd.Timedelta(days=1), today - pd.Timedelta(days=1))]
    log(f"insiders: {len(want)} universe issuers with a CIK · days to read: {', '.join(d.strftime('%d %b') for d in days) or 'none'}")
    t0, rows, done_day, n = time.time(), [], None, 0
    for d in days:
        idx = day_index(s, d)
        mine = {path for cik, path in idx if cik.lstrip("0") in want}
        log(f"  {d:%d %b}: {len(idx)} Form 4 lines, {len(mine)} for universe issuers")
        for path in sorted(mine):
            if n >= MAX_FILINGS or time.time() - t0 > MAX_MINUTES * 60:
                break
            r = get(s, f"https://www.sec.gov/Archives/{path}")
            n += 1
            time.sleep(0.11)                                  # SEC fair-access limit: ≤ 10 requests a second
            if r is None:
                continue
            for x in parse(r.text):
                if x["ticker"] in universe:
                    rows.append({**x, "filed": d.strftime("%Y-%m-%d")})
        else:
            done_day = d
            continue
        break                                                 # ran out of time/budget: resume from here next run
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=new.columns)
        df = pd.concat([old, new]).drop_duplicates(["ticker", "owner", "trade_date", "shares", "price"])
        df = df[pd.to_datetime(df.filed) >= today - pd.Timedelta(days=90)]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values("filed").to_csv(OUT, index=False)
    if done_day is not None:
        STATE.write_text(json.dumps({"last_day": done_day.strftime("%Y-%m-%d")}))
    log(f"insiders: read {n} filings, {len(rows)} open-market purchases in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
