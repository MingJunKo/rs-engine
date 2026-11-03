#!/usr/bin/env python3
"""
Institutional sponsorship from SEC Form 13F (the "I" in CANSLIM).

Every investment manager with >$100M files a 13F within 45 days of each quarter end listing the
US stocks it holds. The SEC publishes all of them as bulk data sets (one zip per 3-month filing
window). For each quarter this script counts how many distinct managers held each stock and
how many shares they held in total, then maps CUSIP → ticker with OpenFIGI.

    python institutions.py              # process any new SEC zips, update data/institutions.csv
    python institutions.py --quarters 9 # how many report quarters to keep (default 9)

Env (set as GitHub Actions secrets):
    SEC_USER_AGENT   required by the SEC — "Your Name your@email.com"
    OPENFIGI_KEY     optional, free — makes the CUSIP→ticker mapping ~20× faster

Outputs
    data/inst/<zipname>.csv.gz   per-zip aggregate (period, cusip, holders, shares) — cache
    data/inst/cusip_map.csv      cusip → ticker (cache; OpenFIGI is only asked about new CUSIPs)
    data/institutions.csv        ticker, period, holders, shares   ← what the engine reads

Notes
    * Counts managers (13F filers): mutual-fund families, hedge funds, banks, pensions, RIAs.
      It is the same idea as "number of funds" in MarketSurge/IBD, but a broader population,
      so compare a stock with itself over time, not with numbers on other sites.
    * Only original 13F-HR holdings reports, shares (not principal), no options.
    * The newest quarter only appears once the SEC publishes the window it was filed in
      (roughly 2–3 weeks after each 45-day deadline).
"""
import argparse
import datetime as dt
import gzip
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "inst"
OUT = ROOT / "data" / "institutions.csv"
MAP = CACHE / "cusip_map.csv"
LIST_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
MIN_HOLDERS_TO_MAP = 15            # only ask OpenFIGI about CUSIPs at least this widely held


def log(*a):
    print(*a, flush=True)


def ua():
    u = os.environ.get("SEC_USER_AGENT", "").strip()
    if not u:
        sys.exit("Set the SEC_USER_AGENT secret (e.g. 'Your Name you@example.com') — the SEC requires it.")
    return {"User-Agent": u, "Accept-Encoding": "gzip, deflate"}


# ── which zips exist ───────────────────────────────────────────────
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def candidate_names(today: dt.date, n_windows: int):
    """Filing windows since 2024 are Dec–Feb, Mar–May, Jun–Aug, Sep–Nov, named like
    01jun2025-31aug2025_form13f.zip. Generate the most recent n (newest first)."""
    out, y, m = [], today.year, today.month
    start_m = [12, 3, 6, 9]
    # find the window that contains today, then step back
    sm = max(s for s in start_m if s <= m) if m >= 3 else 12
    sy = y if sm != 12 or m == 12 else y - 1
    for _ in range(n_windows + 1):
        s = dt.date(sy, sm, 1)
        em, ey = (sm + 2 - 1) % 12 + 1, sy + (1 if sm + 2 > 12 else 0)
        e = (dt.date(ey + (em == 12), em % 12 + 1, 1) - dt.timedelta(days=1))
        if e < today:                                   # only finished windows are published
            out.append(f"{s.day:02d}{MONTHS[s.month-1]}{s.year}-{e.day:02d}{MONTHS[e.month-1]}{e.year}_form13f.zip")
        sm -= 3
        if sm <= 0:
            sm += 12; sy -= 1
    return out[:n_windows]


def list_zips(sess, n_windows):
    try:
        r = sess.get(LIST_URL, timeout=60)
        r.raise_for_status()
        links = re.findall(r'href="([^"]*?form13f\.zip)"', r.text, flags=re.I)
        urls = []
        for h in links:
            u = h if h.startswith("http") else "https://www.sec.gov" + h
            if u not in urls:
                urls.append(u)
        if urls:
            log(f"SEC listing: {len(urls)} zips")
            return urls[:n_windows]
    except Exception as e:
        log(f"  listing page unavailable ({e}); guessing file names")
    return [BASE + n for n in candidate_names(dt.date.today(), n_windows)]


# ── one zip → aggregate per (period, cusip) ────────────────────────
def _member(z, name):
    for n in z.namelist():
        if n.upper().endswith(name):
            return n
    raise KeyError(name)


def aggregate_zip(blob: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(blob))
    sub = pd.read_csv(z.open(_member(z, "SUBMISSION.TSV")), sep="\t", dtype=str,
                      usecols=lambda c: c.upper() in {"ACCESSION_NUMBER", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT", "FILING_DATE"})
    sub.columns = [c.upper() for c in sub.columns]
    sub = sub[sub.SUBMISSIONTYPE.str.upper().eq("13F-HR")].copy()
    sub["period"] = pd.to_datetime(sub.PERIODOFREPORT, format="mixed", dayfirst=False, errors="coerce").dt.strftime("%Y-%m-%d")
    sub["fd"] = pd.to_datetime(sub.FILING_DATE, format="mixed", errors="coerce")
    # one report per manager per quarter: keep the latest filing
    sub = sub.dropna(subset=["period"]).sort_values("fd").drop_duplicates(["CIK", "period"], keep="last")
    acc = sub.set_index("ACCESSION_NUMBER")[["CIK", "period"]]

    parts = []
    reader = pd.read_csv(z.open(_member(z, "INFOTABLE.TSV")), sep="\t", dtype=str, chunksize=750_000,
                         usecols=lambda c: c.upper() in {"ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"})
    for ch in reader:
        ch.columns = [c.upper() for c in ch.columns]
        ch = ch[ch.ACCESSION_NUMBER.isin(acc.index)]
        ch = ch[ch.SSHPRNAMTTYPE.str.upper().eq("SH") & ch.PUTCALL.fillna("").str.strip().eq("")]
        ch = ch.join(acc, on="ACCESSION_NUMBER")
        ch["CUSIP"] = ch.CUSIP.str.strip().str.upper().str[:9]
        ch["sh"] = pd.to_numeric(ch.SSHPRNAMT, errors="coerce").fillna(0)
        parts.append(ch.groupby(["period", "CUSIP", "CIK"], as_index=False).sh.sum())
    if not parts:
        return pd.DataFrame(columns=["period", "cusip", "holders", "shares"])
    h = pd.concat(parts).groupby(["period", "CUSIP", "CIK"], as_index=False).sh.sum()
    agg = h.groupby(["period", "CUSIP"]).agg(holders=("CIK", "nunique"), shares=("sh", "sum")).reset_index()
    agg = agg.rename(columns={"CUSIP": "cusip"})
    filers = sub.groupby("period").CIK.nunique()
    agg["filers"] = agg.period.map(filers)
    return agg


# ── CUSIP → ticker (OpenFIGI) ──────────────────────────────────────
def map_cusips(cusips, known: pd.DataFrame, budget_s=1500):
    todo = [c for c in cusips if c not in set(known.cusip)]
    if not todo:
        return known
    key = os.environ.get("OPENFIGI_KEY", "").strip()
    batch, pause = (100, 0.25) if key else (10, 2.5)       # 25 req/6s with key, 25/min without
    hdr = {"Content-Type": "application/json", **({"X-OPENFIGI-APIKEY": key} if key else {})}
    log(f"OpenFIGI: mapping {len(todo)} new CUSIPs ({'with' if key else 'without'} key)")
    rows, t0 = [], time.time()
    for i in range(0, len(todo), batch):
        if time.time() - t0 > budget_s:
            log(f"  time budget reached — {len(todo) - i} CUSIPs left for the next run")
            break
        part = todo[i:i + batch]
        body = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in part]
        for attempt in range(4):
            try:
                r = requests.post("https://api.openfigi.com/v3/mapping", json=body, headers=hdr, timeout=30)
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1)); continue
                r.raise_for_status()
                res = r.json(); break
            except Exception as e:
                log(f"  retry {attempt + 1}: {e}"); time.sleep(5 * (attempt + 1))
        else:
            continue
        for c, item in zip(part, res):
            data = item.get("data") or []
            eq = [d for d in data if d.get("marketSector") == "Equity" and d.get("ticker")]
            t = (eq[0]["ticker"] if eq else "")
            rows.append({"cusip": c, "ticker": t.replace("/", ".").upper()})
        time.sleep(pause)
    if rows:
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("cusip", keep="last")
    return known


# ── main ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarters", type=int, default=9)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    sess = requests.Session(); sess.headers.update(ua())

    urls = list_zips(sess, a.quarters + 1)
    for u in urls:
        name = u.rsplit("/", 1)[-1]
        out = CACHE / (name.replace(".zip", "") + ".csv.gz")
        if out.exists():
            continue
        log(f"Downloading {name} …")
        try:
            r = sess.get(u, timeout=600)
            if r.status_code == 404:
                log("  not published yet"); continue
            r.raise_for_status()
        except Exception as e:
            log(f"  failed: {e}"); continue
        log(f"  {len(r.content)/1e6:.0f} MB — aggregating")
        agg = aggregate_zip(r.content)
        agg.to_csv(out, index=False, compression="gzip")
        log(f"  {agg.period.nunique()} periods, {len(agg):,} (period, CUSIP) rows")
        time.sleep(1)

    files = sorted(CACHE.glob("*_form13f.csv.gz"))
    if not files:
        sys.exit("No 13F data downloaded.")
    allp = pd.concat([pd.read_csv(f, dtype={"cusip": str}) for f in files])
    # late filers land in later windows: add them up across zips
    tot = allp.groupby(["period", "cusip"], as_index=False)[["holders", "shares"]].sum()
    filers = allp.groupby(["period"]).filers.max()          # filers in the window that holds the deadline
    # drop quarters that are only partly filed (e.g. early filers in a window before the deadline)
    full = filers[filers >= 0.7 * filers.rolling(4, min_periods=1).max()].index
    periods = sorted(full)[-a.quarters:]
    tot = tot[tot.period.isin(periods)]
    log(f"Quarters kept: {', '.join(periods)}")

    # map widely held CUSIPs
    known = pd.read_csv(MAP, dtype=str).fillna("") if MAP.exists() else pd.DataFrame(columns=["cusip", "ticker"])
    latest = tot[tot.period == periods[-1]]
    want = latest[latest.holders >= MIN_HOLDERS_TO_MAP].sort_values("holders", ascending=False).cusip.tolist()
    known = map_cusips(want, known)
    known.to_csv(MAP, index=False)

    m = known[known.ticker.ne("")].set_index("cusip").ticker
    tot["ticker"] = tot.cusip.map(m)
    tot = tot.dropna(subset=["ticker"])
    # a company can have several CUSIPs (old/new after a reorg): keep the most held per quarter
    tot = tot.sort_values("holders").drop_duplicates(["ticker", "period"], keep="last")

    uni = ROOT / "config" / "universe.csv"
    if uni.exists():
        keep = set(pd.read_csv(uni).ticker.str.upper())
        broad = ROOT / "config" / "broad.csv"            # liquid stocks outside the ETF holdings
        if broad.exists():
            keep |= set(pd.read_csv(broad).ticker.str.upper())
        tot = tot[tot.ticker.isin(keep)]
    tot[["ticker", "period", "holders", "shares"]].sort_values(["ticker", "period"]).to_csv(OUT, index=False)
    log(f"Wrote {OUT}: {tot.ticker.nunique()} tickers × {len(periods)} quarters")


if __name__ == "__main__":
    main()
