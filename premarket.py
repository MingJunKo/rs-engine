#!/usr/bin/env python3
"""
Pre-market Episodic Pivot scan. Runs on US trading days shortly before the open (~8:30–9:25 New York).

Lists stocks in the RS Engine universe gapping ≥ 8% pre-market on real volume, adds their catalyst
(earnings today / latest headlines) and sends a Telegram alert. Writes docs/data/premarket.json, which the
Scans page reads live.

    python premarket.py            # only acts inside the pre-market window, once per day
    python premarket.py --force    # run now regardless of the clock (testing)
"""
import argparse
import datetime as dt
import html
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data" / "premarket.json"
NY = ZoneInfo("America/New_York")
MIN_GAP, MIN_PM_VOL, MIN_PM_DVOL = 8.0, 100_000, 1_000_000


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true", help="diagnostics only: no file, no Telegram")
    a = ap.parse_args()
    now = dt.datetime.now(NY)
    today = now.date().isoformat()
    if not a.force:
        if now.weekday() >= 5 or not (dt.time(8, 0) <= now.time() < dt.time(9, 29)):
            log(f"Outside the pre-market window ({now:%a %H:%M} New York) — nothing to do."); return
        if OUT.exists() and json.loads(OUT.read_text()).get("date") == today:
            log("Already ran today."); return

    st = pd.read_csv(ROOT / "docs" / "data" / "stocks.csv")
    st = st[st.price >= 3]
    prev = st.set_index("ticker").price.to_dict()
    dvol = st.set_index("ticker").dollar_vol.to_dict() if "dollar_vol" in st else {}
    tickers = list(prev)
    log(f"{len(tickers)} stocks · scanning pre-market")

    import yfinance as yf
    rows, seen_pm, seen_any, all_gaps = [], 0, 0, []
    for i in range(0, len(tickers), 150):
        part = tickers[i:i + 150]
        try:
            df = yf.download([t.replace(".", "-") for t in part], period="5d", interval="5m", prepost=True,
                             group_by="ticker", threads=True, progress=False, auto_adjust=False)
        except Exception as e:
            log(f"  chunk {i // 150 + 1} failed: {e}"); continue
        if df is None or df.empty:
            continue
        for t in part:
            try:
                sub = df[t.replace(".", "-")] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            seen_any += 1
            ix = sub.index.tz_convert(NY) if sub.index.tz is not None else sub.index.tz_localize("UTC").tz_convert(NY)
            pm = sub[(ix.date == now.date()) & (ix.time < dt.time(9, 30))]
            if pm.empty:
                continue
            seen_pm += 1
            last, vol = float(pm.Close.iloc[-1]), float(pm.Volume.fillna(0).sum())
            gap = (last / prev[t] - 1) * 100
            all_gaps.append((gap, t, vol))
            # Yahoo often reports 0 volume on extended-hours bars: then fall back to the stock's normal liquidity
            liquid = (vol >= MIN_PM_VOL and last * vol >= MIN_PM_DVOL) if vol > 0 else (dvol.get(t, 0) or 0) >= 5e6
            if gap >= MIN_GAP and liquid:
                rows.append({"symbol": t, "gap": round(gap, 1), "pm_price": round(last, 2), "prev_close": round(prev[t], 2),
                             "pm_vol_k": int(vol / 1000)})
        time.sleep(1)
    log(f"{seen_any} tickers returned data, {seen_pm} had pre-market bars today; "
        f"{sum(1 for g in all_gaps if g[2] > 0)} with pre-market volume")
    log("largest gaps: " + ", ".join(f"{t} {g:+.1f}% ({int(v/1000)}K)" for g, t, v in sorted(all_gaps, reverse=True)[:10]))
    rows.sort(key=lambda r: -r["gap"])
    rows = rows[:25]
    log(f"{len(rows)} pre-market gappers ≥ {MIN_GAP}%")

    # catalysts: earnings today (or last night) from the cached Nasdaq calendar, plus headlines
    cal_p = ROOT / "data" / "history" / "earnings_cal.json"
    cal = json.loads(cal_p.read_text()) if cal_p.exists() else {}
    prev_bd = (pd.Timestamp(today) - pd.offsets.BDay(1)).date().isoformat()
    sys.path.insert(0, str(ROOT))
    from scans import catalysts
    news = catalysts([r["symbol"] for r in rows])
    meta = st.set_index("ticker")
    for r in rows:
        r["earnings"] = cal.get(r["symbol"]) in (today, prev_bd)
        r["news"] = news.get(r["symbol"], [])
        for k in ("name", "theme", "rs_rating"):
            if k in meta.columns:
                v = meta.at[r["symbol"], k]
                r[k] = None if pd.isna(v) else (int(v) if k == "rs_rating" else v)

    if a.dry:
        log("dry run: " + ", ".join(f"{r['symbol']} +{r['gap']}%" for r in rows)); return
    OUT.write_text(json.dumps({"date": today, "asof": now.strftime("%a %d %b %H:%M New York"), "rows": rows},
                              separators=(",", ":"), default=str))
    # TradingView import file for the open: the alert-worthy gappers first, then the rest
    good = [r["symbol"] for r in rows if r["earnings"] or (r.get("rs_rating") or 0) >= 70]
    rest = [r["symbol"] for r in rows if r["symbol"] not in good]
    tv = [f"###Pre-market {now:%d %b}: earnings or RS 70+", ",".join(good)] if good else []
    tv += [f"###Pre-market {now:%d %b}: other gappers", ",".join(rest)] if rest else []
    (OUT.parent / "tradingview_premarket.txt").write_text("\n".join(tv) + "\n" if tv else "")
    send(rows, now)


def send(rows, now):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not rows:
        log("No gappers — no Telegram message."); return
    e = lambda s: html.escape(str(s))
    # only the gappers worth a look: earnings-driven or already strong (RS ≥ 70); the rest are on the Scans page
    good = [r for r in rows if r["earnings"] or (r.get("rs_rating") or 0) >= 70]
    if not good:
        log("Gappers found, but none earnings-driven or RS ≥ 70 — no Telegram message."); return
    lines = [f"<b>Pre-market EP watch — {now:%a %d %b}</b>", "<i>Gapping ≥ 8% on volume. Check the catalyst before the open.</i>", ""]
    for r in good[:8]:
        tag = " 📊earnings" if r["earnings"] else ""
        rs = f" · RS {r['rs_rating']}" if r.get("rs_rating") is not None else ""
        pv = f" · {r['pm_vol_k']:,}K sh" if r["pm_vol_k"] else ""
        lines.append(f"<code>{r['symbol']:<6}</code> +{r['gap']}%{pv}{rs}{tag}")
        if r["news"]:
            n = r["news"][0]
            lines.append(f"   ↳ {e(n['title'][:110])}")
    if len(rows) > len(good[:8]):
        lines.append(f"<i>+{len(rows) - len(good[:8])} other gappers on the Scans page</i>")
    try:
        url = json.loads((ROOT / "config" / "settings.json").read_text()).get("dashboard_url")
        if url:
            lines += ["", f'<a href="{e(url.rstrip("/"))}/scans/#ep">Open scans</a>']
    except Exception:
        pass
    msg = "\n".join(lines)
    if not token or not chat:
        log("Telegram not configured. Preview:\n" + msg); return
    import requests
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30, data={
        "chat_id": chat, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": "true"})
    log("Telegram:", r.status_code)


if __name__ == "__main__":
    main()
