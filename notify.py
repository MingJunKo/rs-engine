"""Telegram summary for the RS Engine. Needs TELEGRAM_TOKEN and TELEGRAM_CHAT_ID env vars."""
from __future__ import annotations

import html
import os
from pathlib import Path

import requests


def _arrow(v):
    if v is None:
        return ""
    return "▲" if v > 0 else ("▼" if v < 0 else "·")


def build_message(p: dict, settings: dict) -> str:
    e = lambda s: html.escape(str(s))
    r = p["regime"]
    sc = p["style"]["chg"]
    lines = [f"<b>RS Engine — {e(p['asof'])}</b>" + ("  <i>(demo data)</i>" if p.get("demo") else ""), ""]
    lines.append(f"<b>Regime: {e(r['label'])}</b>  ({r['risk_on']}/{r['gauges_n']} risk-on)")
    lines.append(f"Style 1M vs {e(p['benchmark'])}: Growth {sc['growth']:+.1f}% · "
                 f"Cyclical {sc['cyclical']:+.1f}% · Defensive {sc['defensive']:+.1f}%")
    lines.append(f"Breadth: {r['breadth']:.0f}% of universe above 50-DMA ({round(r['breadth_chg']) + 0:+d} pts 1M)")

    idx = {x["symbol"]: x for x in p["etfs"] if x["category"] == "Index"}
    bm = [f"{s} {idx[s]['chg_1m']:+.1f}%" for s in ("RSP", "SPY", "QQQ", "IWM") if s in idx]
    if bm:
        lines.append("1M: " + " · ".join(bm))

    spdr = sorted([x for x in p["etfs"] if x["category"] == "SPDR Sector" and x["rs_rating"] is not None],
                  key=lambda x: -x["rs_rating"])
    if spdr:
        lines += ["", "<b>Sectors</b>",
                  "Leading: " + ", ".join(f"{x['symbol']} {x['rs_rating']}" for x in spdr[:3]),
                  "Lagging: " + ", ".join(f"{x['symbol']} {x['rs_rating']}" for x in spdr[-3:])]

    lines += ["", "<b>Top themes</b>"]
    for i, t in enumerate([t for t in p["themes"] if t["top"]], 1):
        lines.append(f"{i}. {e(t['theme'])} ({t['symbol']}) {t['rs_rating']} {_arrow(t['rs_chg_1w'])}")

    n = settings.get("telegram_top_n", 15)
    lines += ["", f"<b>Leading stocks</b> (top {min(n, len(p['leaders']))} of {len(p['leaders'])})"]
    for x in p["leaders"][:n]:
        flags = "".join([" 🆕" if x["new"] else "", " ✓TT" if x["trend_template"] else "",
                         " ⚠ext" if x["extended"] else ""])
        lines.append(f"<code>{x['symbol']:<6}</code> {x['rs_rating']} · {e(x['theme'])}{flags}")

    setups = [x for x in p["leaders"] if x.get("setup") and not x.get("er_soon")]
    if setups:
        lines += ["", f"<b>Setups — near pivot, tight, not extended</b> ({len(setups)})"]
        for x in setups[:10]:
            er = f" · ER {e(x['earnings'])}" if x.get("earnings") else ""
            lines.append(f"<code>{x['symbol']:<6}</code> {x['rs_rating']} · {x['pivot_dist']:+.1f}% from pivot{er}")
    soon = [x["symbol"] for x in p["leaders"] if x.get("er_soon")]
    if soon:
        lines += ["", "⚠ Earnings within 7 days: " + ", ".join(soon)]

    new = [x["symbol"] for x in p["leaders"] if x["new"]]
    if new:
        lines += ["", "New this week: " + ", ".join(new)]
    url = settings.get("dashboard_url")
    if url:
        lines += ["", f'<a href="{e(url)}">Open dashboard</a>']
    return "\n".join(lines)


def send_summary(payload: dict, tv_file: Path, settings: dict) -> None:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    msg = build_message(payload, settings)
    if not token or not chat:
        print("Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID missing). Message preview:\n")
        print(msg)
        return
    api = f"https://api.telegram.org/bot{token}"
    r = requests.post(f"{api}/sendMessage", timeout=30, data={
        "chat_id": chat, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": "true"})
    print("Telegram message:", r.status_code, r.text[:200] if r.status_code != 200 else "ok")
    if tv_file.exists() and tv_file.stat().st_size > 5:
        with open(tv_file, "rb") as f:
            r = requests.post(f"{api}/sendDocument", timeout=60,
                              data={"chat_id": chat, "caption": "TradingView watchlist — Import list…"},
                              files={"document": (f"RS_leaders_{payload['asof'][4:].replace(' ', '')}.txt", f)})
        print("Telegram file:", r.status_code, "ok" if r.status_code == 200 else r.text[:200])
