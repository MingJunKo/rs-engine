#!/usr/bin/env python3
"""
Model book annotation tracker → Telegram topic "modelbook".

Morning: today's chart to study (the biggest winner not yet annotated, with a link that opens it), plus progress.
Evening (--evening): a last call, only if nothing was annotated today.

Progress comes from the model book page, which saves notes to the private gist VPA Arena already syncs to
(file modelbook_notes.json). A chart counts as annotated when it has a note or is ticked "Studied".
Env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, VPA_GIST_ID.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
SGT = ZoneInfo("Asia/Singapore")
FILE = "modelbook_notes.json"
GOAL = (1, 2)
TARGET = 300                                  # MJ's goal: 300 annotated charts
PAGE = "https://mingjunko.github.io/rs-engine/modelbook/"
STATE = ROOT / "data" / "history" / "modelbook_reminder.json"


def book() -> list:
    txt = (ROOT / "docs" / "modelbook" / "index.html").read_text()
    return json.loads(re.search(r"const D = (\{.*?\});\n", txt, re.S).group(1).replace("<\\/", "</"))["book"]


def notes(gist_id: str) -> dict | None:
    if not gist_id:
        return None
    try:
        r = requests.get(f"https://gist.githubusercontent.com/MingJunKo/{gist_id}/raw/{FILE}",
                         params={"t": dt.datetime.now().timestamp()}, timeout=30)
        if r.ok:
            return (r.json() or {}).get("notes", {})
        g = requests.get(f"https://api.github.com/gists/{gist_id}", timeout=30).json()
        f = (g.get("files") or {}).get(FILE)
        return json.loads(f["content"]).get("notes", {}) if f else {}
    except Exception as e:
        print("gist read failed:", e)
        return None


def day(ts: str | None):
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(SGT).date()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evening", action="store_true")
    ap.add_argument("--force", action="store_true", help="send even if already sent today")
    a = ap.parse_args()
    today = dt.datetime.now(SGT).date()
    mode = "evening" if a.evening else "morning"
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if st.get(mode) == today.isoformat() and not a.force:
        print(f"{mode} reminder already sent today"); return

    B = book()
    settings = json.loads((ROOT / "config" / "settings.json").read_text())
    N = notes(os.environ.get("VPA_GIST_ID") or settings.get("vpa_gist_id"))
    synced = N is not None
    N = N or {}
    ids = {x["id"] for x in B}
    done = {i for i, o in N.items() if i in ids and ((o.get("n") or "").strip() or o.get("d"))}
    days = {}
    for i in done:
        d = day(N[i].get("t"))
        if d:
            days[d] = days.get(d, 0) + 1
    n_today, n_yday = days.get(today, 0), days.get(today - dt.timedelta(days=1), 0)
    streak, d = 0, today if days.get(today) else today - dt.timedelta(days=1)
    while days.get(d):
        streak += 1; d -= dt.timedelta(days=1)
    week = sum(v for k, v in days.items() if (today - k).days < 7)
    studyable = [x for x in B if x.get("chart")]
    todo = sorted([x for x in studyable if x["id"] not in done], key=lambda x: -(x.get("run_gain") or 0))

    if mode == "evening" and n_today >= GOAL[0]:
        print(f"goal met today ({n_today}) — no last call"); return

    e = html.escape
    link = lambda x: f'<a href="{PAGE}#study={x["id"]}">{e(x["symbol"])}</a>'
    L = []
    if mode == "morning":
        L.append("<b>📚 Model book — today's study</b>")
        for k, x in enumerate(todo[:GOAL[1]]):
            L.append(f"{'①' if k == 0 else '② optional'} {link(x)} +{x['run_gain']}% · {e(x.get('sector') or '')}"
                     f"{' · ' + e(x['theme']) if x.get('theme') else ''} · breakout {dt.date.fromisoformat(x['date']):%d %b %Y}")
        L.append("<i>Mark the base, the tell before the breakout, where you'd have bought, where the stop goes.</i>")
    else:
        x = todo[0] if todo else None
        L.append("<b>📚 Last call — no model book chart annotated today</b>")
        if x:
            L.append(f"Five minutes: {link(x)} +{x['run_gain']}% · {e(x.get('sector') or '')}")
    left = max(0, TARGET - len(done))
    pace = f" · {left} to go: {-(-left // GOAL[1])}–{left // GOAL[0]} days at {GOAL[0]}–{GOAL[1]}/day" if left else " · 🎯 goal reached!"
    bar = "▰" * min(10, len(done) * 10 // TARGET) + "▱" * (10 - min(10, len(done) * 10 // TARGET))
    if synced:
        L.append(f"\n{bar} <b>{len(done)}</b>/{TARGET} · today {n_today} · yesterday {n_yday} · "
                 f"7 days {week} · streak {streak} day{'s' if streak != 1 else ''}{pace}")
    else:
        L.append("\n<i>Progress not visible yet: open VPA Arena → Stats → Sync my devices on the device you annotate on, "
                 "then the model book saves notes to the same private gist.</i>")
    sys.path.insert(0, str(ROOT))
    from notify import tg_send
    topic = "modelbook" if (settings.get("telegram_topics") or {}).get("modelbook") else "vpa"
    tg_send("sendMessage", {"text": "\n".join(L), "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, topic)
    st[mode] = today.isoformat()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))
    print("\n".join(L))


if __name__ == "__main__":
    main()
