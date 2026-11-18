"""Daily VPA Arena practice reminder → Telegram (topic "vpa").

Reads the game's private sync gist (VPA Arena → Stats → Sync my devices), works out today's rounds,
the streak and the weakest skill, and posts a short nudge. Two runs a day (Singapore time):
  20:00  reminder with stats            python vpa_reminder.py
  23:00  last call, only if today's goal is not met yet    python vpa_reminder.py --last-call
Env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, VPA_GIST_ID (or settings.json "vpa_gist_id").
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
SGT = ZoneInfo("Asia/Singapore")
GOAL = 8
GAME_URL = "https://mingjunko.github.io/vpa-arena/"
SKILLS = {
    "momentum": "Trend & momentum reading", "discipline": "No-trade discipline", "climax": "Climactic volume",
    "stopping": "Stopping volume", "no_demand": "No demand", "no_supply": "No supply", "absorption": "Effort vs result",
    "prediction": "Outcome prediction", "breakouts": "Breakout judgement", "risk": "Risk management",
    "sizing": "Sizing & pyramiding", "timing": "Entry timing", "planning": "Trade planning",
}
MODES = {"trigger": "Trigger", "plan": "Plan", "breakout": "Breakout", "trade": "Trade", "position": "Position"}


def load_state(gist_id: str) -> dict:
    """Raw gist URL first (no API rate limit, works for secret gists), API as a fallback."""
    r = requests.get(f"https://gist.githubusercontent.com/MingJunKo/{gist_id}/raw/vpa_arena_sync.json",
                     params={"t": dt.datetime.now().timestamp()}, timeout=30)
    if r.ok:
        return r.json()
    g = requests.get(f"https://api.github.com/gists/{gist_id}", timeout=30).json()
    f = g["files"]["vpa_arena_sync.json"]
    txt = requests.get(f["raw_url"], timeout=30).text if f.get("truncated") else f["content"]
    return json.loads(txt)


def day(ts: str) -> dt.date:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(SGT).date()


def level(xp: int) -> int:
    lv, acc = 1, 0
    while xp >= acc + 300 + 250 * (lv - 1):
        acc += 300 + 250 * (lv - 1)
        lv += 1
    return lv


def build(state: dict, last_call: bool) -> str | None:
    today = dt.datetime.now(SGT).date()
    rounds = state.get("rounds", [])
    by_day: dict[dt.date, list] = {}
    for r in rounds:
        by_day.setdefault(day(r["ts"]), []).append(r)
    n_today = len(by_day.get(today, []))
    if last_call and n_today >= GOAL:
        return None                                  # goal met: no last call

    p = state.get("profile", {})
    streak, best, freezes = p.get("streak", 0), p.get("bestStreak", 0), p.get("freezes", 0)
    lg = dt.date.fromisoformat(p["lastGoalDay"]) if p.get("lastGoalDay") else None
    if lg == today:
        st_line = f"🔥 Streak <b>{streak}</b> — today's goal done. Best {best}."
    elif lg == today - dt.timedelta(days=1):
        st_line = f"🔥 Streak <b>{streak}</b> is alive — {GOAL - n_today} more rounds today makes it <b>{streak + 1}</b>. Best {best}."
    elif lg and (today - lg).days - 1 <= freezes:
        st_line = (f"🧊 Streak <b>{streak}</b> is being held by {(today - lg).days - 1} freeze(s) — "
                   f"play today or it's gone. Best {best}.")
    else:
        since = f"last goal day {lg:%d %b}" if lg else "no goal day yet"
        st_line = f"Streak <b>0</b> ({since}) — {GOAL} rounds today starts a new one. Best {best}."

    week = []
    for i in range(6, -1, -1):
        d = today - dt.timedelta(days=i)
        n = len(by_day.get(d, []))
        week.append("✅" if n >= GOAL else ("🟨" if n else "⬜"))
    days7 = sum(1 for i in range(7) if by_day.get(today - dt.timedelta(days=i)))

    def acc(lo: int, hi: int):
        rs = [r for d, xs in by_day.items() if lo <= (today - d).days < hi for r in xs]
        return (round(100 * sum(1 for r in rs if r.get("correct")) / len(rs)), len(rs)) if rs else (None, 0)
    a7, n7 = acc(0, 7)
    a_prev, n_prev = acc(7, 28)

    sk = [(k, v["correct"] / v["seen"], v["seen"]) for k, v in (state.get("skills") or {}).items() if v.get("seen", 0) >= 3]
    weak = min(sk, key=lambda x: x[1]) if sk else None
    strong = max(sk, key=lambda x: x[1]) if sk else None

    modes = {}
    for r in rounds[-60:]:
        m = modes.setdefault(r.get("mode"), [0, 0]); m[0] += 1; m[1] += bool(r.get("correct"))
    weak_mode = min(((k, c / n) for k, (n, c) in modes.items() if n >= 4 and k in MODES), key=lambda x: x[1], default=None)

    e = lambda s: html.escape(str(s))
    lines = [("⏰ <b>Last call — VPA Arena</b>" if last_call else "📈 <b>VPA Arena — daily reps</b>"),
             f"Today: <b>{n_today}/{GOAL}</b> rounds" + (" ✅" if n_today >= GOAL else ""),
             st_line,
             f"Last 7 days: {''.join(week)}  ({days7}/7 days)"]
    if a7 is not None:
        trend = f" vs {a_prev}% the 3 weeks before" if a_prev is not None else ""
        lines.append(f"Accuracy 7d: {a7}% on {n7} rounds{trend}")
    lines.append(f"Level {level(int(p.get('xp', 0)))} · {p.get('xp', 0):,} XP · {len(rounds)} rounds all-time · freezes {freezes} 🧊")
    if weak:
        focus = f"Focus today: <b>{e(SKILLS.get(weak[0], weak[0]))}</b> ({round(100 * weak[1])}% on {weak[2]})"
        if weak_mode:
            focus += f" — try a few <b>{MODES[weak_mode[0]]}</b> rounds ({round(100 * weak_mode[1])}% recently)"
        lines.append(focus)
    if strong and strong != weak:
        lines.append(f"Strongest: {e(SKILLS.get(strong[0], strong[0]))} ({round(100 * strong[1])}%)")
    if last_call:
        lines.append(f"<i>{GOAL - n_today} rounds ≈ 10 minutes. Pattern recognition is built on reps, not intensity.</i>")
    lines += ["", f'<a href="{GAME_URL}">▶ Play now</a>']
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-call", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    settings = json.loads((ROOT / "config" / "settings.json").read_text())
    gid = os.environ.get("VPA_GIST_ID") or settings.get("vpa_gist_id")
    if not gid:
        print("No VPA_GIST_ID — turn on Sync in VPA Arena and add the gist id."); return
    msg = build(load_state(gid), a.last_call)
    if not msg:
        print("Goal already met today — no last call."); return
    if a.dry:
        print(msg); return
    from notify import tg_send
    tg_send("sendMessage", {"text": msg, "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, "vpa")


if __name__ == "__main__":
    main()
