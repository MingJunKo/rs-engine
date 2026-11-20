#!/usr/bin/env python3
"""
Watchdog: did everything actually run? ("Reliability beats sophistication.")

Checks, each morning after the engine's 7:45am SGT run:
  • the RS Engine output is for the last US trading day (docs/data/alist.json)
  • the pre-market scan saved that day's file (docs/data/premarket.json)
  • the stretch monitor updated (docs/data/stretch.json)
  • the insider reader is keeping up (data/history/insider_state.json)
  • no GitHub Actions run failed in the last 26 hours
Silent when all is well (plus a one-line "all green" on Saturdays); a Telegram warning when anything is stale or failed.
US market holidays can cause a false "stale" warning — the message says so.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")
SGT = ZoneInfo("Asia/Singapore")


def last_session(now_ny: dt.datetime) -> dt.date:
    d = now_ny.date() if now_ny.hour >= 17 else now_ny.date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    now_ny = dt.datetime.now(NY)
    L = last_session(now_ny)
    issues, ok = [], []

    a = read_json(ROOT / "docs" / "data" / "alist.json") or {}
    if a.get("date") == L.isoformat():
        ok.append("RS Engine")
    else:
        issues.append(f"RS Engine output is from {a.get('date') or 'unknown'}, expected {L:%a %d %b}")

    pm = read_json(ROOT / "docs" / "data" / "premarket.json") or {}
    if pm.get("date") == L.isoformat():
        ok.append("pre-market scan")
    else:
        issues.append(f"Pre-market scan last saved {pm.get('date') or 'never'}, expected {L:%a %d %b}")

    st = read_json(ROOT / "docs" / "data" / "stretch.json") or {}
    if st.get("asof") == L.strftime("%a %d %b %Y"):
        ok.append("stretch monitor")
    else:
        issues.append(f"Stretch monitor is at {st.get('asof') or 'never run'}, expected {L:%a %d %b}")

    ins = read_json(ROOT / "data" / "history" / "insider_state.json") or {}
    if ins.get("last_day"):
        lag = (L - dt.date.fromisoformat(ins["last_day"])).days
        (ok.append("insider reader") if lag <= 4 else
         issues.append(f"Insider reader stuck at {ins['last_day']} ({lag} days behind)"))

    # failed workflow runs in the last 26 hours
    tok, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if tok and repo:
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=26)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs",
                             params={"created": f">{since}", "per_page": 100},
                             headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}, timeout=30)
            runs = r.json().get("workflow_runs", [])
            bad = [x for x in runs if x.get("conclusion") in ("failure", "timed_out")]
            for x in bad[:6]:
                t = dt.datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")).astimezone(SGT)
                issues.append(f'GitHub run failed: {x["name"]} ({t:%a %H:%M} SGT) — {x["html_url"]}')
            if runs and not bad:
                ok.append(f"{len(runs)} GitHub runs, none failed")
        except Exception as e:
            issues.append(f"Couldn't read GitHub run history: {e}")

    print("OK:", ", ".join(ok))
    print("Issues:", *issues, sep="\n  ")
    saturday = dt.datetime.now(SGT).weekday() == 5
    if not issues and not saturday:
        return
    settings = json.loads((ROOT / "config" / "settings.json").read_text())
    sys.path.insert(0, str(ROOT))
    from notify import tg_send
    import html
    if issues:
        text = "<b>⚠️ Watchdog — something didn't run</b>\n" + "\n".join(f"• {html.escape(i)}" for i in issues)
        text += ("\n\n<i>If yesterday was a US market holiday, ignore the 'expected' lines. "
                 "To rerun: GitHub → Actions → pick the workflow → Run workflow.</i>")
    else:
        text = "✅ <b>Watchdog</b> — all green this week: " + ", ".join(ok)
    tg_send("sendMessage", {"text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, "alerts")


if __name__ == "__main__":
    main()
