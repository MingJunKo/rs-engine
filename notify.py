"""Telegram summary for the RS Engine. Needs TELEGRAM_TOKEN and TELEGRAM_CHAT_ID env vars."""
from __future__ import annotations

import html
import os
from pathlib import Path

import requests


def tg_send(method: str, data: dict, settings: dict | None = None, topic: str | None = None, files=None):
    """POST to the Telegram bot API. If settings["telegram_topics"] maps `topic` to a forum-topic id,
    the message goes into that topic of the group (TELEGRAM_CHAT_ID = the group's id)."""
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(f"Telegram not configured — preview [{topic or 'main'}]:\n{data.get('text') or data.get('caption') or ''}\n")
        return None
    data = {"chat_id": chat, **data}
    tid = ((settings or {}).get("telegram_topics") or {}).get(topic or "")
    if tid:
        data["message_thread_id"] = tid
    r = requests.post(f"https://api.telegram.org/bot{token}/{method}", timeout=60, data=data, files=files)
    if r.status_code == 400 and tid and "thread" in r.text.lower():
        # topic not found (e.g. TELEGRAM_CHAT_ID still points at the old private chat): send without the topic
        data.pop("message_thread_id", None)
        if files:
            for v in files.values():
                v[1].seek(0)
        r = requests.post(f"https://api.telegram.org/bot{token}/{method}", timeout=60, data=data, files=files)
    print(f"Telegram {method} [{topic or 'main'}]:", r.status_code, "ok" if r.status_code == 200 else r.text[:200])
    return r


def _uses_topic(settings: dict, topic: str) -> bool:
    return bool((settings.get("telegram_topics") or {}).get(topic))


def build_alist_message(p: dict) -> str:
    e = lambda s: html.escape(str(s))
    lines = [f"<b>A-list — {e(p['asof'])}</b>"]
    ex = p.get("exposure")
    if ex:
        lines.append(f"Exposure guide: <b>{ex['pct']}%</b> — {e(ex['note'])}")
    mw = p.get("mkt_wmacd") or {}
    if mw:
        lines.append("Index weekly MACD: " + " · ".join(f"{k} {'🟢 green' if v else '🔴 red'}" for k, v in mw.items()))
    al = p.get("alist") or []
    if not al:
        lines.append("<i>No name passes every A-list rule today.</i>")
    for a in al:
        tag = "" if a["go"] else f"  <i>({e(a['mkt'])} red: watch only)</i>"
        piv = f" · pivot {a['pivot']:.2f}" if a.get("pivot") else ""
        lines.append(f"<code>{e(a['t']):<6}</code> RS {a['rs']}{piv} · from {e(a['src'])}{tag}")
    return "\n".join(lines)


def build_parabolic_message(p: dict) -> str | None:
    sc = p.get("scans") or {}
    xs = sc.get("parabolic") or []
    if not xs:
        return None
    lines = [f"<b>Parabolic short watch — {html.escape(str(p['asof']))}</b>",
             "<i>Short only after the first red day / failed opening range; stop above the high.</i>", ""]
    lines += [f"<code>{x['symbol']:<6}</code> +{x['run_5d']}% in 5d · {x['ext_ema10']:+.0f}% vs 10-EMA{' · FIRST RED' if x['first_red'] else ''}"
              for x in xs[:10]]
    return "\n".join(lines)


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
    ex = p.get("exposure")
    if ex:
        lines.append(f"<b>Exposure guide: {ex['pct']}%</b> — {e(ex['note'])}  <i>({e(ex['why'])})</i>")
    mw = p.get("mkt_wmacd") or {}
    if mw:
        lines.append("Index weekly MACD: " + " · ".join(f"{k} {'🟢 green' if v else '🔴 red'}" for k, v in mw.items()))
    try:
        import environment as ENV
        bl = ENV.brief_line(p.get("environment"))
        if bl:
            lines.append(bl)
    except Exception:
        pass
    X = p.get("extras") or {}
    ev = X.get("econ") or []
    if ev:
        soon = [x for x in ev][:5]
        lines.append("📅 <b>US data ahead (SGT):</b> " + " | ".join(f"{e(x['when'])} {e(x['title'])}" for x in soon))
    al = p.get("alist") or []
    if al and not _uses_topic(settings, "alist"):
        lines += ["", "<b>A-list — open these first</b>"]
        for a in al:
            tag = "" if a["go"] else f"  <i>({e(a['mkt'])} red: watch only)</i>"
            piv = f" · pivot {a['pivot']:.2f}" if a.get("pivot") else ""
            lines.append(f"<code>{e(a['t']):<6}</code> RS {a['rs']}{piv} · from {e(a['src'])}{tag}")

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

    tm = X.get("theme_mco") or []
    ups = [t for t in tm if t["cross_up"]]
    dns = [t for t in tm if t["cross_down"] and t["top"]]
    if ups or dns:
        lines += ["", "<b>Theme breadth (McClellan inside the group)</b>"]
        if ups:
            lines.append("↗ crossed above zero: " + ", ".join(f"{e(t['theme'])} ({t['etf']}) {t['mco']:+d}" for t in ups[:8]))
        if dns:
            lines.append("↘ top themes losing participation: " + ", ".join(f"{e(t['theme'])} {t['mco']:+d}" for t in dns[:6]))
    n = settings.get("telegram_top_n", 15)
    lines += ["", f"<b>Leading stocks</b> (top {min(n, len(p['leaders']))} of {len(p['leaders'])})"]
    for x in p["leaders"][:n]:
        flags = "".join([" 🆕" if x["new"] else "", " ✓TT" if x["trend_template"] else "",
                         " ⚡33" if (x.get("code33") or {}).get("code33") else "", " ⚠ext" if x["extended"] else ""])
        lines.append(f"<code>{x['symbol']:<6}</code> {x['rs_rating']} · {e(x['theme'])}{flags}")

    runners = p.get("runners") or []
    if runners:
        st_ = p.get("stocks") or {}
        lines += ["", f"<b>Runner setups — RS 70+, ADR 4%+, base ≤ 3.5 ADRs</b> ({len(runners)})",
                  "  ".join(f"<code>{t}</code> {(st_.get(t) or [None, None, '–'])[2]}" for t in runners[:12])
                  + (" …" if len(runners) > 12 else "")]
    setups = [x for x in p["leaders"] if x.get("setup") and not x.get("er_soon")]
    if setups:
        lines += ["", f"<b>Setups — near pivot, tight, not extended</b> ({len(setups)})"]
        for x in setups[:10]:
            er = f" · ER {e(x['earnings'])}" if x.get("earnings") else ""
            lines.append(f"<code>{x['symbol']:<6}</code> {x['rs_rating']} · {x['pivot_dist']:+.1f}% from pivot{er}")
    cl = X.get("clusters") or []
    if cl:
        lines += ["", "<b>Clusters — 3+ names from one industry today</b>"]
        for c in cl[:5]:
            lines.append(f"<b>{e(c['industry'])}</b> ({c['n']}): " + ", ".join(f"{x['t']} {x['why']}" for x in c["names"][:6]))
    ins = X.get("insiders") or []
    if ins:
        lines += ["", "<b>Insider buying — last 7 days</b> (open-market, SEC Form 4)"]
        for x in ins[:6]:
            tag = " ◆leader" if x.get("leader") else ""
            lines.append(f"<code>{x['ticker']:<6}</code> {x['insiders']} insider{'s' if x['insiders'] > 1 else ''} · ${x['usd'] / 1e6:.1f}M"
                         f"{' @ ' + str(x['avg_price']) if x.get('avg_price') else ''} · RS {x.get('rs') or '–'}{tag}"
                         + (f" · {e(', '.join(x['titles'][:2]))}" if x.get("titles") else ""))
    c3 = X.get("code33") or []
    if c3:
        lines += ["", "<b>⚡ Code 33 — EPS accelerating 3 qtrs (≥20%), sales ≥10% and accelerating (RS 70+)</b>",
                  ", ".join(f"{x['t']} (EPS {'/'.join('turn' if g >= 999 else str(g) for g in x['eps'])}% · sales +{x['sales'][-1] if x.get('sales') else '–'}%)"
                            for x in c3[:10])]
    tg = p.get("turners") or []
    if tg:
        lines += ["", f"<b>Early turners — watchlist, not buys</b> ({len(tg)})",
                  "<i>Groups outside the top themes whose RS is turning up fast</i>"]
        for g in tg:
            br = (f" · breadth {g['breadth']}% ({g['breadth_chg']:+d})" if g.get("breadth") is not None
                  and g.get("breadth_chg") is not None else "")
            mc = next((t for t in tm if t["theme"] == g["theme"]), None)
            mtag = f" · McClellan {mc['mco']:+d}{' ↗ crossed up' if mc['cross_up'] else ''}" if mc else ""
            lines += ["", f"<b>{e(g['theme'])}</b> ({e(g['etf'])}) RS {g['rs']} ({(g['rs_chg_1w'] or 0):+d} 1W){br} · {e(g['quadrant'] or '')}{mtag}"]
            if not g["stocks"]:
                lines.append("  <i>no stock passes the filters yet</i>")
            for x in g["stocks"]:
                tags = "".join([" 📈RS-first" if x.get("rs_first") else (" RS-high" if x.get("rs_line_hi") else ""),
                                " ◆setup" if x.get("setup") else "", " ★leader" if x.get("leader") else ""])
                lines.append(f"  <code>{x['symbol']:<6}</code> {x['rs_rating']} ({(x['rs_chg_1w'] or 0):+d}) · "
                             f"{(x['off_high'] or 0):+.0f}% off high{tags}")
    sc = p.get("scans")
    if sc:
        fin = [x for x in sc["final"] if not x["biotech"]]
        f = sc["funnel"]
        lines += ["", f"<b>Weekly scan</b> ({len(fin)} names · {f['base']} in watchlists → {f['tight']} tight + {f['rs_scan']} RS scan)",
                  "<i>★ leading theme · ◆ RS leader · ✚ tech + fund screens · ↗ early-turner theme</i>"]
        fin.sort(key=lambda x: -((x["rs_rating"] or 0) + 8 * (x["lead_theme"] + x["leader"] + x["both"] + x["turner"])))
        for x in fin[:15]:
            tags = ("★" if x["lead_theme"] else "") + ("◆" if x["leader"] else "") + ("✚" if x["both"] else "") + ("↗" if x["turner"] else "")
            er = f" · ER {e(x['earnings'])}" if x.get("er_days") is not None and x["er_days"] <= 7 else ""
            lines.append(f"<code>{x['symbol']:<6}</code> {x['rs_rating'] if x['rs_rating'] is not None else '–'} · {e(x['theme'] or '')} {tags}{er}")
        if len(fin) > 15:
            lines.append(f"… +{len(fin) - 15} more on the Scans page")
        if sc.get("ep"):
            lines += ["", "<b>Episodic pivots</b> (last 2 sessions)"]
            for x in sc["ep"][:8]:
                cat = "earnings" if x["earnings_gap"] else (e(x["news"]["title"][:70]) if x.get("news") else "check news")
                lines.append(f"<code>{x['symbol']:<6}</code> gap +{x['gap']}% · {x['rvol']}× vol{' · neglected' if x['neglected'] else ''} — {cat}")
        if sc.get("parabolic") and not _uses_topic(settings, "parabolic"):
            lines += ["", "<b>Parabolic short watch</b>"] + [
                f"<code>{x['symbol']:<6}</code> +{x['run_5d']}% in 5d · {x['ext_ema10']:+.0f}% vs 10-EMA{' · FIRST RED' if x['first_red'] else ''}"
                for x in sc["parabolic"][:6]]
        url_ = settings.get("dashboard_url")
        if url_:
            lines.append(f'<a href="{e(url_.rstrip("/"))}/scans/">Open the Scans page</a>')
    notes = [x for x in p.get("notes", []) if x["tag"] not in ("Regime", "Setups")]
    if notes:
        lines += ["", "<b>Desk notes</b>"] + [f"• <b>{e(x['tag'])}</b> — {e(x['text'])}" for x in notes]
    reg = next((x for x in p.get("notes", []) if x["tag"] == "Regime"), None)
    if reg and "Sizing:" in reg["text"]:
        lines += ["", "<i>Sizing: " + e(reg["text"].split("Sizing:")[1].strip()) + "</i>"]
    url = settings.get("dashboard_url")
    if url:
        lines += ["", f'<a href="{e(url)}">Open dashboard</a>']
    return "\n".join(lines)


def _chunks(msg: str, limit: int = 3900) -> list[str]:
    """Telegram caps a message at 4096 characters: split at section breaks (blank lines)."""
    out, cur = [], ""
    for block in msg.split("\n\n"):
        if cur and len(cur) + len(block) + 2 > limit:
            out.append(cur); cur = block
        else:
            cur = f"{cur}\n\n{block}" if cur else block
    return out + ([cur] if cur else [])


def send_summary(payload: dict, tv_file: Path, settings: dict) -> None:
    msg = build_message(payload, settings)
    for part in _chunks(msg):
        tg_send("sendMessage", {"text": part, "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, "brief")
    stamp = payload["asof"][4:].replace(" ", "")
    if tv_file.exists() and tv_file.stat().st_size > 5:
        with open(tv_file, "rb") as f:
            tg_send("sendDocument", {"caption": "TradingView watchlist — Import list…"}, settings,
                    "files" if _uses_topic(settings, "files") else "brief",
                    files={"document": (f"RS_leaders_{stamp}.txt", f)})
    if _uses_topic(settings, "parabolic"):
        pm = build_parabolic_message(payload)
        if pm:
            tg_send("sendMessage", {"text": pm, "parse_mode": "HTML", "disable_web_page_preview": "true"}, settings, "parabolic")
    if _uses_topic(settings, "alist"):
        tg_send("sendMessage", {"text": build_alist_message(payload), "parse_mode": "HTML",
                                "disable_web_page_preview": "true"}, settings, "alist")
        af = tv_file.parent / "tradingview_alist.txt"
        if af.exists() and af.stat().st_size > 5:
            with open(af, "rb") as f:
                tg_send("sendDocument", {"caption": "A-list — Import list…"}, settings, "alist",
                        files={"document": (f"A_list_{stamp}.txt", f)})


def send_alert(payload: dict, alert: list, settings: dict) -> None:
    """Short midweek message, only sent when the regime changes or new setups appear."""
    e = lambda s: html.escape(str(s))
    lines = [f"<b>RS Engine alert — {e(payload['asof'])}</b>", ""] + [f"• {e(a)}" for a in alert]
    url = settings.get("dashboard_url")
    if url:
        lines += ["", f'<a href="{e(url)}">Open dashboard</a>']
    tg_send("sendMessage", {"text": "\n".join(lines), "parse_mode": "HTML", "disable_web_page_preview": "true"},
            settings, "alerts")
