"""
Extra signals borrowed from Brad Jelinek's "Scattered Focus" build, adapted to the RS Engine:

  theme_mcclellan   McClellan oscillator inside each theme (its member stocks) and zero-line crosses —
                    participation turning up inside a group, not just the ETF price
  clusters          3+ stocks from one industry flagged the same day (EP, setup, new 52-week high)
  code33            O'Neil's "code 33": EPS and sales growth accelerating three quarters in a row
  econ_calendar     high-impact US releases (CPI, FOMC, payrolls…) from the ForexFactory weekly feed
  insider_clusters  open-market insider buying clusters, read from data/history/insider_buys.csv (insiders.py)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SGT = dt.timezone(dt.timedelta(hours=8))


# ───────────────────────── theme breadth
def _mco(close: pd.DataFrame, min_n: int) -> pd.Series:
    r = close.pct_change()
    adv, dec = (r > 0).sum(axis=1), (r < 0).sum(axis=1)
    rana = (adv - dec) / (adv + dec).replace(0, np.nan) * 1000
    out = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    out[r.notna().sum(axis=1) < min_n] = np.nan
    return out


def theme_mcclellan(close: pd.DataFrame, stk: pd.DataFrame, themes: list, min_members: int = 8, lookback: int = 3) -> list:
    """themes = payload['themes'] rows. Returns one row per theme with enough members."""
    out = []
    for t in themes:
        name = t["theme"]
        mem = [s for s, th in stk.all_themes.items() if name in (th or []) and s in close.columns]
        if len(mem) < min_members:
            continue
        m = _mco(close[mem], max(5, len(mem) // 2)).dropna()
        if len(m) < 45:
            continue
        now = float(m.iloc[-1])
        prev = m.iloc[-(lookback + 1):-1]                      # the sessions before today
        up = bool(now > 0 and (prev <= 0).any())                  # crossed above zero in the last `lookback` days
        dn = bool(now < 0 and (prev >= 0).any())
        days = None
        if up or dn:
            sign = np.sign(m)
            flips = np.flatnonzero(sign.values[1:] != sign.values[:-1])
            days = int(len(m) - 2 - flips[-1]) if len(flips) else None
        out.append({"theme": name, "etf": t["symbol"], "members": len(mem), "mco": round(now), "mco_5d": round(float(m.iloc[-6])),
                    "cross_up": up, "cross_down": dn, "days_since": days, "top": t.get("top", False),
                    "rs": t.get("rs_rating")})
    return sorted(out, key=lambda x: -x["mco"])


# ───────────────────────── industry clusters
def clusters(stk: pd.DataFrame, close: pd.DataFrame, ep_today: list, setups: list, min_names: int = 3) -> list:
    hi = close.rolling(252, min_periods=120).max()
    new_hi = set(close.columns[(close.iloc[-1] >= hi.iloc[-1] * 0.999) & close.iloc[-1].notna()])
    flags = {}
    for t in stk.index:
        why = []
        if t in ep_today:
            why.append("EP")
        if t in setups:
            why.append("setup")
        if t in new_hi:
            why.append("new high")
        if why:
            flags[t] = why
    groups = {}
    for t, why in flags.items():
        ind = stk.at[t, "industry"] or stk.at[t, "theme"] or ""
        if ind:
            groups.setdefault(ind, []).append((t, why))
    out = []
    for ind, xs in groups.items():
        if len(xs) >= min_names:
            xs.sort(key=lambda x: -(stk.at[x[0], "rs_rating"] if pd.notna(stk.at[x[0], "rs_rating"]) else 0))
            out.append({"industry": ind, "n": len(xs),
                        "names": [{"t": t, "why": "/".join(w), "rs": None if pd.isna(stk.at[t, "rs_rating"]) else int(stk.at[t, "rs_rating"])}
                                  for t, w in xs[:8]],
                        "eps": sum("EP" in w for _, w in xs), "highs": sum("new high" in w for _, w in xs)})
    return sorted(out, key=lambda x: (-x["eps"], -x["n"]))


# ───────────────────────── code 33
def code33(F: dict, yoy) -> dict:
    """O'Neil's code 33, adapted to the data we have.
    EPS: YoY growth accelerating three quarters in a row (latest > 0).
    Sales: the feed keeps only ~5 quarters (one YoY reading), so sales must be up YoY AND quarter-on-quarter
    growth must be accelerating over the last two quarters (or YoY accelerating 3 in a row where history allows)."""
    out = {}
    acc3 = lambda g: len(g) == 3 and g[0] < g[1] < g[2] and g[2] > 0
    for t, f in F.items():
        e = [x[2] for x in yoy(f.get("eps_q") or []) if x[2] is not None][-3:]
        sy = [x[2] for x in yoy(f.get("sales_q") or []) if x[2] is not None]
        sq = [r[1] for r in (f.get("sales_q") or []) if r[1]][-3:]
        qoq = [(b / a - 1) * 100 for a, b in zip(sq, sq[1:]) if a and a > 0]
        ea = acc3(e) and e[-1] >= 20                                   # CANSLIM floor: latest EPS ≥ +20% YoY
        sa = (acc3(sy[-3:]) or (len(qoq) == 2 and 0 < qoq[0] < qoq[1])) and bool(sy) and sy[-1] >= 10
        if ea or sa:
            out[t] = {"eps": [round(x) for x in e], "sales": [round(x) for x in sy[-3:]],
                      "sales_qoq": [round(x, 1) for x in qoq], "eps_acc": ea, "sales_acc": bool(sa), "code33": bool(ea and sa)}
    return out


# ───────────────────────── economic calendar
# Nasdaq has no impact rating: keep the US releases that move markets, under one short label each
US_FAMILIES = [("nonfarm payrolls", "Nonfarm payrolls"), ("unemployment rate", "Unemployment rate"),
               ("core cpi", "CPI"), ("cpi", "CPI"), ("core ppi", "PPI"), ("ppi", "PPI"),
               ("core pce", "Core PCE"), ("pce price", "PCE"), ("retail sales", "Retail sales"), ("gdp", "GDP"),
               ("interest rate decision", "FOMC decision"), ("fomc statement", "FOMC decision"),
               ("fomc economic projections", "FOMC decision"), ("fomc meeting minutes", "FOMC minutes"),
               ("ism manufacturing pmi", "ISM manufacturing"), ("ism non-manufacturing pmi", "ISM services"),
               ("ism services pmi", "ISM services"), ("jolts job openings", "JOLTS"),
               ("initial jobless claims", "Jobless claims"), ("fed chair", "Fed Chair speaks")]
NOT_HIGH = ("cleveland", "atlanta", "chicago", "dallas", "richmond", "kansas", "philadelphia", "new york", "4-week",
            "continuing", "gdpnow", "sales ex", "gdp sales", "gdp price", "pce prices", "member")


def _nasdaq_day(day: dt.date) -> list:
    """High-impact US releases on `day` (New York date) from Nasdaq's economic calendar.
    Quirks: its `date` parameter returns the PREVIOUS day's events, and times are New York time."""
    import html as _h
    import requests
    from zoneinfo import ZoneInfo
    try:
        r = requests.get("https://api.nasdaq.com/api/calendar/economicevents",
                         params={"date": (day + dt.timedelta(days=1)).isoformat()},
                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=20)
        rows = ((r.json().get("data") or {}).get("rows") or []) if r.ok else []
    except Exception as e:
        print(f"  econ calendar (nasdaq {day}): {e}")
        return []
    groups = {}
    for x in rows:
        low = _h.unescape(x.get("eventName") or "").strip().lower()
        if x.get("country") != "United States" or any(k in low for k in NOT_HIGH):
            continue
        lab = next((l for k, l in US_FAMILIES if k in low), None)
        if not lab:
            continue
        t = (x.get("gmt") or "").strip()
        try:
            hh, mm = (int(v) for v in t.split(":"))
            when = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=ZoneInfo("America/New_York")).astimezone(SGT)
        except Exception:
            continue                                            # all-day / tentative items are not scheduled releases
        g = groups.setdefault(when, [])
        if lab not in g:
            g.append(lab)
    return [{"title": ", ".join(v), "when": k, "forecast": "", "previous": ""} for k, v in groups.items()]


def econ_calendar(days: int = 7) -> list:
    """High-impact USD events from now to `days` ahead, in Singapore time. ForexFactory's weekly feed first (it has an
    impact rating); Nasdaq's calendar fills the days that feed doesn't cover yet (its 'next week' file is often missing)."""
    import requests
    now = dt.datetime.now(SGT)
    cand, covered_to = [], None
    for wk in ("thisweek", "nextweek"):
        try:
            r = requests.get(f"https://nfs.faireconomy.media/ff_calendar_{wk}.json", timeout=20,
                             headers={"User-Agent": "Mozilla/5.0"})
            rows = r.json() if r.ok else []
            print(f"  econ calendar ({wk}): HTTP {r.status_code}, {len(rows)} events")
        except Exception as e:
            print(f"  econ calendar ({wk}): {e}")
            rows = []
        for x in rows:
            try:
                when = dt.datetime.fromisoformat(x["date"]).astimezone(SGT)
            except Exception:
                continue
            covered_to = max(covered_to, when) if covered_to else when
            if x.get("country") == "USD" and x.get("impact") == "High":
                cand.append({"title": x["title"], "when": when, "forecast": x.get("forecast") or "", "previous": x.get("previous") or ""})
    # days the weekly feed doesn't reach (US dates, since releases are scheduled in New York)
    ny_today = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).date()
    last_ff = (covered_to.astimezone(dt.timezone(dt.timedelta(hours=-5))).date() if covered_to else ny_today - dt.timedelta(days=1))
    n_nas = 0
    for k in range(days + 1):
        day = ny_today + dt.timedelta(days=k)
        if day > last_ff and day.weekday() < 5:            # weekday in New York
            got = _nasdaq_day(day); n_nas += len(got); cand += got
    print(f"  econ calendar: ForexFactory covers to {last_ff}, Nasdaq added {n_nas} events")
    out, seen = [], set()
    for x in cand:
        if not (now - dt.timedelta(hours=2) <= x["when"] <= now + dt.timedelta(days=days)):
            continue
        key = (x["title"].lower(), x["when"].date())
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": x["title"], "when": x["when"].strftime("%a %d %b %H:%M"), "iso": x["when"].isoformat(),
                    "forecast": x["forecast"], "previous": x["previous"]})
    return sorted(out, key=lambda x: x["iso"])


# ───────────────────────── insider buying
def insider_clusters(universe: set, days: int = 7, min_insiders: int = 2, big_usd: float = 1e6, min_usd: float = 1e5) -> list:
    p = ROOT / "data" / "history" / "insider_buys.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, parse_dates=["filed"])
    df = df[(df.filed >= pd.Timestamp.now() - pd.Timedelta(days=days)) & df.ticker.isin(universe)]
    out = []
    for t, g in df.groupby("ticker"):
        n, usd = g.owner.nunique(), float(g.value.sum())
        if (n >= min_insiders and usd >= min_usd) or usd >= big_usd:     # token buys don't count
            import html as _h
            titles = sorted({_h.unescape(str(x)).title() for x in g.title.dropna() if str(x) and str(x) != "nan"})[:3]
            out.append({"ticker": t, "insiders": int(n), "usd": round(usd), "titles": titles,
                        "last": g.filed.max().strftime("%d %b"), "avg_price": round(float((g.value.sum() / g.shares.sum())), 2) if g.shares.sum() else None})
    return sorted(out, key=lambda x: (-x["insiders"], -x["usd"]))
