"""
Breakout environment: how many stocks are in a buyable setup, and are breakouts actually working?

Minervini: in a healthy market many stocks set up at once and breakouts follow through; in a poor one setups are
scarce and breakouts fail. Two daily series, using the RS Engine's own definitions:

  setups      liquid stocks above the 50-day with RS ≥ 80 and ADR ≥ 2.5% that are in a SETUP
              (within 6% of the 20-day high, 10-day range ≤ 2.5 ADRs, 0–4 ATRs above the 50-day)
  setup_pct   setups as a % of liquid stocks (normalises for the universe growing over time)
  hold_rate   of the breakouts 10–29 sessions ago (a setup that closed above its prior 20-day high on ≥ 1.4× volume),
              the % still above that pivot 10 sessions later — known on the day, no look-ahead

validate() checks whether either series predicted how the NEXT 20 sessions' breakouts did.
History is rebuilt monthly by backtest10.py (10 years, today's universe: some survivorship bias);
the daily engine refreshes the recent part.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "history" / "environment.csv"
STATS = ROOT / "data" / "history" / "environment_stats.json"


def _atr(h, l, c, n=14):
    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return pd.DataFrame(tr, index=c.index, columns=c.columns).ewm(alpha=1 / n, adjust=False).mean()


def compute(c: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, v: pd.DataFrame, rating: pd.DataFrame,
            settings: dict, with_forward: bool = False) -> pd.DataFrame:
    f32 = lambda x: x.astype("float32")
    c, h, l, v = f32(c), f32(h), f32(l), f32(v.fillna(0))
    rating = rating.reindex_like(c)
    sma50 = c.rolling(50).mean()
    liquid = (c >= settings.get("min_price", 5)) & ((c * v).rolling(20).mean() >= settings.get("min_dollar_volume", 1e7))
    adr = ((h / l - 1).rolling(20).mean() * 100)
    elig = liquid & (c > sma50) & (rating >= settings.get("min_rs_rating", 80)) & (adr >= settings.get("min_adr", 2.5))
    hi20 = h.rolling(20).max()
    pivot_dist = (c / hi20 - 1) * 100
    tight = (h.rolling(10).max() - l.rolling(10).min()) / c * 100 / adr
    atr_ext = (c - sma50) / _atr(h, l, c)
    setup = elig & (pivot_dist >= -6) & (tight <= 2.5) & (atr_ext >= 0) & (atr_ext <= 4)

    pivot = h.shift(1).rolling(20).max()                                   # yesterday's 20-day high
    bo = setup.shift(1, fill_value=False) & (c > pivot) & (v >= 1.4 * v.rolling(50).mean().shift(1))
    held = bo & (c.shift(-10) >= pivot)                                     # still above the pivot 10 sessions later
    gain10 = (c.shift(-10) / c - 1).where(bo)

    out = pd.DataFrame(index=c.index)
    out["liquid"] = liquid.sum(axis=1)
    out["elig"] = elig.sum(axis=1)
    out["setups"] = setup.sum(axis=1)
    out["setup_pct"] = (out.setups / out.liquid.replace(0, np.nan) * 100).round(3)
    out["bo"] = bo.sum(axis=1)
    bo_n, held_n = out.bo, held.sum(axis=1)
    known_bo = bo_n.shift(10).rolling(20).sum()                             # breakouts 10–29 sessions ago
    out["hold_rate"] = (held_n.shift(10).rolling(20).sum() / known_bo.where(known_bo >= 5) * 100).round(1)
    out["bo_gain10"] = (gain10.sum(axis=1).shift(10).rolling(20).sum() / known_bo.where(known_bo >= 5) * 100).round(2)
    if with_forward:                                                        # validation only (uses the future)
        nxt_bo = bo_n[::-1].rolling(20).sum()[::-1].shift(-1)
        nxt_held = held_n[::-1].rolling(20).sum()[::-1].shift(-1)
        out["fwd_hold"] = (nxt_held / nxt_bo.where(nxt_bo >= 5) * 100)
        out["fwd_gain10"] = (gain10.sum(axis=1)[::-1].rolling(20).sum()[::-1].shift(-1) / nxt_bo.where(nxt_bo >= 5) * 100)
        out["fwd_bo"] = nxt_bo
    return out.iloc[260:]                                                   # need a year of history for the inputs


def validate(env: pd.DataFrame, spy: pd.Series | None = None) -> dict:
    e = env.dropna(subset=["setup_pct", "fwd_hold"]).copy()
    if len(e) < 100:
        return {"period": None, "days": int(len(e)), "median_hold": None}
    if spy is not None:
        e["spy_fwd20"] = (spy.shift(-20) / spy - 1).reindex(e.index) * 100
    res = {"period": f"{e.index[0]:%b %Y} – {e.index[-1]:%b %Y}", "days": int(len(e))}
    for col, label in (("setup_pct", "setups"), ("hold_rate", "hold")):
        x = e.dropna(subset=[col])
        q = pd.qcut(x[col].rank(method="first"), 5, labels=False)
        rows = []
        for k in range(5):
            g = x[q == k]
            rows.append({"bucket": k + 1, "lo": round(float(g[col].min()), 2), "hi": round(float(g[col].max()), 2),
                         "fwd_hold": round(float(g.fwd_hold.mean()), 1), "fwd_gain10": round(float(g.fwd_gain10.mean()), 2),
                         "fwd_bo": round(float(g.fwd_bo.mean()), 1),
                         **({"spy_fwd20": round(float(g.spy_fwd20.mean()), 2)} if "spy_fwd20" in g else {})})
        rho = x[col].rank().corr(x.fwd_hold.rank())                       # Spearman without scipy
        res[label] = {"quintiles": rows, "corr_fwd_hold": round(float(rho), 3)}
    res["median_hold"] = round(float(env.hold_rate.median()), 1)
    return res


def save_history(env: pd.DataFrame, stats: dict | None = None):
    keep = ["liquid", "elig", "setups", "setup_pct", "bo", "hold_rate", "bo_gain10"]
    old = pd.read_csv(CSV, index_col=0, parse_dates=True) if CSV.exists() else pd.DataFrame(columns=keep)
    new = env[keep]
    if len(old) and len(new):                                               # fresh values replace the overlap
        old = old[old.index < new.index[0]]
    pd.concat([old, new]).sort_index().to_csv(CSV, float_format="%.3f")
    if stats is not None:
        STATS.write_text(json.dumps(stats, indent=1))


def payload() -> dict | None:
    if not CSV.exists():
        return None
    env = pd.read_csv(CSV, index_col=0, parse_dates=True).dropna(subset=["setup_pct"])
    if not len(env):
        return None
    stats = json.loads(STATS.read_text()) if STATS.exists() else {}
    now = env.iloc[-1]
    pct_rank = round(float((env.setup_pct <= now.setup_pct).mean() * 100))
    hr = env.hold_rate.dropna()
    med = float(hr.median()) if len(hr) else None
    hold_now = None if pd.isna(now.hold_rate) else float(now.hold_rate)
    good_s, good_h = pct_rank >= 60, (hold_now is not None and med is not None and hold_now >= med)
    poor_s, poor_h = pct_rank <= 30, (hold_now is not None and med is not None and hold_now < med - 5)
    verdict = ("Healthy" if good_s and good_h else "Poor" if poor_s and poor_h else
               "Improving" if (good_h and not good_s) else "Mixed")
    tone = {"Healthy": "bull", "Poor": "bear"}.get(verdict, "warn")
    s20 = env.setups.iloc[-21] if len(env) > 21 else None
    return {"verdict": verdict, "tone": tone, "asof": env.index[-1].strftime("%d %b %Y"),
            "setups": int(now.setups), "setup_pct": round(float(now.setup_pct), 2), "setup_rank": pct_rank,
            "setups_20d_ago": None if s20 is None else int(s20), "elig": int(now.elig),
            "hold_rate": hold_now, "hold_median": None if med is None else round(med, 1),
            "bo_gain10": None if pd.isna(now.bo_gain10) else round(float(now.bo_gain10), 1),
            "since": env.index[0].strftime("%Y"),
            "dates": [d.strftime("%Y-%m-%d") for d in env.index],
            "setup_series": [round(float(x), 2) for x in env.setup_pct],
            "setups_n": [int(x) for x in env.setups],
            "hold_series": [None if pd.isna(x) else round(float(x), 1) for x in env.hold_rate],
            "stats": stats}


def brief_line(p: dict | None) -> str | None:
    if not p:
        return None
    hold = (f"{p['hold_rate']:.0f}% of recent breakouts holding (typical {p['hold_median']:.0f}%)"
            if p.get("hold_rate") is not None else "too few recent breakouts to judge")
    icon = {"Healthy": "🟢", "Poor": "🔴"}.get(p["verdict"], "🟡")
    return (f"{icon} <b>Breakout environment: {p['verdict']}</b> — {p['setups']} setups "
            f"({p['setup_rank']}th percentile since {p['since']}) · {hold}")
