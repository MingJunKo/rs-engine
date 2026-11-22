"""
Would the scans have caught the model-book winners (100%+ runs) BEFORE they ran?

Replays every session of the last ~5 years and rebuilds, as of that evening:
  RS Engine  · strong   = liquid, above the 50-day, RS ≥ 80, ADR ≥ 2.5%      (the leader gates)
             · setup    = strong + near pivot, tight, not extended            (the site's SETUP)
             · leader   = strong + in one of the top-8 themes that day        (no 60-name cap)
  Jeff Sun   · final    = your 14 screens → 5-EMA tightness pass, or the weekly RS scan; price > $3
             (fundamental filters — EPS / sales / FCF growth, 13F — have no history: assumed PASS, so this is a best case;
              market cap = today's share count × that day's price; float = today's float)
  EP         · gap ≥ 8% on ≥ 3× volume, $10M+, closes in the upper half
For each winner: was it flagged in the 20 sessions BEFORE its breakout (actionable), or in the first 10 after?
Also: how many names each scan flags a day, the lift (chance a flagged name breaks out into a winner within 20 sessions
vs any liquid stock), and which rule most often kept a winner out.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rs_engine as E           # noqa: E402
import scans as SC              # noqa: E402

t0 = time.time()
log = lambda *a: print(*a, flush=True)
PRE, POST = 20, 10

# ── winners ──────────────────────────────────────────────────────────
html = (ROOT / "docs" / "modelbook" / "index.html").read_text()
MB = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1).replace("<\\/", "</"))
W = pd.DataFrame(MB["book"])[["symbol", "date", "year", "run_gain", "sector", "theme", "price", "dvol", "rs", "adr", "ep"]]
W["date"] = pd.to_datetime(W.date)
log(f"{len(W)} winners {W.date.min().date()} → {W.date.max().date()}")

# ── data ─────────────────────────────────────────────────────────────
st = pd.read_csv(ROOT / "docs" / "data" / "stocks.csv")
etfs = pd.read_csv(ROOT / "config" / "etfs.csv")
meta = etfs.set_index("symbol")
grp = [e for e in etfs.symbol if meta.at[e, "category"] in ("Group", "SPDR Sector")]
uni = pd.read_csv(ROOT / "config" / "universe.csv")
etfs_of = uni.groupby(uni.ticker.str.upper()).etf.apply(lambda s: set(s)).to_dict()
S = sorted(set(st.ticker) | set(W.symbol))
px = E.download_yahoo(S + grp + [E.BENCH], period="6y", want=("Open", "High", "Low", "Close", "Volume"))
c = px["Close"].ffill(limit=3)
c = c.loc[c[E.BENCH].notna()]
o, h, l, v = (px[k].reindex(c.index) for k in ("Open", "High", "Low", "Volume"))
h, l, v = h.ffill(limit=3), l.ffill(limit=3), v.fillna(0)
S = [t for t in S if t in c.columns and c[t].notna().sum() > 260]
G = [e for e in grp if e in c.columns]
idx = c.index
log(f"{len(idx)} sessions {idx[0].date()} → {idx[-1].date()} · {len(S)} stocks · {len(G)} theme ETFs · {time.time() - t0:.0f}s")
b_close = c[E.BENCH]
b_atr = E.atr(h[[E.BENCH]], l[[E.BENCH]], c[[E.BENCH]])[E.BENCH]
C, H, L, O, V = c[S], h[S], l[S], o[S], v[S]

# ── RS Engine flags (whole history at once) ──────────────────────────
rating, _, a = E.rs_rating(C, H, L, b_close, b_atr)
sma50 = C.rolling(50).mean()
liquid = (C >= 5) & ((C * V).rolling(20).mean() >= 1e7)
adr = (H / L - 1).rolling(20).mean() * 100
gates = {"liquid ($10M+, $5+)": liquid, "above the 50-day": C > sma50, "RS ≥ 80": rating >= 80, "ADR ≥ 2.5%": adr >= 2.5}
strong = gates["liquid ($10M+, $5+)"] & gates["above the 50-day"] & gates["RS ≥ 80"] & gates["ADR ≥ 2.5%"]
shape = {"within 6% of the 20-day high": (C / H.rolling(20).max() - 1) * 100 >= -6,
         "10-day range ≤ 2.5 ADRs": (H.rolling(10).max() - L.rolling(10).min()) / C * 100 / adr <= 2.5,
         "0–4 ATRs above the 50-day": (((C - sma50) / a) >= 0) & (((C - sma50) / a) <= 4)}
setup = strong & shape["within 6% of the 20-day high"] & shape["10-day range ≤ 2.5 ADRs"] & shape["0–4 ATRs above the 50-day"]

# top-8 themes by RS each day → leader
g_rating, _, _ = E.rs_rating(c[G], h[G], l[G], b_close, b_atr)
theme_of_etf = meta.theme.to_dict()
th_r = pd.DataFrame({th: g_rating[[e for e in G if theme_of_etf.get(e) == th]].max(axis=1)
                     for th in {theme_of_etf[e] for e in G}})
top8 = th_r.rank(axis=1, ascending=False, method="first") <= 8
in_top = pd.DataFrame(False, index=idx, columns=S)
for t in S:
    ths = {theme_of_etf.get(e) for e in etfs_of.get(t, set())} & set(top8.columns)
    if ths:
        in_top[t] = top8[list(ths)].any(axis=1)
leader = strong & in_top

# EP
gap = (O / C.shift(1) - 1) * 100
ep = (gap >= 8) & (V >= 3 * V.rolling(50, min_periods=30).mean().shift(1)) & (C * V >= 1e7) & (C >= 3) & \
     ((C - L) >= 0.5 * (H - L))
log(f"RS Engine + EP flags built · {time.time() - t0:.0f}s")

# ── Jeff Sun pipeline, replayed day by day ──────────────────────────
names = E.fetch_names()
F = json.loads((ROOT / "data" / "fundamentals.json").read_text()) if (ROOT / "data" / "fundamentals.json").exists() else {}
mc_now = pd.Series({t: np.nan for t in S}, dtype=float)
if "mcap" in names:
    nm = names.drop_duplicates("ticker").set_index("ticker")
    mc_now.update(pd.to_numeric(nm.mcap, errors="coerce").reindex(S))
fb = pd.Series({t: (F.get(t, {}).get("mcap_b") or np.nan) * 1e9 for t in S})
mc_now = mc_now.where(mc_now > 0, fb)
shares = mc_now / C.iloc[-1]
fl = pd.Series({t: (F.get(t, {}).get("float_m") or np.nan) * 1e6 for t in S})
float_ = fl.where(fl > 0, shares)

# rolling fields as frames
low52 = L.rolling(252, min_periods=120).min()
ema5 = C.ewm(span=5, adjust=False).mean()
ema10 = C.ewm(span=10, adjust=False).mean()
sm = {n: C.rolling(n).mean() for n in (10, 20, 50)}
rng = (H / L - 1) * 100
av60 = V.rolling(60, min_periods=40).mean()
av60s = av60.shift(1)
pe = ((O / C.shift(1) - 1) > 0.05) & (V >= 2 * av60s)
pe5 = pe.rolling(5, min_periods=1).max().astype(bool)
FR = {"price": C, "vol": V, "avg_vol60": av60, "dvol20": (C * V).rolling(20).mean(), "above_low": (C / low52 - 1) * 100,
      "ema5_gap": (C / ema5 - 1) * 100, "sma10_gap": (C / sm[10] - 1) * 100, "sma10_gt_20": sm[10] > sm[20],
      "ema10_gt_sma20": ema10 > sm[20], "above_sma20": C > sm[20], "above_sma50": C > sm[50],
      "perf_1w": (C / C.shift(5) - 1) * 100, "perf_1m": (C / C.shift(21) - 1) * 100,
      "perf_3m": (C / C.shift(63) - 1) * 100, "perf_6m": (C / C.shift(126) - 1) * 100,
      "volat_1m": rng.rolling(21).mean(), "volat_1w": rng.rolling(5).mean(), "adr": rng.rolling(20).mean(),
      "above_low20": C >= 1.05 * L.rolling(20).min(), "above_low50": C >= 1.05 * L.rolling(50).min(), "pe_gap_recent": pe5}
js_final = pd.DataFrame(False, index=idx, columns=S)
js_wl = pd.DataFrame(False, index=idx, columns=S)
WLS = [k for _, _, scr in SC.WATCHLISTS for k in scr]
start = max(260, int(np.searchsorted(idx, W.date.min() - pd.Timedelta(days=45))))
for i in range(start, len(idx)):
    d = pd.DataFrame({k: f.iloc[i] for k, f in FR.items()})
    d["mcap"] = shares * d.price
    d["float"] = float_
    for k in ("eps_q_yoy", "rev_q_yoy", "fcf_yoy", "sales_ttm_yoy"):
        d[k] = 999.0                                                    # no history: assume they pass (best case)
    d["inst_chg"] = 1.0
    S_ = SC.screens(d, SC.conditions(d))
    in_wl = pd.concat([S_[k] for k in WLS], axis=1).any(axis=1)
    tb, ta = SC.tight(d)
    fin = ((in_wl & (tb | ta)) | S_["s27"]) & (d.price > 3)
    js_wl.iloc[i] = in_wl.values
    js_final.iloc[i] = fin.values
log(f"Jeff Sun pipeline replayed · {time.time() - t0:.0f}s")

# ── candidate refinements of the RS setup (tested the same way) ──────
rsl = C.div(b_close, axis=0)
rsl_nh = rsl >= rsl.rolling(252, min_periods=120).max() * 0.999          # RS line at a 52-week high
rs_lead = rsl_nh.rolling(10).max().astype(bool) & (C < C.rolling(252, min_periods=120).max() * 0.98)
pivot = (C / H.rolling(20).max() - 1) * 100
tightr = (H.rolling(10).max() - L.rolling(10).min()) / C * 100 / adr
ext = (C - sma50) / a
base_ok = liquid & (C > sma50)
VARIANTS = {
    "A: tight ≤ 3.5 ADRs": strong & (pivot >= -6) & (tightr <= 3.5) & (ext >= 0) & (ext <= 4),
    "B: A + RS ≥ 70": base_ok & (rating >= 70) & (adr >= 2.5) & (pivot >= -6) & (tightr <= 3.5) & (ext >= 0) & (ext <= 4),
    "C: B + ADR ≥ 4%": base_ok & (rating >= 70) & (adr >= 4) & (pivot >= -6) & (tightr <= 3.5) & (ext >= 0) & (ext <= 4),
    "D: C + RS line at a new high (10d)": base_ok & (rating >= 70) & (adr >= 4) & (pivot >= -8) & (tightr <= 4) & (ext >= 0) & (ext <= 5) & rsl_nh.rolling(10).max().astype(bool),
    "E: RS-line lead, ADR ≥ 4%, near pivot": base_ok & (adr >= 4) & (pivot >= -10) & rs_lead,
}
for k_, f_ in VARIANTS.items():
    FLAGS_V = f_

# ── capture per winner ───────────────────────────────────────────────
FLAGS = {**{"V " + k_: f_ for k_, f_ in VARIANTS.items()}, "RS setup": setup, "RS leader": leader, "RS strong (gates)": strong, "Jeff Sun final": js_final,
         "Jeff Sun watchlists": js_wl, "EP": ep}
col = {t: j for j, t in enumerate(S)}
rows = []
for w in W.itertuples():
    if w.symbol not in col:
        rows.append({"symbol": w.symbol, "date": w.date, "in_universe": False}); continue
    b = int(np.searchsorted(idx, w.date))
    if b >= len(idx) or b < start + PRE:
        continue
    j = col[w.symbol]
    rec = {"symbol": w.symbol, "date": w.date, "year": w.year, "gain": w.run_gain, "sector": w.sector, "in_universe": True}
    for k, f in FLAGS.items():
        arr = f.values[:, j]
        rec[k + " · before"] = bool(arr[b - PRE:b].any())
        rec[k + " · after"] = bool(arr[b:b + POST].any())
        first = np.flatnonzero(arr[b - PRE:b + POST])
        rec[k + " · first"] = int(first[0]) - PRE if len(first) else None
    # why the RS setup missed: gates on the day before the breakout
    rec["gate_fail"] = [g for g, f in {**gates, **shape}.items() if not bool(f.values[b - 1, j])]
    rec["js_tight_fail"] = bool(js_wl.values[b - PRE:b, j].any() and not js_final.values[b - PRE:b, j].any())
    rows.append(rec)
R = pd.DataFrame(rows)
U = R[R.in_universe == True].copy()
for k_ in U.columns:
    if k_.endswith(" · before") or k_.endswith(" · after") or k_ == "js_tight_fail":
        U[k_] = U[k_].fillna(False).astype(bool)
log(f"{len(U)} winners scored ({(R.in_universe != True).sum()} outside the universe)")

out = {"winners": int(len(R)), "scored": int(len(U)), "outside_universe": sorted(R[R.in_universe != True].symbol.tolist()),
       "window": {"before": PRE, "after": POST}, "capture": {}, "by_year": {}, "daily_names": {}, "lift": {}}
for k in FLAGS:
    out["capture"][k] = {"before": round(100 * U[k + " · before"].mean(), 1), "after_only": round(100 * (~U[k + " · before"] & U[k + " · after"]).mean(), 1),
                         "lead_days_median": None if U[k + " · first"].dropna().empty else float(U[k + " · first"].dropna().median())}
combos = {"RS setup or Jeff Sun final": U["RS setup · before"] | U["Jeff Sun final · before"],
          **{f"V {k_} or Jeff Sun final": U[f"V {k_} · before"] | U["Jeff Sun final · before"] for k_ in VARIANTS},
          "RS setup or Jeff Sun final or EP": U["RS setup · before"] | U["Jeff Sun final · before"] | U["EP · before"],
          "RS strong or Jeff Sun watchlists": U["RS strong (gates) · before"] | U["Jeff Sun watchlists · before"]}
out["combos"] = {k: round(100 * s.mean(), 1) for k, s in combos.items()}
for y, g in U.groupby("year"):
    out["by_year"][int(y)] = {k: round(100 * g[k + " · before"].mean()) for k in ("RS setup", "RS leader", "Jeff Sun final", "EP")}
    out["by_year"][int(y)]["n"] = int(len(g))

# names per day and lift: P(a flagged name breaks out into a winner within 20 sessions)
win_mat = np.zeros((len(idx), len(S)), bool)
for w in W.itertuples():
    if w.symbol in col:
        b = int(np.searchsorted(idx, w.date))
        if b < len(idx):
            win_mat[max(0, b - PRE):b, col[w.symbol]] = True          # days in the 20 sessions before a winning breakout
rows_ = slice(start, len(idx) - 130)
base = win_mat[rows_][liquid.values[rows_]].mean()
for k, f in FLAGS.items():
    m = f.values[rows_]
    out["daily_names"][k] = round(float(m.sum(axis=1).mean()), 1)
    out["lift"][k] = {"hit_pct": round(100 * float(win_mat[rows_][m].mean()), 2), "lift": round(float(win_mat[rows_][m].mean() / base), 1)}
out["lift"]["base (any liquid stock)"] = {"hit_pct": round(100 * float(base), 2), "lift": 1.0}

# why missed
miss = U[~U["RS setup · before"]]
gf = pd.Series([g for x in miss.gate_fail for g in x]).value_counts()
out["rs_setup_miss_reasons"] = {k: int(v_) for k, v_ in gf.items()}
out["rs_setup_missed_n"] = int(len(miss))
jm = U[~U["Jeff Sun final · before"]]
out["js_miss"] = {"n": int(len(jm)), "never_in_watchlist": int((~jm["Jeff Sun watchlists · before"]).sum()),
                  "in_watchlist_failed_tightness": int(jm.js_tight_fail.sum())}
nothing = U[~(U["RS setup · before"] | U["Jeff Sun final · before"] | U["EP · before"] | U["RS leader · before"])]
out["missed_by_all"] = nothing.sort_values("gain", ascending=False).head(25)[["symbol", "date", "gain", "sector", "gate_fail"]] \
    .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records")
out["missed_by_all_n"] = int(len(nothing))
out["runtime_s"] = round(time.time() - t0)
(ROOT / "research" / "scan_capture.json").write_text(json.dumps(out, indent=1, default=str))
print("RESULT_JSON_START")
print(json.dumps(out, indent=1, default=str))
print("RESULT_JSON_END")
