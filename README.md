# RS Engine

Sector → group → stock relative strength, market regime and rotation. Runs itself on GitHub every weekday morning (SGT), publishes a dashboard, logs history, and sends a Telegram summary with a TradingView watchlist file every Saturday.

```
rs_engine.py            the engine (downloads prices, computes everything, writes outputs)
notify.py               Telegram message + TradingView file
refresh_universe.py     re-downloads ETF holdings from the issuers (monthly)
config/etfs.csv         every index / sector / group ETF, with category, theme and style
config/universe.csv     stock → ETF membership (your 1,381 stocks, 2,327 memberships)
config/settings.json    filters: min $ volume, min RS, how many themes, etc.
templates/dashboard.html
docs/                   the published dashboard + CSVs (GitHub Pages serves this)
data/history/           daily snapshots (RS ratings, leaders) — builds up over time
```

---

## Setup — about 20 minutes, once

### Step 1 — Create the GitHub repository

1. Sign in at github.com (create a free account if you don't have one).
2. Top right **+ → New repository**. Name it `rs-engine`. Choose **Public** (needed for free GitHub Pages; there's nothing private in it — just tickers and code). Don't add a README. Click **Create repository**.
3. On the empty repo page click **uploading an existing file**.
4. Unzip `rs-engine.zip` on your computer, open the folder, select **everything inside it** (including the `.github` folder — on a Mac press `Cmd+Shift+.` to show hidden folders) and drag it into the browser. Click **Commit changes**.
5. Check the repo shows `.github/workflows/rs-engine.yml`. If the `.github` folder didn't upload, click **Add file → Create new file**, type `.github/workflows/rs-engine.yml` as the name, paste the file's contents, and commit. Do the same for `refresh-universe.yml`.

### Step 2 — Let the workflow save results

**Settings → Actions → General → Workflow permissions** → choose **Read and write permissions** → **Save**.

### Step 3 — Create the Telegram bot

1. In Telegram, open a chat with **@BotFather** (blue tick).
2. Send `/newbot`. Give it a name (e.g. `MJ RS Engine`) and a username ending in `bot` (e.g. `mj_rs_engine_bot`).
3. BotFather replies with a **token** like `7412345678:AAH...`. Copy it. Treat it like a password.
4. Open your new bot (the link BotFather gives you) and press **Start**, then send it any message, e.g. `hi`.
5. In a browser open (replace `<TOKEN>`):
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
   Find `"chat":{"id":123456789` — that number is your **chat ID**.

### Step 4 — Store the two secrets in GitHub

**Settings → Secrets and variables → Actions → New repository secret**, twice:

| Name | Value |
|---|---|
| `TELEGRAM_TOKEN` | the token from BotFather |
| `TELEGRAM_CHAT_ID` | the number from step 3.5 |

### Step 5 — First run

1. **Actions** tab → if asked, click **I understand my workflows, go ahead and enable them**.
2. Click **RS Engine** on the left → **Run workflow** → keep *Send the Telegram summary now* ticked → **Run workflow**.
3. It takes about 5–10 minutes (downloading ~1,500 tickers). Green tick = done. Your Telegram should receive the summary and a `.txt` watchlist file.
4. If it goes red, click into the run to see which step failed and send me the log.

### Step 6 — Turn on the dashboard

1. **Settings → Pages** → Source: **Deploy from a branch** → Branch: `main`, folder: `/docs` → **Save**.
2. After a minute the page shows your address: `https://<your-username>.github.io/rs-engine/`. Bookmark it on your phone and desktop.
3. Optional: put that address in `config/settings.json` → `"dashboard_url"` (click the file → pencil icon → edit → commit). The Telegram message then includes an **Open dashboard** link.

### Step 7 — Refresh the stock universe (fills the gaps automatically)

**Actions → Refresh universe → Run workflow**. It pulls full holdings for SPDR and Invesco ETFs directly from the issuers, and top-10 holdings for the rest, then commits `config/universe.csv`. The run summary lists every ticker added or removed. After that it runs on the 1st of every month on its own.

Groups like PEJ (Leisure), BETZ, AMLP, IPAY, BOAT start with no holdings — the dashboard marks them *add holdings* and they can't feed the leaders list until this refresh (or you) adds members.

---

## What runs when

| When (SGT) | What |
|---|---|
| Tue–Sat 6:30am | Engine runs on the previous US close; dashboard and history update |
| Saturday 6:30am | + Telegram summary and TradingView file (Friday close) |
| 1st of the month | Universe refresh from the ETF issuers |
| Any time | Actions → RS Engine → Run workflow |

Want Telegram every day? Set `"telegram_daily": true` in `config/settings.json`.

---

## TradingView

Telegram sends `RS_leaders_<date>.txt` each Saturday. On desktop: save it → in TradingView open the watchlist panel → watchlist name ▾ → **Import list…** → choose the file. Leaders arrive grouped by theme (each `###Theme` becomes a section). The same file is always at `https://<username>.github.io/rs-engine/data/tradingview_leaders.txt`.

(TradingView has no public API for writing watchlists, so a direct push isn't possible; import takes about 10 seconds.)

---

## Keeping your Google Sheet view

Your sheet stays as it is. To pull the engine's results into it, add a tab and put this in A1:

```
=IMPORTDATA("https://<username>.github.io/rs-engine/data/leaders.csv")
```

Other files: `sectors.csv` (every index, sector and group), `stocks.csv` (all 1,381 stocks, one row each, with every ETF they belong to). Google refreshes IMPORTDATA roughly hourly, and these files are one fast request — no GOOGLEFINANCE load.

---

## Editing the lists

All on github.com: click the file → pencil icon → edit → **Commit changes**. The next run uses it.

- **Add an ETF to track:** `config/etfs.csv`, one line: `symbol,name,category,theme,style`.
  - `category`: `Index`, `Segment`, `EW Sector`, `SPDR Sector`, `Group`
  - `theme`: groups with the same theme compete as one (only the strongest ETF of a theme is used), so XLE/XOP/FCG no longer crowd out the top-N.
  - `style`: `growth`, `cyclical`, `defensive` — drives the rotation charts.
- **Add stocks:** `config/universe.csv`, one line per `ticker,etf`. Or just add the ETF and run **Refresh universe**.
- **Change filters:** `config/settings.json` — `min_dollar_volume`, `min_price`, `min_rs_rating`, `require_above_50dma`, `top_themes`, `max_leaders`, `telegram_top_n`, `benchmark`, `min_adr` (default 2.5 — drops pinned / low-volatility names).

---

## How the numbers work

**RS Rating 1–99 (volatility-adjusted, after Jeff Sun's VARS).** Each day's move is converted into ATR units (move ÷ 14-day ATR). The benchmark's ATR-moves are subtracted, summed over 1 week, 1 month and 3 months, each ranked across the universe, then blended 15% / 35% / 50%. A 3%-ADR stock that rose steadily outranks a 12%-ADR stock that got the same return by being wild.

**RS Win %.** Share of the last 20 sessions the ticker beat the benchmark — persistence rather than one spike.

**Rotation.** Growth, cyclical and defensive baskets (equal-weight sector ETFs) relative to RSP; the RRG places each sector in Leading / Weakening / Lagging / Improving from weekly data; the heatmap shows each sector's RS rating week by week.

**Risk appetite gauges.** Seven ratios (growth/defensive, cyclical/defensive, XLY/XLP, RSP/SPY, IWM/SPY, SPMO/RSP, HYG/IEF). Risk-on when the ratio is above its 50-day average and higher than a month ago.

**Regime.** Risk-On when ≥5 gauges are risk-on; Risk-Off when ≤1 and universe breadth < 40%; Defensive Rotation when defensives led the past month; otherwise Mixed.

**Leaders.** Stocks in the top themes (themes that have holdings), RS ≥ `min_rs_rating`, above the 50-DMA, ADR ≥ `min_adr`, price and $-volume above the floors. Flags: **TT** Minervini trend template, **NH** within 15% of the 52-week high, **EXT** more than 4 ATRs above the 50-DMA, **NEW** not on the list a week ago, **SETUP** within 6% of the 20-day high, 10-day range ≤ 2.5 ADRs and 0–4 ATRs above the 50-DMA, **ER** earnings within 7 days.

---

## Running it on your own computer (optional)

```
pip install -r requirements.txt
python rs_engine.py            # real data
python rs_engine.py --demo     # simulated data, no internet needed
python rs_engine.py --notify   # also send Telegram (needs TELEGRAM_TOKEN / TELEGRAM_CHAT_ID env vars)
```

Data comes from Yahoo Finance (free, unofficial). If Yahoo throttles a run, the engine retries; a failed day just means yesterday's dashboard stays up. Not investment advice.
