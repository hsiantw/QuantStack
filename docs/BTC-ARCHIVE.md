# BTC/USD minute archive

The local collection is focused on BTC-USD. MarketData-DailySync and
MarketData-IntradaySync were disabled on 2026-10-25. Their exported definitions
and the previous config are under `data/`. Old downloaded assets remain intact.
Priority pulls and liquidation collection are paused in `config.json`.

Run or resume the archive from the project directory:

```powershell
.\.venv-integration\Scripts\python.exe idle_collector.py .\.venv-integration\Scripts\python.exe btc_backfill.py
```

The worker uses the public Bitstamp BTC/USD OHLC endpoint at 60-second resolution.
Daily discovery starts in 2009; the earliest returned day was 2011-08-18.
This is Bitstamp history, not a claim to every Bitcoin trade since inception.
Provider-supplied zero-volume candles are preserved, and missing minutes are
reported rather than synthesized. Prices are USD; volumes are fractional BTC.

`data/btc-backfill.json` reports completion and coverage. Data and checkpoints
commit in the same SQLite transaction, so restart resumes unfinished windows.
Failures retry with backoff. Only one ingestion process can own the archive.
The worker catches up to the latest completed minute and exits after auditing
stored row counts.

Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-BTC-Schedule.ps1`
to install and immediately start the recurring
`MarketData-BTCMinuteSync` Windows task. It pulls completed one-minute candles
every 15 minutes while logged in on AC power, at below-normal priority. Overlapping
runs are skipped; missed history is caught up from the archive checkpoints.
Each run has a fifteen-minute limit and can resume safely on the next run.
The computer must be awake for collection. This updates the local chart database.
Check `Get-ScheduledTaskInfo -TaskName MarketData-BTCMinuteSync` and
`data/btc-backfill.json` for task and collection status.

## All labeled Stocks and Crypto

`powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-Minute-Schedule.ps1`
installs and starts `MarketData-CatalogMinuteSync`. Every 15 minutes it runs a
45-second budget of Yahoo one-minute requests for the same symbol set used by
the local chart catalog (daily state, intraday state, and configured symbols).
BTC remains on its separate Bitstamp schedule. This explicit catalog collector
does not use the older BTC-only universe restriction.

Symbols rotate by oldest attempt, including failed symbols; a provider rate
limit stops the batch. Initial requests cover the last seven days, then overlap
recent stored bars. Provider availability varies and older minute history is
not guaranteed. Thousands of symbols take multiple cycles to refresh; one-minute
bars do not imply every symbol refreshes every minute. Existing data is retained.
Both collectors share the ingestion lock. Check `data/minute-refresh.json` for
the latest batch results. Runs require an awake PC, an interactive login and AC
power. The legacy daily/hourly tasks remain disabled.

## Idle-only collection

All four market-data task definitions require 15 minutes idle, stop on activity,
and restart when idle. `idle_collector.py` also checks Windows keyboard/mouse
input before starting and every second while the collector runs. It requires
strictly more than 900 seconds without input and refuses to run if input state
cannot be read. Returning input ends the collector process tree; committed SQLite
data/checkpoints survive, and unfinished work is retried on a later eligible run.
Windows may delay starts beyond the idle threshold; this is not an exact timer.
Direct invocation of the underlying Python collectors bypasses this guard; use
the wrapper command above or the installed tasks.

Apply the policy to existing tasks with `Apply-CollectorIdlePolicy.ps1`. It backs
up the task XML under `data/scheduler-backup-*` and preserves disabled tasks.
The installers retain the idle policy on reinstallation.

The app's priority queue also checks Windows idle time before starting each
request. An in-flight priority request may finish, then remaining requests stay
queued until idle. Disabled workers do not open the queue database; policy is
rechecked every 30 seconds, and a newly queued request wakes the worker immediately.
Queue and research status polling pause when the browser tab is hidden and
resume when it becomes visible; research computation continues on the server.

The October 27 review found both enabled tasks firing every minute, a completed
BTC archive, and 4,984 catalog symbols rotating through a 45-second batch. Every
minute was unnecessary for BTC archive maintenance: a catch-up request contains
multiple one-minute candles. Fifteen-minute triggers reduce scheduled launches
by 93%. Catalog freshness is lower with this cadence and idle restriction;
the collector only requests the last seven days, so long gaps can leave missing
catalog history. One-minute bar resolution does not require one-minute polling.

Candles are in `data/market.sqlite`, table `crypto_exchange_prices`, with
symbol `BTC-USD`, interval `1m`, and source `Bitstamp`. The checkpoint tables
are `btc_backfill_windows` and `btc_backfill_meta`.

Open the local chart:

http://127.0.0.1:8501/workspace/?symbol=BTC-USD&interval=1m&period=MAX

The chart starts with the latest 10,000 minutes within the selected range.
Panning toward either edge automatically loads an overlapping earlier or later
window while preserving the zoom and visible candle positions. The browser keeps
at most 10,000 candles in memory. Date controls can also select an earlier range.
All downloaded history remains in SQLite regardless of the chart display limit.
Dragging prefetches overlapping windows, keeps the viewport inside available
candles during loading, and batches pointer-driven drawing once per animation frame.
The maximum visible span is half a page so the incoming page covers the canvas.

The timeframe dropdown offers 1/3/5/15/30 minutes, 1/2/4 hours, and daily/weekly/monthly
views. Star an interval to pin it to the toolbar; preferences persist in this
browser. Higher BTC intraday intervals aggregate complete UTC-aligned Bitstamp
minute buckets, with 1,000 candles per page. Missing or incomplete buckets are
excluded. The 1-minute view retains its existing composite behavior when other
exchange sources overlap. Daily and longer views use the existing daily dataset.

API documentation: https://www.bitstamp.net/api/
