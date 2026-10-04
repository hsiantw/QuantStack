# QuantStack

Use **+ Add symbols** in the chart watchlist to enroll stock/ETF tickers or crypto
USD pairs in the local collector. Paste up to 100 symbols separated by commas or
spaces; crypto inputs such as `BTC ETH` become `BTC-USD ETH-USD`. Saving merges
with the existing universe. Prices appear after successful scheduled collection;
provider ticker availability is checked during collection. Hosted copies cannot
change local settings. If collection is running, retry saving when it finishes.

The Add symbols dialog previews normalized tickers and duplicates before saving.
Use **Refresh status** inside the dialog for the latest hourly and daily reports.
Reports show collection outcomes, not whether Windows tasks are installed.

The default **low background usage** profile uses one hourly worker, two-second
request pauses, below-normal process priority, and ten-minute hourly and scheduled
daily work budgets. An in-flight request may finish after the budget. Both Windows
tasks have a fifteen-minute execution limit. Scheduled daily runs export only
changed symbols. Unfinished symbols rotate into later cycles, including failed tickers.
Large universes therefore do not all refresh every hour. Collector controls do
not poll. The launcher runs only the chart application, without Streamlit.

Run `Install-Schedule.ps1` and `Install-Intraday-Schedule.ps1` to register the
09:00 daily and hourly Windows tasks. They select a working Python environment,
run at low priority while logged in, and avoid battery operation. Daily scheduled
collection uses `--local-only`; it does not publish the website. Existing manual
`market_data.py sync` publishing behavior remains available. Validate with
`python -m unittest test_local_scheduler` and `python smoke_scheduler.py`.

QuantStack is a single **chart-focused workspace**. The right rail provides
an **Analysis library** plus direct icons for portfolio allocation, pairs,
options, fundamentals, volume/liquidity, forecasting, strategy comparison,
market overview and a research journal. All open in the chart's resizable dock
and use the current theme, interval and date range. Existing screener, strategy,
risk, Markov and Brownian panels remain available. There are no embedded pages.

The separate `/?research=1` page is retired and redirects to the chart. The
launcher no longer starts Streamlit. See the [page-by-page migration map](docs/native-workspace-migration.md)
for each rebuilt or consolidated feature and the replacements for unavailable
feeds and broker tools. Legacy source and databases are preserved, not served.

The Workspace menu also provides shareable symbol/interval/date-range links and
keyboard shortcuts. The last symbol and range restore automatically in this
browser; reloading market data preserves the current selection. Shared links
override saved selections. Drawings and notes remain private to the browser.
Watchlists render 100 assets at a time to keep large datasets responsive;
search always covers the entire dataset, and **Show next** loads more rows.

Run the combined app:

For a local review before committing or deploying, double-click **Preview-Local.bat**
or run `powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-LocalSite.ps1`.
This starts the combined app in the background and opens **http://127.0.0.1:8501**.
It reuses a running preview and selects a working installed Python environment.
Use `-Port 8502` if another application uses that port, or `-NoBrowser` to start
without opening a browser. Logs are in `data/local-site-8501*.log`.
Refresh after frontend edits; restart the Python server after backend edits.
This local preview does not commit, push, or deploy anything.

For a foreground server or a fresh environment:

```powershell
python -m pip install -r requirements.txt
python serve_quantstack.py
```

Open **http://127.0.0.1:8501**. The launcher serves the native chart workspace
through one address, including on Render. It reads `data/market.sqlite` when
available. Without stored prices, it serves the merged workspace using the
published daily snapshot. Daily snapshot mode supports the screener, backtests,
drawings, comparisons, and Markov analysis; hourly bars and the full indicator
library require the stored database. `market-snapshot.json` pins and verifies
the release downloaded by `prepare_snapshot.py`.

See [Render deployment and integration checks](docs/quantstack-integration.md).

## Stock and crypto datasets

The main pipeline is `market_data.py`. Daily history is stored in **data/market.sqlite** and exported to **data/exports/** as one CSV per symbol. No editor extensions are required.

## Web interface

Public website: **https://quantstack.onrender.com/**. Repository: **https://github.com/hsiantw/QuantStack**.

QuantStack stays available when this computer is off. The daily 09:00 local collector updates the local dataset when this computer is on, connected, logged in and on AC power. Publishing is a separate manual action. See [HOSTING.md](HOSTING.md) for publishing details.

Open **http://127.0.0.1:8765** while the dashboard is running. To launch it again:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Dashboard.ps1
```

Search by company or ticker, filter stocks/crypto, save favorites, select date ranges, inspect the chart, page through daily records, and export the selected range. Favorites are stored in your browser. The refresh button reloads the local dataset; it does not start a market download. The website is served only on this computer and reads SQLite without changing prices. No internet connection or frontend dependencies are needed to view existing data. Company names are cached during S&P 500 universe refreshes.

The chart workspace has an expandable drawing toolbar on the left and a persistent
panel selector on the right for the watchlist, symbol details, symbol notes and
object tree. Click the arrow above the drawing tools to show their names. Click
the active right-side icon again to collapse its panel; drag the panel's left edge
to resize it. Settings and theme controls sit at the bottom of the right rail.
Notes are saved locally in this browser for each symbol.

The bottom **Stock screener**, **Strategy tester**, **Markov analysis**, **Brownian motion**, **Returns & risk**, and **Price bars** tabs open a dock beneath the chart.
New browser sessions start with this dock collapsed to give the chart more room.
**Brownian motion** simulates geometric Brownian price paths from the selected
chart. Choose historically fitted drift/volatility, zero price drift, or custom
per-bar assumptions, then adjust the window, horizon, path count, random seed,
and adjusted/raw price basis. Pointwise percentile bands, sample paths, an
inspection slider, loss probabilities, and JSON/CSV exports are available.
Changes to settings or chart data invalidate results; simulations run in a
cancelable worker. See [model and coverage notes](docs/brownian-and-assets.md).
**Returns & risk** consolidates the overlapping legacy risk and time-series
summaries into chart-context metrics: annualized return and volatility, Sharpe
and Sortino ratios, maximum drawdown, historical VaR and expected shortfall,
return-distribution moments, and a CSV of observed returns. The selected chart
range and price basis define the sample; annualization assumptions are shown in
the panel. Portfolio, options, and research tools open from the right rail in
the same chart workspace.
Drag its top edge to resize it, or use its maximize and collapse buttons. The
screener's **Filters** button expands presets, rules and saved screens. Panel resize
handles also support arrow keys. Run `.\.venv\Scripts\python.exe smoke_workspace.py`
to check the workspace in desktop and mobile layouts with headless Edge.

Use **Compare** (Alt+C) to add up to three symbols to a linked performance pane.
All plotted series start at 0% on the first visible timestamp shared by the
available series. Panning and zooming recompute that baseline; missing bars stay
as gaps. Hover the main chart to inspect matching returns. These are closing-price
returns, without dividend adjustments, and comparisons use the selected interval
and loaded date range. Symbols with no matching data are labeled explicitly.

The chart-style selector offers candles, OHLC bars, line and area.
With **Auto** enabled, dragging the chart pans through time and keeps the price
scale fitted to the visible candles, including during diagonal drags. Turn Auto
off to pan vertically, or drag the price axis to adjust its scale manually.
Wheel zoom and dragging the time axis can zoom out to five times the loaded
history width. Choose **All** to load older history; zooming out adds space around
the bars already loaded. **Reset chart view** restores the default zoom.

**Settings**
controls grid lines, the last-price line, crosshairs and rising/falling candle
colors. Styles, settings and comparison symbols persist in this browser.
At the bottom of **Chart appearance**, **Edit overview** opens a labeled guide
over the actual workspace. Select a region to highlight it, read its purpose,
and copy its name when describing changes. The right ribbon's top navigation
and bottom settings are labeled separately from the wider right side panel.
Close the guide or press Esc to return without applying unfinished settings.
The theme picker offers **System**, **Light**, **Dark**, **Midnight**, **Forest**,
and **Paper** across the workspace. In **Settings**, preview a theme before
applying it, customize candle wicks/borders, line width, axis label size, grid
style, and watermark visibility. Expand **Custom chart colors** to override
background, grid, labels, line/area, and crosshair colors independently; leave
**Theme** enabled for colors that should follow the selected theme. Canvas
colors also carry into study/comparison panes and PNG exports. **Cancel**
discards edits; **Reset defaults** prepares the original colors and system
theme, which take effect when you choose **Apply**.
Run `python smoke_themes.py` against the local preview to verify appearance.
**My appearance presets** saves up to 20 named combinations of workspace theme,
chart type, and appearance settings. Save / replace stores the current draft;
Load restores a preset into the preview, and Apply changes the chart. Export
JSON transfers the draft to another browser; Import JSON validates and previews
it without changing the chart or saving a preset automatically. These files
contain appearance settings only, without symbols, notes, drawings, or accounts.
You can also choose horizontal/vertical/both grid directions, area opacity,
and independent wick and candle-border colors. Wick colors follow the candle
colors until overridden. Run `python smoke_appearance_profiles.py` to check
profiles, imports, persistence, and mobile controls.
**Go to date** (Alt+G) centers a date in the loaded history, using the next available
bar for non-trading dates. **Snapshot** exports the visible chart, drawings and
analysis panes as PNG; the full-screen button expands the workspace. These tools
also work with daily data in generated static snapshots. Run
`.\.venv\Scripts\python.exe smoke_terminal.py` to verify the analysis controls.

The local chart includes configurable SMA, EMA, Bollinger Bands and volume, plus a
searchable library of 178 studies and candlestick patterns. Use **Study settings**
or the settings button on an active study to edit its periods, supported price
source, output colors, line width and transparency. The library includes VWAP
(with loaded-range or UTC-day reset), Donchian and Keltner channels, Supertrend,
Ichimoku, Bollinger %B/width and Chandelier Exit. Ichimoku shifts its spans within
the loaded dates; it does not project beyond the available bars. Study settings
are saved in this browser. Full-library calculations require the local dashboard;
the quick studies and drawing editor also work in generated static snapshots.

Drawing tools include trend lines, rays, horizontal and vertical levels,
rectangles, Fibonacci retracements and price/bar measurements. Click to place
anchors, then select a drawing to drag it or resize its handles. The appearance
bar edits line/fill colors, their separate transparency, width and dash style.
Selected drawings support exact price edits, duplicate, lock, hide and delete;
the **Objects** list also selects hidden drawings. Undo/redo covers creation,
movement, resizing, styling and deletion (Ctrl+Z / Ctrl+Y). Delete removes the
selected drawing; Esc or right-click cancels unfinished drawing or movement.
OHLC snapping is enabled by default; hold Shift for free anchor placement.
Drawings remain browser-local per symbol and interval. Changing the loaded date
range preserves anchors outside the range; horizontal movement is limited until
both anchors are included again. Undo history resets when switching assets or
intervals, while saved drawings and appearance preferences survive reloads.

To run in a terminal instead: `.\.venv\Scripts\python.exe dashboard.py` (Ctrl+C stops it). Browser smoke tests use the optional packages in `requirements-dev.txt` and your installed Microsoft Edge.

## Stock screener and coverage

Use **Stock screener** to combine company, valuation, liquidity, performance and
technical filters across **41 numeric metrics**. Choose **Match all (AND)** or
**Match any (OR)** for up to 20 numeric rules; company and watchlist filters always
apply. Missing values do not pass a numeric comparison; explicit **Is missing**
and **Is available** rules help inspect coverage. Ten presets cover size,
momentum, oversold stocks, volume surges, value and dividends, profitable growth,
52-week highs, strong uptrends, low Bollinger width, and liquid stocks.

Sort any column, choose six column groups or a custom selection, and save named
screens in this browser. Saved screens retain filters, columns, sorting and page
size. Use the result stars to manage the shared chart watchlist, then enable
**Saved stocks only** to screen it. **Latest price on / after** excludes stocks
with older or missing prices. Active filter chips remove individual criteria.
Results support 20, 50 or 100 rows per page. CSV exports include every match,
with either visible columns or all metrics plus currency and snapshot dates.
Select a downloaded stock to open its daily chart. These features also work in
generated static snapshots. Run the browser checks with
`.\.venv\Scripts\python.exe smoke_screener.py` while the dashboard is running.
Market-cap tiers use USD. Other currencies remain explicit and are not converted
or ranked as if they were dollars. Metadata-only companies remain visible with
chart access disabled until their daily history is collected.

The theme selector beside the chart actions offers **Light**, **Dark**, and
**System** modes. It updates the chart, indicator panes, screener and strategy
results and remembers your choice in this browser. System mode follows your OS.

Use **Sort by** and **Then sort** to rank results by any metric, including hidden
columns. **Top gainers** and **Top losers** use each stock's latest daily change.
Search and reorder custom columns, enable colored values or compact rows, and use
the arrow keys on result symbols to navigate across pages. **Save all matches**
adds the complete result set to the shared watchlist; **Undo save** reverses that
addition. Row display preferences survive changing presets and resetting filters.

## Strategy testing

Open **Strategy tester** below the chart and choose one of **20 presets** or
**Custom entry / exit**. The selector groups presets by their signal type:

| Type | Presets |
| --- | --- |
| Trend | SMA trend, EMA trend, SMA crossover, EMA crossover, triple EMA alignment, golden/death cross, price/EMA crossover |
| Reversion | RSI reversion, RSI recovery, Bollinger reversion, stochastic recovery, Williams %R recovery, CCI recovery |
| Momentum | MACD crossover, MACD zero-line cross, RSI momentum, rate-of-change momentum |
| Breakout | Channel breakout, channel breakout with midpoint exit, Bollinger breakout |

The entry/exit preview shows the exact rules and updates when parameters change.
Triple EMA starts at 5/20/50; golden/death cross uses 50/200 and requires at least
203 bars. Use **Customize this preset** to change fixed periods and thresholds.
Tests use the selected symbol, interval and loaded
date range. Configure periods, initial capital, position size, commission,
slippage, stop loss and take profit, then click **Run backtest**. Settings persist
locally. Changing the chart data or parameters invalidates previous results.

**Customize this preset** copies its rules into the custom builder. Configure
entry and exit separately with up to 12 conditions each and independent **All
(AND)** or **Any (OR)** matching. Compare open/high/low/close, SMA, EMA, RSI,
MACD, its signal, Bollinger bands, prior channel levels/midpoint, rate of change,
fast stochastic %K, Williams %R, CCI, or a number. Each SMA, EMA, RSI, band,
channel, ROC, stochastic, Williams or CCI operand has its own 2–500 bar period. MACD uses 12/26/9;
Bollinger bands use two population standard deviations around the close SMA.
Channels exclude the current bar. Comparisons include above/below, inclusive
thresholds, and crosses above/below (previous bar on the opposite side or equal).
For example, enter when SMA 10 crosses above SMA 30 **and** RSI 14 is above 50;
exit when SMA 10 crosses below SMA 30 **or** RSI 14 is above 70.

ROC is the percentage change from the close N bars earlier. Fast stochastic %K
is the unsmoothed close position in the N-bar high/low range, including the current
bar; Williams %R uses the same range on a -100 to 0 scale. A zero-width range is
missing and cannot pass a condition. CCI uses typical price `(high + low + close)
/ 3`, its N-bar SMA and mean absolute deviation with a 0.015 scaling constant;
zero deviation returns neutral zero. These indicators use the chosen price basis.

**Save setup** stores a named configuration, including rules, costs and risk
settings, in this browser. Saving the same name replaces that setup; loading one
requires a fresh run on the current symbol and range. Trade CSV exports include
the complete entry/exit rules as JSON in the `conditions` column. All configured
indicators finish warming up before entry signals are eligible, even for OR
groups. Stops and targets still apply to custom strategies. There is no same-open
exit and re-entry: a new entry condition is considered at the next completed close.

The tester reports net profit, return, buy-and-hold return, maximum drawdown,
win rate, profit factor, fees and exposure. **Overview** plots equity against the
benchmark with drawdown; **List of trades** includes fills, costs and exit reasons
and supports CSV export. **Performance** shows winning/losing trade statistics
and monthly equity returns, including partial months and open positions.

Execution is long only, one position at a time, with close-based signals filled
at the next open. Costs apply on both sides; stop gaps fill at the open, and the
stop takes priority when a bar touches both stop and target. The last bar closes
any remaining position. The benchmark uses the same eligible period and costs.
Adjusted OHLC uses the provider's adjusted-close ratio; raw prices omit corporate
action adjustments. This is historical simulation on stored data, without order
routing or Pine Script support. The execution assumptions are also shown in the
tester. Run `.\.venv\Scripts\python.exe smoke_research.py` against a running
dashboard to check execution math, themes, screener controls and the tester.
Run `.\.venv\Scripts\python.exe smoke_strategy.py` for custom-rule timing,
AND/OR logic, crossover semantics, preset conversion, saved setups and CSV checks.

## Markov chain analysis

Open **Markov analysis** below the chart and click **Run analysis**. The model
uses the selected symbol, interval, price basis and loaded history. Choose three
or five return states defined by training quantiles, or fixed down/flat/up states
with a configurable log-return band. The estimation window selects the latest
returns within the loaded range; use **All** on the chart to load older history.
At least 61 price bars are required. Settings persist in this browser.

- **Overview:** state boundaries, sample counts, empirical mean returns,
  persistence, expected run lengths, stationary occupancy and recent state history.
- **Transitions:** an interactive probability matrix with raw counts, per-row
  entropy, approximate 95% Wilson intervals and early/late transition drift.
- **Forecasts:** multi-step state probabilities, finite-horizon visit
  probabilities and a seeded return simulation with percentile bands, loss
  probabilities and maximum-drawdown statistics. The horizon slider exposes
  exact percentile values for each bar.
- **Validation:** chronological one-step forecasts compared with a historical
  state-frequency baseline using Brier score, log loss, accuracy, skill,
  calibration bins and a confusion matrix.

Quantile boundaries are learned only from the initial training segment. During
expanding validation, each outcome is scored **before** it updates transition
counts or baseline frequencies; frozen validation keeps training estimates
unchanged. Current forecasts refit transitions on all selected returns while
retaining those boundaries. Smoothing adds the chosen alpha to each transition
cell. Sparse states, empty return pools, periodicity, multiple closed classes,
weak validation and transition drift are reported explicitly.

Simulations hold fitted parameters fixed and bootstrap observed returns within
the destination state. Their ranges describe this model, not parameter uncertainty
or proven trading performance. Returns use observed bars, including session gaps;
raw prices can contain corporate-action jumps. The expandable methodology and
[detailed model guide](docs/markov-analysis.md) explain the formulas and limitations.

Calculations run in a cancelable browser worker, including on generated static
sites, with no extra runtime dependencies or market-data requests. Changing
settings, symbols, intervals or the loaded range invalidates old results. JSON
exports include settings, context, state history, counts, forecasts, simulations
and every validation probability; CSV exports include transition counts,
probabilities and intervals. Both themes, keyboard tabs and mobile layouts work.

Validation commands:

```powershell
.\.venv\Scripts\python.exe -m unittest test_markov -v
.\.venv\Scripts\python.exe smoke_markov.py
.\.venv\Scripts\python.exe smoke_markov.py --static
```

The numerical tests use headless Edge through the existing Playwright development
dependency. The static smoke test builds an isolated fixture, verifies the ZIP
contains all worker assets, and serves it under a nested URL path.

## Screener metric definitions

Returns use adjusted closes when available and 5/21/63/126/252 trading-bar lookbacks.
YTD compares the latest price with the last stored close from the previous
calendar year, and is missing without that baseline.
SMA distance is the percent above/below the moving average. RSI and ATR use
14 bars; ATR and 52-week levels use consistently adjusted OHLC. Relative volume
compares the latest completed daily volume with the prior 20 bars. Volatility
uses 20 daily log returns, annualized with 252 trading days. Insufficient or
incomplete lookbacks display missing values. Fundamentals have separate provider
snapshot dates; this is a stored-data screener, not a real-time quote feed.

Additional trend metrics include distance from EMA 20/50, MACD (12/26/9) and its
signal/histogram as percentages of the latest price, ADX 14, slow stochastic
(14/3/3), and Bollinger position/width (20 bars, two standard deviations).
52-week range position runs from 0% at the low to 100% at the high. Average
turnover is the 20-bar mean of raw close times volume, in the reported currency.
Expand **Metric definitions & data coverage** in the screener for calculation
details. Missing observations restart recursive indicator warmups; incomplete
windows remain missing rather than being filled with zero.

The ranked collection command discovers up to 3,000 US-listed equities and works
from the largest market capitalization downward, retaining configured Taiwan
stocks. Each successful history/profile is saved independently. Rerunning resumes
freshness-aware collection; the dashboard shows progress and refreshes its local
snapshot when requested. Discovery uses the
[yfinance equity screener](https://ranaroussi.github.io/yfinance/reference/api/yfinance.screen.html).
Provider exclusions and duplicate results can make the final queue smaller than
the requested limit. Small companies and new listings may have shorter histories.

```powershell
.\.venv\Scripts\python.exe expand_stocks.py --limit 3000
# Expand further, or refresh the ranked universe explicitly:
.\.venv\Scripts\python.exe expand_stocks.py --limit 5000 --refresh
# Discovery and fundamental quote fields only:
.\.venv\Scripts\python.exe expand_stocks.py --limit 3000 --metadata-only
```

Progress: `data/stock-expansion.json`; log: `data/stock-expansion.log`.
`config.json` sets `stock_universe_limit` for subsequent daily syncs. This command
does not publish the website or install a schedule. Generated static sites include
a screener snapshot; library studies still require the local dashboard.

## Run

```powershell
.\.venv\Scripts\python.exe market_data.py sync
.\.venv\Scripts\python.exe market_data.py status
.\.venv\Scripts\python.exe market_data.py export
.\.venv\Scripts\python.exe market_data.py sync --symbols AAPL 2330.TW BTC-USD ETH-USD
.\.venv\Scripts\python.exe market_data.py sync --full
.\.venv\Scripts\python.exe market_data.py intraday
```

Routine intraday updates now collect native **1-hour candles**. One-minute
collection (including the crypto exchange minute feeds) is no longer part of
scheduled updates. Existing minute records stay archived in SQLite; the chart
controls offer **1D** and **1H**, and 1H never depends on minute rollups.

```powershell
.\.venv\Scripts\python.exe refresh_hourly.py
# Both commands use the same expanded hourly collector:
.\.venv\Scripts\python.exe market_data.py intraday
# Resume an interrupted run, skipping successes from the last 60 minutes:
.\.venv\Scripts\python.exe refresh_hourly.py --resume
# Discover more symbols before collecting their hourly histories:
.\.venv\Scripts\python.exe expand_stocks.py --limit 5000 --metadata-only
```

The collector includes configured symbols, all previously tracked symbols, and
up to `stock_universe_limit` entries in `data/stock-universe.json` (5,000 requested;
the provider may return fewer eligible symbols). New symbols initially request
one year of available hourly history; subsequent updates overlap seven days.
Symbols with only hourly history appear in the local chart catalog and open on
1H. Daily exports and the hosted static snapshot still use daily histories.

`--symbols AAPL MSFT` limits a run; `--days` sets the initial lookback (1?729);
`--full` refetches that lookback. The low profile uses one worker with two seconds between
attempts. Each symbol commits independently, preserves earlier history, retains
the provider's session alignment, and excludes the currently forming candle.
Rate-limit responses stop new requests. Other failures retry up to three times.
The collector stops queuing work after ten minutes and starts with the oldest
attempt on the next run, so interrupted runs do not starve later symbols.
An ingestion lock prevents overlap with daily jobs; a busy lock fails the run,
and the next scheduled hourly run retries. Long daily jobs can therefore delay updates.

Progress is in `data/hourly-refresh.json`; completed run measurements are kept
in `data/hourly-runs.jsonl`, with logs in `data/hourly-refresh.log` (or
`data/ingestion.log` for the compatibility entrypoint). Request timings are
captured inside each worker, excluding the queue for database writes. Counts
represent history-call attempts, which may include internal provider requests.

Capacity uses successful symbols divided by total run wall time, including
retries, pacing, validation and storage. `estimated_symbols_per_cycle` projects
that rate over 60 minutes; `first_symbol_count_over_cycle` is one above it.
`planning_symbols_per_cycle` uses 80% of the configured collection budget.
These are observed-throughput estimates, not a guaranteed Yahoo API quota.
A run that hits rate limits or leaves failures is not evidence that every symbol
can be kept current. The hourly collector does not publish the static site.

Edit `config.json` to add Yahoo Finance symbols or disable S&P 500 collection. Defaults are current S&P 500 constituents plus your existing Taiwan and crypto symbols. Share-class dots become Yahoo hyphens. A cached constituent list is used with a warning on source failure. Membership observations are saved from now onward; historical membership is not reconstructed. Previously tracked symbols continue after index removal.

## Reliability

- Initial downloads request maximum available daily history. Updates overlap the last 14 days. Complete history refreshes every 30 days and when recent corporate actions appear, to reconcile adjustments.
- Each symbol commits atomically. A `(symbol,date)` primary key prevents duplicates. Short provider responses do not delete existing history.
- Successful symbols are skipped on repeated runs on the same UTC date, unless a full refresh is due or `--full` is used. Failed symbols retry on the next run.
- Sequential requests, pauses, three attempts, and exponential retry delays reduce request pressure. An OS lock prevents simultaneous ingestion jobs.
- Current exchange-local dates are excluded. Entirely empty OHLC records are counted and omitted. Partial OHLC and invalid volume fail the symbol.
- `state` stores successful refresh timestamps and errors; `attempts` records results. Progress is in `data/ingestion.log`.
- CSV export streams from SQLite and replaces files atomically. Run `export` to regenerate CSVs after interruption.

The prices table includes OHLC, adjusted close, volume, dividends, splits, currency, exchange, timezone, source and fetch timestamp. Yahoo prices follow its adjustment conventions; `auto_adjust=False` preserves adjusted close separately. This is not a point-in-time vintage database.

## Daily schedule

`Install-Schedule.ps1` registers **MarketData-DailySync** at **09:00 computer-local time**, while the current user is logged in. Missed starts run when available. The computer must be on and connected. Check Windows Task Scheduler and the ingestion log for results.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-Schedule.ps1
```

To update hourly charts every hour while this computer is on and the current
user is logged in, install **MarketData-IntradaySync**. Reinstalling replaces the
old 30-minute minute-data schedule:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-Intraday-Schedule.ps1
```

The task starts one hour after installation and repeats hourly, with a 15-minute
execution limit. Provider retention still limits recovery after long offline gaps.

## Analyze

Open CSVs in `data/exports/` or query SQLite:

```python
import sqlite3
import pandas as pd
with sqlite3.connect('data/market.sqlite') as db:
    prices = pd.read_sql_query(
        'SELECT date, close, adjusted_close, volume FROM prices WHERE symbol=? ORDER BY date',
        db, params=('AAPL',))
print(prices.tail())
```

Use SQLite's backup API for backups while ingestion runs; copying only the database file can omit WAL transactions.

## Scope

Daily coverage means all available **daily OHLCV and corporate-action history for configured symbols**. Minute data is archived and no longer refreshed. Hourly collection covers the expanded universe with up to one year requested initially. It excludes tick data, every global asset, and historical delisted index constituents. Yahoo/yfinance availability and completeness are not guaranteed. Successful ingestion verifies storage of valid returned records, not every expected trading session. Missing sessions and provider truncation require a separate exchange-calendar/provider audit for research requiring guaranteed completeness.

The old PowerShell downloaders and `data/*.csv` remain legacy files. Use this pipeline and `data/exports/` for ongoing data.

## Re-create environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest -v
```

Sources: https://ranaroussi.github.io/yfinance/ and https://github.com/datasets/s-and-p-500-companies


### Priority data pulls

Select a stock or crypto chart and click **Pull data**. The default requests the last **7 days** of **1-hour bars**. Choose 1m, 5m, 15m, 1h or daily bars, a rolling number of days, or an inclusive custom UTC date range, then click **Queue priority pull**. Crypto shorthand such as BTC is normalized to BTC-USD. Save named range/interval favorites in the dialog; they apply to whichever symbol you choose and persist in this browser. Delete favorites from the same selector.

The local queue persists in `data/pull-queue.sqlite`. The app starts a worker automatically, and both daily and hourly scheduled collectors drain priority requests before their next regular download. Already-running requests finish first. Interrupted priority jobs return to the queue after restart; provider failures are shown per request and can be resubmitted. The dialog refreshes status only while open. After completion, use the chart Refresh control to load the stored bars. Short requests do not mark a full scheduled backfill complete. Successfully downloaded daily/hourly symbols join the existing tracked-history universe. **Manage scheduled symbols** opens the explicit enrollment controls.

The app enforces conservative lookback limits: 7 days for 1m, 59 days for 5m/15m, 729 days for hourly, and 36,500 days for daily bars. Provider availability may be shorter. This action is local-only; hosted snapshots remain read-only.

The Workspace menu has been removed. Share chart link and Keyboard shortcuts are in chart settings. Analysis tools are organized in one searchable library under Markets & data, Strategies & relationships, Portfolio & risk, Models & forecasts, and Notes & research.


### Watchlists and color labels

Use the selector at the top of the right watchlist panel to switch between **All instruments**, existing **Saved favorites**, and named watchlists. Open the **...** menu beside it to create, rename or delete a list. **+ Add symbol** searches stored instruments and adds/removes them in the chosen list; the same dialog links to local collection for missing instruments.

Right-click an instrument, press Shift+F10 while it is focused, or use its **...** row button to open instrument options. Choose one of seven color flags, clear a flag, add/remove the instrument in multiple lists, compare it with the current chart, or open its notes. Right-clicking keeps the current chart selection. The Label filter shows a chosen color or unlabeled instruments. Flags follow the symbol across all lists.

Lists, labels and the selected list/filter are stored in this browser and sync between tabs. Existing Saved favorites remain shared with the chart and screener. Deleting a named list keeps price data, other lists, labels and notes. These controls also work in hosted snapshots; collecting missing data still requires the local app.

To reuse a color group, choose a color in **Label**, click **Save color as watchlist**, and name it (for example, **Red flags**). The saved list appears in the watchlist selector and automatically includes every instrument carrying that color, across all lists. Changing or clearing an instrument's label updates these lists immediately. Adding an instrument through a color watchlist assigns that color, replacing its previous label; removing it clears the label. Deleting the color watchlist itself keeps all labels. Successful label changes display a save confirmation and persist after reloading the browser.

Use **Sort** and its ascending/descending arrow, or click a column heading, to order the entire filtered watchlist before pagination. **Columns ±** adds/removes last price, daily/weekly/monthly percentage changes, market cap, daily volume, and instrument names. Preferences persist in this browser. Extra columns scroll horizontally inside the watchlist, keeping the symbol visible. Missing values show “—” and sort last in both directions.

Watchlist percentage changes use completed daily bars, with adjusted closes when available (raw closes only when the recent history has no adjusted prices). Weekly/monthly returns span 5/21 stored trading sessions for stocks and 7/30 daily bars for crypto. Insufficient or missing prices produce unavailable returns. These fields are separate from the chart's latest intraday quote change. Market cap comes from stored metadata; price and market-cap tooltips show the currency and available timestamps. Values are not converted between currencies for sorting. Hosted snapshots show the metrics included in their most recent build.
