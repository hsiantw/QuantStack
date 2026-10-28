# Trading analysis tools

Research libraries run in `.venv-analysis`, separate from the collector/server
environment. VectorBT handles strategy simulations, QuantStats return reports,
Alphalens Reloaded factor evaluation, and CCXT exchange market-data access.
The chart's **Research tools** button opens four workflows backed by these
libraries. The strategy, screener, risk and price-bar panels also have contextual
launch buttons. Research is available on the local app only.

## Chart workflows

The chart toolbar also has **Database** (or **Price bars → Browse database
tables**). This read-only browser shows table schemas, filtered row counts and
paged records. Filter by symbol, interval, source, status and date where those
columns exist, refresh while collection runs, and export the displayed page as
CSV. Initial inspection uses the chart symbol and the last seven days; clear
filters to inspect older data. Pages contain 25–200 rows. Broad queries time out
after eight seconds; use symbol/date filters for large tables. The endpoint is
`GET /workspace/api/database`, with the same local-only access checks as research.
Only `data/market.sqlite` is exposed; account and preference databases are excluded.

- **Strategy validation:** a long-only SMA regime with editable periods, capital,
  fees and slippage. VectorBT evaluates the final 30% of the chosen range with
  next-open fills; QuantStats computes daily risk metrics. This workflow has its
  own settings and does not execute the browser tester's other presets/custom rules.
- **Portfolio report:** enter symbols and initial weights totaling 100%. QuantStats
  reports buy-and-hold portfolio performance and correlations on common daily
  observations, with drifting weights and no assumed rebalancing.
- **Factor research:** Alphalens evaluates momentum across 5–20 same-currency
  symbols. Lookback and horizon count common observations. Reports include rank
  IC, quintile forward returns and top-quintile turnover; this is descriptive
  research rather than an executable backtest.
- **Exchange data:** CCXT pulls public spot USD candles from Bitstamp, Kraken or
  Coinbase and saves source-labeled rows. Up to 31 calendar days and 20 requests
  per job; venue retention can be shorter. **View pulled candles on chart** loads
  that exact venue/timeframe. The chart's **Candle source** selector can switch
  between Automatic, Yahoo Finance and each supported exchange. Source selection
  is session-local. Only the requested source's stored intervals are served;
  Bitstamp BTC minute history also supports the existing intraday rollups.

Choose a source, symbols, interval and range, then run the job. Job progress and
cancellation are available in the dialog. Closing it leaves work running; opening
it restores the last job for this browser. Each result contains its input snapshot,
source coverage, assumptions and package versions, with JSON and curve-CSV exports.
The plotted curve is sampled to at most 1,201 points; calculations use all loaded
bars. Trade tables include the first 500 records. Open positions remain marked
to the final close. No order execution is implemented.

## Backend and limits

`analysis_jobs.py` queues up to eight active jobs and runs one isolated
`.venv-analysis` subprocess at a time, hidden on Windows, with a ten-minute limit.
`analysis_worker.py` reads market data through SQLite read-only connections.
CCXT writes alone acquire the existing ingestion lock. Cancelling after a commit
retains the candles already stored. Interrupted jobs are marked failed after a
server restart; resubmit them explicitly. Request, status, worker log and result
files remain under `data/analysis/jobs/<id>/`.

`POST /workspace/api/analysis` submits a validated request;
`GET /workspace/api/analysis?id=<id>` returns progress/result;
`POST /workspace/api/analysis/cancel` cancels `{ "id": "..." }`.
Gateway and backend enforce local same-origin access. There are no client-supplied
Python expressions or file paths. Hosted snapshots cannot execute research.

Research loads at most 50,000 source bars per symbol and never silently truncates
that input. Portfolio/factor jobs use daily bars, one currency and one asset class
per report, with no FX conversion. Yahoo daily OHLC uses adjusted-close ratios;
exchange prices remain venue-specific. Source gaps and incomplete aggregation
buckets are disclosed. Stock calendar gaps include market closures and are not
classified as missing trades. Missing data is not filled. These limits are shown
in the UI rather than substituting a different dataset.

Recreate the tested environment from the project directory:

```powershell
.\.venv-integration\Scripts\python.exe -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis-lock.txt
.\.venv-analysis\Scripts\python.exe -m pip check
.\.venv-analysis\Scripts\python.exe smoke_analysis.py
.\.venv-analysis\Scripts\python.exe -m unittest test_analysis
.\.venv-integration\Scripts\python.exe smoke_analysis_ui.py --live-exchange
```

`requirements-analysis.txt` lists direct dependencies; the lock file records
the tested Python 3.13 Windows environment. The smoke check uses synthetic
prices and factors, validates a backtest with costs, a known drawdown, factor
correlation, and CCXT's Bitstamp adapter without making network requests.
It is a dependency compatibility check, not a profitable trading strategy.
`test_analysis` checks the real calculation libraries with isolated fixtures.
The browser smoke test exercises jobs, cancellation, local access, downloads and
responsive layouts; `--live-exchange` adds a public Bitstamp pull into its isolated
fixture database. Restart the local chart server after Python code changes.

Four custom skills have versioned source in `skills/` and are installed under
the user's Codex skills directory:

- `$market-data-audit`: inspect source, coverage, freshness and data quality.
- `$strategy-validation`: run chronological out-of-sample tests with costs.
- `$trading-analysis`: explain trend, momentum, volatility and volume.
- `$portfolio-risk`: assess supplied holdings, exposures and scenario losses.

Examples: "Use $market-data-audit on BTC-USD minute candles for the last week"
or "Use $strategy-validation to evaluate a moving-average strategy with fees."
If skills do not appear in the current session, start a new session.

Use read-only SQLite queries for research and write outputs to `data/analysis/`.
Daily equity prices, Yahoo intraday data and exchange-specific crypto candles
have different adjustment, volume and calendar semantics; the skills describe
these differences. New dependencies do not change scheduled collection.
