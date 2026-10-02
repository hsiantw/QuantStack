# Brownian scenarios and expanded assets

## Local coverage added October 3, 2026 (Asia/Taipei)

Added 40 international equities across Taiwan, Hong Kong, Japan, the Netherlands,
Germany, France, Switzerland, the UK, Canada, Australia, and India, plus 26
cryptocurrencies. New histories contain 337,927 daily bars and 298,158 native
hourly bars. All 66 hourly downloads completed successfully. The local chart
catalog now contains 4,907 assets; 2,955 have daily history, including 28 crypto
assets. Existing US coverage is preserved.

Examples: MediaTek, Tencent, Toyota, Sony, ASML, SAP, LVMH, Roche, Shell,
Shopify, BHP, Reliance, Solana, XRP, Cardano, Dogecoin, Chainlink, Uniswap,
Sui, and Aptos. Yahoo uses disambiguated tickers such as `UNI7083-USD`,
`SUI20947-USD`, `APT21794-USD`, and `ARB11841-USD`. `ARB-USD` is ARbit,
which is retained under its correct name and is distinct from Arbitrum.

`expand_assets.py` collects the curated additions through the existing strict
normalization and transaction code. Newest daily sessions initially contained
partial OHLC values, so this backfill requests an exclusive UTC-day cutoff;
daily data through October 1 was loaded without fabricating missing fields.
Failed ticker probes remain in the attempt log but are not enrolled in future
collection. Successful symbols are merged into `config.json` and picked up by
the regular collectors. Rerunning the script reuses successfully collected
histories. `data/asset-expansion.json` and `data/hourly-refresh.json` contain
the latest collection results. Downloads are provider data, not a guarantee of
complete exchange coverage.

The collector writes these database additions locally. Publishing the updated
market-data snapshot exposes the daily histories on Render; committing the
config alone does not publish stored prices. Hourly histories remain local
because the hosted snapshot contains daily data only.

## Brownian motion panel

Open the bottom **Brownian motion** tab. The selected symbol, interval and loaded
date range determine the available estimation data. Defaults use the latest
252 log returns (or fewer if fewer bars are loaded), 2,000 paths, 60 future bars,
adjusted closes, and seed 42. At least 21 valid price bars are required.

The geometric Brownian model is `dS = μS dt + σS dW`. With one observed bar as
the time unit, each exact transition is `S_next = S exp(μ − σ²/2 + σZ)`.
Historical calibration uses sample log-return variance for `σ²` and
`μ = mean(log return) + σ²/2`. Zero-price-drift mode sets μ to zero while
retaining estimated volatility; custom mode accepts per-bar drift and
volatility per square-root bar. There is no automatic annualization or mapping
to future calendar dates. Price basis remains consistent between estimation
and the starting price. Missing, nonpositive or unordered prices reject a run.

Results show pointwise simulated 5th/25th/50th/75th/95th percentiles, means,
and probabilities of ending below the initial price. A slider inspects each
future bar; 12 optional sample paths illustrate individual realizations.
Analytic lognormal horizon values provide a reference for simulation noise.
JSON exports include assumptions, calibration context, fan values and sample
paths; CSV exports include all fan rows. Identical data, settings and seed
reproduce the same paths. Changing chart data or settings clears stale results
and disables exports; canceling terminates the worker.

This model assumes independent Gaussian log returns and fixed parameters. It
does not capture volatility clustering, jumps, trading costs or parameter
uncertainty. Bands are scenario percentiles, not guaranteed or simultaneous
confidence bounds, and loss probability is terminal rather than pathwise.
No forecast validation or trading recommendation is implied.

Reference: [Columbia University: Geometric Brownian motion](https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf).

## Checks

```powershell
python -m unittest test_brownian test_market_data test_quantstack_gateway
python smoke_brownian.py --url http://127.0.0.1:8502/workspace/
python smoke_brownian.py --static
```

Numerical tests cover sample-variance calibration, the drift correction, exact
zero-volatility paths, closed-form moments and probabilities, seeded
reproducibility, and invalid data. Browser checks cover worker execution,
exports, cancellation, stale results, recovery, themes, mobile layout, and
static packaging.
