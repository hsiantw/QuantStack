# Native chart workspace migration

The supported application is `serve_quantstack.py` → `/workspace/`. It starts
no Streamlit process, iframe, websocket session, account page or upload handler.
`/?research=1` redirects to `/workspace/`, preserving chart parameters. The old
Python source in `QuantStack-main/` is retained as an implementation reference;
none of its pages is served by the supported launcher. User databases are not
deleted or exposed through the chart API. At migration, the local legacy
database had zero saved portfolios, strategies and preferences.

Use the right rail's **Analysis library** grid icon to search all tools. Nine
direct icons open native panels under the current chart. These panels share its
theme, date range, interval and stored dataset. The dock can be resized, maximized
or collapsed. Re-clicking an active tool icon collapses it. Missing data produces
an explicit explanation. Calculations run only on request in a cancelable worker;
chart/input changes invalidate results. JSON exports include chart context.

## Page-by-page disposition

| Previous file in `QuantStack-main/pages/` | Native destination / replacement |
| --- | --- |
| `market_workspace.py` | Main chart, without an iframe |
| `portfolio_manager.py` | Portfolio & allocation: editable holdings, cost basis, named browser-local portfolios |
| `portfolio_optimization.py` | Portfolio & allocation: equal weight, inverse volatility, long-only minimum variance |
| `dynamic_portfolio_rebalancing.py` | Portfolio & allocation: target weights and hypothetical rebalance units |
| `risk_management.py` | Returns & risk, Brownian simulations, Options lab, portfolio shock scenario |
| `advanced_risk_management.py` | Returns & risk: VaR, expected shortfall, drawdown; portfolio uniform shock |
| `strategy_backtesting.py` | Strategy tester: native execution model, costs, trades and performance |
| `interactive_backtest_visualization.py` | Strategy tester: equity, benchmark, drawdown, trades and monthly performance |
| `trading_strategies.py` | Strategy tester: presets, configurable rules and saved setups |
| `strategy_comparison.py` | Strategy comparison: all native presets, same inputs/costs, inspect in tester |
| `orb_strategy.py` | Replaced by native channel breakout rules. Opening-range execution is unavailable without session-aware minute bars; it is not mislabeled as ORB |
| `market_performance_tracker.py` | Market overview, Stock screener rankings and Strategy tester momentum presets |
| `statistical_arbitrage.py` | Pairs & statistical analysis: return correlation, log-price OLS, residual spread and z-score. No cointegration claim |
| `ai_pairs_trading.py` | Pairs panel; unsupported AI selection/live signals replaced by reproducible stored-data calculations |
| `time_series_analysis.py` | Returns & risk, Markov analysis, Forecast lab |
| `ai_analysis.py` | Forecast lab: walk-forward AR(1) vs no-change baseline; native indicators and custom strategy rules replace external AI recommendations/code generation |
| `options_analysis.py` | Options lab: European Black–Scholes, dividend yield, Greeks and expiry payoff. No live chains or American-exercise model |
| `fundamental_analysis.py` | Company fundamentals and Stock screener, using stored metadata with source and date |
| `liquidity_analysis.py` | Volume & liquidity: turnover, relative volume, OBV, A/D, Amihud and approximate volume buckets |
| `advanced_market_analysis.py` | Volume & liquidity plus native indicator library; unsupported dark-pool/order-flow estimates removed |
| `crypto_analysis.py` | Market overview crypto filter, symbol charts, indicators, risk and strategies; unsupported on-chain/DeFi proxies removed |
| `commodities_forex_futures.py` | Market overview and chart tools for instruments actually present in the local catalog; no synthetic curves or positioning |
| `market_data_stream.py` | Daily/hourly chart, stored bars, fundamentals, Options lab and local scheduler; no pretend streaming quotes |
| `news_and_economic_data.py` | Research journal and source links replace unavailable live news/calendar feeds |
| `market_information_sources.py` | Per-symbol Research journal/source reference workflow |
| `trading_monitor.py` | Portfolio holdings and Strategy tester performance replace unavailable broker accounts/orders; no trade submission |
| `user_settings.py` | Existing theme/settings, saved strategy setups, saved portfolios and symbol notes; obsolete account/subscription UI removed |

These are consolidated replacements, not claims of identical legacy feature
coverage. Old simulation proxies, broker integrations and credential-dependent
feeds are not silently reproduced. Mathematical assumptions and data limitations
appear next to each native form.

## Data and calculations

Portfolio and pair calculations use the intersection of price timestamps without
forward filling. Portfolios require a known common currency and 2–12 holdings;
no FX conversion occurs. Portfolio history is fixed initial allocation with no
rebalance costs. Last stored quotes used for hypothetical rebalancing display
their dates. Minimum variance is fitted to the same history and is not a validated
strategy. An all-stock or mixed portfolio assumes 252 daily / 1,638 hourly bars per
year; all-crypto portfolios assume 365 / 8,760.

Forecast evaluation uses expanding past-only fits over the final 20% of observed
log returns. It does not claim predictive superiority; baseline MAE remains
visible. Volume buckets use closing prices, not transactions. Options use
user-entered assumptions; Greeks are per underlying unit.

## Verification

`python -m unittest test_research test_quantstack_gateway test_local_scheduler`
checks numerical reference values, alignment, missing data, forecast leakage,
route retirement, snapshot delivery and local scheduler access.
`python smoke_research_workspace.py` exercises all nine native panels with
deterministic fixtures, saved portfolios, stale results, library navigation and
desktop/mobile themes. `python smoke_quantstack.py` checks the running application
with its actual stored data.
