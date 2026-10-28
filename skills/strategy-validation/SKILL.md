---
name: strategy-validation
description: Validate stock or crypto strategies in QuantStack with chronological holdouts, realistic execution costs and benchmark comparisons using VectorBT.
---

# Strategy validation

## QuantStack environment

Project: `C:/Users/i/Desktop/quant stack/QuantStack`. Verify this directory before use; outside QuantStack adapt to the user's actual data instead of assuming this schema.
Use `.venv-analysis/Scripts/python.exe` for research. Setup and smoke checks are in `docs/TRADING-ANALYSIS.md` and `smoke_analysis.py` at the project root.
Read local SQLite with a read-only connection (URI `file:.../data/market.sqlite?mode=ro`); do not call `market_data.connect` for analysis because it initializes schema. Bound queries by symbol and date. Keep generated research in `data/analysis/`.
Daily bars: `prices` (symbol, date, OHLC, adjusted_close, volume, dividends, splits, currency, exchange, timezone, source). Yahoo intraday: `intraday_prices` (symbol, timestamp, interval, OHLC, volume, currency, exchange, timezone, source). BTC exchange bars: `crypto_exchange_prices` (symbol, timestamp, interval, source, OHLC, fractional BTC volume). Select one exchange source explicitly; never sum or silently splice overlapping sources.

## Research workflow

- Establish entry/exit rules, sizing, assets, timeframe, holding period and benchmark before tuning. Make fees, spread, slippage, borrow/funding costs and execution timing explicit.
- Inspect data coverage and corporate actions. Separate stock sessions from 24/7 crypto. Account for survivorship bias when the universe consists of today's listed symbols.
- Use VectorBT for repeatable simulations. A signal computed from a closing candle cannot fill at that same close by default; shift execution to the next available tradable bar and document the price used. OHLC alone cannot establish intrabar stop/target ordering.
- Split chronologically into training, validation and an untouched holdout. Fit transforms and tune parameters only on earlier data. Use walk-forward evaluation and purge overlapping label/holding windows where appropriate.
- Record parameter searches; show parameter sensitivity and out-of-sample results rather than reporting only the best trial. Compare net returns with buy-and-hold and cash on identical periods and capital assumptions.
- Use QuantStats for return/risk summaries. Aggregate minute equity to daily returns before daily annualization; use appropriate stock/crypto calendars. Report costs, trades, turnover, drawdown, exposure and uncertainty along with returns.
- Use Alphalens Reloaded for cross-sectional signal evaluation when requested: align factors to information available at formation, evaluate forward-return horizons and quantile spreads, and inspect dropped observations and turnover. Predictive factors are not automatically executable strategies.
- Save data boundaries, source, package versions and parameters with results. Research authorization does not authorize broker orders or a live strategy deployment.

