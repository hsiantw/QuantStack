---
name: portfolio-risk
description: Assess stock and crypto portfolio exposure, concentration, correlation, drawdowns and scenario losses in QuantStack using explicit holdings and return assumptions.
---

# Portfolio risk

## QuantStack environment

Project: `C:/Users/i/Desktop/quant stack/QuantStack`. Verify this directory before use; outside QuantStack adapt to the user's actual data instead of assuming this schema.
Use `.venv-analysis/Scripts/python.exe` for research. Setup and smoke checks are in `docs/TRADING-ANALYSIS.md` and `smoke_analysis.py` at the project root.
Read local SQLite with a read-only connection (URI `file:.../data/market.sqlite?mode=ro`); do not call `market_data.connect` for analysis because it initializes schema. Bound queries by symbol and date. Keep generated research in `data/analysis/`.
Daily bars: `prices` (symbol, date, OHLC, adjusted_close, volume, dividends, splits, currency, exchange, timezone, source). Yahoo intraday: `intraday_prices` (symbol, timestamp, interval, OHLC, volume, currency, exchange, timezone, source). BTC exchange bars: `crypto_exchange_prices` (symbol, timestamp, interval, source, OHLC, fractional BTC volume). Select one exchange source explicitly; never sum or silently splice overlapping sources.

## Risk workflow

- Obtain holdings/weights, cash, leverage, base currency and valuation time from supplied data. A watchlist is not a portfolio; if holdings are absent, request them or clearly label an illustrative scenario.
- Align returns to a common frequency and currency. Account for FX or disclose its omission. For stock/crypto mixes, distinguish valuation carry on stock-closed days from actual observed returns; do not create fictitious stock trading sessions.
- Specify fixed-weight versus drifting-weight assumptions and rebalancing costs. Use total-return treatment for equities when supported; avoid double-counting dividends in adjusted prices.
- Calculate concentration, gross/net exposure, drawdown, volatility and correlation over stated windows. Use QuantStats on consistent portfolio/benchmark return series with explicit annualization and risk-free-rate conventions.
- Stress price shocks, correlation increases and illiquidity where relevant. If reporting VaR/expected shortfall, state horizon, method, confidence, sample length and tail-sample limitations; these are estimates, not maximum possible losses.
- Show which holdings drive risk and compare candidate changes under the same assumptions. Do not infer permission to rebalance or execute transactions from a risk-analysis request.

