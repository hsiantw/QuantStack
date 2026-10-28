---
name: trading-analysis
description: Analyze stock or crypto price action in QuantStack using trend, momentum, volatility and volume with explicit data timestamps and reproducible calculations.
---

# Trading analysis

## QuantStack environment

Project: `C:/Users/i/Desktop/quant stack/QuantStack`. Verify this directory before use; outside QuantStack adapt to the user's actual data instead of assuming this schema.
Use `.venv-analysis/Scripts/python.exe` for research. Setup and smoke checks are in `docs/TRADING-ANALYSIS.md` and `smoke_analysis.py` at the project root.
Read local SQLite with a read-only connection (URI `file:.../data/market.sqlite?mode=ro`); do not call `market_data.connect` for analysis because it initializes schema. Bound queries by symbol and date. Keep generated research in `data/analysis/`.
Daily bars: `prices` (symbol, date, OHLC, adjusted_close, volume, dividends, splits, currency, exchange, timezone, source). Yahoo intraday: `intraday_prices` (symbol, timestamp, interval, OHLC, volume, currency, exchange, timezone, source). BTC exchange bars: `crypto_exchange_prices` (symbol, timestamp, interval, source, OHLC, fractional BTC volume). Select one exchange source explicitly; never sum or silently splice overlapping sources.

## Analysis workflow

- Establish symbol, venue, interval, horizon and cutoff. Read stored data first and report its actual latest completed bar. Do not describe delayed local data as live.
- Assess trend, momentum, volatility and volume with a small set of complementary measures. Explain indicator windows in bars and elapsed time, warmup requirements and how each supports the conclusion.
- TA-Lib is listed in the collector environment's requirements. The separate analysis environment may use explicit pandas formulas; label smoothing/initialization conventions rather than silently assuming indicator equivalence.
- Use CCXT for additional public exchange OHLCV only when needed by the request. Inspect exchange capabilities and symbol mapping; distinguish BTC/USD from BTC/USDT and spot from derivatives. Respect pagination/rate limits, omit incomplete candles and preserve source/quote currency. CCXT does not guarantee full historical coverage.
- Present observations separately from hypotheses. Include supporting values and timestamps, alternate scenarios and what would invalidate each interpretation. Do not assign probabilities without an evaluated model.
- If the user asks for predictive performance, validate it with chronological out-of-sample tests; an indicator reading is not evidence of profitability. Do not place trades from an analysis request.

