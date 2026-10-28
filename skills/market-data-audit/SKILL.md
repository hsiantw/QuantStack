---
name: market-data-audit
description: Audit QuantStack stock and crypto OHLCV coverage, freshness, calendar gaps, corporate actions and source consistency before research.
---

# Market data audit

## QuantStack environment

Project: `C:/Users/i/Desktop/quant stack/QuantStack`. Verify this directory before use; outside QuantStack adapt to the user's actual data instead of assuming this schema.
Use `.venv-analysis/Scripts/python.exe` for research. Setup and smoke checks are in `docs/TRADING-ANALYSIS.md` and `smoke_analysis.py` at the project root.
Read local SQLite with a read-only connection (URI `file:.../data/market.sqlite?mode=ro`); do not call `market_data.connect` for analysis because it initializes schema. Bound queries by symbol and date. Keep generated research in `data/analysis/`.
Daily bars: `prices` (symbol, date, OHLC, adjusted_close, volume, dividends, splits, currency, exchange, timezone, source). Yahoo intraday: `intraday_prices` (symbol, timestamp, interval, OHLC, volume, currency, exchange, timezone, source). BTC exchange bars: `crypto_exchange_prices` (symbol, timestamp, interval, source, OHLC, fractional BTC volume). Select one exchange source explicitly; never sum or silently splice overlapping sources.

## Audit workflow

- Identify requested universe, interval, venue, time range and analysis cutoff. Compare the catalog/configured universe against stored symbols; never treat a configured symbol as proof of downloaded data.
- Report first/last completed candle, row count, duplicates, invalid OHLC bounds, nonfinite values, negative volume and freshness per symbol/source. Preserve valid zero-volume bars.
- Count missing crypto minutes on the UTC grid; stock gaps require exchange sessions, holidays, DST and halts. If the calendar is unavailable label gaps as unclassified, not missing trades.
- Distinguish collection backlog, provider unavailability and market closure. Check `data/minute-refresh.json`, `data/btc-backfill.json` and collector attempts as supporting evidence; process/task status alone is not proof of coverage.
- Check splits/dividends and adjusted-price treatment for equities. Intraday Yahoo volumes are stored as integers; do not claim precision matching fractional exchange volumes.
- Return a coverage summary, concrete exceptions and which proposed analyses those exceptions affect. Do not fabricate or forward-fill missing candles to make an audit pass. Download or repair only within the user's requested scope.

