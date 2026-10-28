"""Read one explicitly selected candle source for the main chart."""
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from analysis_jobs import SOURCES


def source_history(database, query):
    source = query['candle_source'][0]
    if source not in SOURCES:
        raise ValueError('Unknown candle source.')
    symbol = query.get('symbol', ['BTC-USD'])[0]
    interval = query.get('interval', ['1m'])[0]
    start, end = query.get('start', ['0001-01-01'])[0], query.get('end', ['9999-12-31'])[0]
    lower = start + 'T00:00:00+00:00'
    upper = ((date.fromisoformat(end) + timedelta(days=1)).isoformat() + 'T00:00:00+00:00') if end != '9999-12-31' else '9999-12-31T23:59:59+00:00'
    before, after = query.get('before', [None])[0], query.get('after', [None])[0]
    if before and after:
        raise ValueError('Use one history cursor.')
    if before or after:
        stamp = datetime.fromisoformat(before or after)
        if stamp.tzinfo is None:
            raise ValueError('History cursor requires a timezone.')
        stamp = stamp.astimezone(timezone.utc)
        if before:
            upper = min(upper, stamp.isoformat())
        else:
            lower = max(lower, (stamp + timedelta(seconds=1)).isoformat())
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        if source == 'Yahoo Finance' and interval in ('1d', '1w', '1mo'):
            from dashboard import aggregate_daily_bars
            rows = db.execute('SELECT date,open,high,low,close,adjusted_close,volume,dividends,splits,source FROM prices WHERE symbol=? AND source=? AND date>=? AND date<=? ORDER BY date', (symbol, source, start, end)).fetchall()
            return [dict(r) for r in rows] if interval == '1d' else aggregate_daily_bars(rows, interval)
        native = '60m' if interval == '1h' else interval
        if source == 'Yahoo Finance':
            table, extra = 'intraday_prices', 'adjusted_close'
        else:
            table, extra = 'crypto_exchange_prices', 'close AS adjusted_close'
        direction = 'ASC' if after else 'DESC'
        rows = db.execute(f'SELECT timestamp AS date,open,high,low,close,{extra},volume,NULL AS dividends,NULL AS splits,source FROM {table} WHERE symbol=? AND source=? AND interval=? AND timestamp>=? AND timestamp<? ORDER BY timestamp {direction} LIMIT 10000', (symbol, source, native, lower, upper)).fetchall()
        if not rows and source == 'Bitstamp' and symbol == 'BTC-USD' and interval in ('3m', '5m', '15m', '30m', '1h', '2h', '4h'):
            from dashboard import btc_archive_bars
            return btc_archive_bars(db, {'3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240}[interval], lower, upper, bool(after))
        return [dict(r) for r in (rows if after else reversed(rows))]
