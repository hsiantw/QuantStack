"""Export bounded, deterministic chart snapshots for Git (not a full backup)."""
import csv
import hashlib
import json
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def snapshot(database=ROOT / 'data' / 'market.sqlite', output=ROOT / 'chart-snapshots'):
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=60)) as db:
        db.execute('BEGIN')  # Both timeframes see one consistent database snapshot.
        for name, table, timecol, limit, condition in (
            ('daily', 'prices', 'date', 30, ''),
            ('hourly', 'intraday_prices', 'timestamp', 48, " AND interval='60m'"),
        ):
            fields = f'symbol,{timecol},open,high,low,close,adjusted_close,volume,currency,exchange,timezone,source'
            # State tables are small and contain all symbols saved by the collectors.
            symbols = db.execute('SELECT symbol FROM state UNION SELECT symbol FROM intraday_state ORDER BY symbol').fetchall()
            counts[name] = 0
            paths = [output / f'{name}-{bucket:x}.csv' for bucket in range(16)]
            with ExitStack() as stack:
                writers = [csv.writer(stack.enter_context(path.with_suffix('.tmp').open('w', encoding='utf-8', newline='')))
                           for path in paths]
                for writer in writers:
                    writer.writerow(fields.split(','))
                for (symbol,) in symbols:
                    rows = db.execute(f'SELECT {fields} FROM {table} WHERE symbol=?{condition} ORDER BY {timecol} DESC LIMIT ?',
                                      (symbol, limit)).fetchall()
                    bucket = hashlib.sha256(symbol.encode()).digest()[0] % 16
                    writers[bucket].writerows(reversed(rows))
                    counts[name] += len(rows)
            for path in paths:
                path.with_suffix('.tmp').replace(path)
    metadata = dict(rows=counts, daily_bars_per_symbol=30, hourly_bars_per_symbol=48,
                    scope='Latest stored completed daily and hourly candles; full history remains in local SQLite.')
    temporary = output / 'manifest.tmp'
    temporary.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output / 'manifest.json')
    print(json.dumps(metadata))
    return counts


if __name__ == '__main__':
    snapshot()
