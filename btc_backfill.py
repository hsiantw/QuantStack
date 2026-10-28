"""Resumable BTC/USD 1-minute Bitstamp archive; retries until every window succeeds.

Run with .venv-integration/Scripts/python.exe btc_backfill.py.
Progress: data/btc-backfill.json. Checkpoints and candles commit atomically.
"""
import concurrent.futures
from datetime import datetime, timezone
import json
import math
import time
import threading
import urllib.request

from market_data import DATA, connect, process_lock

URL = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'
START = 1230768000  # 2009-01-01: probe before the exchange existed, without assuming inception.
STEP = 60000
_rate_lock = threading.Lock()
_next_request = 0.0


def iso(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def fetch(start, end, step=60):
    global _next_request
    with _rate_lock:
        pause = max(0, _next_request - time.monotonic())
        _next_request = max(_next_request, time.monotonic()) + 0.1
    if pause:
        time.sleep(pause)
    limit = min(1000, (end - start) // step)
    url = f'{URL}?step={step}&limit={limit}&start={start}&end={end-step}'
    request = urllib.request.Request(url, headers={'User-Agent': 'QuantStack-BTC-Archive/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.load(response)
    candles = payload['data']['ohlc']
    if not isinstance(candles, list):
        raise ValueError('Invalid OHLC response')
    rows = []
    seen = set()
    fetched = iso(time.time())
    for candle in candles:
        stamp = int(candle['timestamp'])
        values = [float(candle[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
        o, h, l, c, v = values
        if stamp in seen or stamp % step or not start <= stamp < end:
            raise ValueError('Duplicate, unaligned or out-of-window timestamp')
        if not all(math.isfinite(x) for x in values) or min(o, h, l, c) <= 0 or v < 0 or not l <= min(o, c) <= max(o, c) <= h:
            raise ValueError('Invalid OHLCV')
        seen.add(stamp)
        rows.append(('BTC-USD', iso(stamp), '1m', 'Bitstamp', *values, fetched))
    return rows


def report(db, state, error=None):
    total, done, count, missing = db.execute('SELECT COUNT(*),SUM(done),SUM(rows),SUM(missing) FROM btc_backfill_windows').fetchone()
    payload = dict(status=state, source='Bitstamp', symbol='BTC-USD', interval='1m',
                   updated_at=iso(time.time()), windows=total, completed=done or 0,
                   rows=count or 0, missing_minutes=missing or 0, error=error,
                   **dict(db.execute('SELECT key,value FROM btc_backfill_meta')))
    temp = DATA / 'btc-backfill.tmp'
    temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temp.replace(DATA / 'btc-backfill.json')
    print(json.dumps(payload), flush=True)


def archive_totals(db, first, target):
    # Combining COUNT/MIN/MAX makes SQLite compare both timestamps for every
    # candle. Separate scalar queries keep the exact full count while allowing
    # MIN/MAX to seek to the index edges. One statement keeps a single snapshot.
    clause = """FROM crypto_exchange_prices
        WHERE symbol='BTC-USD' AND interval='1m' AND source='Bitstamp'
        AND timestamp>=:first AND timestamp<:target"""
    return db.execute(f'''SELECT
        (SELECT COUNT(*) {clause}),
        (SELECT MIN(timestamp) {clause}),
        (SELECT MAX(timestamp) {clause})''',
        {'first': iso(first), 'target': iso(target)}).fetchone()


def run():
    with process_lock(DATA / 'ingestion.lock'), connect(DATA / 'market.sqlite') as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS btc_backfill_windows (
            start INTEGER PRIMARY KEY, end INTEGER NOT NULL, done INTEGER DEFAULT 0,
            rows INTEGER DEFAULT 0, missing INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS btc_backfill_meta(key TEXT PRIMARY KEY,value TEXT);''')
        if not db.execute('SELECT 1 FROM btc_backfill_windows LIMIT 1').fetchone():
            # Daily discovery finds the first listed day without thousands of empty minute calls.
            cursor = START
            while True:
                daily = fetch(cursor, cursor + 1000 * 86400, 86400)
                if daily:
                    first = int(datetime.fromisoformat(min(r[1] for r in daily)).timestamp())
                    break
                cursor += 1000 * 86400
                if cursor >= time.time():
                    raise ValueError('No historical data found')
            target = int(time.time()) // 60 * 60
            with db:
                db.executemany('INSERT INTO btc_backfill_meta VALUES (?,?)',
                               [('start', iso(first)), ('target_exclusive', iso(target))])
                db.executemany('INSERT INTO btc_backfill_windows(start,end) VALUES (?,?)',
                               [(s, min(s+STEP, target)) for s in range(first, target, STEP)])
        delay = 5
        while True:
            pending = db.execute('SELECT start,end FROM btc_backfill_windows WHERE done=0 ORDER BY start DESC').fetchall()
            if not pending:
                target = db.execute('SELECT MAX(end) FROM btc_backfill_windows').fetchone()[0]
                latest = int(time.time()) // 60 * 60
                if latest > target:
                    with db:
                        db.executemany('INSERT INTO btc_backfill_windows(start,end) VALUES (?,?)',
                                       [(s, min(s+STEP, latest)) for s in range(target, latest, STEP)])
                        db.execute("UPDATE btc_backfill_meta SET value=? WHERE key='target_exclusive'", (iso(latest),))
                    continue
                first = db.execute('SELECT MIN(start) FROM btc_backfill_windows').fetchone()[0]
                actual = archive_totals(db, first, target)
                expected = db.execute('SELECT SUM(rows) FROM btc_backfill_windows').fetchone()[0]
                if actual[0] != expected:
                    raise ValueError(f'Archive audit failed: stored {actual[0]}, checkpointed {expected}')
                first_stamp = int(datetime.fromisoformat(actual[1]).timestamp())
                last_stamp = int(datetime.fromisoformat(actual[2]).timestamp())
                with db:
                    db.executemany('INSERT OR REPLACE INTO btc_backfill_meta VALUES (?,?)',
                                   [('verified_rows', str(actual[0])), ('first_candle', actual[1]),
                                    ('last_candle', actual[2]),
                                    ('leading_unavailable_minutes', str((first_stamp-first)//60)),
                                    ('internal_missing_minutes', str((last_stamp-first_stamp)//60+1-actual[0]))])
                report(db, 'complete')
                return
            report(db, 'running')
            failures = 0
            # Ten requests/second is below the documented 10,000 per ten minutes.
            # Bounded batches avoid retaining a full archive in RAM.
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                for offset in range(0, len(pending), 80):
                    futures = {pool.submit(fetch, s, e): (s, e) for s, e in pending[offset:offset+80]}
                    for future in concurrent.futures.as_completed(futures):
                        s, e = futures[future]
                        try:
                            rows = future.result()
                            if not rows:
                                raise ValueError('Empty minute window inside listed history; retry required')
                            with db:
                                db.executemany('''INSERT INTO crypto_exchange_prices VALUES (?,?,?,?,?,?,?,?,?,?)
                                    ON CONFLICT(symbol,timestamp,interval,source) DO UPDATE SET
                                    open=excluded.open,high=excluded.high,low=excluded.low,
                                    close=excluded.close,volume=excluded.volume,fetched_at=excluded.fetched_at''', rows)
                                db.execute('UPDATE btc_backfill_windows SET done=1,rows=?,missing=? WHERE start=?',
                                           (len(rows), (e-s)//60-len(rows), s))
                        except Exception as exc:
                            failures += 1
                            report(db, 'retrying', f'{iso(s)}: {exc}')
                    report(db, 'running')
                    time.sleep(1)
            if failures:
                time.sleep(delay)
                delay = min(delay * 2, 300)


if __name__ == '__main__':
    run()
