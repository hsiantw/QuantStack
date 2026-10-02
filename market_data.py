"""Resumable daily and hourly Yahoo Finance ingestion. Run --help for commands."""
import argparse
import csv
import io
import json
import logging
import math
import sqlite3
import sys
import time
import urllib.request
import urllib.parse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SOURCE = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'


def connect(path):
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL,
        close REAL, adjusted_close REAL, volume INTEGER, dividends REAL, splits REAL,
        currency TEXT, exchange TEXT, timezone TEXT, fetched_at TEXT,
        source TEXT NOT NULL DEFAULT 'Yahoo Finance', PRIMARY KEY(symbol,date));
      CREATE INDEX IF NOT EXISTS prices_date ON prices(date);
      CREATE TABLE IF NOT EXISTS state (
        symbol TEXT PRIMARY KEY, last_success TEXT, last_full TEXT, error TEXT);
      CREATE TABLE IF NOT EXISTS attempts (
        id INTEGER PRIMARY KEY, symbol TEXT, started TEXT, mode TEXT,
        status TEXT, rows INTEGER, empty_rows INTEGER, error TEXT);
      CREATE TABLE IF NOT EXISTS membership (
        observed_date TEXT, symbol TEXT, PRIMARY KEY(observed_date,symbol));
      CREATE TABLE IF NOT EXISTS intraday_prices (
        symbol TEXT NOT NULL, timestamp TEXT NOT NULL, interval TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, adjusted_close REAL,
        volume INTEGER, currency TEXT, exchange TEXT, timezone TEXT,
        fetched_at TEXT, source TEXT NOT NULL DEFAULT 'Yahoo Finance',
        PRIMARY KEY(symbol,timestamp,interval));
      CREATE INDEX IF NOT EXISTS intraday_prices_timestamp ON intraday_prices(timestamp);
      CREATE TABLE IF NOT EXISTS intraday_state (
        symbol TEXT NOT NULL, interval TEXT NOT NULL, last_success TEXT,
        error TEXT, PRIMARY KEY(symbol,interval));
      CREATE TABLE IF NOT EXISTS crypto_exchange_prices (
        symbol TEXT NOT NULL, timestamp TEXT NOT NULL, interval TEXT NOT NULL,
        source TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL,
        volume REAL, fetched_at TEXT, PRIMARY KEY(symbol,timestamp,interval,source));
      CREATE INDEX IF NOT EXISTS crypto_exchange_timestamp ON crypto_exchange_prices(symbol,timestamp);
      CREATE TABLE IF NOT EXISTS api_requests (
        id INTEGER PRIMARY KEY, provider TEXT NOT NULL, endpoint TEXT NOT NULL,
        requested_at TEXT NOT NULL, status TEXT NOT NULL, rows INTEGER,
        duration_ms INTEGER, error TEXT);
      CREATE INDEX IF NOT EXISTS api_requests_time ON api_requests(requested_at);
    ''')
    return db


@contextmanager
def process_lock(path):
    # OS releases the byte lock even after a crash; an existing file is harmless.
    import msvcrt
    with open(path, 'a+b') as lock:
        lock.seek(0, 2)
        if not lock.tell():
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another ingestion process is already running') from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def universe(db, config):
    symbols = set(config['symbols'])
    if config['sp500']:
        cache = DATA / 'sp500.json'
        try:
            request = urllib.request.Request(SOURCE, headers={'User-Agent': 'market-data/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                source_csv = response.read().decode('utf-8-sig')
                entries = list(csv.DictReader(io.StringIO(source_csv)))
            members = sorted({row['Symbol'].replace('.', '-') for row in entries})
            if not 450 <= len(members) <= 550:
                raise ValueError('Unexpected constituent count')
            names_temp = DATA / 'constituents.tmp'
            names_temp.write_text(source_csv, encoding='utf-8')
            names_temp.replace(DATA / 'constituents.csv')
            snapshot = {'observed_at': datetime.now(timezone.utc).isoformat(), 'source': SOURCE, 'symbols': members}
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
            temporary.replace(cache)
            with db:
                db.executemany('INSERT OR IGNORE INTO membership VALUES (?,?)',
                               [(date.today().isoformat(), s) for s in members])
        except Exception:
            if not cache.exists():
                raise
            snapshot = json.loads(cache.read_text(encoding='utf-8'))
            members = snapshot['symbols']
            logging.warning('Universe refresh failed; using snapshot from %s', snapshot['observed_at'], exc_info=True)
        symbols.update(members)
    # Continue tracking previously downloaded symbols after index removal.
    symbols.update(row[0] for row in db.execute('SELECT symbol FROM state'))
    expanded = DATA / 'stock-universe.json'
    if config.get('stock_universe_limit', 0) and expanded.exists():
        ranked = json.loads(expanded.read_text(encoding='utf-8'))['symbols'][:config['stock_universe_limit']]
        return list(dict.fromkeys(ranked + sorted(symbols)))
    return sorted(symbols)


def normalize(symbol, frame, meta, today):
    rows, empty = [], 0
    seen = set()
    for stamp, item in frame.iterrows():
        day = stamp.date()
        if day >= today:
            continue
        values = [item.get(name) for name in ('Open', 'High', 'Low', 'Close')]
        if all(value is None or math.isnan(float(value)) for value in values):
            empty += 1
            continue
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(f'{symbol}: partial OHLC on {day}')
        volume = float(item['Volume'])
        if not math.isfinite(volume) or volume < 0 or not volume.is_integer():
            raise ValueError(f'{symbol}: invalid volume on {day}')
        if day in seen:
            raise ValueError(f'{symbol}: duplicate date {day}')
        seen.add(day)
        def optional(name):
            value = item.get(name)
            return float(value) if value is not None and math.isfinite(float(value)) else None
        rows.append((symbol, day.isoformat(), *map(float, values), optional('Adj Close'),
                     int(volume), optional('Dividends'), optional('Stock Splits'),
                     meta.get('currency'), meta.get('exchangeName'), meta.get('exchangeTimezoneName'),
                     datetime.now(timezone.utc).isoformat()))
    return rows, empty


def normalize_intraday(symbol, frame, meta, interval, now=None):
    """Validate intraday bars and store timestamps in UTC."""
    rows, empty = [], 0
    seen = set()
    now = now or datetime.now(timezone.utc)
    # Yahoo can return the currently-forming candle. Keep only closed candles.
    cutoff = now - timedelta(minutes=int(interval[:-1]))
    for stamp, item in frame.iterrows():
        instant = stamp.to_pydatetime()
        if instant.tzinfo is None:
            zone = meta.get('exchangeTimezoneName')
            if not zone:
                raise ValueError('Provider did not supply exchange timezone')
            instant = instant.replace(tzinfo=ZoneInfo(zone))
        instant = instant.astimezone(timezone.utc)
        if instant > cutoff:
            continue
        values = [item.get(name) for name in ('Open', 'High', 'Low', 'Close')]
        if all(value is None or math.isnan(float(value)) for value in values):
            empty += 1
            continue
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(f'{symbol}: partial OHLC at {instant.isoformat()}')
        volume = float(item['Volume'])
        if not math.isfinite(volume) or volume < 0 or not volume.is_integer():
            raise ValueError(f'{symbol}: invalid volume at {instant.isoformat()}')
        stamp_text = instant.isoformat()
        if stamp_text in seen:
            raise ValueError(f'{symbol}: duplicate timestamp {stamp_text}')
        seen.add(stamp_text)
        adjusted = item.get('Adj Close')
        adjusted = float(adjusted) if adjusted is not None and math.isfinite(float(adjusted)) else None
        rows.append((symbol, stamp_text, interval, *map(float, values), adjusted, int(volume),
                     meta.get('currency'), meta.get('exchangeName'), meta.get('exchangeTimezoneName'),
                     datetime.now(timezone.utc).isoformat()))
    return rows, empty


def persist_intraday(db, symbol, interval, rows, started, empty):
    if not rows:
        raise ValueError(f'{symbol}: no completed {interval} bars returned')
    with db:
        db.executemany('''INSERT INTO intraday_prices
          (symbol,timestamp,interval,open,high,low,close,adjusted_close,volume,currency,exchange,timezone,fetched_at)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,timestamp,interval) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
          adjusted_close=excluded.adjusted_close, volume=excluded.volume,
          currency=excluded.currency, exchange=excluded.exchange,
          timezone=excluded.timezone, fetched_at=excluded.fetched_at''', rows)
        db.execute('''INSERT INTO intraday_state VALUES (?,?,?,NULL)
          ON CONFLICT(symbol,interval) DO UPDATE SET last_success=excluded.last_success,error=NULL''',
                   (symbol, interval, started))
        db.execute('INSERT INTO attempts(symbol,started,mode,status,rows,empty_rows) VALUES (?,?,?,?,?,?)',
                   (symbol, started, f'intraday-{interval}', 'ok', len(rows), empty))


def record_api_request(db, provider, endpoint, started, status, rows=None, error=None, duration_ms=None):
    elapsed = duration_ms if duration_ms is not None else int((time.perf_counter() - started[1]) * 1000)
    with db:
        db.execute('''INSERT INTO api_requests(provider,endpoint,requested_at,status,rows,duration_ms,error)
          VALUES (?,?,?,?,?,?,?)''', (provider, endpoint, started[0], status, rows, elapsed, error))


def ingest_intraday(db, config, symbols, force=False):
    """Backfill available one-minute bars in small windows, then overlap updates."""
    import yfinance as yf
    interval = config.get('intraday_interval', '1m')
    lookback_days = config.get('intraday_lookback_days', 29)
    chunk_days = config.get('intraday_chunk_days', 7)
    overlap_minutes = config.get('intraday_overlap_minutes', 120)
    failed = []
    utc_now = datetime.now(timezone.utc)
    for number, symbol in enumerate(symbols, 1):
        started = datetime.now(timezone.utc).isoformat()
        latest = db.execute('SELECT MAX(timestamp) FROM intraday_prices WHERE symbol=? AND interval=?',
                            (symbol, interval)).fetchone()[0]
        if force or not latest:
            range_start = utc_now - timedelta(days=lookback_days)
        else:
            range_start = datetime.fromisoformat(latest) - timedelta(minutes=overlap_minutes)
        windows = []
        cursor = range_start
        while cursor < utc_now:
            end = min(cursor + timedelta(days=chunk_days), utc_now)
            windows.append((cursor, end))
            cursor = end
        for attempt in range(config['attempts']):
            try:
                ticker = yf.Ticker(symbol)
                meta = ticker.get_history_metadata()
                if not meta.get('exchangeTimezoneName'):
                    raise ValueError('Provider did not supply exchange timezone')
                all_rows, empty = [], 0
                for start, end in windows:
                    request_started = (datetime.now(timezone.utc).isoformat(), time.perf_counter())
                    try:
                        frame = ticker.history(start=start, end=end, interval=interval, auto_adjust=False,
                                               actions=False, keepna=True, raise_errors=True, timeout=30)
                        record_api_request(db, 'Yahoo Finance', 'chart/history', request_started, 'ok', len(frame))
                    except Exception as request_error:
                        record_api_request(db, 'Yahoo Finance', 'chart/history', request_started, 'failed', error=str(request_error))
                        raise
                    rows, skipped = normalize_intraday(symbol, frame, meta, interval, utc_now)
                    all_rows.extend(rows)
                    empty += skipped
                    if len(windows) > 1:
                        time.sleep(config['request_pause_seconds'])
                persist_intraday(db, symbol, interval, all_rows, started, empty)
                logging.info('[%s/%s] %s: %s %s rows, %s empty records',
                             number, len(symbols), symbol, len(all_rows), interval, empty)
                break
            except Exception as exc:
                if attempt + 1 < config['attempts']:
                    delay = 10 * 2 ** attempt
                    logging.warning('%s intraday attempt %s failed: %s; retry in %ss', symbol, attempt + 1, exc, delay)
                    time.sleep(delay)
                else:
                    failed.append(symbol)
                    with db:
                        db.execute('''INSERT INTO intraday_state(symbol,interval,error) VALUES (?,?,?)
                          ON CONFLICT(symbol,interval) DO UPDATE SET error=excluded.error''',
                                   (symbol, interval, str(exc)))
                        db.execute('INSERT INTO attempts(symbol,started,mode,status,error) VALUES (?,?,?,?,?)',
                                   (symbol, started, f'intraday-{interval}', 'failed', str(exc)))
                    logging.error('%s intraday failed: %s', symbol, exc)
        time.sleep(config['request_pause_seconds'])
    return failed


def fetch_json(url, provider='Unknown', events=None):
    event = {'provider': provider, 'endpoint': url.split('?', 1)[0],
             'requested_at': datetime.now(timezone.utc).isoformat(), 'started': time.perf_counter()}
    if events is not None:
        events.append(event)
    request = urllib.request.Request(url, headers={'User-Agent': 'MarketAtlas/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode())
        event.update(status='ok', duration_ms=int((time.perf_counter() - event['started']) * 1000))
        return value
    except Exception as exc:
        event.update(status='failed', duration_ms=int((time.perf_counter() - event['started']) * 1000), error=str(exc))
        raise


def crypto_source_rows(now=None, start=None, events=None):
    """Fetch recent keyless BTC/USD one-minute candles from major spot exchanges."""
    now = (now or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    start = start or (now - timedelta(days=2)).replace(hour=0, minute=0)
    output = {}
    # Coinbase caps candle responses at 300; use four-hour windows with overlap-free bounds.
    coinbase = []
    cursor = start
    while cursor < now:
        end = min(cursor + timedelta(hours=4), now)
        params = urllib.parse.urlencode({'granularity': 60, 'start': cursor.isoformat(), 'end': end.isoformat()})
        payload = fetch_json('https://api.exchange.coinbase.com/products/BTC-USD/candles?' + params, 'Coinbase', events)
        if events is not None: events[-1]['rows'] = len(payload)
        for item in payload:
            coinbase.append((item[0], item[3], item[2], item[1], item[4], item[5]))
        cursor = end
    output['Coinbase'] = coinbase
    kraken_data = fetch_json(f'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1&since={int(start.timestamp())}', 'Kraken', events)
    if kraken_data.get('error'):
        raise ValueError('Kraken: ' + ', '.join(kraken_data['error']))
    kraken_key = next(key for key in kraken_data['result'] if key != 'last')
    if events is not None: events[-1]['rows'] = len(kraken_data['result'][kraken_key])
    output['Kraken'] = [(r[0], r[1], r[2], r[3], r[4], r[6]) for r in kraken_data['result'][kraken_key]]
    bitstamp = []
    cursor = start
    while cursor < now:
        end = min(cursor + timedelta(hours=15), now)
        params = urllib.parse.urlencode({'step': 60, 'limit': 1000, 'start': int(cursor.timestamp()),
                                         'end': int(end.timestamp()), 'exclude_current_candle': 'true'})
        payload = fetch_json('https://www.bitstamp.net/api/v2/ohlc/btcusd/?' + params, 'Bitstamp', events)['data']['ohlc']
        if events is not None: events[-1]['rows'] = len(payload)
        bitstamp.extend(payload)
        cursor = end
    output['Bitstamp'] = [(r['timestamp'], r['open'], r['high'], r['low'], r['close'], r['volume']) for r in bitstamp]
    gemini = fetch_json('https://api.gemini.com/v2/candles/btcusd/1m', 'Gemini', events)
    if events is not None: events[-1]['rows'] = len(gemini)
    output['Gemini'] = [(int(r[0]) // 1000, r[1], r[2], r[3], r[4], r[5]) for r in gemini]
    fetched = datetime.now(timezone.utc).isoformat()
    normalized = {}
    for source, values in output.items():
        rows = []
        for stamp, open_, high, low, close, volume in values:
            instant = datetime.fromtimestamp(int(stamp), timezone.utc).replace(second=0, microsecond=0)
            if not start <= instant < now:
                continue
            numbers = [float(open_), float(high), float(low), float(close), float(volume)]
            if not all(math.isfinite(v) and v >= 0 for v in numbers):
                continue
            rows.append(('BTC-USD', instant.isoformat(), '1m', source, *numbers, fetched))
        normalized[source] = rows
    return normalized


def ingest_crypto_sources(db, force=False):
    failed = []
    events = []
    try:
        latest = db.execute("SELECT MAX(timestamp) FROM crypto_exchange_prices WHERE symbol='BTC-USD'").fetchone()[0]
        start = None if force or not latest else datetime.fromisoformat(latest) - timedelta(hours=2)
        sources = crypto_source_rows(start=start, events=events)
    except Exception as exc:
        logging.error('Crypto exchange collection failed: %s', exc)
        with db:
            db.executemany('''INSERT INTO api_requests(provider,endpoint,requested_at,status,rows,duration_ms,error)
              VALUES (:provider,:endpoint,:requested_at,:status,:rows,:duration_ms,:error)''',
                           [{**e, 'rows': e.get('rows'), 'error': e.get('error')} for e in events])
        return ['crypto-exchanges']
    with db:
        db.executemany('''INSERT INTO api_requests(provider,endpoint,requested_at,status,rows,duration_ms,error)
          VALUES (:provider,:endpoint,:requested_at,:status,:rows,:duration_ms,:error)''',
                       [{**e, 'rows': e.get('rows'), 'error': e.get('error')} for e in events])
        for source, rows in sources.items():
            if not rows:
                failed.append(source)
                continue
            db.executemany('''INSERT INTO crypto_exchange_prices
              (symbol,timestamp,interval,source,open,high,low,close,volume,fetched_at)
              VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,timestamp,interval,source) DO UPDATE SET
              open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,
              volume=excluded.volume,fetched_at=excluded.fetched_at''', rows)
            logging.info('BTC-USD: %s %s exchange rows', len(rows), source)
    return failed


def persist(db, symbol, rows, full, started, empty):
    if not rows:
        raise ValueError(f'{symbol}: no completed daily bars returned')
    with db:
        db.executemany('''INSERT INTO prices
          (symbol,date,open,high,low,close,adjusted_close,volume,dividends,splits,currency,exchange,timezone,fetched_at)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,date) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
          adjusted_close=excluded.adjusted_close, volume=excluded.volume, dividends=excluded.dividends,
          splits=excluded.splits, currency=excluded.currency, exchange=excluded.exchange,
          timezone=excluded.timezone, fetched_at=excluded.fetched_at''', rows)
        db.execute('''INSERT INTO state VALUES (?,?,?,NULL) ON CONFLICT(symbol) DO UPDATE SET
          last_success=excluded.last_success, last_full=COALESCE(excluded.last_full,state.last_full), error=NULL''',
                   (symbol, started, started if full else None))
        db.execute('INSERT INTO attempts(symbol,started,mode,status,rows,empty_rows) VALUES (?,?,?,?,?,?)',
                   (symbol, started, 'full' if full else 'incremental', 'ok', len(rows), empty))


def ingest(db, config, symbols, force=False, end=None):
    import yfinance as yf
    failed = []
    for number, symbol in enumerate(symbols, 1):
        started = datetime.now(timezone.utc).isoformat()
        state = db.execute('SELECT last_full,last_success,error FROM state WHERE symbol=?', (symbol,)).fetchone()
        full = force or not state or not state[0] or (datetime.now(timezone.utc) - datetime.fromisoformat(state[0])).days >= config['full_refresh_days']
        latest = db.execute('SELECT MAX(date) FROM prices WHERE symbol=?', (symbol,)).fetchone()[0]
        if not force and state and state[1] and not state[2] and datetime.fromisoformat(state[1]).date() == datetime.now(timezone.utc).date() and not full:
            continue
        for attempt in range(config['attempts']):
            try:
                ticker = yf.Ticker(symbol)
                kwargs = dict(interval='1d', auto_adjust=False, actions=True, keepna=True, raise_errors=True, timeout=30)
                if end is not None:
                    kwargs['end'] = end
                if full or not latest:
                    kwargs['period'] = 'max'
                else:
                    kwargs['start'] = (date.fromisoformat(latest) - timedelta(days=config['overlap_days'])).isoformat()
                frame = ticker.history(**kwargs)
                meta = ticker.get_history_metadata()
                zone = meta.get('exchangeTimezoneName')
                if not zone:
                    raise ValueError('Provider did not supply exchange timezone')
                rows, empty = normalize(symbol, frame, meta, datetime.now(ZoneInfo(zone)).date())
                # New corporate actions can revise the entire adjusted history.
                if not full and any(row[8] or row[9] for row in rows):
                    full = True
                    frame = ticker.history(period='max', **{k: v for k, v in kwargs.items() if k not in ('start', 'period')})
                    rows, empty = normalize(symbol, frame, meta, datetime.now(ZoneInfo(zone)).date())
                persist(db, symbol, rows, full, started, empty)
                logging.info('[%s/%s] %s: %s rows, %s empty records, %s', number, len(symbols), symbol, len(rows), empty, 'full' if full else 'incremental')
                break
            except Exception as exc:
                if attempt + 1 < config['attempts']:
                    delay = 10 * 2 ** attempt
                    logging.warning('%s attempt %s failed: %s; retry in %ss', symbol, attempt + 1, exc, delay)
                    time.sleep(delay)
                else:
                    failed.append(symbol)
                    with db:
                        db.execute('INSERT INTO state(symbol,error) VALUES (?,?) ON CONFLICT(symbol) DO UPDATE SET error=excluded.error', (symbol, str(exc)))
                        db.execute('INSERT INTO attempts(symbol,started,mode,status,error) VALUES (?,?,?,?,?)', (symbol, started, 'full' if full else 'incremental', 'failed', str(exc)))
                    logging.error('%s failed: %s', symbol, exc)
        time.sleep(config['request_pause_seconds'])
    return failed


def export(db):
    folder = DATA / 'exports'
    folder.mkdir(exist_ok=True)
    for (symbol,) in db.execute('SELECT DISTINCT symbol FROM prices'):
        # Quote punctuation for Windows-safe filenames.
        from urllib.parse import quote
        path = folder / (quote(symbol, safe='-._') + '.csv')
        temporary = path.with_suffix('.tmp')
        cursor = db.execute('SELECT * FROM prices WHERE symbol=? ORDER BY date', (symbol,))
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([column[0] for column in cursor.description])
            writer.writerows(cursor)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sync', 'intraday', 'status', 'export'])
    parser.add_argument('--symbols', nargs='+', help='Override configured universe for this run')
    parser.add_argument('--full', action='store_true', help='Force complete history refresh')
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), RotatingFileHandler(DATA / 'ingestion.log', maxBytes=10_000_000, backupCount=5, encoding='utf-8')])
    config = json.loads((ROOT / 'config.json').read_text())
    if args.command == 'intraday':
        # Keep existing scheduled-task entrypoints, but stop routine minute pulls.
        from refresh_hourly import refresh
        return refresh(symbols=args.symbols, full=args.full)
    with connect(DATA / 'market.sqlite') as db:
        if args.command == 'status':
            print('symbols, rows, earliest, latest:', db.execute('SELECT COUNT(DISTINCT symbol),COUNT(*),MIN(date),MAX(date) FROM prices').fetchone())
            print('intraday symbols, rows, earliest, latest:', db.execute('SELECT COUNT(DISTINCT symbol),COUNT(*),MIN(timestamp),MAX(timestamp) FROM intraday_prices').fetchone())
            print('Integrity:', db.execute('PRAGMA quick_check').fetchone()[0])
            print('Failed symbols:', db.execute('SELECT symbol,error FROM state WHERE error IS NOT NULL').fetchall())
            print('Failed intraday symbols:', db.execute('SELECT symbol,error FROM intraday_state WHERE error IS NOT NULL').fetchall())
            return 0
        with process_lock(DATA / 'ingestion.lock'):
            if args.command == 'export':
                export(db)
                return 0
            symbols = sorted(set(args.symbols)) if args.symbols else universe(db, config)
            logging.info('Starting sync for %s symbols', len(symbols))
            failed = ingest(db, config, symbols, args.full)
            export(db)
            logging.info('Sync finished: %s failed symbols: %s', len(failed), failed)
            if (ROOT / 'deployment.json').exists():
                from publish_site import publish
                publish()
            return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
