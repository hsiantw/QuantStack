"""Refresh daily and hourly charts in liquidity/market-cap/watchlist order."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
import json
import logging
import threading
import time
from zoneinfo import ZoneInfo

from market_data import (DATA, ROOT, connect, normalize, persist, persist_intraday,
                         process_lock, universe)
from refresh_hourly import fetch_hourly, rate_limited


def ranked_symbols(db, config, symbols, live=True):
    """Interleave leaders so neither stocks nor crypto wait for the other group."""
    import yfinance as yf
    from expand_stocks import metadata_schema
    metadata_schema(db)
    available = set(symbols)
    cap, volume, crypto = [], [], []
    if live:
        from expand_stocks import write_metadata
        query = yf.EquityQuery('and', [yf.EquityQuery('eq', ['region', 'us']),
                                    yf.EquityQuery('gt', ['intradaymarketcap', 0])])
        for field, target in [('intradaymarketcap', cap), ('dayvolume', volume)]:
            try:
                result = yf.screen(query, size=100, sortField=field, sortAsc=False)
                for quote in result.get('quotes', []):
                    symbol = quote.get('symbol')
                    if symbol in available:
                        target.append(symbol)
                        write_metadata(db, symbol, quote)
            except Exception as exc:
                logging.warning('Live ranking %s unavailable: %s', field, exc)
    if not cap:
        cap = [r[0] for r in db.execute("SELECT symbol FROM stock_metadata WHERE currency='USD' ORDER BY market_cap DESC") if r[0] in available][:100]
    if not volume:
        # USD instruments only: raw prices/market caps in different currencies
        # must not be compared as though they were the same currency.
        volume = [r[0] for r in db.execute("""SELECT p.symbol FROM prices p
          WHERE p.date=(SELECT MAX(q.date) FROM prices q WHERE q.symbol=p.symbol)
          AND p.currency='USD' ORDER BY p.volume DESC LIMIT 100""") if r[0] in available]
    popular = [s for s in ['NVDA','AAPL','MSFT','AMZN','GOOGL','META','TSLA','AMD','AVGO','TSM',
                          'PLTR','COIN','MSTR','2330.TW','SPY','QQQ'] if s in available]
    crypto = [s for s in config['symbols'] if s.endswith('-USD') and s in available]
    preferred = []
    groups = [popular, cap[:50], volume[:50], crypto[:30], config['symbols'][:45]]
    for index in range(max(map(len, groups))):
        for group in groups:
            if index < len(group):
                preferred.append(group[index])
    preferred = list(dict.fromkeys(preferred))
    (DATA / 'refresh-priority.json').write_text(json.dumps({
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'basis': 'Live Yahoo US market cap and share volume when available; cached fallback; configured popular stocks and crypto',
        'symbols': preferred}, indent=2), encoding='utf-8')
    return list(dict.fromkeys(preferred + list(symbols)))


def fetch_daily(symbol, latest, last_full, config):
    import yfinance as yf
    now = datetime.now(timezone.utc)
    full = not latest or not last_full or (now - datetime.fromisoformat(last_full)).days >= config['full_refresh_days']
    ticker = yf.Ticker(symbol)
    kwargs = dict(interval='1d', auto_adjust=False, actions=True, keepna=True, raise_errors=True, timeout=25)
    if full:
        kwargs['period'] = 'max'
    else:
        kwargs['start'] = (date.fromisoformat(latest) - timedelta(days=config['overlap_days'])).isoformat()
    frame = ticker.history(**kwargs)
    meta = ticker.get_history_metadata()
    zone = meta.get('exchangeTimezoneName')
    if not zone:
        raise ValueError('Provider did not supply exchange timezone')
    rows, empty = normalize(symbol, frame, meta, datetime.now(ZoneInfo(zone)).date())
    if not full and any(row[8] or row[9] for row in rows):
        full = True
        kwargs.pop('start', None)
        frame = ticker.history(period='max', **kwargs)
        rows, empty = normalize(symbol, frame, meta, datetime.now(ZoneInfo(zone)).date())
    if not rows:
        raise ValueError('No completed daily bars returned')
    return rows, empty, full


def refresh(workers=6, hourly=True, resume=False):
    import yfinance as yf
    DATA.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(DATA / 'provider-cache'))
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    from local_scheduler import lower_priority
    lower_priority(config)
    report_path = DATA / 'all-refresh.json'
    previous = json.loads(report_path.read_text()) if resume and report_path.exists() else {}
    today = datetime.now(timezone.utc).date().isoformat()
    same_run = previous.get('run_date') == today and previous.get('hourly') == hourly
    completed = set(previous.get('successful_symbols', [])) if same_run else set()
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), status='running',
                  run_date=today, hourly=hourly,
                  successful_symbols=list(completed), failed={}, completed=0, total=0, daily_rows=0, hourly_rows=0)
    stop = threading.Event()
    with process_lock(DATA / 'ingestion.lock'), connect(DATA / 'market.sqlite') as db:
        symbols = ranked_symbols(db, config, universe(db, config))
        jobs = []
        now = datetime.now(timezone.utc)
        for symbol in symbols:
            if symbol in completed:
                continue
            latest = db.execute('SELECT MAX(date) FROM prices WHERE symbol=?', (symbol,)).fetchone()[0]
            state = db.execute('SELECT last_full FROM state WHERE symbol=?', (symbol,)).fetchone()
            hourly_latest = db.execute("SELECT MAX(timestamp) FROM intraday_prices WHERE symbol=? AND interval='60m'", (symbol,)).fetchone()[0]
            start = max(now-timedelta(days=config.get('hourly_lookback_days', 365)),
                        datetime.fromisoformat(hourly_latest)-timedelta(days=7)) if hourly_latest else now-timedelta(days=365)
            jobs.append((symbol, latest, state[0] if state else None, start))
        report['total'] = len(symbols)
        report['completed'] = len(completed)
        def save():
            report['updated_at'] = datetime.now(timezone.utc).isoformat()
            report['pending'] = report['total'] - report['completed']
            temp = report_path.with_suffix('.tmp')
            temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
            temp.replace(report_path)
        def fetch(job):
            symbol, latest, last_full, start = job
            daily_result = hourly_result = None
            errors = {}
            for attempt in range(2):
                try:
                    daily_result = fetch_daily(symbol, latest, last_full, config)
                    break
                except Exception as exc:
                    if rate_limited(str(exc)):
                        stop.set()
                    if attempt == 0 and not stop.is_set():
                        time.sleep(2)
                    else:
                        errors['daily'] = str(exc)
            if hourly and not stop.is_set():
                hourly_result = fetch_hourly(symbol, start, now, 2, 0.5, stop)
                if hourly_result[3]:
                    errors['hourly'] = hourly_result[3]
            elif hourly:
                errors['hourly'] = 'Deferred after provider rate limit'
            time.sleep(0.3)
            return daily_result, hourly_result, errors
        queue = iter(jobs)
        save()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            def submit():
                job = next(queue, None) if not stop.is_set() else None
                if job:
                    futures[pool.submit(fetch, job)] = job[0]
            for _ in range(workers):
                submit()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    symbol = futures.pop(future)
                    daily_result, hourly_result, errors = future.result()
                    stamp = datetime.now(timezone.utc).isoformat()
                    if daily_result:
                        rows, empty, full = daily_result
                        persist(db, symbol, rows, full, stamp, empty)
                        report['daily_rows'] += len(rows)
                    if hourly_result and not hourly_result[3] and hourly_result[0]:
                        persist_intraday(db, symbol, '60m', hourly_result[0], stamp, hourly_result[1])
                        report['hourly_rows'] += len(hourly_result[0])
                    if errors:
                        report['failed'][symbol] = errors
                    else:
                        report['successful_symbols'].append(symbol)
                    report['completed'] += 1
                    report['last_symbol'] = symbol
                    save()
                    if report['completed'] % 25 == 0 or errors:
                        logging.info('%s/%s %s %s', report['completed'], report['total'], symbol, errors or 'updated')
                    submit()
        report['status'] = 'rate_limited' if stop.is_set() else ('complete_with_errors' if report['failed'] else 'complete')
        save()
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=6)
    parser.add_argument('--daily-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Retry unfinished symbols from this catch-up run')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    raise SystemExit(refresh(args.workers, not args.daily_only, args.resume))
