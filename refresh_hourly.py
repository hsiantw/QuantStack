"""Collect native hourly candles and measure the capacity of an hourly update cycle."""
import argparse
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import threading
import time

from market_data import ROOT, DATA, connect, normalize_intraday, persist_intraday, process_lock, record_api_request


def rate_limited(error):
    message = str(error).lower()
    return any(part in message for part in ('rate limit', 'too many requests', '429'))


def fetch_bars(symbol, start, end, attempts=3, pause=0.5, stop=None, interval="60m"):
    from market_data import normalize
    from zoneinfo import ZoneInfo
    import yfinance as yf
    events = []
    for attempt in range(attempts):
        if stop is not None and stop.is_set():
            return [], 0, events, 'Deferred after provider rate limit'
        request = (datetime.now(timezone.utc).isoformat(), time.perf_counter())
        try:
            ticker = yf.Ticker(symbol)
            frame = ticker.history(start=start, end=end, interval=interval, auto_adjust=False,
                                   actions=interval == '1d', keepna=True, raise_errors=True, timeout=30)
            meta = ticker.get_history_metadata()
            if interval == '1d':
                rows, empty = normalize(symbol, frame, meta, datetime.now(ZoneInfo(meta['exchangeTimezoneName'])).date())
            else:
                rows, empty = normalize_intraday(symbol, frame, meta, interval, datetime.now(timezone.utc))
            if not rows:
                raise ValueError('Provider returned no completed candles in the requested range')
            events.append((request, 'ok', len(rows), None, int((time.perf_counter()-request[1])*1000)))
            return rows, empty, events, None
        except Exception as error:
            events.append((request, 'failed', None, str(error), int((time.perf_counter()-request[1])*1000)))
            if rate_limited(error):
                if stop is not None:
                    stop.set()
                return [], 0, events, str(error)
            if attempt + 1 == attempts:
                return [], 0, events, str(error)
            time.sleep(10 * 2**attempt)
        finally:
            time.sleep(pause)


def fetch_hourly(symbol, start, end, attempts=3, pause=0.5, stop=None):
    return fetch_bars(symbol, start, end, attempts, pause, stop)


def tracked_symbols(db, config):
    symbols = set(config.get('symbols', []))
    symbols.update(row[0] for row in db.execute('SELECT symbol FROM state'))
    symbols.update(row[0] for row in db.execute("SELECT symbol FROM intraday_state WHERE interval='60m'"))
    path = DATA / 'stock-universe.json'
    if path.exists():
        symbols.update(json.loads(path.read_text(encoding='utf-8'))['symbols'][:config.get('stock_universe_limit', 5000)])
    return sorted(symbols)


def capacity(progress, elapsed, cycle_minutes, budget_minutes):
    """End-to-end successful throughput, including retries, pacing and database writes."""
    rate = progress['succeeded'] / elapsed if elapsed > 0 else 0
    attempted = progress['completed'] - progress['skipped']
    measured = attempted / elapsed if elapsed > 0 else 0
    limit = math.floor(rate * cycle_minutes * 60)
    planning = math.floor(rate * budget_minutes * 60 * 0.8)
    return dict(elapsed_seconds=round(elapsed, 3), symbols_per_minute=round(rate*60, 2),
                attempts=progress['requests'], request_failures=progress['request_failures'],
                rate_limit_events=progress['rate_limit_events'],
                cycle_minutes=cycle_minutes, budget_minutes=budget_minutes,
                estimated_symbols_per_cycle=limit,
                first_symbol_count_over_cycle=limit+1 if rate else None,
                planning_symbols_per_cycle=planning,
                estimated_total_refresh_minutes=round(progress['total']/measured/60, 2) if measured else None,
                exceeds_planning_capacity=progress['total'] > planning if rate else None,
                basis='Successful symbols / total wall time; includes retries, pacing and storage. '
                      'Planning uses 80% of the run budget. Extrapolation, not a provider quota. '
                      'History calls can include extra internal HTTP requests. Shared daily jobs can delay starts.')


def refresh(symbols=None, days=None, workers=None, resume=False, full=False, profile=None, budget_minutes=None):
    import yfinance as yf
    DATA.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(DATA / 'provider-cache'))
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    if profile is not None:
        from local_scheduler import resource_profile
        config.update(resource_profile(profile))
    if budget_minutes is not None:
        config['hourly_budget_minutes'] = budget_minutes
    from local_scheduler import lower_priority
    lower_priority(config)
    days = days if days is not None else config.get('hourly_lookback_days', 365)
    workers = workers if workers is not None else config.get('hourly_workers', 6)
    pause = config.get('hourly_pause_seconds', 0.5)
    cycle = config.get('hourly_cycle_minutes', 60)
    budget = config.get('hourly_budget_minutes', 55)
    if not 1 <= days <= 729 or not 1 <= workers <= 8 or pause < 0 or not 0 < budget <= cycle:
        raise ValueError('Invalid hourly lookback, workers, pacing or cycle budget')
    now = datetime.now(timezone.utc)
    started = now.isoformat()
    clock_start = time.perf_counter()
    progress_path = DATA / 'hourly-refresh.json'
    stop = threading.Event()
    with process_lock(DATA / 'ingestion.lock'), closing(connect(DATA / 'market.sqlite')) as db:
        from pull_queue import drain
        drain(db, config, ROOT)
        symbols = sorted(set(symbols)) if symbols is not None else tracked_symbols(db, config)
        progress = dict(started_at=started, status='running', total=len(symbols), completed=0,
                        succeeded=0, skipped=0, failed={}, rows=0, days=days, workers=workers,
                        requests=0, request_failures=0, rate_limit_events=0, backfills=0,
                        refreshes=0, new_symbols=0)
        def report():
            progress['updated_at'] = datetime.now(timezone.utc).isoformat()
            progress['pending'] = progress['total'] - progress['completed']
            progress['capacity'] = capacity(progress, time.perf_counter()-clock_start, cycle, budget)
            temporary = progress_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(progress, indent=2), encoding='utf-8')
            temporary.replace(progress_path)
        jobs = []
        last_attempts = dict(db.execute("SELECT symbol,MAX(started) FROM attempts WHERE mode='intraday-60m' GROUP BY symbol"))
        for symbol in symbols:
            state = db.execute("SELECT last_success,error FROM intraday_state WHERE symbol=? AND interval='60m'", (symbol,)).fetchone()
            if resume and not full and state and not state[1] and state[0] and datetime.fromisoformat(state[0]) >= now - timedelta(minutes=cycle):
                progress['skipped'] += 1
                progress['completed'] += 1
                continue
            latest = db.execute("SELECT MAX(timestamp) FROM intraday_prices WHERE symbol=? AND interval='60m'", (symbol,)).fetchone()[0]
            earliest = now - timedelta(days=days)
            start = max(earliest, datetime.fromisoformat(latest) - timedelta(days=7)) if latest and not full else earliest
            # Rotate failed tickers too, so a small budget cannot retry the same
            # unavailable symbols forever and starve the rest of the universe.
            jobs.append((last_attempts.get(symbol) or (state[0] if state and state[0] else ''), symbol, start, not latest))
        jobs.sort()
        queue = iter(jobs)
        report()
        def submit(pool, futures):
            drain(db, config, ROOT)
            if stop.is_set() or time.perf_counter()-clock_start >= budget*60:
                return
            job = next(queue, None)
            if job is not None:
                _, symbol, start, is_new = job
                futures[pool.submit(fetch_hourly, symbol, start, now, config.get('attempts', 3), pause, stop)] = (symbol, is_new)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for _ in range(workers):
                submit(pool, futures)
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    symbol, is_new = futures.pop(future)
                    rows, empty, events, error = future.result()
                    if not events:
                        continue  # Never attempted; keep pending for the next cycle.
                    for request, status, count, message, duration in events:
                        record_api_request(db, 'Yahoo Finance', 'chart/history/60m', request, status, count, message, duration_ms=duration)
                        progress['requests'] += 1
                        progress['request_failures'] += status == 'failed'
                        progress['rate_limit_events'] += bool(message and rate_limited(message))
                    if error:
                        with db:
                            db.execute('''INSERT INTO intraday_state(symbol,interval,error) VALUES (?,'60m',?)
                              ON CONFLICT(symbol,interval) DO UPDATE SET error=excluded.error''', (symbol, error))
                            db.execute("INSERT INTO attempts(symbol,started,mode,status,error) VALUES (?,?,'intraday-60m','failed',?)", (symbol, started, error))
                        progress['failed'][symbol] = error
                    else:
                        persist_intraday(db, symbol, '60m', rows, datetime.now(timezone.utc).isoformat(), empty)
                        progress['succeeded'] += 1
                        progress['rows'] += len(rows)
                        progress['new_symbols'] += is_new
                        progress['backfills' if is_new or full else 'refreshes'] += 1
                    progress['completed'] += 1
                    progress['last_symbol'] = symbol
                    report()
                    logging.info('[%s/%s] %s: %s', progress['completed'], progress['total'], symbol, error or f'{len(rows)} hourly bars')
                    submit(pool, futures)
        if stop.is_set():
            progress['status'] = 'rate_limited'
        elif progress['completed'] < progress['total']:
            progress['status'] = 'budget_exhausted'
        else:
            progress['status'] = 'complete_with_errors' if progress['failed'] else 'complete'
        report()
        # Preserve measurements even when a later scheduled run replaces the progress file.
        with (DATA / 'hourly-runs.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(progress) + '\n')
        logging.info('Hourly refresh finished: %s; %s', progress['status'], json.dumps(progress['capacity']))
        return 0 if progress['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', help='Limit this run to selected symbols; default is the expanded universe')
    parser.add_argument('--days', type=int, choices=range(1, 730), metavar='1-729')
    parser.add_argument('--workers', type=int, choices=range(1, 9), metavar='1-8')
    parser.add_argument('--resume', action='store_true', help='Skip successful collections from the last update cycle')
    parser.add_argument('--full', action='store_true', help='Refetch the complete configured hourly lookback')
    parser.add_argument('--profile', choices=['high', 'low'], help='Resource settings for this run only')
    parser.add_argument('--budget-minutes', type=float, help='Override the collection time budget')
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(DATA / 'hourly-refresh.log', encoding='utf-8')])
    raise SystemExit(refresh(args.symbols, args.days, args.workers, args.resume, args.full,
                             args.profile, args.budget_minutes))
