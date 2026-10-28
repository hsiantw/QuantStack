"""Budgeted one-minute pulls for every symbol in the local Stocks/Crypto catalog."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import time

from market_data import ROOT, DATA, connect, process_lock, persist_intraday, record_api_request
from refresh_hourly import fetch_bars, rate_limited


def jobs(db, config):
    symbols = set(config.get('symbols', []))
    symbols.update(row[0] for row in db.execute('SELECT symbol FROM state UNION SELECT symbol FROM intraday_state'))
    # BTC has its own full-history Bitstamp collector.
    symbols.discard('BTC-USD')
    attempts = dict(db.execute("SELECT symbol,MAX(started) FROM attempts WHERE mode='catalog-1m' GROUP BY symbol"))
    return sorted(symbols, key=lambda symbol: (attempts.get(symbol, ''), symbol))


def range_start(latest, now):
    earliest = now - timedelta(days=7)
    return max(earliest, datetime.fromisoformat(latest) - timedelta(minutes=2)) if latest else earliest


def run(budget_seconds=45):
    from local_scheduler import lower_priority
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    lower_priority({'background_usage': 'low'})
    deadline = time.monotonic() + budget_seconds
    report = dict(status='running', interval='1m', succeeded=0, completed=0, rows=0, failed={})
    path = DATA / 'minute-refresh.json'

    def save():
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(path)

    with closing(connect(DATA / 'market.sqlite')) as db:
        pending = jobs(db, config)
        report['total'] = len(pending)
        report['btc_collector'] = 'MarketData-BTCMinuteSync'
        save()
        while report['completed'] < len(pending):
            if time.monotonic() >= deadline:
                break
            symbol = pending[report['completed']]
            try:
                # Release between symbols so the BTC collector can also run.
                with process_lock(DATA / 'ingestion.lock'):
                    report['status'] = 'running'
                    now = datetime.now(timezone.utc)
                    started = now.isoformat()
                    latest = db.execute("SELECT MAX(timestamp) FROM intraday_prices WHERE symbol=? AND interval='1m'", (symbol,)).fetchone()[0]
                    rows, empty, events, error = fetch_bars(symbol, range_start(latest, now), now,
                        attempts=1, pause=2, interval='1m')
                    for request, state, count, message, duration in events:
                        record_api_request(db, 'Yahoo Finance', 'chart/history/1m', request, state, count, message, duration_ms=duration)
                    if not error:
                        persist_intraday(db, symbol, '1m', rows, started, empty)
                        report['succeeded'] += 1
                        report['rows'] += len(rows)
                    else:
                        report['failed'][symbol] = error
                        with db:
                            db.execute("INSERT INTO intraday_state(symbol,interval,error) VALUES (?,'1m',?) ON CONFLICT(symbol,interval) DO UPDATE SET error=excluded.error", (symbol, error))
                    with db:
                        db.execute('INSERT INTO attempts(symbol,started,mode,status,error) VALUES (?,?,?,?,?)',
                            (symbol, started, 'catalog-1m', 'failed' if error else 'ok', error))
                    report['completed'] += 1
                    report['last_symbol'] = symbol
                    save()
                    if error and rate_limited(error):
                        report['status'] = 'rate_limited'
                        break
            except RuntimeError as exc:
                if str(exc) != 'Another ingestion process is already running':
                    raise
                report['status'] = 'waiting_for_collector'
                time.sleep(1)
        if report['status'] == 'running':
            report['status'] = 'complete' if report['completed'] == len(pending) else 'budget_exhausted'
        report['pending'] = len(pending) - report['completed']
        save()


if __name__ == '__main__':
    run()
