"""Durable, local priority pulls. Consumers must hold ingestion.lock."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading
import time

from market_data import ROOT

INTERVALS = {'1m': 7, '5m': 59, '15m': 59, '60m': 729, '1d': 36500}
_worker = None
_guard = threading.Lock()


def connection(root=ROOT):
    (root / 'data').mkdir(exist_ok=True)
    db = sqlite3.connect(root / 'data' / 'pull-queue.sqlite', timeout=15)
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY, symbol TEXT, interval TEXT, start TEXT, end TEXT,
        status TEXT DEFAULT 'queued', created TEXT, rows INTEGER, error TEXT)''')
    return db


def validate(payload, now=None):
    from local_scheduler import normalize_symbols
    if not isinstance(payload, dict):
        raise ValueError('Expected a request object.')
    symbols = normalize_symbols(payload.get('symbol', ''), payload.get('kind', 'stocks'))
    if len(symbols) != 1:
        raise ValueError('Choose one stock or crypto symbol.')
    interval = payload.get('interval', '60m')
    if not isinstance(interval, str) or interval not in INTERVALS:
        raise ValueError('Choose 1m, 5m, 15m, 60m or 1d bars.')
    now = now or datetime.now(timezone.utc)
    if payload.get('range', 'relative') == 'custom':
        try:
            start = datetime.strptime(payload['start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end = min(datetime.strptime(payload['end'], '%Y-%m-%d').replace(tzinfo=timezone.utc) + timedelta(days=1), now)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Enter valid start and end dates.') from exc
    elif payload.get('range', 'relative') == 'relative':
        days = payload.get('days', 7)
        if type(days) is not int or not 1 <= days <= INTERVALS[interval]:
            raise ValueError(f'{interval} bars support 1–{INTERVALS[interval]} days per request.')
        start, end = now - timedelta(days=days), now
    else:
        raise ValueError('Choose a relative or custom date range.')
    if start >= end or start < now - timedelta(days=INTERVALS[interval]):
        raise ValueError(f'Choose an ordered range within the last {INTERVALS[interval]} days for {interval} bars.')
    return symbols[0], interval, start.isoformat(), end.isoformat()


def enqueue(payload, root=ROOT):
    values = validate(payload)
    with closing(connection(root)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        existing = db.execute("SELECT id FROM requests WHERE symbol=? AND interval=? AND start=? AND end=? AND status IN ('queued','running')", values).fetchone()
        if existing:
            return dict(id=existing['id'], status='queued')
        if db.execute("SELECT COUNT(*) FROM requests WHERE status IN ('queued','running')").fetchone()[0] >= 100:
            raise ValueError('Queue is full. Wait for current requests to finish.')
        item = db.execute('INSERT INTO requests(symbol,interval,start,end,created) VALUES (?,?,?,?,?)', (*values, datetime.now(timezone.utc).isoformat()))
        return dict(id=item.lastrowid, status='queued')


def status(root=ROOT):
    with closing(connection(root)) as db:
        return {'requests': [dict(row) for row in db.execute("SELECT * FROM requests ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,id DESC LIMIT 100")]}


def drain(db, config, root=ROOT):
    """Run FIFO priority requests before the next regular job; recover interrupted work."""
    from refresh_hourly import fetch_bars
    from market_data import persist, persist_intraday, record_api_request
    with closing(connection(root)) as queue:
        with queue:
            queue.execute("UPDATE requests SET status='queued' WHERE status='running'")
        while True:
            with queue:
                job = queue.execute("SELECT * FROM requests WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
                if job is None:
                    return
                queue.execute("UPDATE requests SET status='running',error=NULL WHERE id=?", (job['id'],))
            try:
                rows, empty, events, error = fetch_bars(job['symbol'], datetime.fromisoformat(job['start']), datetime.fromisoformat(job['end']),
                    attempts=config.get('attempts', 3), pause=config.get('request_pause_seconds', 2), interval=job['interval'])
                for request, state, count, message, duration in events:
                    record_api_request(db, 'Yahoo Finance', 'chart/history/' + job['interval'], request, state, count, message, duration_ms=duration)
                if error:
                    raise ValueError(error)
                started = datetime.now(timezone.utc).isoformat()
                if job['interval'] == '1d':
                    previous = db.execute('SELECT last_success FROM state WHERE symbol=?', (job['symbol'],)).fetchone()
                    persist(db, job['symbol'], rows, False, started, empty)
                    with db:
                        db.execute('UPDATE state SET last_success=? WHERE symbol=?', (previous[0] if previous else None, job['symbol']))
                else:
                    previous = db.execute('SELECT last_success FROM intraday_state WHERE symbol=? AND interval=?', (job['symbol'], job['interval'])).fetchone()
                    persist_intraday(db, job['symbol'], job['interval'], rows, started, empty)
                    # A short priority range must not mark the full scheduled backfill fresh.
                    with db:
                        db.execute('UPDATE intraday_state SET last_success=? WHERE symbol=? AND interval=?', (previous[0] if previous else None, job['symbol'], job['interval']))
                with queue:
                    queue.execute("UPDATE requests SET status='complete',rows=? WHERE id=?", (len(rows), job['id']))
            except Exception as exc:
                with queue:
                    queue.execute("UPDATE requests SET status='failed',error=? WHERE id=?", (str(exc), job['id']))


def start_worker(root=ROOT):
    global _worker
    def run():
        from market_data import connect, process_lock
        while True:
            try:
                with closing(connection(root)) as queue:
                    pending = queue.execute("SELECT 1 FROM requests WHERE status IN ('queued','running') LIMIT 1").fetchone()
                if pending:
                    with process_lock(root / 'data' / 'ingestion.lock'), closing(connect(root / 'data' / 'market.sqlite')) as db:
                        drain(db, json.loads((root / 'config.json').read_text(encoding='utf-8')), root)
            except (OSError, RuntimeError, sqlite3.Error):
                pass  # A scheduled collector may own the lock; it also drains this queue.
            time.sleep(2)
    with _guard:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=run, daemon=True, name='priority-pulls')
            _worker.start()
