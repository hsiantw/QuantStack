"""Local-only, bounded research jobs executed outside the chart server environment."""
from datetime import date
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
SOURCES = ('Yahoo Finance', 'Bitstamp', 'Kraken', 'Coinbase')
KINDS = ('strategy', 'portfolio', 'factor', 'exchange')


def validate(payload):
    if not isinstance(payload, dict) or payload.get('kind') not in KINDS:
        raise ValueError('Choose a supported research tool.')
    p = {key: payload.get(key) for key in ('kind', 'start', 'end')}
    try:
        start, end = date.fromisoformat(p['start']), date.fromisoformat(p['end'])
    except (ValueError, TypeError):
        raise ValueError('Choose valid start and end dates.') from None
    if start > end or (end - start).days > 3660:
        raise ValueError('Choose an ordered date range of at most ten years.')
    p['source'] = payload.get('source', 'Yahoo Finance')
    p['interval'] = payload.get('interval', '1d')
    if p['source'] not in SOURCES or p['interval'] not in ('1m', '5m', '15m', '60m', '1d'):
        raise ValueError('Choose a supported source and interval.')
    symbols = payload.get('symbols')
    if not isinstance(symbols, list) or not 1 <= len(symbols) <= 20:
        raise ValueError('Enter between one and twenty symbols.')
    if any(not isinstance(s, str) or not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]{0,29}', s) for s in symbols):
        raise ValueError('Use uppercase provider symbols, such as AAPL or BTC-USD.')
    if len(set(symbols)) != len(symbols):
        raise ValueError('Remove duplicate symbols.')
    p['symbols'] = symbols
    if p['kind'] in ('strategy', 'exchange') and len(symbols) != 1:
        raise ValueError('Choose one symbol for this tool.')
    if p['kind'] == 'factor' and len(symbols) < 5:
        raise ValueError('Factor research requires at least five symbols.')
    if p['kind'] in ('factor', 'portfolio'):
        if p['interval'] != '1d':
            raise ValueError('Portfolio and factor research use daily bars.')
        if len({s.endswith('-USD') for s in symbols}) > 1:
            raise ValueError('Use either stocks or crypto in one report to keep trading calendars consistent.')
    def number(key, default, low, high, integer=False):
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
            raise ValueError(f'{key} must be between {low} and {high}.')
        p[key] = int(value) if integer else value
    if p['kind'] == 'strategy':
        number('fast', 10, 2, 499, True)
        number('slow', 30, 3, 500, True)
        number('capital', 10000, 100, 100000000)
        number('fee_bps', 10, 0, 500)
        number('slippage_bps', 5, 0, 500)
        if p['fast'] >= p['slow']:
            raise ValueError('Fast moving average must be shorter than slow.')
    if p['kind'] == 'portfolio':
        weights = payload.get('weights')
        if not isinstance(weights, list) or len(weights) != len(symbols) or any(isinstance(w, bool) or not isinstance(w, (float, int)) or not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 100, abs_tol=1e-6):
            raise ValueError('Enter a positive weight for each holding, totaling 100%.')
        p['weights'] = weights
    if p['kind'] == 'factor':
        number('lookback', 20, 2, 252, True)
        number('horizon', 5, 1, 60, True)
    if p['kind'] == 'exchange':
        if p['source'] == 'Yahoo Finance' or not re.fullmatch(r'[A-Z0-9]+-USD', symbols[0]):
            raise ValueError('Choose an exchange and a USD crypto pair, such as ETH-USD.')
        if (end - start).days > 30:
            raise ValueError('Exchange pulls are limited to 31 calendar days per job.')
    return p


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


class Jobs:
    def __init__(self, root=ROOT, database=None):
        self.root = Path(root)
        self.directory = self.root / 'data' / 'analysis' / 'jobs'
        self.python = self.root / '.venv-analysis' / 'Scripts' / 'python.exe'
        self.database = Path(database) if database is not None else self.root / 'data' / 'market.sqlite'
        self.lock = threading.RLock()
        self.thread = None
        self.records = {}
        if self.directory.exists():
            for path in self.directory.glob('*/status.json'):
                try:
                    item = json.loads(path.read_text(encoding='utf-8'))
                    if item['status'] in ('queued', 'running'):
                        item.update(status='failed', message='Server restarted; submit this job again.')
                        write_json(path, item)
                    self.records[item['id']] = item
                except (OSError, ValueError, KeyError):
                    continue

    def available(self):
        return {'available': self.python.exists(), 'kinds': list(KINDS), 'sources': list(SOURCES)}

    def save(self, item):
        write_json(self.directory / item['id'] / 'status.json', item)

    def submit(self, payload):
        request = validate(payload)
        if not self.python.exists():
            raise ValueError('Research environment is not installed. See docs/TRADING-ANALYSIS.md.')
        with self.lock:
            if sum(r['status'] in ('queued', 'running') for r in self.records.values()) >= 8:
                raise ValueError('Research queue is full. Wait for a job or cancel one.')
            job_id = uuid.uuid4().hex
            folder = self.directory / job_id
            folder.mkdir(parents=True)
            write_json(folder / 'request.json', request)
            item = dict(id=job_id, status='queued', kind=request['kind'], created=time.time(), message='Waiting for the research worker.')
            self.records[job_id] = item
            self.save(item)
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.work, daemon=True, name='research-jobs')
                self.thread.start()
            return dict(item)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise ValueError('Research job not found.')
            item = dict(self.records[job_id])
            folder = self.directory / job_id
            if item['status'] == 'running' and (folder / 'progress.json').exists():
                try:
                    item['message'] = json.loads((folder / 'progress.json').read_text(encoding='utf-8'))['message']
                except (OSError, ValueError, KeyError):
                    pass
            if item['status'] == 'complete':
                item['result'] = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            return item

    def cancel(self, job_id):
        with self.lock:
            item = self.records.get(job_id)
            if item is None:
                raise ValueError('Research job not found.')
            if item['status'] in ('queued', 'running'):
                item.update(status='cancelled', message='Cancelled. Any exchange candles already committed are retained.')
                self.save(item)
            return dict(item)

    def work(self):
        while True:
            with self.lock:
                item = next((r for r in self.records.values() if r['status'] == 'queued'), None)
                if item is None:
                    self.thread = None
                    return
                item.update(status='running', message='Loading research libraries.')
                self.save(item)
            folder = self.directory / item['id']
            process = None
            try:
                with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen([str(self.python), str(self.root / 'analysis_worker.py'), str(folder), str(self.database)],
                        cwd=self.root, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    deadline = time.monotonic() + 600
                    while process.poll() is None:
                        if item['status'] == 'cancelled' or time.monotonic() >= deadline:
                            process.terminate()
                            process.wait(timeout=10)
                            if item['status'] != 'cancelled':
                                raise ValueError('Research exceeded the ten-minute limit. Use a shorter range.')
                            break
                        time.sleep(.2)
                with self.lock:
                    if item['status'] != 'cancelled':
                        result_path = folder / 'result.json'
                        if not result_path.exists():
                            raise ValueError('Research worker failed. See its worker.log under data/analysis/jobs.')
                        result = json.loads(result_path.read_text(encoding='utf-8'))
                        if result.get('error'):
                            raise ValueError(result['error'])
                        if process.returncode:
                            raise ValueError('Research worker exited unsuccessfully.')
                        item.update(status='complete', message='Research complete.')
                        self.save(item)
            except Exception as exc:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                with self.lock:
                    if item['status'] != 'cancelled':
                        item.update(status='failed', message=str(exc))
                        self.save(item)


_manager = None
_guard = threading.Lock()


def manager():
    global _manager
    with _guard:
        if _manager is None:
            _manager = Jobs()
        return _manager
