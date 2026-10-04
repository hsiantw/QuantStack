"""Local collector enrollment and conservative resource defaults."""
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
LOW = {'background_usage': 'low', 'hourly_workers': 1,
       'hourly_pause_seconds': 2.0, 'request_pause_seconds': 2.0,
       'hourly_budget_minutes': 10, 'daily_budget_minutes': 10}


def local_request(peer, host, origin=None):
    """Reject remote clients, DNS rebinding and cross-origin browser writes."""
    try:
        hostname = urlsplit('http://' + host).hostname
        return (peer in ('127.0.0.1', '::1') and
                hostname in ('localhost', '127.0.0.1', '::1') and
                (origin is None or (urlsplit(origin).scheme == 'http' and
                                   urlsplit(origin).netloc == host)))
    except ValueError:
        return False


def normalize_symbols(value, kind):
    if not isinstance(value, str) or len(value) > 4000 or kind not in ('stocks', 'crypto'):
        raise ValueError('Enter stock tickers or crypto symbols (up to 100 at once).')
    symbols = list(dict.fromkeys(re.split(r'[\s,;]+', value.strip().upper()))) if value.strip() else []
    if len(symbols) > 100:
        raise ValueError('Add at most 100 symbols at once.')
    if kind == 'crypto':
        symbols = [s if s.endswith('-USD') else s + '-USD' for s in symbols]
    if any(not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]{0,29}', s) for s in symbols):
        raise ValueError('Use provider tickers such as AAPL, 2330.TW or BTC-USD.')
    return list(dict.fromkeys(symbols))


def status(root=ROOT):
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    result = {key: config.get(key) for key in ('symbols', 'background_usage', 'hourly_workers',
            'hourly_pause_seconds', 'hourly_budget_minutes', 'hourly_cycle_minutes', 'daily_budget_minutes')}
    result['collection'] = {}
    for name in ('hourly', 'daily'):
        try:
            progress = json.loads((root / 'data' / f'{name}-refresh.json').read_text(encoding='utf-8'))
            if not isinstance(progress, dict):
                raise ValueError('Invalid collection report')
            result['collection'][name] = {key: progress.get(key) for key in
                ('status', 'updated_at', 'total', 'completed', 'pending', 'succeeded')}
            result['collection'][name]['failed'] = len(progress.get('failed', {}))
        except (OSError, ValueError, TypeError):
            result['collection'][name] = None
    return result


def enroll(payload, root=ROOT):
    from market_data import process_lock
    if not isinstance(payload, dict):
        raise ValueError('Expected an object.')
    symbols = normalize_symbols(payload.get('symbols', ''), payload.get('kind', 'stocks'))
    if payload.get('background_usage', 'low') != 'low':
        raise ValueError('Only the low background profile is supported here.')
    (root / 'data').mkdir(exist_ok=True)
    with process_lock(root / 'data' / 'ingestion.lock'):
        path = root / 'config.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        added = [s for s in symbols if s not in config['symbols']]
        config['symbols'] = list(dict.fromkeys(config['symbols'] + symbols))
        config.update(LOW)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    return dict(status(root), added=added)


def lower_priority(config):
    if config.get('background_usage') != 'low':
        return
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.nice(10)
