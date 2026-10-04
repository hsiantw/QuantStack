"""Local market dashboard. Run with .venv/Scripts/python.exe dashboard.py."""
import argparse
import csv
import io
import json
import sqlite3
import statistics
import math
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / 'data' / 'market.sqlite'


@contextmanager
def database():
    connection = sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def daily_performance(bars, crypto=False):
    """Completed daily-bar returns, newest first; never splice adjusted/raw prices."""
    def positive(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    adjusted = any(positive(row['adjusted_close']) for row in bars)
    field = 'adjusted_close' if adjusted else 'close'
    result = {'performance_asof': bars[0]['date'] if bars else None,
              'performance_basis': 'adjusted' if adjusted else 'unadjusted'}
    for key, periods in (('change_1d_pct', 1), ('return_1w_pct', 7 if crypto else 5), ('return_1m_pct', 30 if crypto else 21)):
        window = bars[:periods + 1]
        valid = len(window) == periods + 1 and all(positive(row[field]) for row in window)
        value = (window[0][field] / window[-1][field] - 1) * 100 if valid else None
        result[key] = value if value is not None and math.isfinite(value) else None
    return result


def catalog():
    names = {'2330.TW': 'Taiwan Semiconductor', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'}
    source = ROOT / 'data' / 'constituents.csv'
    if source.exists():
        with source.open(encoding='utf-8-sig') as handle:
            names.update({r['Symbol'].replace('.', '-'): r['Security'] for r in csv.DictReader(handle)})
    with database() as db:
        metadata = {}
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_metadata'").fetchone():
            metadata = {row['symbol']: dict(row) for row in db.execute('SELECT * FROM stock_metadata')}
            names.update({symbol: row['name'] for symbol, row in metadata.items() if row.get('name')})
        symbols = db.execute('''SELECT symbol,last_success,error FROM state UNION ALL
          SELECT symbol,MAX(last_success),error FROM intraday_state
          WHERE symbol NOT IN (SELECT symbol FROM state) GROUP BY symbol ORDER BY symbol''').fetchall()
        result = []
        for symbol in symbols:
            bars = db.execute('SELECT date,close,adjusted_close,volume,currency,exchange FROM prices WHERE symbol=? ORDER BY date DESC LIMIT 31', (symbol['symbol'],)).fetchall()
            latest = bars[:2]
            performance = daily_performance(bars, symbol['symbol'].endswith('-USD'))
            meta = metadata.get(symbol['symbol'], {})
            market_cap = meta.get('market_cap')
            if not isinstance(market_cap, (int, float)) or not math.isfinite(market_cap) or market_cap < 0:
                market_cap = None
            hourly = db.execute('''SELECT timestamp,close,currency,exchange,interval FROM intraday_prices
              WHERE symbol=? ORDER BY (interval='60m') DESC, timestamp DESC LIMIT 1''', (symbol['symbol'],)).fetchone()
            has_daily = bool(latest)
            if not latest and not hourly:
                continue
            latest, previous = (latest[0], latest[1] if len(latest) > 1 else None) if latest else (
                dict(date=hourly['timestamp'][:10], close=hourly['close'], currency=hourly['currency'], exchange=hourly['exchange']), None)
            # Prefer the freshest stored quote, but retain the completed daily date
            # separately for date-range defaults and daily charts.
            use_hourly = bool(hourly and hourly['timestamp'][:10] >= latest['date'])
            quote = hourly if use_hourly else latest
            baseline = (latest['close'] if has_daily else None) if use_hourly else (previous['close'] if previous else None)
            change = (quote['close'] / baseline - 1) * 100 if baseline else None
            daily = {key: latest[key] for key in ('date', 'close', 'currency', 'exchange')}
            daily.update(close=quote['close'], currency=quote['currency'], exchange=quote['exchange'])
            result.append(dict(symbol=symbol['symbol'], name=names.get(symbol['symbol'], symbol['symbol']),
                               **daily, quote_timestamp=quote['timestamp'] if use_hourly else latest['date'],
                               quote_interval=('1h' if hourly['interval'] == '60m' else hourly['interval']) if use_hourly else '1d', change=change,
                               has_daily=has_daily,
                               **performance, market_cap=market_cap,
                               market_cap_currency=meta.get('currency'), market_cap_asof=meta.get('fetched_at'),
                               volume=bars[0]['volume'] if bars else None,
                               kind='Crypto' if symbol['symbol'].endswith('-USD') else 'Stocks',
                               updated=symbol['last_success'], error=symbol['error']))
    return result


def history(query):
    symbol = query.get('symbol', ['AAPL'])[0]
    interval = query.get('interval', ['1d'])[0]
    intervals = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '1d': 0}
    if interval not in intervals:
        raise ValueError('Interval must be 1m, 5m, 15m, 1h, or 1d.')
    start, end = query.get('start', ['0001-01-01'])[0], query.get('end', ['9999-12-31'])[0]
    date.fromisoformat(start)
    date.fromisoformat(end)
    if start > end:
        raise ValueError('Start date must be before the end date.')
    with database() as db:
        if interval == '1d':
            rows = db.execute('''SELECT date,open,high,low,close,adjusted_close,volume,dividends,splits
              FROM prices WHERE symbol=? AND date>=? AND date<=? ORDER BY date''', (symbol, start, end)).fetchall()
            return [dict(row) for row in rows]
        lower = start + 'T00:00:00+00:00'
        upper = (date.fromisoformat(end) + timedelta(days=1)).isoformat() + 'T00:00:00+00:00' if end != '9999-12-31' else '9999-12-31T23:59:59+00:00'
        if interval in ('1h', '5m', '15m'):
            # Native provider candles retain their exchange/session alignment.
            # Do not mix them with UTC-hour rollups of partial minute coverage.
            native = db.execute('''SELECT timestamp AS date,open,high,low,close,adjusted_close,
              volume,NULL AS dividends,NULL AS splits,source FROM intraday_prices
              WHERE symbol=? AND interval=? AND timestamp>=? AND timestamp<?
              ORDER BY timestamp''', (symbol, '60m' if interval == '1h' else interval, lower, upper)).fetchall()
            if native or interval == '1h':
                return [dict(row) for row in native[-10000:]]
        if symbol == 'BTC-USD':
            exchange_rows = db.execute('''SELECT timestamp,source,open,high,low,close,volume
              FROM crypto_exchange_prices WHERE symbol=? AND interval='1m'
              AND timestamp>=? AND timestamp<? ORDER BY timestamp,source''', (symbol, lower, upper)).fetchall()
            yahoo_rows = db.execute('''SELECT timestamp,open,high,low,close,adjusted_close,volume
              FROM intraday_prices WHERE symbol=? AND interval='1m'
              AND timestamp>=? AND timestamp<? ORDER BY timestamp''', (symbol, lower, upper)).fetchall()
            grouped = {}
            for item in exchange_rows:
                grouped.setdefault(item['timestamp'], []).append(item)
            raw = []
            for timestamp, values in grouped.items():
                raw.append(dict(timestamp=timestamp,
                                open=statistics.median(v['open'] for v in values),
                                high=statistics.median(v['high'] for v in values),
                                low=statistics.median(v['low'] for v in values),
                                close=statistics.median(v['close'] for v in values),
                                adjusted_close=statistics.median(v['close'] for v in values),
                                volume=sum(v['volume'] for v in values), sources=len(values),
                                source=' + '.join(v['source'] for v in values)))
            exchange_times = set(grouped)
            raw.extend(dict(timestamp=item['timestamp'], open=item['open'], high=item['high'], low=item['low'],
                            close=item['close'], adjusted_close=item['adjusted_close'], volume=item['volume'],
                            sources=1, source='Yahoo Finance')
                       for item in yahoo_rows if item['timestamp'] not in exchange_times)
            raw.sort(key=lambda item: item['timestamp'])
        else:
            raw = db.execute('''SELECT timestamp,open,high,low,close,adjusted_close,volume
              FROM intraday_prices WHERE symbol=? AND interval='1m'
              AND timestamp>=? AND timestamp<? ORDER BY timestamp''', (symbol, lower, upper)).fetchall()
    minutes = intervals[interval]
    result = []
    for item in raw:
        stamp = datetime.fromisoformat(item['timestamp']).astimezone(timezone.utc)
        bucket = stamp.replace(minute=(stamp.minute // minutes) * minutes, second=0, microsecond=0)
        key = bucket.isoformat()
        if not result or result[-1]['date'] != key:
            result.append(dict(date=key, open=item['open'], high=item['high'], low=item['low'],
                               close=item['close'], adjusted_close=item['adjusted_close'],
                               volume=item['volume'], dividends=None, splits=None,
                               sources=item.get('sources') if isinstance(item, dict) else None,
                               source=item.get('source') if isinstance(item, dict) else 'Yahoo Finance'))
        else:
            bar = result[-1]
            bar['high'] = max(bar['high'], item['high'])
            bar['low'] = min(bar['low'], item['low'])
            bar['close'] = item['close']
            bar['adjusted_close'] = item['adjusted_close']
            bar['volume'] += item['volume']
            if bar.get('sources') is not None:
                bar['sources'] = min(bar['sources'], item['sources'])
                if bar.get('source') != item.get('source'):
                    bar['source'] = 'Mixed sources'
    # Bound browser work while keeping the most recent bars in very large ranges.
    return result if symbol == 'BTC-USD' else result[-5000:]


def api_usage():
    limits = {
        'Yahoo Finance': {'access': 'Unofficial, keyless', 'candle_limit': 'Native 1h candles; one year requested initially'},
        'Coinbase': {'access': 'Public, keyless', 'candle_limit': 'Maximum 300 candles/request'},
        'Kraken': {'access': 'Public, keyless', 'candle_limit': 'Most recent 720 OHLC candles'},
        'Bitstamp': {'access': 'Public, keyless', 'candle_limit': 'Maximum 1,000 candles/request'},
        'Gemini': {'access': 'Public, keyless', 'candle_limit': '1m candles supported; response cap not documented'},
    }
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with database() as db:
        has_table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='api_requests'").fetchone()
        activity = db.execute('''SELECT provider,COUNT(*) requests,
          COALESCE(SUM(rows),0) rows,SUM(status='failed') failures,
          MAX(requested_at) last_request FROM api_requests WHERE requested_at>=? GROUP BY provider''',
                              (cutoff,)).fetchall() if has_table else []
        latest = db.execute('''SELECT a.provider,a.status,a.duration_ms,a.error FROM api_requests a
          JOIN (SELECT provider,MAX(id) id FROM api_requests GROUP BY provider) b ON a.id=b.id''').fetchall() if has_table else []
    by_provider = {row['provider']: dict(row) for row in activity}
    last_by_provider = {row['provider']: dict(row) for row in latest}
    providers = []
    for name, details in limits.items():
        providers.append({'provider': name, **details,
                          **by_provider.get(name, {'requests': 0, 'rows': 0, 'failures': 0, 'last_request': None}),
                          **last_by_provider.get(name, {'status': None, 'duration_ms': None, 'error': None})})
    return {'window_hours': 24, 'generated_at': datetime.now(timezone.utc).isoformat(), 'providers': providers}


PRICE_SOURCES = ['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4']


def parameter_metadata(parameters):
    """Publish the dashboard's supported ranges alongside each numeric input."""
    import talib
    result = {}
    for key, default in parameters.items():
        meta = {'label': key.replace('_', ' ').capitalize(), 'min': 0, 'max': 1000,
                'step': 1 if isinstance(default, int) else 0.01}
        if 'matype' in key:
            options = talib.MA_Type._lookup
            meta.update(label='Moving average type', max=max(options), options=options)
        elif 'period' in key or key in {'leftbars', 'rightbars'}:
            meta.update(min=2, max=10000, step=1)
        elif key in {'fastlimit', 'slowlimit'}:
            meta.update(min=0.01, max=0.99)
        elif key in {'vfactor', 'penetration'}:
            meta.update(max=1)
        elif key == 'percentile':
            meta.update(max=100)
        elif key == 'startvalue':
            meta.update(min=-1e12, max=1e12)
        elif key == 'multiplier' or key.startswith('nbdev'):
            meta.update(min=0.01, max=100, step=0.1)
        result[key] = meta
    return result


def indicator_catalog():
    import talib
    from talib import abstract
    excluded = {'Math Operators', 'Math Transform'}
    result = []
    for group, names in talib.get_function_groups().items():
        if group in excluded:
            continue
        for name in names:
            info = abstract.Function(name).info
            parameters = dict(info['parameters'])
            sources = PRICE_SOURCES if info['input_names'].get('price') == 'close' else ['close']
            result.append({'id': name, 'name': info['display_name'], 'group': group,
                           'parameters': parameters, 'parameter_meta': parameter_metadata(parameters),
                           'outputs': ['uptrend', 'downtrend'] if name == 'SUPERTREND' else list(info['output_names']),
                           'sources': sources, 'source': 'close',
                           'overlay': group in {'Overlap Studies', 'Price Transform'},
                           'pattern': group == 'Pattern Recognition'})
    custom = [
        ('VWAP', 'Volume Weighted Average Price', {'anchor': 0}, ['vwap'], True, PRICE_SOURCES,
         'hlc3', 'Volume-weighted price from the loaded range, or reset at each UTC day.'),
        ('ICHIMOKU', 'Ichimoku Cloud',
         {'conversionperiod': 9, 'baseperiod': 26, 'spanperiod': 52, 'displacement': 26},
         ['conversion', 'base', 'span_a', 'span_b', 'lagging'], True, ['close'], 'close',
         'Leading and lagging spans are shifted within the loaded dates; future dates are not projected.'),
        ('BBP', 'Bollinger Bands %B', {'timeperiod': 20, 'nbdev': 2.0}, ['percent_b'], False,
         PRICE_SOURCES, 'close', 'Position within the Bollinger Bands: lower band = 0, upper band = 1.'),
        ('BBWIDTH', 'Bollinger Band Width', {'timeperiod': 20, 'nbdev': 2.0}, ['width'], False,
         PRICE_SOURCES, 'close', 'Bollinger Band width as a percentage of the middle band.'),
        ('CHANDELIER', 'Chandelier Exit', {'timeperiod': 22, 'atrperiod': 22, 'multiplier': 3.0},
         ['long_exit', 'short_exit'], True, ['close'], 'close',
         'Rolling price extremes offset by a configurable multiple of Average True Range.'),
    ]
    custom_names = {item[0] for item in custom}
    result = [item for item in result if item['id'] not in custom_names]
    for name, title, parameters, outputs, overlay, sources, source, description in custom:
        meta = parameter_metadata(parameters)
        if name == 'VWAP':
            meta['anchor'].update(label='Reset period', min=0, max=1,
                                  options={0: 'Loaded range', 1: 'UTC day'})
        if name == 'ICHIMOKU':
            meta['displacement'].update(label='Displacement (bars)', max=1000)
        result.append({'id': name, 'name': title, 'group': 'Overlap Studies' if overlay else 'Volatility Indicators',
                       'parameters': parameters, 'parameter_meta': meta, 'outputs': outputs,
                       'sources': sources, 'source': source, 'overlay': overlay, 'pattern': False,
                       'description': description})
    return result


def indicator_parameters(query, item):
    try:
        supplied = json.loads(query.get('params', ['{}'])[0])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('Indicator params must be a JSON object of numeric values.') from exc
    if not isinstance(supplied, dict):
        raise ValueError('Indicator params must be a JSON object of numeric values.')
    parameters = dict(item['parameters'])
    for key, value in supplied.items():
        if key not in parameters:
            raise ValueError(f'Unknown parameter "{key}" for {item["id"]}.')
        if type(value) not in (int, float):
            raise ValueError(f'{key} must be a finite number.')
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f'{key} must be a finite number.')
        meta = item['parameter_meta'][key]
        if not meta['min'] <= value <= meta['max']:
            raise ValueError(f'{key} must be between {meta["min"]} and {meta["max"]}.')
        if isinstance(parameters[key], int) and value != int(value):
            raise ValueError(f'{key} must be a whole number.')
        parameters[key] = int(value) if isinstance(parameters[key], int) else float(value)
    if 'minperiod' in parameters and parameters['minperiod'] > parameters['maxperiod']:
        raise ValueError('minperiod must not exceed maxperiod.')
    source = query.get('source', [item['source']])[0]
    if source not in item['sources']:
        raise ValueError(f'Source for {item["id"]} must be one of: {", ".join(item["sources"])}.')
    return parameters, source


def custom_indicator(name, inputs, parameters, bars):
    """Additional studies, calculated on the same bars as the price chart."""
    import numpy as np
    import talib
    close, high, low = inputs['close'], inputs['high'], inputs['low']
    if name == 'VWAP':
        values = np.full(len(bars), np.nan)
        total_volume = total_price = 0.0
        previous_day = None
        for index, bar in enumerate(bars):
            day = bar['date'][:10]
            if parameters['anchor'] == 1 and day != previous_day:
                total_volume = total_price = 0.0
            volume = max(0.0, inputs['volume'][index])
            total_volume += volume
            total_price += close[index] * volume
            if total_volume:
                values[index] = total_price / total_volume
            previous_day = day
        return [values]
    if name == 'ICHIMOKU':
        def midpoint(period):
            return (talib.MAX(high, period) + talib.MIN(low, period)) / 2

        def shift(values, offset):
            result = np.full(len(values), np.nan)
            if offset == 0:
                return values
            if abs(offset) < len(values):
                if offset > 0:
                    result[offset:] = values[:-offset]
                else:
                    result[:offset] = values[-offset:]
            return result

        conversion = midpoint(parameters['conversionperiod'])
        base = midpoint(parameters['baseperiod'])
        displacement = parameters['displacement']
        return [conversion, base, shift((conversion + base) / 2, displacement),
                shift(midpoint(parameters['spanperiod']), displacement), shift(close, -displacement)]
    if name in {'BBP', 'BBWIDTH'}:
        upper, middle, lower = talib.BBANDS(close, parameters['timeperiod'],
                                           parameters['nbdev'], parameters['nbdev'])
        numerator = close - lower if name == 'BBP' else (upper - lower) * 100
        denominator = upper - lower if name == 'BBP' else middle
        return [np.divide(numerator, denominator, out=np.full(len(bars), np.nan), where=denominator != 0)]
    if name == 'CHANDELIER':
        distance = talib.ATR(high, low, close, parameters['atrperiod']) * parameters['multiplier']
        return [talib.MAX(high, parameters['timeperiod']) - distance,
                talib.MIN(low, parameters['timeperiod']) + distance]
    return None


def indicator_data(query):
    import numpy as np
    from talib import abstract
    name = query.get('name', [''])[0].upper()
    catalog = {item['id']: item for item in indicator_catalog()}
    if name not in catalog:
        raise ValueError('Unknown technical indicator.')
    item = catalog[name]
    parameters, source = indicator_parameters(query, item)
    bars = history(query)
    inputs = {key: np.asarray([float(row[key] or 0) for row in bars], dtype=float)
              for key in ('open', 'high', 'low', 'close', 'volume')}
    if source == 'hl2':
        inputs['close'] = (inputs['high'] + inputs['low']) / 2
    elif source == 'hlc3':
        inputs['close'] = (inputs['high'] + inputs['low'] + inputs['close']) / 3
    elif source == 'ohlc4':
        inputs['close'] = (inputs['open'] + inputs['high'] + inputs['low'] + inputs['close']) / 4
    elif source != 'close':
        inputs['close'] = inputs[source]
    if not bars:
        arrays = [[] for _ in item['outputs']]
    else:
        arrays = custom_indicator(name, inputs, parameters, bars)
        if arrays is None:
            if name == 'MAVP':
                period = min(parameters['maxperiod'], max(parameters['minperiod'], 14))
                inputs['periods'] = np.full(len(bars), float(period), dtype=float)
            try:
                values = abstract.Function(name)(inputs, **parameters)
            except Exception as exc:
                raise ValueError(f'Invalid settings for {name}: {exc}') from exc
            arrays = list(values) if isinstance(values, (list, tuple)) else [values]
            if name == 'SUPERTREND':
                line, direction = arrays
                arrays = [np.where(direction > 0, line, np.nan), np.where(direction < 0, line, np.nan)]
    outputs = {}
    for output_name, array in zip(item['outputs'], arrays):
        outputs[output_name] = [float(value) if math.isfinite(float(value)) else None for value in array]
    return {'id': name, 'name': item['name'], 'group': item['group'],
            'overlay': item['overlay'], 'pattern': item['pattern'], 'parameters': parameters, 'source': source,
            'dates': [row['date'] for row in bars], 'outputs': outputs}


class Handler(BaseHTTPRequestHandler):
    def scheduler_allowed(self):
        from local_scheduler import local_request
        return local_request(self.client_address[0], self.headers.get('Host', ''), self.headers.get('Origin'))

    def do_POST(self):
        if urlparse(self.path).path not in ('/api/local-scheduler', '/api/pull-queue'):
            self.send(404, b'Not found', 'text/plain')
            return
        if not self.scheduler_allowed() or self.headers.get('Content-Type') != 'application/json':
            self.send(403, b'{"error":"Local same-origin JSON requests only."}', 'application/json')
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError('Invalid request size.')
            from local_scheduler import enroll
            payload = json.loads(self.rfile.read(size))
            if urlparse(self.path).path == '/api/pull-queue':
                from pull_queue import enqueue, start_worker
                value = enqueue(payload)
                start_worker()
            else:
                value = enroll(payload)
            self.send(200, json.dumps(value).encode(), 'application/json')
        except (ValueError, RuntimeError) as exc:
            self.send(400, json.dumps({'error': str(exc)}).encode(), 'application/json')
        except OSError:
            self.send(500, b'{"error":"Could not save collector settings."}', 'application/json')

    def send(self, status, content, mime, attachment=False):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="prices.csv"')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ('/api/local-scheduler', '/api/pull-queue'):
                if not self.scheduler_allowed():
                    self.send(403, b'{"error":"Available on the local app only."}', 'application/json')
                    return
                from local_scheduler import status
                if url.path == '/api/pull-queue':
                    from pull_queue import status
                value = status()
            elif url.path == '/api/symbols':
                value = catalog()
            elif url.path == '/api/screener':
                from screener import snapshot
                value = dict(snapshot(DATABASE))
                progress = ROOT / 'data' / 'stock-expansion.json'
                if progress.exists():
                    value['expansion'] = json.loads(progress.read_text(encoding='utf-8'))
            elif url.path == '/api/usage':
                value = api_usage()
            elif url.path == '/api/indicators':
                value = indicator_catalog()
            elif url.path == '/api/indicator':
                value = indicator_data(parse_qs(url.query))
            elif url.path in ('/api/history', '/api/export'):
                value = history(parse_qs(url.query))
                if url.path == '/api/export':
                    output = io.StringIO(newline='')
                    fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividends', 'splits', 'sources', 'source']
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    symbol = parse_qs(url.query).get('symbol', ['AAPL'])[0]
                    for row in value:
                        writer.writerow({'symbol': symbol, **row})
                    self.send(200, output.getvalue().encode(), 'text/csv; charset=utf-8', True)
                    return
            else:
                files = {'/interview-prep.html': ('interview-prep.html', 'text/html'),
                         '/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                         '/indicators.js': ('indicators.js', 'text/javascript'),
                         '/drawings.js': ('drawings.js', 'text/javascript'),
                         '/chart-scale.js': ('chart-scale.js', 'text/javascript'),
                         '/chart-scale.css': ('chart-scale.css', 'text/css'),
                         '/workspace.js': ('workspace.js', 'text/javascript'),
                         '/terminal.js': ('terminal.js', 'text/javascript'),
                         '/data-pulls.js': ('data-pulls.js', 'text/javascript'),
                         '/watchlists.js': ('watchlists.js', 'text/javascript'),
                         '/watchlists.css': ('watchlists.css', 'text/css'),
                         '/theme.js': ('theme.js', 'text/javascript'),
                         '/theme.css': ('theme.css', 'text/css'),
                         '/strategy-engine.js': ('strategy-engine.js', 'text/javascript'),
                         '/strategy.js': ('strategy.js', 'text/javascript'),
                         '/markov-engine.js': ('markov-engine.js', 'text/javascript'),
                         '/markov-worker.js': ('markov-worker.js', 'text/javascript'),
                         '/markov.js': ('markov.js', 'text/javascript'),
                         '/markov.css': ('markov.css', 'text/css'),
                         '/brownian-engine.js': ('brownian-engine.js', 'text/javascript'),
                         '/brownian-worker.js': ('brownian-worker.js', 'text/javascript'),
                         '/brownian.js': ('brownian.js', 'text/javascript'),
                         '/risk-engine.js': ('risk-engine.js', 'text/javascript'),
                         '/risk.js': ('risk.js', 'text/javascript'),
                         '/research.js': ('research.js', 'text/javascript'),
                         '/research-engine.js': ('research-engine.js', 'text/javascript'),
                         '/research-worker.js': ('research-worker.js', 'text/javascript'),
                         '/research.css': ('research.css', 'text/css'),
                         '/terminal.css': ('terminal.css', 'text/css'),
                         '/workspace.css': ('workspace.css', 'text/css'),
                         '/indicators.css': ('indicators.css', 'text/css'), '/style.css': ('style.css', 'text/css'),
                         '/screener.js': ('screener.js', 'text/javascript'), '/screener.css': ('screener.css', 'text/css')}
                if url.path not in files:
                    self.send(404, b'Not found', 'text/plain')
                    return
                filename, mime = files[url.path]
                self.send(200, (ROOT / 'web' / filename).read_bytes(), mime + '; charset=utf-8')
                return
            self.send(200, json.dumps(value, allow_nan=False).encode(), 'application/json')
        except ValueError as exc:
            self.send(400, json.dumps({'error': str(exc)}).encode(), 'application/json')
        except Exception:
            self.send(500, b'{"error":"Could not read the local dataset. Check that the database exists."}', 'application/json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'QuantStack is running at http://127.0.0.1:{args.port}', flush=True)
    from pull_queue import start_worker
    start_worker()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
