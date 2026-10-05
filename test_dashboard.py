import csv
import io
import json
import math
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlencode
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import dashboard
from market_data import connect, persist, persist_intraday


class DashboardTests(unittest.TestCase):
    def test_hourly_only_symbol_is_visible_without_minute_or_daily_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market.sqlite'
            db = connect(path)
            persist_intraday(db, 'NEW', '60m', [
                ('NEW', '2026-01-02T14:30:00+00:00', '60m', 10., 15., 9., 14., 14., 1000, 'USD', 'NMS', 'America/New_York', 'now')], 'now', 0)
            db.close()
            with patch.object(dashboard, 'DATABASE', path), patch.object(dashboard, 'ROOT', Path(directory)):
                asset = dashboard.catalog()[0]
                self.assertEqual((asset['symbol'], asset['close'], asset['quote_interval']), ('NEW', 14., '1h'))
                self.assertFalse(asset['has_daily'])
                self.assertIsNone(asset['change'])
                self.assertEqual(len(dashboard.history({'symbol': ['NEW'], 'interval': ['1h']})), 1)

    def test_configured_symbols_are_visible_before_their_first_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'symbols': ['BTC-USD', 'DOGE-USD']}))
            path = root / 'market.sqlite'
            db = connect(path)
            persist(db, 'BTC-USD', [
                ('BTC-USD', '2026-01-01', 10., 11., 9., 10., 10., 100, 0., 0., 'USD', 'CCC', 'UTC', 'now')
            ], True, 'now', 0)
            db.close()
            with patch.object(dashboard, 'DATABASE', path), patch.object(dashboard, 'ROOT', root):
                assets = {asset['symbol']: asset for asset in dashboard.catalog()}
            self.assertEqual(set(assets), {'BTC-USD', 'DOGE-USD'})
            self.assertEqual(assets['BTC-USD']['close'], 10.)
            self.assertTrue(assets['BTC-USD']['has_data'])
            self.assertIsNone(assets['DOGE-USD']['close'])
            self.assertFalse(assets['DOGE-USD']['has_data'])
            self.assertEqual(assets['DOGE-USD']['kind'], 'Crypto')

    def test_native_hourly_history_precedes_minute_rollups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market.sqlite'
            db = connect(path)
            persist_intraday(db, 'TEST', '60m', [
                ('TEST', '2026-01-02T14:30:00+00:00', '60m', 10., 15., 9., 14., 14., 1000, 'USD', 'NMS', 'America/New_York', 'now')], 'now', 0)
            persist_intraday(db, 'TEST', '1m', [
                ('TEST', '2026-01-02T14:30:00+00:00', '1m', 10., 11., 9., 10., 10., 10, 'USD', 'NMS', 'America/New_York', 'now')], 'now', 0)
            db.close()
            with patch.object(dashboard, 'DATABASE', path):
                hourly = dashboard.history({'symbol': ['TEST'], 'interval': ['1h']})
                self.assertEqual(len(hourly), 1)
                self.assertEqual((hourly[0]['date'], hourly[0]['close'], hourly[0]['volume']),
                                 ('2026-01-02T14:30:00+00:00', 14., 1000))
                self.assertEqual(dashboard.history({'symbol': ['TEST'], 'interval': ['1m']})[0]['close'], 10.)
                self.assertEqual(dashboard.history({'symbol': ['TEST'], 'interval': ['1h'], 'start': ['2026-01-03']}), [])
                # An archived minute record alone must never supply a 1H chart.
                db = connect(path)
                db.execute("DELETE FROM intraday_prices WHERE interval='60m'")
                db.commit()
                db.close()
                self.assertEqual(dashboard.history({'symbol': ['TEST'], 'interval': ['1h']}), [])

    def test_weekly_and_monthly_history_aggregates_daily_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market.sqlite'
            db = connect(path)
            rows = [
                ('TEST', day.isoformat(), 10. + index, 12. + index, 9. + index,
                 11. + index, 10. + index, 100 + index, 1., 0., 'USD', 'NMS',
                 'America/New_York', 'now')
                for index, day in enumerate([
                    date(2026, 1, 30), date(2026, 2, 2), date(2026, 2, 3),
                    date(2026, 2, 27), date(2026, 3, 2)])
            ]
            persist(db, 'TEST', rows, True, 'now', 0)
            db.close()
            with patch.object(dashboard, 'DATABASE', path):
                weekly = dashboard.history({'symbol': ['TEST'], 'interval': ['1w']})
                monthly = dashboard.history({'symbol': ['TEST'], 'interval': ['1mo']})
                limited = dashboard.history({
                    'symbol': ['TEST'], 'interval': ['1mo'], 'start': ['2026-02-01'], 'end': ['2026-02-28']
                })
            self.assertEqual([bar['date'] for bar in weekly],
                             ['2026-01-26', '2026-02-02', '2026-02-23', '2026-03-02'])
            self.assertEqual((weekly[1]['open'], weekly[1]['high'], weekly[1]['low'],
                              weekly[1]['close'], weekly[1]['volume'], weekly[1]['dividends']),
                             (11., 14., 10., 13., 203, 2.))
            self.assertEqual([bar['date'] for bar in monthly],
                             ['2026-01-01', '2026-02-01', '2026-03-01'])
            self.assertEqual((monthly[1]['open'], monthly[1]['high'], monthly[1]['low'],
                              monthly[1]['close'], monthly[1]['volume']),
                             (11., 15., 10., 14., 306))
            self.assertEqual(len(limited), 1)
            self.assertEqual(limited[0]['volume'], 306)
            with self.assertRaises(ValueError):
                dashboard.history({'symbol': ['TEST'], 'interval': ['1q']})

    def test_api_filters_exports_and_rejects_invalid_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market.sqlite'
            db = connect(path)
            rows = [('TEST', d, 1., 2., 1., c, c, 10, 0., 0., 'USD', 'NMS', 'America/New_York', 'now') for d,c in [('2026-01-01',1.),('2026-01-02',2.)]]
            persist(db, 'TEST', rows, True, 'now', 0)
            intraday = [
                ('TEST', '2026-01-02T15:30:00+00:00', '1m', 10., 12., 9., 11., 11., 100, 'USD', 'NMS', 'America/New_York', 'now'),
                ('TEST', '2026-01-02T15:31:00+00:00', '1m', 11., 13., 10., 12., 12., 200, 'USD', 'NMS', 'America/New_York', 'now')]
            persist_intraday(db, 'TEST', '1m', intraday, 'now', 0)
            persist_intraday(db, 'TEST', '60m', [
                ('TEST', '2026-01-02T14:30:00+00:00', '60m', 10., 13., 9., 12., 12., 300, 'USD', 'NMS', 'America/New_York', 'now')], 'now', 0)
            db.close()
            with patch.object(dashboard, 'DATABASE', path), patch.object(dashboard, 'ROOT', Path(directory)):
                server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.Handler)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                base = f'http://127.0.0.1:{server.server_port}'
                def get(route):
                    with urllib.request.urlopen(base + route) as response:
                        return response.read().decode()
                try:
                    catalog = json.loads(get('/api/symbols'))
                    self.assertEqual(catalog[0]['close'], 12)
                    self.assertEqual(catalog[0]['change'], 500)
                    self.assertEqual(catalog[0]['quote_interval'], '1h')
                    self.assertEqual(catalog[0]['quote_timestamp'], '2026-01-02T14:30:00+00:00')
                    usage = json.loads(get('/api/usage'))
                    self.assertEqual(usage['window_hours'], 24)
                    self.assertEqual({p['provider'] for p in usage['providers']},
                                     {'Yahoo Finance', 'Coinbase', 'Kraken', 'Bitstamp', 'Gemini'})
                    indicators = json.loads(get('/api/indicators'))
                    self.assertGreaterEqual(len(indicators), 100)
                    self.assertIn('RSI', {item['id'] for item in indicators})
                    rsi = json.loads(get('/api/indicator?symbol=TEST&name=RSI'))
                    self.assertEqual((rsi['id'], len(rsi['outputs']['real'])), ('RSI', 2))
                    custom_sma = json.loads(get('/api/indicator?' + urlencode({
                        'symbol': 'TEST', 'name': 'SMA', 'params': json.dumps({'timeperiod': 2}), 'source': 'high'})))
                    self.assertEqual(custom_sma['parameters'], {'timeperiod': 2})
                    self.assertEqual(custom_sma['source'], 'high')
                    self.assertEqual(custom_sma['outputs']['real'], [None, 2.0])
                    for bad_params in ('[]', '{', '{"timeperiod":0}', '{"timeperiod":2.5}',
                                       '{"timeperiod":"10"}', '{"timeperiod":true}', '{"timeperiod":NaN}',
                                       '{"timeperiod":1e999}', '{"timeperiod":10001}', '{"unknown":14}'):
                        with self.subTest(params=bad_params):
                            with self.assertRaises(urllib.error.HTTPError) as error:
                                get('/api/indicator?' + urlencode({'name': 'RSI', 'params': bad_params}))
                            self.assertEqual(error.exception.code, 400)
                            self.assertTrue(json.loads(error.exception.read())['error'])
                    for bad_source in ('volume', 'not-a-source'):
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            get('/api/indicator?name=RSI&source=' + bad_source)
                        self.assertEqual(error.exception.code, 400)
                    prices = json.loads(get('/api/history?symbol=TEST&start=2026-01-02'))
                    self.assertEqual(len(prices), 1)
                    self.assertEqual(prices[0]['close'], 2)
                    minute = json.loads(get('/api/history?symbol=TEST&start=2026-01-02&end=2026-01-02&interval=5m'))
                    self.assertEqual((len(minute), minute[0]['open'], minute[0]['close'], minute[0]['volume']), (1, 10, 12, 300))
                    exported = list(csv.DictReader(io.StringIO(get('/api/export?symbol=TEST&end=2026-01-01'))))
                    self.assertEqual(len(exported), 1)
                    self.assertEqual(exported[0]['symbol'], 'TEST')
                    self.assertEqual(json.loads(get('/api/history?symbol=%27%20OR%201%3D1--')), [])
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        get('/api/history?start=2026-02-01&end=2026-01-01')
                    self.assertEqual(error.exception.code, 400)
                    with self.assertRaises(urllib.error.HTTPError):
                        get('/../config.json')
                finally:
                    server.shutdown()
                    server.server_close()
                    worker.join()

    @staticmethod
    def sample_bars(count=100):
        return [dict(date=(date(2026, 1, 1) + timedelta(days=index)).isoformat(),
                     open=100.0 + index, high=104.0 + index, low=98.0 + index,
                     close=102.0 + index, volume=10 + index)
                for index in range(count)]

    def indicator(self, name, bars=None, source=None, **parameters):
        query = {'name': [name], 'params': [json.dumps(parameters)]}
        if source is not None:
            query['source'] = [source]
        with patch.object(dashboard, 'history', return_value=bars if bars is not None else self.sample_bars()):
            return dashboard.indicator_data(query)

    def test_catalog_and_all_indicators_return_aligned_finite_outputs(self):
        catalog = dashboard.indicator_catalog()
        ids = [item['id'] for item in catalog]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({'VWAP', 'ICHIMOKU', 'BBP', 'BBWIDTH', 'CHANDELIER',
                         'SUPERTREND', 'KC', 'DONCHIAN'}.issubset(ids))
        for item in catalog:
            with self.subTest(indicator=item['id']):
                self.assertEqual(set(item['parameters']), set(item['parameter_meta']))
                for key, value in item['parameters'].items():
                    meta = item['parameter_meta'][key]
                    self.assertLessEqual(meta['min'], value)
                    self.assertGreaterEqual(meta['max'], value)
                result = self.indicator(item['id'])
                self.assertEqual(result['source'], item['source'])
                self.assertEqual(result['parameters'], item['parameters'])
                self.assertEqual(list(result['outputs']), item['outputs'])
                self.assertEqual(len(result['dates']), 100)
                for values in result['outputs'].values():
                    self.assertEqual(len(values), 100)
                    self.assertTrue(all(value is None or math.isfinite(value) for value in values))
                empty = self.indicator(item['id'], bars=[])
                self.assertTrue(all(values == [] for values in empty['outputs'].values()))

    def test_source_and_period_change_the_computation(self):
        bars = self.sample_bars(10)
        result = self.indicator('SMA', bars, timeperiod=3)
        self.assertEqual(result['outputs']['real'][:4], [None, None, 103.0, 104.0])
        result = self.indicator('SMA', bars, source='open', timeperiod=3)
        self.assertEqual(result['outputs']['real'][2], 101.0)
        result = self.indicator('SMA', bars, source='hlc3', timeperiod=2)
        self.assertAlmostEqual(result['outputs']['real'][1], 101.83333333333333)
        # OHLC-dependent studies keep their original price relationships.
        with self.assertRaisesRegex(ValueError, 'Source for ATR'):
            self.indicator('ATR', bars, source='open')
        with self.assertRaisesRegex(ValueError, 'minperiod'):
            self.indicator('MAVP', bars, minperiod=20, maxperiod=10)

    def test_vwap_weights_zero_volume_and_utc_day_reset(self):
        bars = self.sample_bars(3)
        bars[0].update(date='2026-01-01T12:00:00+00:00', close=10.0, volume=0)
        bars[1].update(date='2026-01-01T12:01:00+00:00', close=20.0, volume=2)
        bars[2].update(date='2026-01-02T12:00:00+00:00', close=40.0, volume=6)
        self.assertEqual(self.indicator('VWAP', bars, source='close')['outputs']['vwap'], [None, 20.0, 35.0])
        self.assertEqual(self.indicator('VWAP', bars, source='close', anchor=1)['outputs']['vwap'],
                         [None, 20.0, 40.0])

    def test_ichimoku_displacement_and_chandelier_levels(self):
        result = self.indicator('ICHIMOKU', self.sample_bars(10), conversionperiod=2,
                                baseperiod=3, spanperiod=4, displacement=2)['outputs']
        self.assertEqual(result['conversion'][1], 101.5)
        self.assertEqual(result['base'][2], 102.0)
        self.assertEqual(result['span_a'][:4], [None] * 4)
        self.assertEqual(result['span_a'][4], 102.25)
        self.assertEqual(result['span_b'][5], 102.5)
        self.assertEqual(result['lagging'][-2:], [None, None])
        self.assertEqual(result['lagging'][0], 104.0)
        result = self.indicator('CHANDELIER', self.sample_bars(10), timeperiod=3, atrperiod=3,
                                multiplier=2)['outputs']
        self.assertEqual(result['long_exit'][3], 95.0)
        self.assertEqual(result['short_exit'][3], 111.0)

    def test_band_derivatives_and_supertrend_price_outputs(self):
        bars = self.sample_bars(5)
        bands = self.indicator('BBANDS', bars, timeperiod=3)['outputs']
        percent = self.indicator('BBP', bars, timeperiod=3)['outputs']['percent_b']
        width = self.indicator('BBWIDTH', bars, timeperiod=3)['outputs']['width']
        upper, middle, lower = (bands[key][-1] for key in ('upperband', 'middleband', 'lowerband'))
        self.assertAlmostEqual(percent[-1], (bars[-1]['close'] - lower) / (upper - lower))
        self.assertAlmostEqual(width[-1], (upper - lower) / middle * 100)
        flat = [dict(bar, open=100, high=100, low=100, close=100) for bar in bars]
        self.assertEqual(self.indicator('BBP', flat, timeperiod=3)['outputs']['percent_b'], [None] * 5)
        supertrend = self.indicator('SUPERTREND')['outputs']
        self.assertEqual(set(supertrend), {'uptrend', 'downtrend'})
        self.assertIsNotNone(supertrend['uptrend'][-1])
        for index in range(100):
            self.assertFalse(supertrend['uptrend'][index] is not None and supertrend['downtrend'][index] is not None)


if __name__ == '__main__':
    unittest.main()
