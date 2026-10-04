import json
import http.client
import tempfile
import threading
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer

import market_data
import pull_queue as queue
import refresh_hourly


class PullQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        self.db = market_data.connect(self.root / 'data' / 'market.sqlite')
        self.addCleanup(self.db.close)

    def enqueue(self, symbol='AAPL', **kwargs):
        return queue.enqueue(dict(symbol=symbol, days=7, interval='60m', **kwargs), self.root)

    def test_validation(self):
        now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        symbol, interval, start, end = queue.validate({'symbol':'btc', 'kind':'crypto', 'days':7}, now)
        self.assertEqual((symbol, interval, start), ('BTC-USD', '60m', '2026-09-26T00:00:00+00:00'))
        for payload in ({'symbol':'AAPL', 'days':True}, {'symbol':'A B'}, {'symbol':'A', 'interval':'bad'},
                        {'symbol':'A', 'interval':'1m', 'days':8}, {'symbol':'A', 'range':'custom', 'start':'2026-10-04','end':'2026-10-05'},
                        {'symbol':'A', 'range':'custom', 'start':'2026-10-02','end':'2026-10-01'}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                queue.validate(payload, now)

    def test_custom_inclusive_end_and_deduplication(self):
        request = dict(symbol='AAPL', range='custom', start='2026-01-01', end='2026-01-02', interval='1d')
        first = queue.enqueue(request, self.root)
        self.assertEqual(queue.enqueue(request, self.root), first)
        self.assertEqual(queue.status(self.root)['requests'][0]['end'], '2026-01-03T00:00:00+00:00')

    def test_fifo_failure_recovery_and_new_arrivals(self):
        self.enqueue('FIRST'); self.enqueue('FAIL')
        with closing(queue.connection(self.root)) as db, db:
            db.execute("UPDATE requests SET status='running' WHERE symbol='FIRST'")
        calls = []
        def fetch(symbol, start, end, **kwargs):
            calls.append(symbol)
            if symbol == 'FIRST':
                self.enqueue('LATE')
            if symbol == 'FAIL':
                return [], 0, [], 'Provider rejected symbol'
            stamp = start.isoformat()
            return [(symbol, stamp, '60m', 1, 2, 1, 2, 2, 10, 'USD', 'NMS', 'UTC', stamp)], 0, [], None
        with patch.object(refresh_hourly, 'fetch_bars', side_effect=fetch):
            queue.drain(self.db, {}, self.root)
        self.assertEqual(calls, ['FIRST', 'FAIL', 'LATE'])
        states = {r['symbol']: r for r in queue.status(self.root)['requests']}
        self.assertEqual(states['FIRST']['status'], 'complete')
        self.assertEqual(states['FAIL']['status'], 'failed')
        self.assertEqual(states['LATE']['rows'], 1)
        self.assertIsNone(self.db.execute("SELECT last_success FROM intraday_state WHERE symbol='FIRST'").fetchone()[0])

    def test_hourly_priority_before_regular_fetch(self):
        self.enqueue('URGENT')
        config = dict(symbols=['NORMAL'], hourly_workers=1, hourly_pause_seconds=0)
        (self.root / 'config.json').write_text(json.dumps(config))
        order=[]
        def priority(*args, **kwargs):
            order.append('priority');return [], 0, [], 'Unavailable'
        def normal(*args, **kwargs):
            order.append('normal');return [], 0, [], 'Unavailable'
        with patch.object(refresh_hourly, 'ROOT', self.root), patch.object(refresh_hourly, 'DATA', self.root/'data'), \
             patch.object(refresh_hourly, 'fetch_bars', side_effect=priority), patch.object(refresh_hourly, 'fetch_hourly', side_effect=normal):
            refresh_hourly.refresh()
        self.assertEqual(order, ['priority', 'normal'])

    def test_daily_drains_even_before_budget_check(self):
        self.enqueue('URGENT')
        with patch.object(market_data, 'ROOT', self.root), patch.object(refresh_hourly, 'fetch_bars', return_value=([],0,[],'Unavailable')) as fetch, \
             patch.object(market_data.time, 'monotonic', side_effect=[0, 100]):
            market_data.ingest(self.db, {}, ['NORMAL'], budget_minutes=1)
        fetch.assert_called_once()
        self.assertEqual(queue.status(self.root)['requests'][0]['status'], 'failed')

    def test_native_requested_bars_are_visible_in_chart(self):
        import dashboard
        stamp = '2026-01-02T12:00:00+00:00'
        market_data.persist_intraday(self.db, 'TEST', '5m', [('TEST', stamp, '5m', 1, 2, 1, 2, 2, 10, 'USD', 'NMS', 'UTC', stamp)], stamp, 0)
        with patch.object(dashboard, 'DATABASE', self.root/'data'/'market.sqlite'):
            bars = dashboard.history({'symbol':['TEST'], 'interval':['5m']})
            assets = dashboard.catalog()
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0]['date'], stamp)
        self.assertEqual(assets[0]['quote_interval'], '5m')
        self.assertFalse(assets[0]['has_daily'])

    def test_http_enqueue_and_origin_boundary(self):
        from dashboard import Handler
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with patch.object(queue, 'enqueue', return_value={'id':1,'status':'queued'}) as enqueue, patch.object(queue, 'start_worker') as worker:
                for origin, expected in [('https://evil.example',403), (f'http://127.0.0.1:{server.server_port}',200)]:
                    client = http.client.HTTPConnection('127.0.0.1', server.server_port)
                    client.request('POST', '/api/pull-queue', json.dumps({'symbol':'AAPL'}), {'Content-Type':'application/json','Origin':origin})
                    response=client.getresponse();response.read();self.assertEqual(response.status,expected);client.close()
                enqueue.assert_called_once_with({'symbol':'AAPL'});worker.assert_called_once()
        finally:
            server.shutdown();server.server_close()


if __name__ == '__main__':
    unittest.main()
