import tempfile
import json
import threading
import unittest
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import dashboard
from liquidations import force_order_rows, persist_force_orders, read_liquidations
from market_data import connect


def event(order_id, side, event_time, quantity, price, status='FILLED', symbol='BTCUSDT'):
    timestamp = int(datetime.fromisoformat(event_time).replace(tzinfo=timezone.utc).timestamp() * 1000)
    return {
        'e': 'forceOrder',
        'E': timestamp,
        'o': {
            's': symbol, 'S': side, 'X': status, 'i': order_id,
            'T': timestamp, 'z': str(quantity), 'q': str(quantity),
            'ap': str(price), 'p': str(price),
        },
    }


class LiquidationTests(unittest.TestCase):
    def test_normalizes_only_filled_configured_usdt_crypto_orders(self):
        payload = [
            event(1, 'SELL', '2026-01-05T10:04:05', 2, 100),
            event(2, 'BUY', '2026-01-05T10:04:06', 3, 110),
            event(3, 'SELL', '2026-01-05T10:04:07', 1, 100, status='NEW'),
            event(4, 'SELL', '2026-01-05T10:04:08', 1, 100, symbol='ETHUSDT'),
            event(5, 'SELL', '2026-01-05T10:04:09', 1, 100, symbol='BTCBUSD'),
        ]
        rows = force_order_rows(payload, {'BTC-USD'})
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0][0], rows[0][2], rows[0][3], rows[0][6], rows[0][7]),
                         ('BTC-USD', '1', 'long', 200., 'USDT'))
        self.assertEqual((rows[1][3], rows[1][6]), ('short', 330.))

    def test_persists_deduplicated_events_and_groups_by_week(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'market.sqlite'
            rows = force_order_rows([
                event(1, 'SELL', '2026-01-05T10:04:05', 2, 100),
                event(2, 'BUY', '2026-01-06T11:04:06', 3, 110),
            ], {'BTC-USD'})
            self.assertEqual(persist_force_orders(database, rows), 2)
            self.assertEqual(persist_force_orders(database, rows), 0)
            result = read_liquidations(
                database, 'BTC-USD', '2026-01-05T00:00:00+00:00',
                '2026-01-12T00:00:00+00:00', '1w')
            self.assertEqual(result['venue'], 'Binance USD-M Futures')
            self.assertEqual(result['event_count'], 2)
            self.assertEqual(result['bars'], [{
                'date': '2026-01-05', 'longs': 200., 'shorts': 330., 'count': 2,
            }])
            self.assertIn('partial', result['coverage'])

    def test_requires_configured_crypto_pair_and_supported_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'market.sqlite'
            with self.assertRaises(ValueError):
                read_liquidations(database, 'AAPL', '2026-01-01', '2026-01-02', '1d')
            with self.assertRaises(ValueError):
                read_liquidations(database, 'BTC-USD', '2026-01-01', '2026-01-02', '1q')

    def test_dashboard_api_exposes_observed_events_and_capture_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'market.sqlite'
            db = connect(database)
            db.close()
            rows = force_order_rows([
                event(9, 'SELL', '2026-01-05T10:04:05', 2, 100),
            ], {'BTC-USD'})
            persist_force_orders(database, rows)
            from liquidations import update_collector_state
            update_collector_state(database, 'connected', started_at='2026-01-05T10:00:00+00:00')
            with patch.object(dashboard, 'DATABASE', database):
                server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.Handler)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    client = HTTPConnection('127.0.0.1', server.server_port)
                    client.request('GET', '/api/liquidations?symbol=BTC-USD&interval=1d&start=2026-01-05&end=2026-01-05')
                    response = client.getresponse()
                    self.assertEqual(response.status, 200)
                    result = json.loads(response.read())
                    self.assertEqual(result['event_count'], 1)
                    self.assertEqual(result['bars'][0]['longs'], 200.)
                    self.assertEqual(result['collector']['status'], 'connected')
                    client.close()
                finally:
                    server.shutdown()
                    server.server_close()
                    worker.join()


if __name__ == '__main__':
    unittest.main()
