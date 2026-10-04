import json
import tempfile
import unittest
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from local_scheduler import enroll, local_request, normalize_symbols, status, LOW


class SchedulerTests(unittest.TestCase):
    def test_collection_reports_are_optional_and_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'symbols': ['AAPL']}))
            self.assertIsNone(status(root)['collection']['hourly'])
            (root / 'data').mkdir()
            (root / 'data' / 'hourly-refresh.json').write_text(json.dumps({
                'status': 'budget_exhausted', 'pending': 8, 'succeeded': 2,
                'failed': {'BAD': 'provider error'}, 'private': 'not exposed'}))
            report = status(root)['collection']['hourly']
            self.assertEqual((report['pending'], report['failed']), (8, 1))
            self.assertNotIn('private', report)
            (root / 'data' / 'daily-refresh.json').write_text('partial JSON')
            self.assertIsNone(status(root)['collection']['daily'])

    def test_http_write_boundary(self):
        from dashboard import Handler
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with patch('local_scheduler.enroll', return_value={'added': ['MSFT']}) as save:
                for origin, mime, expected in [('http://evil.example', 'application/json', 403),
                                               (None, 'text/plain', 403),
                                               (None, 'application/json', 200)]:
                    client = HTTPConnection('127.0.0.1', server.server_port)
                    headers = {'Content-Type': mime}
                    if origin:
                        headers['Origin'] = origin
                    client.request('POST', '/api/local-scheduler', '{"symbols":"MSFT"}', headers)
                    response = client.getresponse()
                    self.assertEqual(response.status, expected)
                    response.read()
                    client.close()
                save.assert_called_once_with({'symbols': 'MSFT'})
        finally:
            server.shutdown()
            server.server_close()

    def test_symbols(self):
        self.assertEqual(normalize_symbols('btc, ETH btc-USD', 'crypto'), ['BTC-USD', 'ETH-USD'])
        self.assertEqual(normalize_symbols('aapl; 2330.tw BRK-B', 'stocks'), ['AAPL', '2330.TW', 'BRK-B'])
        for value in ('../../file', '<script>', 'A' * 31):
            with self.assertRaises(ValueError):
                normalize_symbols(value, 'stocks')

    def test_local_boundary(self):
        self.assertTrue(local_request('127.0.0.1', 'localhost:8501', 'http://localhost:8501'))
        self.assertFalse(local_request('10.0.0.1', 'localhost:8501'))
        self.assertFalse(local_request('127.0.0.1', 'evil.example'))
        self.assertFalse(local_request('127.0.0.1', 'localhost:8501', 'http://evil.example'))

    def test_additive_atomic_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'config.json'
            path.write_text(json.dumps({'symbols': ['AAPL'], 'sp500': True, 'custom': 42}))
            result = enroll({'symbols': 'ETH BTC', 'kind': 'crypto'}, root)
            self.assertEqual(result['added'], ['ETH-USD', 'BTC-USD'])
            self.assertEqual(enroll({'symbols': 'ETH', 'kind': 'crypto'}, root)['added'], [])
            config = json.loads(path.read_text())
            self.assertEqual(config['symbols'], ['AAPL', 'ETH-USD', 'BTC-USD'])
            self.assertEqual(config['custom'], 42)
            self.assertTrue(config['sp500'])
            for key, value in LOW.items():
                self.assertEqual(config[key], value)
            with patch('market_data.process_lock', side_effect=RuntimeError('busy')):
                with self.assertRaises(RuntimeError):
                    enroll({'symbols': 'MSFT'}, root)
            self.assertEqual(json.loads(path.read_text()), config)


if __name__ == '__main__':
    unittest.main()
