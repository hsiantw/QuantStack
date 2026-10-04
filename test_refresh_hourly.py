import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from unittest.mock import patch

import refresh_hourly as hourly
import market_data
from market_data import connect, persist_intraday


class HourlyTests(unittest.TestCase):
    def test_existing_intraday_entrypoint_routes_to_hourly_only(self):
        with patch('sys.argv', ['market_data.py', 'intraday', '--symbols', 'AAPL', '--full']), patch.object(hourly, 'refresh', return_value=0) as refresh, patch.object(market_data, 'ingest_intraday') as minute, patch.object(market_data, 'ingest_crypto_sources') as crypto:
            self.assertEqual(market_data.main(), 0)
            refresh.assert_called_once_with(symbols=['AAPL'], full=True)
            minute.assert_not_called()
            crypto.assert_not_called()

    def test_capacity_includes_elapsed_time_and_headroom(self):
        progress = dict(total=5000, completed=600, skipped=100, succeeded=480,
                        requests=530, request_failures=50, rate_limit_events=0)
        result = hourly.capacity(progress, 120, 60, 55)
        self.assertEqual(result['symbols_per_minute'], 240)
        self.assertEqual(result['estimated_symbols_per_cycle'], 14400)
        self.assertEqual(result['first_symbol_count_over_cycle'], 14401)
        self.assertEqual(result['planning_symbols_per_cycle'], 10560)
        self.assertEqual(result['estimated_total_refresh_minutes'], 20)
        self.assertIsNone(hourly.capacity(dict(progress, succeeded=0), 0, 60, 55)['first_symbol_count_over_cycle'])

    def test_rate_limit_stops_retries(self):
        stop = Event()
        with patch('yfinance.Ticker', side_effect=RuntimeError('Too Many Requests. Rate limited.')) as ticker, patch('refresh_hourly.time.sleep'):
            rows, _, events, error = hourly.fetch_hourly('TEST', None, None, stop=stop)
        self.assertTrue(stop.is_set())
        self.assertEqual(ticker.call_count, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], 'failed')
        self.assertGreaterEqual(events[0][4], 0)
        self.assertEqual(rows, [])

    def test_refresh_expands_preserves_history_and_resumes_one_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(dict(symbols=['CORE'], stock_universe_limit=2, hourly_workers=1)))
            (root / 'stock-universe.json').write_text(json.dumps(dict(symbols=['NEW', 'CORE', 'OUTSIDE'])))
            db = connect(root / 'market.sqlite')
            recent = datetime.now(timezone.utc)
            old = (recent - timedelta(days=2)).isoformat()
            row = ('CORE', old, '60m', 1., 2., 1., 2., 2., 10, 'USD', 'NMS', 'UTC', old)
            persist_intraday(db, 'CORE', '60m', [row], old, 0)
            db.close()
            calls = []
            def fetch(symbol, start, end, *args):
                calls.append((symbol, start))
                current = (end-timedelta(hours=2)).isoformat()
                new_row = (symbol, current, '60m', 1., 2., 1., 2., 2., 10, 'USD', 'NMS', 'UTC', current)
                return [new_row], 0, [((current, time.perf_counter()-5), 'ok', 1, None, 17)], None
            with patch.object(hourly, 'ROOT', root), patch.object(hourly, 'DATA', root), patch.object(hourly, 'fetch_hourly', side_effect=fetch):
                self.assertEqual(hourly.refresh(), 0)
                report = json.loads((root/'hourly-refresh.json').read_text())
                self.assertEqual((report['succeeded'], report['new_symbols']), (2, 1))
                self.assertEqual(calls[0][0], 'NEW')
                self.assertEqual(hourly.refresh(resume=True), 0)
                self.assertEqual(len(calls), 2)
                db = connect(root/'market.sqlite')
                self.assertEqual(db.execute('SELECT COUNT(*) FROM intraday_prices').fetchone()[0], 3)
                self.assertEqual(db.execute('SELECT DISTINCT duration_ms FROM api_requests').fetchall(), [(17,)])
                db.close()
                # Default cycles must refresh even when the previous run was recent.
                self.assertEqual(hourly.refresh(), 0)
                self.assertEqual(len(calls), 4)

    def test_limit_leaves_unattempted_symbols_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps(dict(symbols=['A','B','C'], hourly_workers=1)))
            def fetch(*args):
                args[-1].set()
                return [], 0, [((datetime.now(timezone.utc).isoformat(), time.perf_counter()), 'failed', None, 'HTTP 429', 12)], 'HTTP 429'
            with patch.object(hourly, 'ROOT', root), patch.object(hourly, 'DATA', root), patch.object(hourly, 'fetch_hourly', side_effect=fetch) as mock:
                self.assertEqual(hourly.refresh(), 1)
                self.assertEqual(mock.call_count, 1)
                result = json.loads((root/'hourly-refresh.json').read_text())
                self.assertEqual((result['status'], result['pending'], result['rate_limit_events']), ('rate_limited', 2, 1))
                # The failed first ticker must not monopolize every later cycle.
                self.assertEqual(hourly.refresh(), 1)
                self.assertEqual(mock.call_args.args[0], 'B')


if __name__ == '__main__':
    unittest.main()
