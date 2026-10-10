from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import tempfile

import pandas as pd
from refresh_all import fetch_daily
import refresh_all


class RefreshTests(unittest.TestCase):
    def test_progress_retries_a_brief_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            original = Path.replace
            attempts = []
            def replace(path, destination):
                attempts.append(path)
                if len(attempts) == 1:
                    raise PermissionError('reader still has the old file open')
                return original(path, destination)
            with patch.object(Path, 'replace', autospec=True, side_effect=replace), patch('refresh_all.time.sleep'):
                refresh_all.write_report(target, {'completed':2903})
            self.assertEqual(json.loads(target.read_text()), {'completed':2903})
            self.assertEqual(len(attempts), 2)

    def test_incremental_preserves_adjustments_by_refetching_after_action(self):
        day = datetime.now(timezone.utc).date() - timedelta(days=2)
        frame = pd.DataFrame({'Open':[10.], 'High':[11.], 'Low':[9.], 'Close':[10.],
                              'Adj Close':[9.], 'Volume':[100], 'Dividends':[1.], 'Stock Splits':[0.]},
                             index=pd.DatetimeIndex([day], tz='UTC'))
        ticker = MagicMock()
        ticker.history.return_value = frame
        ticker.get_history_metadata.return_value = {'exchangeTimezoneName':'UTC','currency':'USD','exchangeName':'NMS'}
        with patch('yfinance.Ticker', return_value=ticker):
            rows, empty, full = fetch_daily('TEST', day.isoformat(), datetime.now(timezone.utc).isoformat(),
                                            {'full_refresh_days':30,'overlap_days':14})
        self.assertTrue(full)
        self.assertEqual(rows[0][7], 100)
        self.assertEqual(rows[0][6], 9.)
        self.assertIn('start', ticker.history.call_args_list[0].kwargs)
        self.assertEqual(ticker.history.call_args_list[1].kwargs['period'], 'max')
        self.assertNotIn('start', ticker.history.call_args_list[1].kwargs)

    def test_empty_response_does_not_replace_existing_history(self):
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        ticker.get_history_metadata.return_value = {'exchangeTimezoneName':'UTC'}
        with patch('yfinance.Ticker', return_value=ticker), self.assertRaisesRegex(ValueError, 'No completed'):
            fetch_daily('TEST', None, None, {'full_refresh_days':30,'overlap_days':14})

    def test_rate_limit_stops_new_work_and_preserves_pending_symbols(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({'symbols':['A','B']}))
            with patch.object(refresh_all, 'ROOT', root), patch.object(refresh_all, 'DATA', root), \
                 patch.object(refresh_all, 'universe', return_value=['A','B']), \
                 patch.object(refresh_all, 'tracked_symbols', return_value=[]), \
                 patch.object(refresh_all, 'ranked_symbols', return_value=['A','B']), \
                 patch.object(refresh_all, 'fetch_daily', side_effect=RuntimeError('HTTP 429 Too Many Requests')) as fetch, \
                 patch('refresh_all.time.sleep'):
                self.assertEqual(refresh_all.refresh(workers=1, hourly=False), 1)
                self.assertEqual(fetch.call_count, 1)
            report = json.loads((root / 'all-refresh.json').read_text())
            self.assertEqual(report['status'], 'rate_limited')
            self.assertEqual(report['pending'], 1)
            self.assertEqual(report['successful_symbols'], [])


if __name__ == '__main__':
    unittest.main()
