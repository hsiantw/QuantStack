from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from refresh_all import fetch_daily


class RefreshTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
