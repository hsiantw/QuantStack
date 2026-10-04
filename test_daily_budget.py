import unittest
from unittest.mock import patch

from market_data import connect, ingest


class DailyBudgetTests(unittest.TestCase):
    def test_budget_prevents_retries_and_rotates_failures(self):
        db = connect(':memory:')
        config = dict(full_refresh_days=30, overlap_days=14, attempts=3, request_pause_seconds=2)
        clock = [0]
        def offline(symbol):
            clock[0] += 61
            raise RuntimeError('offline')
        try:
            with patch('market_data.time.monotonic', side_effect=lambda: clock[0]), patch('market_data.time.sleep'), patch('yfinance.Ticker', side_effect=offline) as ticker:
                progress = {}
                self.assertEqual(ingest(db, config, ['A', 'B'], budget_minutes=1, progress=progress), ['A'])
                self.assertEqual((progress['completed'], progress['pending'], progress['status']), (1, 1, 'budget_exhausted'))
                self.assertEqual(ticker.call_count, 1)
                self.assertEqual(ingest(db, config, ['A', 'B'], budget_minutes=1), ['B'])
                self.assertEqual(ticker.call_count, 2)
        finally:
            db.close()

    def test_zero_budget_is_rejected(self):
        db = connect(':memory:')
        try:
            with self.assertRaises(ValueError):
                ingest(db, {}, [], budget_minutes=0)
        finally:
            db.close()


if __name__ == '__main__':
    unittest.main()
