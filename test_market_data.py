import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from datetime import date
from pathlib import Path
import pandas as pd
from market_data import connect, normalize, normalize_intraday, persist, persist_intraday, ingest

class StorageTests(unittest.TestCase):
    def test_expansion_cutoff_passed_without_weakening_validation(self):
        db = connect(':memory:')
        config = {'attempts': 1, 'full_refresh_days': 30, 'overlap_days': 14, 'request_pause_seconds': 0}
        frame = pd.DataFrame({'Open': [1.], 'High': [2.], 'Low': [1.], 'Close': [2.], 'Volume': [1]},
                             index=pd.date_range('2020-01-01', periods=1, tz='UTC'))
        with patch('yfinance.Ticker') as ticker:
            ticker.return_value.history.return_value = frame
            ticker.return_value.get_history_metadata.return_value = {'exchangeTimezoneName': 'UTC'}
            self.assertEqual(ingest(db, config, ['TEST'], end='2020-01-02'), [])
            self.assertEqual(ticker.return_value.history.call_args.kwargs['end'], '2020-01-02')
            self.assertEqual(db.execute('SELECT count(*) FROM prices').fetchone()[0], 1)
        db.close()

    def test_resume_and_retry_failed_symbol(self):
        db = connect(':memory:')
        now = datetime.now(timezone.utc).isoformat()
        with db:
            db.execute('INSERT INTO state VALUES (?,?,?,?)', ('TEST', now, now, None))
        config = {'attempts': 2, 'full_refresh_days': 30, 'overlap_days': 14, 'request_pause_seconds': 0}
        with patch('yfinance.Ticker', side_effect=RuntimeError('provider offline')) as ticker, patch('market_data.time.sleep'):
            self.assertEqual(ingest(db, config, ['TEST']), [])
            ticker.assert_not_called()
            with db:
                db.execute("UPDATE state SET error='previous failure'")
            self.assertEqual(ingest(db, config, ['TEST']), ['TEST'])
            self.assertEqual(ticker.call_count, 2)
            self.assertEqual(db.execute('SELECT last_success,error FROM state').fetchone(), (now, 'provider offline'))
        db.close()

    def test_upsert_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            db = connect(Path(folder) / 'test.sqlite')
            row = ('TEST', '2026-01-01', 1., 2., 1., 2., 2., 10, 0., 0., 'USD', 'NMS', 'America/New_York', 'now')
            persist(db, 'TEST', [row], True, '2026-01-02T00:00:00+00:00', 0)
            updated = list(row)
            updated[6] = 1.5
            persist(db, 'TEST', [updated], False, '2026-01-03T00:00:00+00:00', 0)
            self.assertEqual(db.execute('SELECT COUNT(*),adjusted_close FROM prices').fetchone(), (1, 1.5))
            self.assertEqual(db.execute('SELECT last_full FROM state').fetchone()[0], '2026-01-02T00:00:00+00:00')
            bad = list(row)
            bad[0] = None
            with self.assertRaises(sqlite3.IntegrityError):
                persist(db, 'TEST', [row, bad], True, 'bad', 0)
            self.assertEqual(db.execute('SELECT adjusted_close FROM prices').fetchone()[0], 1.5)
            db.close()

    def test_current_and_missing_bars(self):
        frame = pd.DataFrame({'Open': [1, float('nan'), 3], 'High': [2, float('nan'), 4], 'Low': [1, float('nan'), 3], 'Close': [2, float('nan'), 4], 'Volume': [1, 0, 1]}, index=pd.date_range('2026-01-01', periods=3, tz='America/New_York'))
        rows, empty = normalize('TEST', frame, {}, date(2026, 1, 3))
        self.assertEqual((len(rows), empty), (1, 1))
        frame.iloc[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            normalize('TEST', frame, {}, date(2026, 1, 3))

    def test_intraday_normalize_and_upsert(self):
        db = connect(':memory:')
        frame = pd.DataFrame(
            {'Open': [1.0, 2.0], 'High': [2.0, 3.0], 'Low': [0.5, 1.5],
             'Close': [1.5, 2.5], 'Adj Close': [1.5, 2.5], 'Volume': [10, 20]},
            index=pd.date_range('2026-01-01T10:00:00', periods=2, freq='min', tz='America/New_York'))
        meta = {'currency': 'USD', 'exchangeName': 'NMS', 'exchangeTimezoneName': 'America/New_York'}
        rows, empty = normalize_intraday(
            'TEST', frame, meta, '1m', datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc))
        self.assertEqual((len(rows), empty), (2, 0))
        self.assertEqual(rows[0][1], '2026-01-01T15:00:00+00:00')
        persist_intraday(db, 'TEST', '1m', rows, '2026-01-01T16:00:00+00:00', empty)
        changed = list(rows[0])
        changed[6] = 9.0
        persist_intraday(db, 'TEST', '1m', [changed], '2026-01-01T16:01:00+00:00', 0)
        self.assertEqual(db.execute('SELECT COUNT(*),close FROM intraday_prices').fetchone(), (2, 9.0))
        db.close()

    def test_hourly_excludes_current_bar_and_preserves_session_alignment(self):
        frame = pd.DataFrame({'Open': [10., 11.], 'High': [12., 13.], 'Low': [9., 10.],
                              'Close': [11., 12.], 'Adj Close': [11., 12.], 'Volume': [100, 200]},
                             index=pd.date_range('2026-01-02T09:30:00', periods=2, freq='h', tz='America/New_York'))
        rows, _ = normalize_intraday('TEST', frame, {}, '60m', datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1:3], ('2026-01-02T14:30:00+00:00', '60m'))

if __name__ == '__main__':
    unittest.main()
