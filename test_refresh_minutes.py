import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import refresh_minutes as minute
from market_data import connect


class MinuteTests(unittest.TestCase):
    def test_catalog_includes_archived_symbols_and_rotates_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / 'market.sqlite')
            with db:
                db.execute("INSERT INTO state(symbol) VALUES ('OLD')")
                db.execute("INSERT INTO intraday_state(symbol,interval) VALUES ('ETH-USD','60m')")
                db.execute("INSERT INTO attempts(symbol,started,mode,status,error) VALUES ('OLD','2026-10-26','catalog-1m','failed','unavailable')")
            self.assertEqual(minute.jobs(db, {'symbols': ['BTC-USD', 'NEW'], 'collection_symbols_only': True}), ['ETH-USD', 'NEW', 'OLD'])
            db.close()

    def test_range_clamps_stale_history_and_overlaps_recent_bars(self):
        now = datetime(2026, 10, 26, tzinfo=timezone.utc)
        self.assertEqual(minute.range_start(None, now), now - timedelta(days=7))
        self.assertEqual(minute.range_start((now - timedelta(days=30)).isoformat(), now), now - timedelta(days=7))
        self.assertEqual(minute.range_start((now - timedelta(minutes=5)).isoformat(), now), now - timedelta(minutes=7))

    def test_rate_limit_stops_batch_and_records_attempt_for_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'symbols': ['A', 'B']}))
            with patch.object(minute, 'ROOT', root), patch.object(minute, 'DATA', root), \
                 patch.object(minute, 'fetch_bars', return_value=([], 0, [], '429 Too many requests')) as fetch, \
                 patch('local_scheduler.lower_priority'):
                minute.run()
            self.assertEqual(fetch.call_count, 1)
            report = json.loads((root / 'minute-refresh.json').read_text())
            self.assertEqual(report['status'], 'rate_limited')
            db = connect(root / 'market.sqlite')
            self.assertEqual(minute.jobs(db, {'symbols': ['A', 'B']}), ['B', 'A'])
            db.close()


if __name__ == '__main__':
    unittest.main()
