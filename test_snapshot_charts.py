import csv
import tempfile
import unittest
from pathlib import Path

from local_scheduler import resource_profile, LOW
from market_data import connect
from snapshot_charts import snapshot


class SnapshotTests(unittest.TestCase):
    def test_profiles_do_not_mutate_defaults(self):
        low = resource_profile('low')
        low['hourly_workers'] = 9
        self.assertEqual(LOW['hourly_workers'], 1)
        self.assertGreater(resource_profile('high')['hourly_workers'], LOW['hourly_workers'])
        with self.assertRaises(ValueError):
            resource_profile('invalid')

    def test_bounded_deterministic_snapshot_excludes_minute_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'market.sqlite'
            db = connect(database)
            db.execute("INSERT INTO state(symbol) VALUES ('TEST')")
            for day in range(1, 41):
                db.execute('INSERT INTO prices(symbol,date,close) VALUES (?,?,?)', ('TEST', f'2026-{day:03}', day))
            for hour in range(60):
                for interval in ('60m', '1m'):
                    db.execute('INSERT INTO intraday_prices(symbol,timestamp,interval,close) VALUES (?,?,?,?)',
                               ('TEST', f'2026-{hour:03}', interval, hour))
            db.commit()
            db.close()
            output = root / 'snapshots'
            self.assertEqual(snapshot(database, output), {'daily': 30, 'hourly': 48})
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            snapshot(database, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            rows = []
            for path in output.glob('daily-*.csv'):
                with path.open(newline='') as handle:
                    rows.extend(csv.DictReader(handle))
            self.assertEqual(float(rows[0]['close']), 11)
