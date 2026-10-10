import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from refresh_priority import order_symbols


class PriorityTests(unittest.TestCase):
    def test_due_leaders_first_then_fair_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'refresh-priority.json').write_text(json.dumps({'symbols':['BTC-USD','NVDA']}))
            now = datetime.now(timezone.utc)
            attempts = {'NVDA': now.isoformat(), 'BTC-USD': (now-timedelta(days=2)).isoformat(),
                        'OLD': (now-timedelta(days=5)).isoformat()}
            self.assertEqual(order_symbols(['NVDA','OLD','BTC-USD','NEW'], attempts, root, 60),
                             ['BTC-USD','NEW','OLD','NVDA'])

    def test_future_clock_and_missing_priority_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempts = {'A': (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
            self.assertEqual(order_symbols(['A','B'], attempts, root, 60), ['B','A'])
            (root / 'refresh-priority.json').write_text(json.dumps({'symbols':['A']}))
            self.assertEqual(order_symbols(['A','B'], attempts, root, 60), ['A','B'])


if __name__ == '__main__':
    unittest.main()
