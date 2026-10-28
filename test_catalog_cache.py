"""Catalog reuse must never hide committed quotes or settings changes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import dashboard


class CatalogCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'data').mkdir()
        self.path = self.root / 'data' / 'market.sqlite'
        for target, value in [('ROOT', self.root), ('DATABASE', self.path),
                              ('_catalog_cache', None)]:
            self.enterContext(patch.object(dashboard, target, value))

    def test_parallel_requests_share_serialized_result(self):
        with patch.object(dashboard, 'catalog', return_value=[{'symbol': 'BTC-USD'}]) as build:
            with ThreadPoolExecutor(max_workers=6) as pool:
                responses = list(pool.map(lambda _: dashboard.catalog_response(), range(12)))
            build.assert_called_once()
            self.assertTrue(all(result is responses[0] for result in responses))
            self.assertEqual(json.loads(responses[0]), [{'symbol': 'BTC-USD'}])

    def test_wal_commit_invalidates_without_database_checkpoint(self):
        db = sqlite3.connect(self.path)
        self.addCleanup(db.close)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE quotes (close REAL)')
        db.execute('INSERT INTO quotes VALUES (1)')
        db.commit()
        def build():
            return [{'close': db.execute('SELECT close FROM quotes').fetchone()[0]}]
        with patch.object(dashboard, 'catalog', side_effect=build) as load:
            self.assertEqual(json.loads(dashboard.catalog_response()), [{'close': 1}])
            original = self.path.stat().st_mtime_ns
            db.execute('UPDATE quotes SET close=2')
            db.commit()
            self.assertEqual(self.path.stat().st_mtime_ns, original)
            self.assertEqual(json.loads(dashboard.catalog_response()), [{'close': 2}])
            self.assertEqual(load.call_count, 2)

    def test_config_and_names_changes_invalidate(self):
        with patch.object(dashboard, 'catalog', return_value=[]) as build:
            dashboard.catalog_response()
            (self.root / 'config.json').write_text('{"symbols":["NEW"]}')
            dashboard.catalog_response()
            (self.root / 'data' / 'constituents.csv').write_text('Symbol,Security\nNEW,New name')
            dashboard.catalog_response()
            self.assertEqual(build.call_count, 3)

    def test_expiration_rebuilds_even_if_file_metadata_is_unchanged(self):
        with patch.object(dashboard, 'catalog', return_value=[]) as build, \
             patch.object(dashboard.time, 'monotonic', return_value=0) as clock:
            dashboard.catalog_response()
            clock.return_value = 29
            dashboard.catalog_response()
            clock.return_value = 30
            dashboard.catalog_response()
            self.assertEqual(build.call_count, 2)

    def test_change_during_build_is_not_cached(self):
        with patch.object(dashboard, 'catalog', return_value=[]) as build, \
             patch.object(dashboard, 'catalog_revision', side_effect=['old', 'new', 'new', 'new']):
            dashboard.catalog_response()
            dashboard.catalog_response()
            self.assertEqual(build.call_count, 2)

    def test_failed_build_is_retried(self):
        with patch.object(dashboard, 'catalog', side_effect=[ValueError('bad data'), []]):
            with self.assertRaises(ValueError):
                dashboard.catalog_response()
            self.assertEqual(dashboard.catalog_response(), b'[]')
