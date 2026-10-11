import json
from pathlib import Path
import tempfile
import unittest

from snapshot_catalog import snapshot_catalog


class SnapshotCatalogTests(unittest.TestCase):
    def test_configuration_expands_old_snapshot_without_overwriting_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'symbols': ['BTC-USD', 'SOL-USD', 'SOL-USD']}))
            bitcoin = dict(symbol='BTC-USD', name='Bitcoin', close=100, has_daily=True, kind='Crypto')
            assets = snapshot_catalog([bitcoin], path)
            self.assertEqual(len(assets), 2)
            self.assertEqual(assets[0], bitcoin)
            self.assertEqual(assets[1]['kind'], 'Crypto')
            self.assertFalse(assets[1]['has_data'])
            self.assertIsNone(assets[1]['close'])
            self.assertEqual(assets[1]['quote_interval'], '1d')

    def test_old_snapshot_crypto_classification_is_repaired(self):
        original = dict(symbol='SOL-USD', name='Solana', kind='Stocks', close=100, has_daily=True)
        with tempfile.TemporaryDirectory() as directory:
            asset = snapshot_catalog([original], Path(directory) / 'missing.json')[0]
        self.assertEqual(asset['kind'], 'Crypto')
        self.assertEqual(asset['close'], 100)
        self.assertEqual(original['kind'], 'Stocks')

    def test_intraday_quotes_are_not_advertised_as_available_daily_history(self):
        original = dict(symbol='TEST', name='Test', has_daily=False, has_data=True,
                        close=10, quote_interval='1h', change=2)
        with tempfile.TemporaryDirectory() as directory:
            asset = snapshot_catalog([original], Path(directory) / 'missing.json')[0]
        self.assertFalse(asset['has_data'])
        self.assertIsNone(asset['close'])
        self.assertIsNone(asset['change'])
        self.assertEqual(original['close'], 10)


if __name__ == '__main__':
    unittest.main()
