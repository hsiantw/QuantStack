import io
import json
import sqlite3
import unittest
from unittest.mock import patch

import btc_backfill as btc
from market_data import universe
from refresh_hourly import tracked_symbols


class BackfillTests(unittest.TestCase):
    def response(self, candles):
        return patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(
            {'data': {'ohlc': candles}}).encode()))

    def candle(self, **changes):
        return dict(timestamp='120', open='10', high='12', low='9', close='11',
                    volume='0.12345678', **changes)

    def test_fractional_volume_and_source(self):
        with self.response([self.candle()]) as request:
            rows = btc.fetch(120, 180)
        self.assertIn('limit=1&', request.call_args.args[0].full_url)
        self.assertEqual(rows[0][0:4], ('BTC-USD', btc.iso(120), '1m', 'Bitstamp'))
        self.assertEqual(rows[0][8], 0.12345678)

    def test_reject_outside_window_and_invalid_prices(self):
        for changes in ({'timestamp': '180'}, {'close': 'NaN'}, {'high': '8'}, {'volume': '-1'}):
            candle = self.candle(); candle.update(changes)
            with self.response([candle]), self.assertRaises(ValueError):
                btc.fetch(120, 180)

    def test_duplicate_rejected(self):
        with self.response([self.candle(), self.candle()]), self.assertRaises(ValueError):
            btc.fetch(120, 180)

    def test_btc_only_does_not_reenroll_archived_symbols(self):
        config = {'collection_symbols_only': True, 'symbols': ['BTC-USD']}
        self.assertEqual(universe(None, config), ['BTC-USD'])
        self.assertEqual(tracked_symbols(None, config), ['BTC-USD'])

    def test_archive_audit_counts_every_matching_candle_and_preserves_bounds(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('''CREATE TABLE crypto_exchange_prices (
                symbol TEXT, timestamp TEXT, interval TEXT, source TEXT,
                PRIMARY KEY(symbol,timestamp,interval,source))''')
            db.executemany('INSERT INTO crypto_exchange_prices VALUES (?,?,?,?)', [
                ('BTC-USD', btc.iso(stamp), interval, source)
                for stamp, interval, source in [(60, '1m', 'Bitstamp'),
                    (120, '1m', 'Bitstamp'), (180, '1m', 'Bitstamp'),
                    (240, '1m', 'Bitstamp'), (120, '1m', 'Other'),
                    (120, '5m', 'Bitstamp')]])
            self.assertEqual(btc.archive_totals(db, 120, 240),
                             (2, btc.iso(120), btc.iso(180)))
            db.execute("DELETE FROM crypto_exchange_prices WHERE timestamp=? AND source='Bitstamp' AND interval='1m'",
                       (btc.iso(120),))
            self.assertEqual(btc.archive_totals(db, 120, 240),
                             (1, btc.iso(180), btc.iso(180)))
            self.assertEqual(btc.archive_totals(db, 300, 360), (0, None, None))


if __name__ == '__main__':
    unittest.main()
