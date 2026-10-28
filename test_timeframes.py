"""Check archive OHLCV aggregation, UTC alignment, gaps and page boundaries."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard
from market_data import connect


class TimeframeTests(unittest.TestCase):
    def test_archive_intervals_and_cursors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market.sqlite'
            db = connect(path)
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            with db:
                db.executemany('INSERT INTO crypto_exchange_prices VALUES (?,?,?,?,?,?,?,?,?,?)',
                    [('BTC-USD', (start + timedelta(minutes=i)).isoformat(), '1m', 'Bitstamp',
                      100.+i, 101.+i, 99.+i, 100.5+i, .25, 'now') for i in range(30030)])
            with patch.object(dashboard, 'DATABASE', path):
                for interval, minutes in [('3m',3),('5m',5),('15m',15),('30m',30),('1h',60),('2h',120),('4h',240)]:
                    with self.subTest(interval=interval):
                        rows = dashboard.history({'symbol':['BTC-USD'],'interval':[interval]})
                        self.assertEqual(len(rows), min(1000, 30030//minutes))
                        for bar in (rows[0], rows[-1]):
                            timestamp = datetime.fromisoformat(bar['date'])
                            offset = int((timestamp-start).total_seconds()/60)
                            self.assertEqual(int(timestamp.timestamp()) % (minutes*60), 0)
                            self.assertEqual((bar['open'],bar['high'],bar['low'],bar['close'],bar['volume']),
                                             (100.+offset,100.+offset+minutes,99.+offset,99.5+offset+minutes,.25*minutes))
                            self.assertEqual(bar['source'],'Bitstamp')
                query = {'symbol':['BTC-USD'],'interval':['30m']}
                latest = dashboard.history(query)
                older = dashboard.history(dict(query,before=[latest[0]['date']]))
                self.assertEqual(len(older),1)
                self.assertEqual(dashboard.history(dict(query,after=[older[0]['date']])),latest)
                with db:
                    db.execute("DELETE FROM crypto_exchange_prices WHERE timestamp=?",((start+timedelta(minutes=15)).isoformat(),))
                # Never fabricate a full candle across a missing minute.
                self.assertEqual(dashboard.history(dict(query,before=[latest[0]['date']])),[])
            db.close()


if __name__ == '__main__':
    unittest.main()
