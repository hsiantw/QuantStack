from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard
from market_data import connect, persist


class WatchlistMetricsTests(unittest.TestCase):
    def bars(self):
        return [dict(date=(date(2026, 10, 1)-timedelta(days=i)).isoformat(), close=200-i*2, adjusted_close=100-i) for i in range(31)]

    def test_stock_and_crypto_horizons(self):
        bars=self.bars()
        for crypto, week, month in ((False,5,21),(True,7,30)):
            values=dashboard.daily_performance(bars,crypto)
            self.assertAlmostEqual(values['change_1d_pct'],(100/99-1)*100)
            self.assertAlmostEqual(values['return_1w_pct'],(100/(100-week)-1)*100)
            self.assertAlmostEqual(values['return_1m_pct'],(100/(100-month)-1)*100)
            self.assertEqual(values['performance_basis'],'adjusted')
            self.assertEqual(values['performance_asof'],'2026-10-01')

    def test_missing_history_and_adjustments(self):
        bars=self.bars()
        values=dashboard.daily_performance(bars[:2])
        self.assertIsNone(values['return_1w_pct'])
        self.assertIsNone(values['return_1m_pct'])
        bars[4]['adjusted_close']=None
        values=dashboard.daily_performance(bars)
        self.assertIsNotNone(values['change_1d_pct'])
        self.assertIsNone(values['return_1w_pct'])
        self.assertIsNone(values['return_1m_pct'])
        for bar in bars:bar['adjusted_close']=None
        self.assertEqual(dashboard.daily_performance(bars)['performance_basis'],'unadjusted')
        self.assertIsNotNone(dashboard.daily_performance(bars)['return_1m_pct'])
        split=[dict(date='2026-10-01',close=50,adjusted_close=50),dict(date='2026-09-30',close=100,adjusted_close=50)]
        self.assertEqual(dashboard.daily_performance(split)['change_1d_pct'],0)

    def test_catalog_reports_market_cap_and_daily_metrics_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'market.sqlite'
            db=connect(path)
            rows=[('TEST',bar['date'],bar['close'],bar['close'],bar['close'],bar['close'],bar['adjusted_close'],1000,0,0,'USD','TEST','UTC','now') for bar in self.bars()]
            persist(db,'TEST',rows,True,'now',0)
            db.execute('CREATE TABLE stock_metadata (symbol TEXT,name TEXT,market_cap REAL,currency TEXT,fetched_at TEXT)')
            db.execute("INSERT INTO stock_metadata VALUES ('TEST','Test asset',1500000000,'USD','2026-09-30')")
            db.commit();db.close()
            with patch.object(dashboard,'DATABASE',path):
                asset=dashboard.catalog()[0]
            self.assertEqual(asset['market_cap'],1500000000)
            self.assertEqual(asset['market_cap_currency'],'USD')
            self.assertEqual(asset['market_cap_asof'],'2026-09-30')
            self.assertEqual(asset['volume'],1000)
            self.assertAlmostEqual(asset['return_1m_pct'],(100/79-1)*100)


if __name__=='__main__':unittest.main()
