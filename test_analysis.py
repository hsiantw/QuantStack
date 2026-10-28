"""Run with .venv-analysis: real research libraries on isolated market fixtures."""
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis_jobs import Jobs, validate
from analysis_worker import load_bars, run
from analysis_history import source_history
from market_data import connect


def fixture(path):
    with connect(path) as db:
        for j, symbol in enumerate(['AAA', 'BBB', 'CCC', 'DDD', 'EEE']):
            rows = []
            for i in range(260):
                day = (datetime(2024, 1, 1) + timedelta(days=i)).date().isoformat()
                close = 100 + i * (.03 + .01 * j) + math.sin(i / (5 + j)) * (6 + j)
                op = close * .997
                rows.append((symbol, day, op, close * 1.01, op * .99, close, close, 1000., 0, 0, 'USD', 'TEST', 'UTC', day, 'Yahoo Finance'))
            db.executemany('INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
            db.execute('INSERT INTO state(symbol) VALUES (?)', (symbol,))
    db.close()


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / 'market.sqlite'
        fixture(self.db)
        self.base = dict(kind='strategy', symbols=['AAA'], source='Yahoo Finance', interval='1d', start='2024-01-01', end='2024-09-16')

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_rejects_unbounded_or_ambiguous_requests(self):
        for changes in [dict(symbols=['../x']), dict(kind='orders'), dict(fast=True), dict(slow=2),
                        dict(source='https://example.com'), dict(interval='tick'), dict(start='2000-01-01'),
                        dict(symbols=['AAA', 'AAA']), dict(kind='portfolio', symbols=['AAA','BTC-USD'],weights=[50,50]),
                        dict(kind='portfolio', weights=[80]), dict(kind='factor',symbols=['AAA']),
                        dict(kind='exchange', source='Kraken', symbols=['BTC-USDT'])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(dict(self.base, **changes))

    def test_strategy_real_engine_holdout_and_costs(self):
        request = dict(self.base, fast=4, slow=12)
        result = run(request, self.db)
        self.assertEqual(result['engines'], ['vectorbt', 'quantstats'])
        self.assertGreater(result['trade_count'], 0)
        self.assertGreaterEqual(result['evaluation_start'], '2024-06-30')
        self.assertTrue(all(trade['Entry Timestamp'].isoformat() >= result['evaluation_start'] if hasattr(trade['Entry Timestamp'], 'isoformat') else trade['Entry Timestamp'] >= result['evaluation_start'] for trade in result['trades']))
        self.assertGreater(run(dict(request,fee_bps=0,slippage_bps=0),self.db)['net_return'], result['net_return'])
        json.dumps(result, allow_nan=False)

    def test_daily_adjustment_and_read_only_database(self):
        with connect(self.db) as db, db:
            db.execute("UPDATE prices SET adjusted_close=close/2 WHERE symbol='AAA'")
        db.close()
        frame, audit = load_bars(validate(self.base), 'AAA', self.db)
        self.assertAlmostEqual(frame.iloc[0].close, 50)
        self.assertAlmostEqual(frame.iloc[0].open, 49.85)
        self.assertTrue(audit['adjusted'])
        with connect(self.db) as db:
            self.assertEqual(db.execute("SELECT close FROM prices WHERE symbol='AAA' ORDER BY date LIMIT 1").fetchone()[0],100)
        db.close()

    def test_portfolio_buys_initial_weights_without_implicit_rebalancing(self):
        request = dict(self.base, kind='portfolio', symbols=['AAA','BBB'], weights=[60,40])
        result = run(request,self.db)
        a,_=load_bars(validate(request),'AAA',self.db)
        b,_=load_bars(validate(request),'BBB',self.db)
        expected=(.6*a.close.iloc[-1]/a.close.iloc[0]+.4*b.close.iloc[-1]/b.close.iloc[0])*10000
        self.assertAlmostEqual(result['curve'][-1]['value'],expected)
        self.assertEqual(result['metrics']['annualization'],252)

    def test_factor_real_alphalens_outputs_five_quintiles(self):
        result=run(dict(self.base,kind='factor',symbols=['AAA','BBB','CCC','DDD','EEE']),self.db)
        self.assertEqual(len(result['quintiles']),5)
        self.assertGreater(result['metrics']['ic_observations'],100)
        self.assertTrue(-1 <= result['metrics']['mean_rank_ic'] <= 1)

    def test_exchange_validates_stores_fractional_volume_and_keeps_sources_separate(self):
        stamp=int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()*1000)
        class Exchange:
            markets={'ETH/USD':{'spot':True}}
            has={'fetchOHLCV':True}
            timeframes={'1m':'1m'}
            def load_markets(self): pass
            def fetch_ohlcv(self,*args,**kwargs):
                return [[stamp+i*60000,10,12,9,11+i*.1,.125] for i in range(3)] if kwargs['since']==stamp else []
        request=dict(self.base,kind='exchange',symbols=['ETH-USD'],source='Bitstamp',interval='1m',end='2024-01-01')
        with patch('ccxt.bitstamp', return_value=Exchange()):
            result=run(request,self.db)
        self.assertEqual(result['stored_rows'],3)
        query={key:[str(value)] for key,value in dict(symbol='ETH-USD',candle_source='Bitstamp',interval='1m',start='2024-01-01',end='2024-01-01').items()}
        bars=source_history(self.db,query)
        self.assertEqual(bars[0]['volume'],.125)
        self.assertEqual(source_history(self.db,dict(query,candle_source=['Kraken'])),[])
        loaded,audit=load_bars(validate(dict(request,kind='strategy')),'ETH-USD',self.db)
        self.assertEqual(len(loaded),3)
        self.assertEqual(audit['missing_grid_slots'],0)

    def test_job_cancel_and_restart_mark_interrupted_jobs(self):
        manager=Jobs(self.root)
        manager.python=Path(__file__)
        with patch('threading.Thread.start'):
            item=manager.submit(self.base)
        self.assertEqual(manager.cancel(item['id'])['status'],'cancelled')
        with patch('threading.Thread.start'):
            interrupted=manager.submit(self.base)
        recovered=Jobs(self.root)
        self.assertEqual(recovered.get(interrupted['id'])['status'],'failed')
        self.assertEqual(recovered.get(item['id'])['status'],'cancelled')
        with self.assertRaises(ValueError): recovered.get('../../config.json')


if __name__=='__main__':
    unittest.main()
