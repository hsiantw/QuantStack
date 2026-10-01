"""Offline regression tests for equity discovery, metadata, and resumable collection."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import call, patch

import expand_stocks as expansion
from market_data import connect, persist


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        expansion.metadata_schema(self.db)
        self.addCleanup(self.db.close)

    def metadata(self, symbol='TEST'):
        return dict(self.db.execute('SELECT * FROM stock_metadata WHERE symbol=?', (symbol,)).fetchone())

    def test_quote_fields_are_numeric_and_dividend_yield_is_a_ratio(self):
        quote = dict(longName='Example Corp', shortName='Example', marketCap='1234567890',
                     sector='Technology', industry='Software', country='United States',
                     exchange='NMS', currency='USD', trailingPE='23.5', forwardPE=20,
                     priceToBook='4.75', trailingAnnualDividendYield=0.021,
                     dividendYield=9.9, revenueGrowth=0.18, profitMargins=0.27, beta=1.1)
        with patch.object(expansion, 'now', return_value='2026-09-25T00:00:00+00:00'):
            expansion.write_metadata(self.db, 'TEST', quote)
        row = self.metadata()
        expected = dict(symbol='TEST', name='Example Corp', market_cap=1234567890.0,
                        sector='Technology', industry='Software', country='United States',
                        exchange='NMS', currency='USD', pe=23.5, forward_pe=20.0,
                        pb=4.75, dividend_yield=0.021, revenue_growth=0.18,
                        profit_margin=0.27, beta=1.1,
                        fetched_at='2026-09-25T00:00:00+00:00', source='Yahoo Finance')
        self.assertEqual(row, expected)

    def test_displayed_dividend_percent_converts_once_including_zero(self):
        for index, (value, expected) in enumerate([(4.25, 0.0425), ('2.5', 0.025), (0, 0.0)]):
            with self.subTest(value=value):
                symbol = f'YIELD{index}'
                expansion.write_metadata(self.db, symbol, {'dividendYield': value})
                self.assertEqual(self.metadata(symbol)['dividend_yield'], expected)
        expansion.write_metadata(self.db, 'ZERO', {'trailingAnnualDividendYield': 0, 'dividendYield': 4})
        self.assertEqual(self.metadata('ZERO')['dividend_yield'], 0)

    def test_nonfinite_boolean_and_missing_numeric_values_stay_unknown(self):
        fields = ('marketCap', 'trailingPE', 'forwardPE', 'priceToBook',
                  'trailingAnnualDividendYield', 'revenueGrowth', 'profitMargins', 'beta')
        columns = ('market_cap', 'pe', 'forward_pe', 'pb', 'dividend_yield',
                   'revenue_growth', 'profit_margin', 'beta')
        for index, value in enumerate([None, True, False, float('nan'), float('inf'), '-inf', '', 'unknown']):
            with self.subTest(value=value):
                symbol = f'INVALID{index}'
                expansion.write_metadata(self.db, symbol, dict.fromkeys(fields, value))
                self.assertTrue(all(self.metadata(symbol)[key] is None for key in columns))

    def test_sparse_quote_refresh_preserves_profile_and_valid_prior_metrics(self):
        expansion.write_metadata(self.db, 'TEST', {
            'longName': 'Full Company Name', 'sector': 'Healthcare', 'industry': 'Biotechnology',
            'country': 'United States', 'marketCap': 50_000_000, 'trailingPE': 12.5,
            'profitMargins': 0.31, 'currency': 'USD', 'dividendYield': 2,
        }, profile=True)
        expansion.write_metadata(self.db, 'TEST', {'marketCap': 60_000_000, 'trailingPE': float('nan')})
        row = self.metadata()
        self.assertEqual(row['market_cap'], 60_000_000)
        self.assertEqual((row['name'], row['sector'], row['industry'], row['country']),
                         ('Full Company Name', 'Healthcare', 'Biotechnology', 'United States'))
        self.assertEqual((row['pe'], row['profit_margin'], row['currency'], row['dividend_yield']),
                         (12.5, 0.31, 'USD', 0.02))
        # A known zero is an update, rather than a missing value to preserve.
        expansion.write_metadata(self.db, 'TEST', {'profitMargins': 0, 'trailingAnnualDividendYield': 0})
        self.assertEqual((self.metadata()['profit_margin'], self.metadata()['dividend_yield']), (0, 0))

    def test_successful_profile_clears_retry_error_and_records_success(self):
        self.db.execute('INSERT INTO stock_enrichment VALUES (?,?,?)', ('TEST', None, 'offline'))
        with patch.object(expansion, 'now', return_value='2026-09-25T01:02:03+00:00'):
            expansion.write_metadata(self.db, 'TEST', {'shortName': 'Test'}, profile=True)
        self.assertEqual(tuple(self.db.execute('SELECT * FROM stock_enrichment').fetchone()),
                         ('TEST', '2026-09-25T01:02:03+00:00', None))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='stock-tests-', dir=Path(__file__).resolve().parent / 'data')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.universe = self.root / 'stock-universe.json'
        self.db = sqlite3.connect(':memory:')
        expansion.metadata_schema(self.db)
        self.addCleanup(self.db.close)

    @staticmethod
    def quote(symbol, cap, **extra):
        return dict(symbol=symbol, marketCap=cap, quoteType='EQUITY', **extra)

    def test_discovery_pages_in_market_cap_order_and_deduplicates_provider_results(self):
        pages = [
            {'total': 999, 'quotes': [self.quote('B', 200), self.quote('A', 100),
                                     self.quote('BAD SYMBOL', 9999), self.quote('NAN', float('nan')),
                                     dict(symbol='ETF', marketCap=9999, quoteType='ETF')]},
            {'total': 999, 'quotes': [self.quote('A', 350), self.quote('C', 200)]},
            {'total': 999, 'quotes': [self.quote('D', 500), self.quote('BRK-B', 50)]},
        ]
        with patch.object(expansion, 'UNIVERSE', self.universe), \
                patch('yfinance.screen', side_effect=pages) as screen, \
                patch.object(expansion.time, 'sleep') as sleep:
            symbols = expansion.discover(self.db, 503)
        self.assertEqual(symbols, ['D', 'A', 'B', 'C', 'BRK-B'])
        self.assertEqual([item.kwargs for item in screen.call_args_list], [
            dict(offset=0, size=250, sortField='intradaymarketcap', sortAsc=False),
            dict(offset=250, size=250, sortField='intradaymarketcap', sortAsc=False),
            dict(offset=500, size=3, sortField='intradaymarketcap', sortAsc=False),
        ])
        self.assertEqual(sleep.call_args_list, [call(0.6)] * 3)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM stock_metadata').fetchone()[0], 5)
        self.assertEqual(self.db.execute("SELECT market_cap FROM stock_metadata WHERE symbol='A'").fetchone()[0], 350)
        saved = json.loads(self.universe.read_text(encoding='utf-8'))
        self.assertEqual(saved['symbols'], symbols)
        self.assertEqual((saved['requested_limit'], saved['provider_total'], saved['sort']),
                         (503, 999, 'market_cap_desc'))
        self.assertFalse(self.universe.with_suffix('.tmp').exists())

    def test_empty_provider_result_preserves_last_usable_universe(self):
        original = json.dumps({'symbols': ['KEEP'], 'observed_at': 'previous'})
        self.universe.write_text(original, encoding='utf-8')
        with patch.object(expansion, 'UNIVERSE', self.universe), \
                patch('yfinance.screen', return_value={'quotes': []}) as screen, \
                patch.object(expansion.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'existing coverage was preserved'):
                expansion.discover(self.db, 1000)
        self.assertEqual(screen.call_count, 1)
        self.assertEqual(self.universe.read_text(encoding='utf-8'), original)

    def test_discovery_stops_on_exhausted_page(self):
        with patch.object(expansion, 'UNIVERSE', self.universe), \
                patch('yfinance.screen', side_effect=[{'quotes': [self.quote('ONLY', 100)], 'total': 1},
                                                   {'quotes': [], 'total': 1}]) as screen, \
                patch.object(expansion.time, 'sleep'):
            self.assertEqual(expansion.discover(self.db, 1000), ['ONLY'])
        self.assertEqual(screen.call_count, 2)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='stock-resume-', dir=Path(__file__).resolve().parent / 'data')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        (self.root / 'config.json').write_text(json.dumps({'symbols': ['BIG', 'MED', '2330.TW', 'BTC-USD']}), encoding='utf-8')
        self.universe = self.data / 'stock-universe.json'
        self.progress = self.data / 'stock-expansion.json'
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in [('ROOT', self.root), ('DATA', self.data), ('UNIVERSE', self.universe), ('PROGRESS', self.progress)]:
            stack.enter_context(patch.object(expansion, name, value))
        stack.enter_context(patch('yfinance.set_tz_cache_location'))

    def cache(self, age_days=0):
        snapshot = {'observed_at': (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat(),
                    'requested_limit': 3, 'symbols': ['BIG', 'MED', 'SMALL']}
        self.universe.write_text(json.dumps(snapshot), encoding='utf-8')
        return snapshot

    def test_fresh_stored_universe_is_reused_without_network_discovery(self):
        original = self.cache()
        with patch.object(expansion, 'discover') as discover, patch.object(expansion, 'ingest') as ingest:
            status = expansion.expand(limit=2, metadata_only=True)
        discover.assert_not_called()
        ingest.assert_not_called()
        self.assertEqual((status['status'], status['target']), ('complete', 2))
        self.assertEqual(json.loads(self.universe.read_text(encoding='utf-8')), original)
        self.assertEqual(json.loads(self.progress.read_text(encoding='utf-8'))['status'], 'complete')

    def test_stale_or_explicitly_refreshed_universe_is_rediscovered(self):
        for age, refresh in [(8, False), (0, True)]:
            with self.subTest(age=age, refresh=refresh):
                self.cache(age)
                with patch.object(expansion, 'discover', return_value=['NEW']) as discover:
                    status = expansion.expand(limit=2, refresh=refresh, metadata_only=True)
                self.assertEqual(discover.call_count, 1)
                self.assertEqual(discover.call_args.args[1], 2)
                self.assertEqual(status['target'], 1)

    @staticmethod
    def store_bar(db, symbol):
        timestamp = datetime.now(timezone.utc).isoformat()
        row = (symbol, '2026-09-24', 10., 12., 9., 11., 11., 100, 0., 0., 'USD', 'NMS', 'America/New_York', timestamp)
        persist(db, symbol, [row], True, timestamp, 0)

    def test_ranked_resume_keeps_taiwan_extras_and_does_not_recount_existing_history(self):
        self.cache()
        db = connect(self.data / 'market.sqlite')
        expansion.metadata_schema(db)
        self.store_bar(db, 'BIG')
        with db:
            db.executemany('INSERT INTO stock_enrichment VALUES (?,?,NULL)',
                           [(symbol, datetime.now(timezone.utc).isoformat()) for symbol in ('BIG', 'MED', '2330.TW')])
        db.close()

        def collect(connection, config, symbols):
            self.assertEqual(config['attempts'], 1)
            self.assertEqual(config['request_pause_seconds'], 0.35)
            for symbol in symbols:
                self.store_bar(connection, symbol)
            return []

        with patch.object(expansion, 'discover') as discover, \
                patch.object(expansion, 'ingest', side_effect=collect) as ingest, \
                patch('yfinance.Ticker') as ticker:
            first = expansion.expand(limit=3, history_limit=2)
            second = expansion.expand(limit=3, history_limit=2)
        discover.assert_not_called()
        ticker.assert_not_called()
        self.assertEqual([item.args[2] for item in ingest.call_args_list],
                         [['BIG'], ['MED'], ['2330.TW']] * 2)
        self.assertEqual((first['status'], first['target'], first['histories_added']), ('complete', 3, 2))
        self.assertEqual((second['status'], second['target'], second['histories_added']), ('complete', 3, 0))
        self.assertEqual(second['profiles_updated'], 0)
        with expansion.process_lock(self.data / 'ingestion.lock'):
            pass  # The completed queue must release its OS lock for the next run.


if __name__ == '__main__':
    unittest.main()
