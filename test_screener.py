import json
import math
import sqlite3
import statistics
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import talib

import screener


class ScreenerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'market.sqlite'
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE prices (
                symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
                adjusted_close REAL, volume REAL, currency TEXT, exchange TEXT,
                source TEXT, PRIMARY KEY(symbol,date));
            CREATE TABLE stock_metadata (
                symbol TEXT PRIMARY KEY, name TEXT, market_cap REAL, sector TEXT,
                industry TEXT, country TEXT, exchange TEXT, currency TEXT, pe REAL,
                forward_pe REAL, pb REAL, dividend_yield REAL, revenue_growth REAL,
                profit_margin REAL, beta REAL, fetched_at TEXT, source TEXT);
        ''')

    def prices(self, symbol='TEST', count=260, split=False, varying=False, currency='USD'):
        records = []
        for index in range(count):
            close = 100 + index * .1 + math.sin(index / 3) if varying else 100
            factor = 2 if split and index < count // 2 else 1
            volume = 1000 if index == count - 1 else 100
            records.append((symbol, (date(2025, 1, 1) + timedelta(days=index)).isoformat(),
                            close * factor, (close + 2) * factor, (close - 2) * factor,
                            close * factor, close, volume, currency, 'TESTEX', 'Test daily feed'))
        self.db.executemany('INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?,?)', records)
        self.db.commit()
        return records

    def metadata(self, symbol='TEST', **values):
        values = {'symbol': symbol, 'name': 'Test Company', 'currency': 'USD',
                  'source': 'Test fundamentals', 'fetched_at': '2026-09-24T00:00:00+00:00', **values}
        fields = list(values)
        self.db.execute(f'INSERT INTO stock_metadata ({",".join(fields)}) VALUES ({",".join("?" for _ in fields)})',
                        [values[key] for key in fields])
        self.db.commit()

    def row(self, symbol='TEST'):
        return next(item for item in screener.snapshot(self.path)['rows'] if item['symbol'] == symbol)

    def test_adjustments_neutralize_split_and_preserve_volume_baseline(self):
        self.prices(split=True)
        self.metadata(market_cap=2e12, sector='Technology', country='United States', pe=25,
                      dividend_yield=.0044, revenue_growth=.15, profit_margin=.2)
        row = self.row()
        self.assertTrue(row['has_history'])
        self.assertEqual(row['price_basis'], 'adjusted')
        for key in ('change_1d_pct', 'return_1w_pct', 'return_1m_pct', 'return_3m_pct',
                    'return_6m_pct', 'return_1y_pct', 'sma20_distance_pct', 'sma50_distance_pct',
                    'sma200_distance_pct', 'ema20_distance_pct', 'ema50_distance_pct',
                    'macd_pct', 'macd_signal_pct', 'macd_histogram_pct'):
            self.assertEqual(row[key], 0, key)
        self.assertEqual(row['atr14_pct'], 4)
        self.assertEqual(row['rsi14'], 0)
        self.assertEqual(row['volatility20_pct'], 0)
        self.assertEqual(row['relative_volume'], 10)
        self.assertEqual(row['avg_volume20'], 145)
        self.assertEqual(row['avg_turnover20'], 14500)
        self.assertEqual(row['adx14'], 0)
        self.assertEqual((row['stochastic_k'], row['stochastic_d']), (50, 50))
        self.assertEqual(row['bollinger_width_pct'], 0)
        self.assertIsNone(row['bollinger_position_pct'])
        self.assertEqual((row['high_52w'], row['low_52w']), (102, 98))
        self.assertAlmostEqual(row['distance_52w_high_pct'], (100 / 102 - 1) * 100)
        self.assertAlmostEqual(row['distance_52w_low_pct'], (100 / 98 - 1) * 100)
        self.assertEqual(row['range_52w_position_pct'], 50)
        self.assertEqual(row['market_cap'], 2e12)
        self.assertAlmostEqual(row['dividend_yield_pct'], .44)
        self.assertEqual((row['revenue_growth_pct'], row['profit_margin_pct']), (15, 20))

    def test_rsi_returns_and_volatility_use_daily_adjusted_observations(self):
        records = self.prices(varying=True, split=True)
        row = self.row()
        adjusted = np.array([record[6] for record in records], dtype=float)
        self.assertAlmostEqual(row['rsi14'], talib.RSI(adjusted, timeperiod=14)[-1])
        self.assertAlmostEqual(row['return_1y_pct'], (adjusted[-1] / adjusted[-253] - 1) * 100)
        self.assertAlmostEqual(row['return_6m_pct'], (adjusted[-1] / adjusted[-127] - 1) * 100)
        self.assertAlmostEqual(row['return_1w_pct'], (adjusted[-1] / adjusted[-6] - 1) * 100)
        self.assertAlmostEqual(row['sma20_distance_pct'], (adjusted[-1] / adjusted[-20:].mean() - 1) * 100)
        returns = [math.log(adjusted[index] / adjusted[index - 1]) for index in range(len(adjusted) - 20, len(adjusted))]
        self.assertAlmostEqual(row['volatility20_pct'], statistics.stdev(returns) * math.sqrt(252) * 100)

    def test_extended_indicators_match_talib_on_split_adjusted_data(self):
        records = self.prices(varying=True, split=True)
        row = self.row()
        close = np.array([record[6] for record in records], dtype=float)
        high = np.array([record[3] * record[6] / record[5] for record in records], dtype=float)
        low = np.array([record[4] * record[6] / record[5] for record in records], dtype=float)
        for periods in (20, 50):
            expected = (close[-1] / talib.EMA(close, timeperiod=periods)[-1] - 1) * 100
            self.assertAlmostEqual(row[f'ema{periods}_distance_pct'], expected)
        for key, expected in zip(('macd_pct', 'macd_signal_pct', 'macd_histogram_pct'),
                                 talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)):
            self.assertAlmostEqual(row[key], expected[-1] / close[-1] * 100)
        self.assertAlmostEqual(row['adx14'], talib.ADX(high, low, close, timeperiod=14)[-1])
        k, d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3,
                           slowk_matype=0, slowd_period=3, slowd_matype=0)
        self.assertAlmostEqual(row['stochastic_k'], k[-1])
        self.assertAlmostEqual(row['stochastic_d'], d[-1])
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.assertAlmostEqual(row['bollinger_position_pct'], (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) * 100)
        self.assertAlmostEqual(row['bollinger_width_pct'], (upper[-1] - lower[-1]) / middle[-1] * 100)
        self.assertAlmostEqual(row['distance_52w_low_pct'], (close[-1] / low[-252:].min() - 1) * 100)
        self.assertAlmostEqual(row['range_52w_position_pct'],
                               (close[-1] - low[-252:].min()) / (high[-252:].max() - low[-252:].min()) * 100)
        self.db.execute('UPDATE prices SET close=adjusted_close, high=adjusted_close+2, low=adjusted_close-2, adjusted_close=NULL')
        self.db.commit()
        raw_row = self.row()
        self.assertEqual(raw_row['price_basis'], 'unadjusted')
        for key in screener.TECHNICAL_FIELDS:
            if key == 'avg_turnover20':
                continue
            if row[key] is None:
                self.assertIsNone(raw_row[key], key)
            else:
                self.assertAlmostEqual(raw_row[key], row[key], msg=key)

    def test_ytd_uses_prior_year_close_and_requires_complete_interval(self):
        records = self.prices(count=400, varying=True, split=True)
        row = self.row()
        baseline = next(record for record in records if record[1] == '2025-12-31')
        self.assertAlmostEqual(row['return_ytd_pct'], (records[-1][6] / baseline[6] - 1) * 100)
        # A missing input before the baseline does not invalidate this year's return.
        self.db.execute('UPDATE prices SET adjusted_close=NULL WHERE date=?', (records[100][1],))
        self.db.commit()
        self.assertEqual(self.row()['return_ytd_pct'], row['return_ytd_pct'])
        self.db.execute('UPDATE prices SET adjusted_close=NULL WHERE date=?', ('2026-01-15',))
        self.db.commit()
        self.assertIsNone(self.row()['return_ytd_pct'])
        self.db.execute('UPDATE prices SET adjusted_close=close WHERE date=?', ('2026-01-15',))
        self.db.execute('UPDATE prices SET adjusted_close=NULL WHERE date=?', ('2025-12-31',))
        self.db.commit()
        self.assertIsNone(self.row()['return_ytd_pct'])
        self.db.execute("DELETE FROM prices WHERE date<'2026-01-01'")
        self.db.commit()
        self.assertIsNone(self.row()['return_ytd_pct'])

    def test_extended_indicators_require_their_full_warmup(self):
        minimum_bars = {'ema20_distance_pct': 20, 'ema50_distance_pct': 50,
                        'macd_pct': 34, 'macd_signal_pct': 34, 'macd_histogram_pct': 34,
                        'adx14': 28, 'stochastic_k': 18, 'stochastic_d': 18,
                        'bollinger_position_pct': 20, 'bollinger_width_pct': 20,
                        'avg_turnover20': 20, 'return_6m_pct': 127,
                        'distance_52w_low_pct': 252, 'range_52w_position_pct': 252}
        for count in sorted({value + offset for value in minimum_bars.values() for offset in (-1, 0)}):
            symbol = f'N{count}'
            self.prices(symbol=symbol, count=count, varying=True)
            row = self.row(symbol)
            for key, minimum in minimum_bars.items():
                with self.subTest(count=count, field=key):
                    if count < minimum:
                        self.assertIsNone(row[key])
                    else:
                        self.assertIsNotNone(row[key])

    def test_turnover_uses_raw_prices_and_does_not_depend_on_adjusted_close(self):
        records = self.prices(count=30, split=True, varying=True)
        expected = statistics.mean(record[5] * record[7] for record in records[-20:])
        self.assertAlmostEqual(self.row()['avg_turnover20'], expected)
        self.db.execute('UPDATE prices SET adjusted_close=NULL WHERE date=?', (records[-5][1],))
        self.db.commit()
        self.assertAlmostEqual(self.row()['avg_turnover20'], expected)
        self.db.execute('UPDATE prices SET close=NULL WHERE date=?', (records[-5][1],))
        self.db.commit()
        self.assertIsNone(self.row()['avg_turnover20'])

    def test_missing_ohlc_restarts_ohlc_indicators_without_breaking_close_indicators(self):
        records = self.prices(count=80, varying=True)
        self.db.execute('UPDATE prices SET high=low-1 WHERE date=?', (records[-10][1],))
        self.db.commit()
        row = self.row()
        for key in ('adx14', 'stochastic_k', 'stochastic_d', 'atr14_pct'):
            self.assertIsNone(row[key], key)
        for key in ('ema50_distance_pct', 'macd_pct', 'bollinger_width_pct'):
            self.assertIsNotNone(row[key], key)
        # Recursive indicators use only the valid tail after a missing close.
        self.db.execute('UPDATE prices SET adjusted_close=NULL WHERE date=?', (records[-36][1],))
        self.db.commit()
        row = self.row()
        close_tail = np.array([record[6] for record in records[-35:]], dtype=float)
        macd = talib.MACD(close_tail, fastperiod=12, slowperiod=26, signalperiod=9)[0]
        self.assertAlmostEqual(row['macd_pct'], macd[-1] / close_tail[-1] * 100)
        self.assertIsNone(row['ema50_distance_pct'])

    def test_zero_width_year_range_has_no_position(self):
        self.prices()
        self.db.execute('UPDATE prices SET high=close,low=close')
        self.db.commit()
        row = self.row()
        self.assertIsNone(row['range_52w_position_pct'])
        self.assertEqual(row['distance_52w_low_pct'], 0)
        self.assertEqual(row['distance_52w_high_pct'], 0)
        json.dumps(row, allow_nan=False)

    def test_new_listing_has_nulls_for_incomplete_windows(self):
        self.prices(count=20)
        row = self.row()
        self.assertEqual(row['return_1w_pct'], 0)
        self.assertEqual(row['sma20_distance_pct'], 0)
        self.assertEqual(row['avg_volume20'], 145)
        for key in ('return_1m_pct', 'return_3m_pct', 'return_6m_pct', 'return_ytd_pct', 'return_1y_pct', 'relative_volume',
                    'volatility20_pct', 'sma50_distance_pct', 'sma200_distance_pct', 'high_52w', 'low_52w'):
            self.assertIsNone(row[key], key)

    def test_missing_adjusted_price_or_volume_is_never_zero_filled(self):
        records = self.prices(count=30)
        self.db.execute('UPDATE prices SET adjusted_close=NULL,volume=NULL WHERE date=?', (records[-10][1],))
        self.db.commit()
        row = self.row()
        self.assertEqual(row['change_1d_pct'], 0)
        self.assertEqual(row['return_1w_pct'], 0)
        for key in ('return_1m_pct', 'sma20_distance_pct', 'rsi14', 'atr14_pct', 'relative_volume',
                    'avg_volume20', 'avg_turnover20', 'volatility20_pct', 'ema20_distance_pct',
                    'ema50_distance_pct', 'macd_pct', 'macd_signal_pct', 'macd_histogram_pct',
                    'adx14', 'stochastic_k', 'stochastic_d', 'bollinger_position_pct', 'bollinger_width_pct'):
            self.assertIsNone(row[key], key)
        self.db.execute('UPDATE prices SET adjusted_close=NULL')
        self.db.commit()
        row = self.row()
        self.assertEqual(row['price_basis'], 'unadjusted')
        self.assertEqual(row['return_1m_pct'], 0)
        self.assertIsNone(row['adjusted_close'])

    def test_union_includes_uncovered_stocks_and_keeps_currencies_separate(self):
        self.prices(count=3)
        self.prices(symbol='BTC-USD', count=3)
        self.prices(symbol='2330.TW', count=3, currency='TWD')
        self.metadata(symbol='NEW', market_cap=3e9, pe=math.inf, revenue_growth=math.inf)
        self.metadata(symbol='2330.TW', currency='TWD', country='Taiwan', market_cap=40e12)
        self.metadata(symbol='BTC-USD', currency='USD')
        result = screener.snapshot(self.path)
        self.assertEqual({row['symbol'] for row in result['rows']}, {'TEST', 'NEW', '2330.TW'})
        uncovered = self.row('NEW')
        self.assertFalse(uncovered['has_history'])
        self.assertEqual(uncovered['market_cap'], 3e9)
        self.assertIsNone(uncovered['close'])
        self.assertIsNone(uncovered['pe'])
        self.assertIsNone(uncovered['revenue_growth_pct'])
        self.assertTrue(all(uncovered[key] is None for key in screener.TECHNICAL_FIELDS))
        taiwan = self.row('2330.TW')
        self.assertEqual((taiwan['currency'], taiwan['country'], taiwan['market_cap']), ('TWD', 'Taiwan', 40e12))
        self.assertEqual(result['coverage']['with_history'], 2)
        self.assertEqual(result['coverage']['without_history'], 1)
        self.assertEqual(result['coverage']['with_metadata'], 2)
        self.assertEqual(result['coverage']['by_currency'], {'TWD': 1, 'USD': 2})
        self.assertEqual(result['data_dates'], {'earliest': '2025-01-03', 'latest': '2025-01-03'})
        json.dumps(result, allow_nan=False)

    def test_optional_metadata_table_and_constituent_name_fallback(self):
        self.prices(count=3)
        self.db.execute('DROP TABLE stock_metadata')
        self.db.commit()
        self.path.with_name('constituents.csv').write_text(
            'Symbol,Security,GICS Sector,GICS Sub-Industry\nTEST,Example Company,Technology,Software\n',
            encoding='utf-8')
        row = self.row()
        self.assertEqual((row['name'], row['sector'], row['industry']), ('Example Company', 'Technology', 'Software'))
        self.assertEqual(row['currency'], 'USD')
        self.assertIsNone(row['country'])
        self.assertIsNone(row['market_cap'])
        self.assertIsNone(row['metadata_profile_at'])

    def test_profile_freshness_is_separate_from_bulk_quote_freshness(self):
        self.metadata(market_cap=1e9)
        self.db.execute('CREATE TABLE stock_enrichment (symbol TEXT PRIMARY KEY,last_success TEXT,error TEXT)')
        self.db.execute('INSERT INTO stock_enrichment VALUES (?,?,NULL)', ('TEST', '2026-09-20T00:00:00+00:00'))
        self.db.commit()
        row = self.row()
        self.assertEqual(row['metadata_date'], '2026-09-24T00:00:00+00:00')
        self.assertEqual(row['metadata_profile_at'], '2026-09-20T00:00:00+00:00')

    def test_wal_update_invalidates_cache_and_returned_objects_are_isolated(self):
        self.prices(count=30)
        self.metadata(market_cap=1e9)
        with patch.object(screener, '_read_snapshot', wraps=screener._read_snapshot) as read:
            initial = screener.snapshot(self.path)
            initial['rows'][0]['market_cap'] = 999
            self.assertEqual(screener.snapshot(self.path)['rows'][0]['market_cap'], 1e9)
            self.assertEqual(read.call_count, 1)
            self.db.execute('UPDATE stock_metadata SET market_cap=2e9 WHERE symbol=?', ('TEST',))
            self.db.commit()  # Connection remains open: this update lives in the WAL.
            self.assertEqual(screener.snapshot(self.path)['rows'][0]['market_cap'], 2e9)
            self.assertEqual(read.call_count, 2)
            with patch.object(screener, 'CACHE_SECONDS', 0):
                screener.snapshot(self.path)
            self.assertEqual(read.call_count, 3)

    def test_empty_dataset_and_zero_volume_produce_json_safe_nulls(self):
        result = screener.snapshot(self.path)
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['coverage']['total'], 0)
        self.assertEqual(result['data_dates'], {'earliest': None, 'latest': None})
        self.prices(count=21)
        self.db.execute('UPDATE prices SET volume=0')
        self.db.commit()
        row = self.row()
        self.assertEqual(row['avg_volume20'], 0)
        self.assertIsNone(row['relative_volume'])
        json.dumps(row, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
