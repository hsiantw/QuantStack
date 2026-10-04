"""Numerical checks for the chart workspace historical return analysis."""
import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


class ReturnsRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel='msedge', headless=True)
        cls.page = cls.browser.new_page()
        cls.page.add_script_tag(path=str(Path('web/risk-engine.js').resolve()))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    @staticmethod
    def bars(log_returns):
        price = 100
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rows = [{'date': start.isoformat(), 'close': price, 'adjusted_close': price}]
        for index, value in enumerate(log_returns, start=1):
            price *= math.exp(value)
            rows.append({'date': (start + timedelta(days=index)).isoformat(), 'close': price, 'adjusted_close': price})
        return rows

    def run_engine(self, bars, parameters=None):
        return self.page.evaluate('({bars, parameters}) => AtlasRisk.run(bars, parameters || {})',
                                  {'bars': bars, 'parameters': parameters or {}})

    def test_distribution_risk_and_annualization(self):
        result = self.run_engine(self.bars([.01, -.02, .005, -.003] * 25), {'window': 100, 'barsPerYear': 252})
        self.assertEqual(result['data']['count'], 100)
        self.assertEqual(len(result['histogram']), 20)
        self.assertEqual(sum(item['count'] for item in result['histogram']), 100)
        self.assertLessEqual(result['maxDrawdown'], 0)
        self.assertGreaterEqual(result['expectedShortfall'], result['valueAtRisk'])
        self.assertGreater(result['annualizedVolatility'], 0)
        self.assertIsNotNone(result['sharpe'])

    def test_zero_variance_and_zero_loss(self):
        result = self.run_engine(self.bars([0] * 20))
        self.assertEqual(result['annualizedVolatility'], 0)
        self.assertIsNone(result['sharpe'])
        self.assertIsNone(result['sortino'])
        self.assertEqual(result['maxDrawdown'], 0)
        self.assertEqual(result['valueAtRisk'], 0)

    def test_rejects_short_and_invalid_histories(self):
        with self.assertRaises(Exception):
            self.run_engine(self.bars([.01] * 19))
        invalid = self.bars([.01] * 20)
        invalid[-1]['date'] = invalid[-2]['date']
        with self.assertRaises(Exception):
            self.run_engine(invalid)
        invalid = self.bars([.01] * 20)
        invalid[-1]['adjusted_close'] = None
        with self.assertRaises(Exception):
            self.run_engine(invalid)

    def test_rejects_invalid_settings(self):
        with self.assertRaises(Exception):
            self.run_engine(self.bars([.01] * 20), {'window': 10})
        with self.assertRaises(Exception):
            self.run_engine(self.bars([.01] * 20), {'confidence': .8})


if __name__ == '__main__':
    unittest.main()