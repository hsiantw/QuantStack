"""Offline compatibility check for the optional trading research environment."""
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'data' / 'analysis' / '.matplotlib'))

import alphalens.performance as ap
import ccxt
import numpy as np
import pandas as pd
import quantstats as qs
import vectorbt as vbt


def main():
    dates = pd.date_range('2025-01-01', periods=5, freq='D', tz='UTC')
    prices = pd.Series([100., 110., 105., 115., 120.], index=dates)
    portfolio = vbt.Portfolio.from_signals(
        prices, entries=pd.Series([True, False, False, False, False], index=dates),
        exits=pd.Series([False, False, False, False, True], index=dates),
        init_cash=1000., fees=0.001, slippage=0.001, freq='1D')
    result = float(portfolio.total_return())
    assert 0 < result < 0.20, result
    returns = prices.pct_change().dropna()
    drawdown = float(qs.stats.max_drawdown(returns))
    assert math.isclose(drawdown, 105 / 110 - 1, abs_tol=1e-10), drawdown
    idx = pd.MultiIndex.from_product([dates[:3], ['A', 'B', 'C', 'D']], names=['date', 'asset'])
    factors = pd.DataFrame({'factor': np.tile([1., 2., 3., 4.], 3),
                            '1D': np.tile([.01, .02, .03, .04], 3)}, index=idx)
    ic = ap.factor_information_coefficient(factors)
    assert np.allclose(ic['1D'], 1.), ic
    exchange = ccxt.bitstamp({'enableRateLimit': True})
    assert exchange.has['fetchOHLCV'] and exchange.parse_timeframe('1m') == 60
    print(json.dumps({'status': 'passed', 'versions': {name: version(name) for name in
        ['vectorbt', 'quantstats', 'alphalens-reloaded', 'ccxt']},
        'net_test_return': result, 'test_drawdown': drawdown,
        'factor_ic': float(ic['1D'].mean()), 'ccxt': 'Bitstamp OHLCV adapter checked offline'}, indent=2))


if __name__ == '__main__':
    main()
