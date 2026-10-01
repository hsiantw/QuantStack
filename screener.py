"""Read-only stock screener snapshots from locally stored daily prices.

All return and distance fields ending in ``_pct`` use percentage points. Metadata
yield/growth/margin fields retain provider fractions and have percentage aliases.
No quotes, currencies, or fundamentals are fabricated for uncovered symbols.
"""
import copy
import csv
import math
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import talib


CACHE_SECONDS = 60
HISTORY_BARS = 400  # Enough observations for one-year/YTD returns and indicator warmup.
_cache = {}
_cache_lock = threading.Lock()

METADATA_FIELDS = (
    'name', 'market_cap', 'sector', 'industry', 'country', 'exchange', 'currency',
    'pe', 'forward_pe', 'pb', 'dividend_yield', 'revenue_growth', 'profit_margin',
    'beta', 'fetched_at', 'source',
)
NUMERIC_METADATA = {
    'market_cap', 'pe', 'forward_pe', 'pb', 'dividend_yield', 'revenue_growth',
    'profit_margin', 'beta',
}
TECHNICAL_FIELDS = (
    'change_1d_pct', 'return_1w_pct', 'return_1m_pct', 'return_3m_pct', 'return_6m_pct',
    'return_ytd_pct', 'return_1y_pct',
    'sma20_distance_pct', 'sma50_distance_pct', 'sma200_distance_pct',
    'ema20_distance_pct', 'ema50_distance_pct', 'rsi14',
    'macd_pct', 'macd_signal_pct', 'macd_histogram_pct', 'adx14',
    'stochastic_k', 'stochastic_d', 'bollinger_position_pct', 'bollinger_width_pct',
    'atr14_pct', 'relative_volume', 'avg_volume20', 'avg_turnover20', 'volatility20_pct',
    'low_52w', 'high_52w', 'distance_52w_high_pct', 'distance_52w_low_pct',
    'range_52w_position_pct',
)


def _number(value, *, positive=False, nonnegative=False):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        return None
    return result


def _file_signature(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def _signature(path):
    # SQLite WAL commits need not change the main database's modification time.
    return tuple(_file_signature(item) for item in (
        path, Path(str(path) + '-wal'), path.parent / 'constituents.csv',
    ))


def _constituents(path):
    names = {}
    source = path.parent / 'constituents.csv'
    if source.exists():
        with source.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                symbol = row.get('Symbol', '').replace('.', '-')
                if symbol:
                    names[symbol] = {'name': row.get('Security') or None,
                                     'sector': row.get('GICS Sector') or None,
                                     'industry': row.get('GICS Sub-Industry') or None}
    return names


def _complete_tail(*arrays):
    """Do not join observations across a missing daily price or OHLC input."""
    valid = np.logical_and.reduce([np.isfinite(values) for values in arrays])
    missing = np.flatnonzero(~valid)
    start = int(missing[-1]) + 1 if len(missing) else 0
    return [values[start:] for values in arrays]


def _complete_window(values, length):
    return len(values) >= length and bool(np.isfinite(values[-length:]).all())


def _stock_row(symbol, bars, metadata, fallback, profile_date=None):
    row = {'symbol': symbol, **dict.fromkeys(METADATA_FIELDS)}
    row.update({key: value for key, value in metadata.items() if key in METADATA_FIELDS})
    for key in ('name', 'sector', 'industry'):
        row[key] = row[key] or fallback.get(key)
    row['name'] = row['name'] or symbol
    for key in NUMERIC_METADATA:
        row[key] = _number(row[key], nonnegative=key == 'market_cap')
    row['metadata_date'] = row.pop('fetched_at')
    row['metadata_source'] = row.pop('source')
    row['metadata_profile_at'] = profile_date
    for key in ('dividend_yield', 'revenue_growth', 'profit_margin'):
        row[key + '_pct'] = _number(row[key] * 100) if row[key] is not None else None
    row.update(dict.fromkeys(TECHNICAL_FIELDS))
    row.update(has_history=bool(bars), history_bars=len(bars), date=None, open=None, high=None,
               low=None, close=None, adjusted_close=None, volume=None, price_source=None,
               price_basis=None)
    if not bars:
        return row

    latest = bars[-1]
    row['date'] = latest['date']
    for key in ('open', 'high', 'low', 'close', 'adjusted_close'):
        row[key] = _number(latest[key], positive=True)
    row['volume'] = _number(latest['volume'], nonnegative=True)
    row['currency'] = row['currency'] or latest['currency']
    row['exchange'] = row['exchange'] or latest['exchange']
    row['price_source'] = latest['source']

    raw = np.array([_number(bar['close'], positive=True) for bar in bars], dtype=float)
    adjusted = np.array([_number(bar['adjusted_close'], positive=True) for bar in bars], dtype=float)
    # A raw-only dataset is useful; partial adjusted data must not be spliced with
    # incompatible raw prices around splits or dividend adjustments.
    use_adjusted = bool(np.isfinite(adjusted).any())
    prices = adjusted if use_adjusted else raw
    row['price_basis'] = 'adjusted' if use_adjusted else 'unadjusted'
    factors = np.divide(prices, raw, out=np.full(len(raw), np.nan), where=np.isfinite(raw))
    highs = np.array([_number(bar['high'], positive=True) for bar in bars], dtype=float) * factors
    lows = np.array([_number(bar['low'], positive=True) for bar in bars], dtype=float) * factors
    invalid_ranges = highs < lows
    highs[invalid_ranges] = lows[invalid_ranges] = np.nan
    volume = np.array([_number(bar['volume'], nonnegative=True) for bar in bars], dtype=float)

    for key, periods in (('change_1d_pct', 1), ('return_1w_pct', 5), ('return_1m_pct', 21),
                         ('return_3m_pct', 63), ('return_6m_pct', 126), ('return_1y_pct', 252)):
        if _complete_window(prices, periods + 1):
            row[key] = _number((prices[-1] / prices[-periods - 1] - 1) * 100)
    # YTD requires an actual prior-year baseline; a new listing's first close
    # must not silently become a substitute for the previous year-end close.
    prior_year = str(int(latest['date'][:4]) - 1)
    prior_year_indices = [index for index, bar in enumerate(bars) if bar['date'][:4] == prior_year]
    if prior_year_indices:
        baseline = prior_year_indices[-1]
        if _complete_window(prices, len(prices) - baseline):
            row['return_ytd_pct'] = _number((prices[-1] / prices[baseline] - 1) * 100)
    for periods in (20, 50, 200):
        if _complete_window(prices, periods):
            row[f'sma{periods}_distance_pct'] = _number((prices[-1] / prices[-periods:].mean() - 1) * 100)

    close_tail, = _complete_tail(prices)
    for periods in (20, 50):
        if len(close_tail) >= periods:
            average = talib.EMA(close_tail, timeperiod=periods)[-1]
            row[f'ema{periods}_distance_pct'] = _number((close_tail[-1] / average - 1) * 100)
    if len(close_tail) >= 15:
        row['rsi14'] = _number(talib.RSI(close_tail, timeperiod=14)[-1])
    if len(close_tail) >= 34:
        macd, signal, histogram = talib.MACD(close_tail, fastperiod=12, slowperiod=26, signalperiod=9)
        for key, values in (('macd_pct', macd), ('macd_signal_pct', signal), ('macd_histogram_pct', histogram)):
            row[key] = _number(values[-1] / close_tail[-1] * 100)
    if len(close_tail) >= 20:
        upper, middle, lower = talib.BBANDS(close_tail, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        width = upper[-1] - lower[-1]
        row['bollinger_width_pct'] = _number(width / middle[-1] * 100)
        if width > 0:
            row['bollinger_position_pct'] = _number((close_tail[-1] - lower[-1]) / width * 100)
    high_tail, low_tail, atr_close = _complete_tail(highs, lows, prices)
    if len(atr_close) >= 15:
        row['atr14_pct'] = _number(talib.ATR(high_tail, low_tail, atr_close, timeperiod=14)[-1]
                                  / atr_close[-1] * 100)
    if len(atr_close) >= 28:
        row['adx14'] = _number(talib.ADX(high_tail, low_tail, atr_close, timeperiod=14)[-1])
    if len(atr_close) >= 18:
        stochastic_k, stochastic_d = talib.STOCH(
            high_tail, low_tail, atr_close, fastk_period=14, slowk_period=3,
            slowk_matype=0, slowd_period=3, slowd_matype=0)
        row['stochastic_k'] = _number(stochastic_k[-1])
        row['stochastic_d'] = _number(stochastic_d[-1])
    if _complete_window(volume, 20):
        row['avg_volume20'] = _number(volume[-20:].mean())
    if _complete_window(raw, 20) and _complete_window(volume, 20):
        with np.errstate(over='ignore', invalid='ignore'):
            row['avg_turnover20'] = _number((raw[-20:] * volume[-20:]).mean())
    if _complete_window(volume, 21) and volume[-21:-1].mean() > 0:
        row['relative_volume'] = _number(volume[-1] / volume[-21:-1].mean())
    if _complete_window(prices, 21):
        log_returns = np.diff(np.log(prices[-21:]))
        row['volatility20_pct'] = _number(log_returns.std(ddof=1) * math.sqrt(252) * 100)
    if (_complete_window(highs, 252) and _complete_window(lows, 252)
            and math.isfinite(prices[-1]) and math.isfinite(factors[-1])):
        highest, lowest = float(highs[-252:].max()), float(lows[-252:].min())
        # Express the split/dividend-adjusted range in the latest quote's units.
        row['high_52w'] = _number(highest / factors[-1])
        row['low_52w'] = _number(lowest / factors[-1])
        row['distance_52w_high_pct'] = _number((prices[-1] / highest - 1) * 100)
        row['distance_52w_low_pct'] = _number((prices[-1] / lowest - 1) * 100)
        if highest > lowest:
            row['range_52w_position_pct'] = _number((prices[-1] - lowest) / (highest - lowest) * 100)
    return row


def _read_snapshot(path):
    fallback = _constituents(path)
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('BEGIN')
        tables = {item[0] for item in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        metadata = ({item['symbol']: dict(item) for item in connection.execute('SELECT * FROM stock_metadata')}
                    if 'stock_metadata' in tables else {})
        profiles = (dict(connection.execute('SELECT symbol,last_success FROM stock_enrichment'))
                    if 'stock_enrichment' in tables else {})
        history_symbols = ({item[0] for item in connection.execute('SELECT DISTINCT symbol FROM prices')}
                           if 'prices' in tables else set())
        symbols = sorted(symbol for symbol in history_symbols | metadata.keys() if not symbol.endswith('-USD'))
        rows = []
        for symbol in symbols:
            bars = list(connection.execute('''SELECT date,open,high,low,close,adjusted_close,volume,
                                               currency,exchange,source FROM prices
                                               WHERE symbol=? ORDER BY date DESC LIMIT ?''',
                                           (symbol, HISTORY_BARS))) if symbol in history_symbols else []
            bars.reverse()
            rows.append(_stock_row(symbol, bars, metadata.get(symbol, {}), fallback.get(symbol, {}),
                                   profiles.get(symbol)))
    finally:
        connection.close()

    dates = [row['date'] for row in rows if row['date']]
    currencies = Counter(row['currency'] or 'Unknown' for row in rows)
    covered = sum(row['has_history'] for row in rows)
    with_metadata = sum(row['symbol'] in metadata for row in rows)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'rows': rows,
        'data_dates': {'earliest': min(dates, default=None), 'latest': max(dates, default=None)},
        'coverage': {
            'total': len(rows), 'with_history': covered, 'without_history': len(rows) - covered,
            'with_metadata': with_metadata, 'with_market_cap': sum(row['market_cap'] is not None for row in rows),
            'with_1y_history': sum(row['return_1y_pct'] is not None for row in rows),
            'by_currency': dict(sorted(currencies.items())),
        },
        'universe': {
            'total': len(rows), 'scope': 'Locally stored stocks',
            'currencies': sorted({row['currency'] for row in rows if row['currency']}),
            'metadata_source': sorted({row['metadata_source'] for row in rows if row['metadata_source']}),
        },
        'methodology': {
            'returns': 'Percentage change in adjusted close over 1, 5, 21, 63, 126 or 252 daily bars; raw close only for raw-only series.',
            'return_ytd_pct': 'Return from the last stored close in the previous calendar year to the latest bar; null without that baseline or with missing prices in the interval.',
            'averages': 'SMA/EMA distances and RSI use adjusted close, with raw close for raw-only series. OHLC indicators scale highs/lows by each bar adjusted-close/close ratio. Indicators restart after missing inputs.',
            'macd': 'TA-Lib MACD with EMA periods 12/26 and signal 9. MACD, signal and histogram are each divided by latest selected close and multiplied by 100.',
            'adx14': 'TA-Lib 14-period ADX using consistently adjusted high/low/close; at least 28 complete consecutive bars.',
            'stochastic': 'Slow stochastic: 14-period fast K, 3-period SMA slow K, 3-period SMA slow D; at least 18 complete consecutive OHLC bars.',
            'bollinger': '20-period SMA bands at +/-2 population standard deviations. Position is (close-lower)/(upper-lower)*100, null for zero width; width is (upper-lower)/middle*100. Position may be outside 0-100.',
            'relative_volume': 'Latest daily volume divided by the preceding 20 daily volumes; current day excluded from the baseline.',
            'avg_volume20': 'Average of the latest 20 daily volumes, including the current bar.',
            'avg_turnover20': 'Average raw daily close times reported volume over the latest 20 bars, in reported currency; approximate traded value, not actual transaction turnover.',
            'volatility20_pct': 'Sample standard deviation of 20 daily log returns, annualized by sqrt(252), in percent.',
            'range_52w': '252 daily highs/lows, adjusted consistently and expressed in the latest quote units. Distances are (close/extreme-1)*100; position is (close-low)/(high-low)*100 and null for zero width. Incomplete windows are null.',
            'currencies': 'Market caps and prices retain their reported currencies; no currency conversion is applied.',
        },
    }


def snapshot(database_path):
    """Return a thread-safe, at-most-60-second cached read-only stock snapshot."""
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Market database does not exist: {path}')
    with _cache_lock:
        signature = _signature(path)
        entry = _cache.get(path)
        if entry and entry[0] == signature and time.monotonic() - entry[1] < CACHE_SECONDS:
            return copy.deepcopy(entry[2])
        result = _read_snapshot(path)
        # Cache under the pre-read signature. A concurrent commit then triggers
        # another read rather than being mistaken for part of this transaction.
        if len(_cache) >= 8 and path not in _cache:
            del _cache[next(iter(_cache))]
        _cache[path] = (signature, time.monotonic(), result)
        return copy.deepcopy(result)
