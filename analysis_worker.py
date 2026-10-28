"""Research calculations in .venv-analysis; only exchange pulls write market data."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data' / 'analysis' / '.matplotlib'))
from analysis_jobs import validate, write_json

SECONDS = {'1m': 60, '5m': 300, '15m': 900, '60m': 3600, '1d': 86400}
MAX_BARS = 50000


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, 'item'):
        return clean(value.item())
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value


def bounds(p):
    import pandas as pd
    start = pd.Timestamp(p['start'], tz='UTC')
    end = min(pd.Timestamp(p['end'], tz='UTC') + pd.Timedelta(days=1), pd.Timestamp.now(tz='UTC'))
    return start, end


def load_bars(p, symbol, database):
    import numpy as np
    import pandas as pd
    start, end = bounds(p)
    if start >= end:
        raise ValueError('The requested range contains no completed history.')
    interval, source = p['interval'], p['source']
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True, timeout=15)) as db:
        if source == 'Yahoo Finance' and interval == '1d':
            frame = pd.read_sql_query('SELECT date AS timestamp,open,high,low,close,adjusted_close,volume,currency FROM prices WHERE symbol=? AND source=? AND date>=? AND date<? ORDER BY date LIMIT ?', db,
                params=(symbol, source, start.date().isoformat(), end.date().isoformat(), MAX_BARS + 1))
        else:
            table = 'intraday_prices' if source == 'Yahoo Finance' else 'crypto_exchange_prices'
            columns = 'timestamp,open,high,low,close,volume,' + ('currency' if source == 'Yahoo Finance' else "'USD' AS currency")
            def read(bar_interval):
                return pd.read_sql_query(f'SELECT {columns} FROM {table} WHERE symbol=? AND source=? AND interval=? AND timestamp>=? AND timestamp<? ORDER BY timestamp LIMIT ?', db,
                    params=(symbol, source, bar_interval, start.isoformat(), end.isoformat(), MAX_BARS + 1))
            frame = read(interval)
            if frame.empty and source != 'Yahoo Finance' and interval in ('5m', '15m', '60m'):
                frame = read('1m')
                frame.attrs['aggregate'] = True
    if len(frame) > MAX_BARS:
        raise ValueError(f'{symbol}: range exceeds {MAX_BARS:,} source bars; choose a shorter range.')
    if frame.empty:
        raise ValueError(f'{symbol}: no {source} {interval} candles in this range. Choose another source/range or pull exchange candles first.')
    aggregate = frame.attrs.get('aggregate', False)
    frame.index = pd.to_datetime(frame.pop('timestamp'), utc=True)
    step = 60 if aggregate else SECONDS[interval]
    frame = frame[frame.index + pd.Timedelta(seconds=step) <= end].copy()
    if frame.empty:
        raise ValueError(f'{symbol}: no completed candles.')
    if frame.index.has_duplicates:
        raise ValueError(f'{symbol}: duplicate timestamps; audit the data first.')
    values = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
    valid &= (values[:, 2] <= np.minimum(values[:, 0], values[:, 3])) & (values[:, 1] >= np.maximum(values[:, 0], values[:, 3]))
    if not valid.all():
        raise ValueError(f'{symbol}: {int((~valid).sum())} invalid OHLCV rows; audit the data first.')
    currencies = sorted(frame.currency.dropna().unique().tolist())
    adjusted = 'adjusted_close' in frame
    if adjusted:
        if frame.adjusted_close.isna().any() or not np.isfinite(frame.adjusted_close.to_numpy(dtype=float)).all() or (frame.adjusted_close <= 0).any():
            raise ValueError(f'{symbol}: adjusted daily prices are incomplete.')
        ratio = frame.adjusted_close / frame.close
        frame.loc[:, ['open', 'high', 'low', 'close']] = frame[['open', 'high', 'low', 'close']].mul(ratio, axis=0)
    dropped = 0
    if aggregate:
        grouped = frame.resample(f'{SECONDS[interval]}s', origin='epoch')
        counts = grouped.close.count()
        result = grouped.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        complete = counts == SECONDS[interval] // 60
        dropped = int((~complete).sum())
        frame = result[complete].dropna()
        frame = frame[frame.index + pd.Timedelta(seconds=SECONDS[interval]) <= end]
    if len(frame) < 3:
        raise ValueError(f'{symbol}: fewer than three complete candles available.')
    gaps = int(((frame.index.to_series().diff().dt.total_seconds() / SECONDS[interval] - 1).clip(lower=0)).sum())
    audit = dict(symbol=symbol, source=source, interval=interval, bars=len(frame), first=frame.index[0].isoformat(), last=frame.index[-1].isoformat(),
        adjusted=adjusted, currencies=currencies, incomplete_buckets_dropped=dropped,
        missing_grid_slots=gaps, gap_interpretation='Missing UTC slots' if symbol.endswith('-USD') else 'Unclassified calendar gaps; includes market closures',
        age_hours=round((pd.Timestamp.now(tz='UTC') - frame.index[-1]).total_seconds() / 3600, 2))
    return frame, audit


def metrics(returns, periods):
    import quantstats as qs
    if len(returns) < 2:
        return {'observations': len(returns), 'note': 'At least two daily returns required for risk statistics.'}
    return clean(dict(observations=len(returns), total_return=float((1 + returns).prod() - 1),
        max_drawdown=float(qs.stats.max_drawdown(returns)),
        sharpe=float(qs.stats.sharpe(returns, periods=periods, rf=0)),
        sortino=float(qs.stats.sortino(returns, periods=periods, rf=0)),
        annualized_volatility=float(qs.stats.volatility(returns, periods=periods)), annualization=periods))


def curve(series, benchmark=None):
    # Bounded display; calculations always use the full series. Include final point.
    indices = sorted(set(range(0, len(series), max(1, math.ceil(len(series) / 1200)))) | {len(series) - 1})
    return [dict(time=series.index[i].isoformat(), value=float(series.iloc[i]),
                 benchmark=float(benchmark.iloc[i]) if benchmark is not None else None) for i in indices]


def same_currency(audits):
    currencies = {c for a in audits for c in a['currencies']}
    if any(len(a['currencies']) != 1 for a in audits) or len(currencies) != 1:
        raise ValueError('Symbols must have one common known currency; FX conversion is not available in this report.')
    return next(iter(currencies))


def strategy(p, database, progress):
    import pandas as pd
    import vectorbt as vbt
    progress('Checking source candles and building moving-average signals.')
    frame, audit = load_bars(p, p['symbols'][0], database)
    if len(frame) < max(100, p['slow'] * 2):
        raise ValueError('Choose a range with at least 100 bars and twice the slow moving-average period.')
    regime = frame.close.rolling(p['fast']).mean() > frame.close.rolling(p['slow']).mean()
    # No parameter search: last 30% is a chronological evaluation segment.
    split = max(p['slow'] + 1, int(len(frame) * .7))
    evaluation = frame.iloc[split:]
    entries = regime.shift(1, fill_value=False).iloc[split:]
    exits = (~regime).shift(1, fill_value=False).iloc[split:]
    progress('Running VectorBT on the final 30% with next-open execution and costs.')
    options = dict(price=evaluation.open, init_cash=p['capital'], fees=p['fee_bps'] / 10000,
        slippage=p['slippage_bps'] / 10000, freq=f'{SECONDS[p["interval"]]}s')
    portfolio = vbt.Portfolio.from_signals(evaluation.close, entries, exits, **options)
    benchmark = vbt.Portfolio.from_holding(evaluation.close, **options)
    equity, baseline = portfolio.value(), benchmark.value()
    daily = equity.resample('1D').last().dropna()
    returns = daily.pct_change().fillna(daily.iloc[0] / p['capital'] - 1)
    periods = 365 if p['symbols'][0].endswith('-USD') else 252
    progress('Computing QuantStats risk and benchmark reports.')
    trades = portfolio.trades.records_readable
    return dict(title='Moving-average strategy validation', engines=['vectorbt', 'quantstats'], audit=[audit],
        metrics=metrics(returns, periods), net_return=float(portfolio.total_return()), benchmark_return=float(benchmark.total_return()),
        trade_count=len(trades), trades=clean(trades.head(500).to_dict(orient='records')), trades_display_limit=500,
        curve=curve(equity, baseline), currency=same_currency([audit]), evaluation_start=evaluation.index[0].isoformat(),
        assumptions=['Long only; signals from the previous completed bar fill at the next available open.',
            'Fixed parameters; first 70% is warmup/research history, final 30% is evaluation. Repeated tuning on this segment invalidates the holdout.',
            'Fees and slippage apply on both sides. Open positions remain marked to the final close.',
            'Buy-and-hold benchmark uses the same capital, execution prices and costs. No stops or leverage.',
            'Risk metrics use daily equity returns, zero risk-free rate; displayed curves are sampled to at most 1,201 points.'])


def portfolio(p, database, progress):
    import pandas as pd
    frames, audits = {}, []
    for symbol in p['symbols']:
        progress(f'Checking daily history for {symbol}.')
        frame, audit = load_bars(p, symbol, database)
        frames[symbol] = frame.close
        audits.append(audit)
    currency = same_currency(audits)
    prices = pd.DataFrame(frames).dropna()
    if len(prices) < 20:
        raise ValueError('At least twenty common daily observations are required.')
    weights = pd.Series(p['weights'], index=p['symbols']) / 100
    # Initial allocations drift with prices; no implicit daily rebalancing.
    equity = (prices / prices.iloc[0]).mul(weights).sum(axis=1) * 10000
    returns = equity.pct_change().dropna()
    periods = 365 if p['symbols'][0].endswith('-USD') else 252
    progress('Computing QuantStats portfolio risk and correlations.')
    return dict(title='Portfolio performance & risk', engines=['quantstats'], audit=audits, currency=currency,
        metrics=metrics(returns, periods), curve=curve(equity),
        weights=dict(zip(p['symbols'], p['weights'])), correlation=prices.pct_change().dropna().corr().to_dict(),
        max_initial_weight=float(weights.max()), common_observations=len(prices),
        assumptions=['Illustrative 10,000-unit portfolio using supplied initial weights; buy-and-hold with drifting weights.',
            'Common observed dates only; no forward-filled prices. No transaction costs, taxes, FX or leverage.',
            'Adjusted daily prices used for Yahoo; exchange daily candles are unadjusted.',
            'Daily risk metrics use zero risk-free rate. Historical correlation does not bound future losses.'])


def factor(p, database, progress):
    import alphalens.performance as ap
    import numpy as np
    import pandas as pd
    prices, audits = {}, []
    for symbol in p['symbols']:
        progress(f'Loading factor universe: {symbol}.')
        frame, audit = load_bars(p, symbol, database)
        prices[symbol] = frame.close
        audits.append(audit)
    same_currency(audits)
    panel = pd.DataFrame(prices).dropna()
    if len(panel) < p['lookback'] + p['horizon'] + 20:
        raise ValueError('Not enough common dates for the selected factor lookback and forward horizon.')
    momentum = panel.pct_change(p['lookback'])
    forward = panel.shift(-p['horizon']) / panel - 1
    data = pd.concat([momentum.stack().rename('factor'), forward.stack().rename('1D')], axis=1).dropna()
    data.index.names = ['date', 'asset']
    # Ranking creates equal-count portfolios; explicit tie policy is disclosed.
    data['factor_quantile'] = data.groupby(level='date')['factor'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1).astype(int)
    progress('Evaluating momentum information coefficients, quintiles and turnover with Alphalens.')
    ic = ap.factor_information_coefficient(data)['1D']
    means, _ = ap.mean_return_by_quantile(data, demeaned=False)
    turnover = ap.quantile_turnover(data.factor_quantile, 5, period=1)
    return dict(title='Cross-sectional momentum factor', engines=['alphalens-reloaded'], audit=audits,
        metrics=clean(dict(mean_rank_ic=float(ic.mean()), ic_observations=len(ic),
            top_minus_bottom=float(means.loc[5, '1D'] - means.loc[1, '1D']),
            mean_top_quintile_turnover=float(turnover.mean()))),
        quintiles=[dict(quantile=int(q), mean_forward_return=float(row['1D'])) for q, row in means.iterrows()],
        curve=curve(ic), horizon=p['horizon'], lookback=p['lookback'], common_observations=len(panel), factor_observations=len(data),
        assumptions=[f'Momentum measured over {p["lookback"]} common observations; forward returns over {p["horizon"]} common observations.',
            'Only dates observed for every symbol are included. Missing dates are not filled; horizons count common observations, not calendar days.',
            'Equal-count quintiles; ties broken in input-symbol order. Forward returns are raw horizon returns, not annualized.',
            'Descriptive factor research with overlapping horizons, not an executable strategy or an out-of-sample profitability claim.',
            'User-selected current universe may contain survivorship/selection bias. No fees or turnover costs included.'])


def exchange(p, database, progress):
    import ccxt
    import pandas as pd
    source = p['source']
    client = getattr(ccxt, {'Bitstamp': 'bitstamp', 'Kraken': 'kraken', 'Coinbase': 'coinbase'}[source])(
        {'enableRateLimit': True, 'timeout': 20000})
    pair = p['symbols'][0][:-4] + '/USD'
    interval = '1h' if p['interval'] == '60m' else p['interval']
    start, end = bounds(p)
    progress(f'Loading public {source} spot markets.')
    client.load_markets()
    if pair not in client.markets or not client.markets[pair].get('spot') or not client.has.get('fetchOHLCV') or interval not in (client.timeframes or {}):
        raise ValueError(f'{source} does not expose this spot pair/timeframe through CCXT.')
    step = SECONDS[p['interval']] * 1000
    cursor, stop = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    candles, calls = {}, 0
    for _ in range(20):
        if cursor + step > stop:
            break
        progress(f'Fetching {source} {pair}: {len(candles):,} closed candles, request {calls + 1}/20.')
        batch = client.fetch_ohlcv(pair, timeframe=interval, since=cursor, limit=1000)
        calls += 1
        if not batch:
            break
        advance = max(row[0] for row in batch) + step
        if advance <= cursor:
            break
        for row in batch:
            stamp, o, h, l, c, volume = row[:6]
            if not cursor <= stamp or stamp + step > stop:
                continue
            if stamp % step or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row[:6]) or min(o, h, l, c) <= 0 or volume < 0 or not l <= min(o, c) <= max(o, c) <= h:
                raise ValueError('Exchange returned invalid OHLCV; no data stored.')
            candles[stamp] = (o, h, l, c, volume)
        cursor = advance
    if not candles:
        raise ValueError('Exchange returned no completed candles for this range.')
    progress('Waiting for collector lock to save validated exchange candles.')
    from market_data import connect, process_lock
    fetched = datetime.now(timezone.utc).isoformat()
    rows = [(p['symbols'][0], datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat(), p['interval'], source, *values, fetched)
            for stamp, values in sorted(candles.items())]
    for attempt in range(30):
        try:
            with process_lock(Path(database).parent / 'ingestion.lock'), closing(connect(database)) as db, db:
                db.executemany('INSERT INTO crypto_exchange_prices VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,timestamp,interval,source) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume,fetched_at=excluded.fetched_at', rows)
                db.execute('INSERT INTO intraday_state(symbol,interval,last_success) VALUES (?,?,?) ON CONFLICT(symbol,interval) DO NOTHING', (p['symbols'][0], p['interval'], fetched))
            break
        except RuntimeError:
            if attempt == 29:
                raise ValueError('Collector remained busy; retry the exchange pull shortly.') from None
            time.sleep(1)
    close = pd.Series([r[7] for r in rows], index=pd.to_datetime([r[1] for r in rows], utc=True))
    return dict(title=f'{source} {pair} candle pull', engines=['ccxt'], stored_rows=len(rows), requests=calls,
        first=rows[0][1], last=rows[-1][1], curve=curve(close), currency='USD',
        preview=[dict(time=r[1], open=r[4], high=r[5], low=r[6], close=r[7], volume=r[8]) for r in rows[-100:]],
        assumptions=['Public spot OHLCV only; quote currency USD. Incomplete candles excluded.',
            'Saved to the local exchange table with its venue label; existing other-source data is preserved.',
            'Exchange retention varies. At most twenty requests; returned history may not cover the requested start or every interval.',
            'Cancelling after a database commit retains those candles.'])


def run(p, database, progress=lambda message: None):
    p = validate(p)
    result = {'strategy': strategy, 'portfolio': portfolio, 'factor': factor, 'exchange': exchange}[p['kind']](p, database, progress)
    result.update(request=p, generated_at=datetime.now(timezone.utc).isoformat(),
                  versions={name: version(name) for name in result['engines']})
    return clean(result)


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    try:
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        result = run(request, Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'data' / 'market.sqlite', lambda message: write_json(folder / 'progress.json', {'message': message}))
    except Exception as exc:
        result = {'error': str(exc)}
    write_json(folder / 'result.json', result)
