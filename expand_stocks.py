"""Discover US-listed equities by market cap, then resume history and profile collection.

Uses the existing Yahoo Finance collector. No publishing or schedule installation.
"""
import argparse
import csv
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_data import DATA, ROOT, connect, ingest, process_lock

METADATA_FIELDS = ('symbol', 'name', 'market_cap', 'sector', 'industry', 'country',
                   'exchange', 'currency', 'pe', 'forward_pe', 'pb', 'dividend_yield',
                   'revenue_growth', 'profit_margin', 'beta', 'fetched_at', 'source')
UNIVERSE = DATA / 'stock-universe.json'
PROGRESS = DATA / 'stock-expansion.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def metadata_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS stock_metadata (
            symbol TEXT PRIMARY KEY, name TEXT, market_cap REAL, sector TEXT,
            industry TEXT, country TEXT, exchange TEXT, currency TEXT,
            pe REAL, forward_pe REAL, pb REAL, dividend_yield REAL,
            revenue_growth REAL, profit_margin REAL, beta REAL,
            fetched_at TEXT, source TEXT);
        CREATE TABLE IF NOT EXISTS stock_enrichment (
            symbol TEXT PRIMARY KEY, last_success TEXT, error TEXT);
    ''')


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def write_metadata(db, symbol, quote, profile=False):
    dividend_yield = finite(quote.get('trailingAnnualDividendYield'))
    # Yahoo's dividendYield is displayed in percent; trailingAnnualDividendYield
    # is a ratio. Keep the ratio as the storage contract for the screener.
    if dividend_yield is None and finite(quote.get('dividendYield')) is not None:
        dividend_yield = finite(quote['dividendYield']) / 100
    values = dict(symbol=symbol, name=quote.get('longName') or quote.get('shortName'),
                  market_cap=finite(quote.get('marketCap')), sector=quote.get('sector'),
                  industry=quote.get('industry'), country=quote.get('country'),
                  exchange=quote.get('exchange'), currency=quote.get('currency'),
                  pe=finite(quote.get('trailingPE')), forward_pe=finite(quote.get('forwardPE')),
                  pb=finite(quote.get('priceToBook')), dividend_yield=dividend_yield,
                  revenue_growth=finite(quote.get('revenueGrowth')),
                  profit_margin=finite(quote.get('profitMargins')), beta=finite(quote.get('beta')),
                  fetched_at=now(), source='Yahoo Finance')
    columns = ','.join(METADATA_FIELDS)
    financials = {'pe', 'forward_pe', 'pb', 'dividend_yield', 'revenue_growth', 'profit_margin', 'beta'}
    # A new full profile can legitimately stop reporting a P/E or dividend.
    # Do not stamp an obsolete financial figure with a fresh profile date.
    updates = ','.join(f'{key}=excluded.{key}' if profile and key in financials else
                       f'{key}=COALESCE(excluded.{key},stock_metadata.{key})'
                       for key in METADATA_FIELDS if key != 'symbol')
    with db:
        db.execute(f'INSERT INTO stock_metadata ({columns}) VALUES ({",".join("?" for _ in METADATA_FIELDS)}) '
                   f'ON CONFLICT(symbol) DO UPDATE SET {updates}', tuple(values[k] for k in METADATA_FIELDS))
        if profile:
            db.execute('INSERT INTO stock_enrichment VALUES (?,?,NULL) ON CONFLICT(symbol) '
                       'DO UPDATE SET last_success=excluded.last_success,error=NULL', (symbol, now()))


def discover(db, limit):
    import yfinance as yf
    query = yf.EquityQuery('and', [
        yf.EquityQuery('eq', ['region', 'us']),
        yf.EquityQuery('is-in', ['exchange', 'NMS', 'NYQ', 'NGM', 'NCM', 'ASE']),
        yf.EquityQuery('gt', ['intradaymarketcap', 0]),
    ])
    quotes = {}
    provider_total = None
    for offset in range(0, limit, 250):
        result = yf.screen(query, offset=offset, size=min(250, limit-offset),
                           sortField='intradaymarketcap', sortAsc=False)
        page = result.get('quotes', [])
        provider_total = result.get('total')
        if not page:
            break
        for quote in page:
            symbol = quote.get('symbol', '')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.^=-]{0,24}', symbol):
                continue
            if quote.get('quoteType') != 'EQUITY' or finite(quote.get('marketCap')) is None:
                continue
            quotes[symbol] = quote
            write_metadata(db, symbol, quote)
        logging.info('Discovered %s / %s ranked equities', len(quotes), min(limit, provider_total or limit))
        time.sleep(0.6)
    if not quotes:
        raise RuntimeError('The provider returned no stock universe; existing coverage was preserved.')
    ordered = sorted(quotes, key=lambda symbol: (-float(quotes[symbol]['marketCap']), symbol))
    snapshot = dict(observed_at=now(), source='Yahoo Finance equity screener',
                    market='US listings', sort='market_cap_desc', provider_total=provider_total,
                    requested_limit=limit, symbols=ordered)
    atomic_json(UNIVERSE, snapshot)
    return ordered


def seed_sectors(db):
    path = DATA / 'constituents.csv'
    if not path.exists():
        return
    with path.open(encoding='utf-8-sig') as handle, db:
        for row in csv.DictReader(handle):
            symbol = row['Symbol'].replace('.', '-')
            sector = row.get('GICS Sector') or row.get('Sector')
            industry = row.get('GICS Sub-Industry') or row.get('Industry')
            db.execute('UPDATE stock_metadata SET sector=COALESCE(sector,?),industry=COALESCE(industry,?) '
                       'WHERE symbol=?', (sector, industry, symbol))


def expand(limit=3000, history_limit=None, refresh=False, metadata_only=False, stop_after_minutes=None):
    import yfinance as yf
    DATA.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(DATA / 'provider-cache'))
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    # Provider failures remain retryable on a subsequent run instead of stalling
    # the ranked queue for minutes on an unavailable ticker.
    config.update(attempts=1, request_pause_seconds=0.35)
    started = time.monotonic()
    status = dict(status='running', stage='discovering', started_at=now(), updated_at=now(),
                  target=limit, completed=0, histories_added=0, profiles_updated=0, failures=[])
    atomic_json(PROGRESS, status)
    db = connect(DATA / 'market.sqlite')
    try:
        metadata_schema(db)
        cached = json.loads(UNIVERSE.read_text(encoding='utf-8')) if UNIVERSE.exists() else None
        fresh = cached and datetime.fromisoformat(cached['observed_at']) > datetime.now(timezone.utc)-timedelta(days=7)
        symbols = cached['symbols'][:limit] if not refresh and fresh and cached.get('requested_limit', 0) >= limit else discover(db, limit)
        seed_sectors(db)
        status.update(stage='metadata ready' if metadata_only else 'collecting', target=len(symbols))
        atomic_json(PROGRESS, status)
        if metadata_only:
            status['status']='complete'
            return status
        # Retain configured Taiwan equities and collect their company metadata too.
        extras = [s for s in config['symbols'] if s not in symbols and not s.endswith('-USD')]
        queue = symbols[:history_limit] if history_limit else symbols
        queue = list(dict.fromkeys(queue + extras))
        status['target'] = len(queue)
        consecutive_errors = 0
        for index, symbol in enumerate(queue, 1):
            if stop_after_minutes and time.monotonic()-started >= stop_after_minutes*60:
                status.update(status='paused', stage='time limit reached; rerun to resume')
                break
            status.update(current_symbol=symbol, updated_at=now())
            atomic_json(PROGRESS, status)
            had_history = bool(db.execute('SELECT 1 FROM prices WHERE symbol=? LIMIT 1', (symbol,)).fetchone())
            # Give scheduled collectors a chance between symbols. A busy lock
            # leaves this symbol queued for the next invocation.
            try:
                with process_lock(DATA / 'ingestion.lock'):
                    failures = ingest(db, config, [symbol])
            except RuntimeError as exc:
                status.update(status='paused', stage=str(exc))
                break
            if failures:
                consecutive_errors += 1
                status['failures'].append(symbol)
                if consecutive_errors >= 8:
                    status.update(status='paused', stage='provider errors; rerun later to resume')
                    break
            else:
                consecutive_errors = 0
                if not had_history:
                    status['histories_added'] += 1
            previous = db.execute('SELECT last_success FROM stock_enrichment WHERE symbol=?', (symbol,)).fetchone()
            needs_profile = not previous or not previous[0] or datetime.fromisoformat(previous[0]) < datetime.now(timezone.utc)-timedelta(days=7)
            if needs_profile and not failures:
                try:
                    info = yf.Ticker(symbol).get_info()
                    if not info or not info.get('symbol'):
                        raise ValueError('Company profile was not returned')
                    write_metadata(db, symbol, info, profile=True)
                    status['profiles_updated'] += 1
                except Exception as exc:
                    with db:
                        db.execute('INSERT INTO stock_enrichment(symbol,error) VALUES (?,?) '
                                   'ON CONFLICT(symbol) DO UPDATE SET error=excluded.error', (symbol, str(exc)))
                    logging.warning('%s profile unavailable: %s', symbol, exc)
            status.update(completed=index, updated_at=now())
            atomic_json(PROGRESS, status)
            logging.info('Rank %s/%s %s; %s new histories, %s profiles', index, len(queue), symbol,
                         status['histories_added'], status['profiles_updated'])
        else:
            status.update(status='complete', stage='complete')
        return status
    except Exception as exc:
        status.update(status='failed', stage=str(exc))
        raise
    finally:
        status['updated_at']=now()
        atomic_json(PROGRESS, status)
        db.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--history-limit', type=int)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--stop-after-minutes', type=float)
    args=parser.parse_args()
    if not 1 <= args.limit <= 10000 or (args.history_limit is not None and args.history_limit < 1):
        parser.error('Use a universe limit between 1 and 10000 and a positive history limit.')
    DATA.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(DATA/'stock-expansion.log',encoding='utf-8')])
    with process_lock(DATA/'stock-expansion.lock'):
        print(json.dumps(expand(args.limit,args.history_limit,args.refresh,args.metadata_only,args.stop_after_minutes),indent=2))
