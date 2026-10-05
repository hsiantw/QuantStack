"""Capture observed Binance USD-M forced liquidation orders for local charts."""
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sqlite3

from aiohttp import WSMsgType

from market_data import connect

STREAM_URL = 'wss://fstream.binance.com/ws/!forceOrder@arr'


def force_order_rows(payload, symbols):
    """Normalize filled USDT-margined force orders for configured USD symbols."""
    messages = payload if isinstance(payload, list) else [payload]
    output = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        order = message.get('o')
        if not isinstance(order, dict) or order.get('X') != 'FILLED':
            continue
        contract = order.get('s')
        if not isinstance(contract, str) or not contract.endswith('USDT'):
            continue
        symbol = contract[:-4] + '-USD'
        if symbol not in symbols:
            continue
        side = {'SELL': 'long', 'BUY': 'short'}.get(order.get('S'))
        if side is None:
            continue
        try:
            price = float(order.get('ap') or order.get('p'))
            quantity = float(order.get('z') or order.get('q'))
            timestamp = int(order.get('T') or message['E'])
            order_id = str(order['i'])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity <= 0:
            continue
        instant = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        output.append((symbol, instant.isoformat(timespec='milliseconds'), order_id,
                       side, price, quantity, price * quantity, 'USDT'))
    return output


def persist_force_orders(database_path, events):
    if not events:
        return 0
    db = connect(database_path)
    try:
        with db:
            cursor = db.executemany('''INSERT OR IGNORE INTO liquidation_events
              (symbol,timestamp,order_id,side,price,quantity,notional,quote_asset)
              VALUES (?,?,?,?,?,?,?,?)''', events)
            inserted = cursor.rowcount
            db.execute('''UPDATE liquidation_collector_state
              SET status='connected',last_event_at=?,error=NULL WHERE id=1''',
                       (max(event[1] for event in events),))
        return inserted
    finally:
        db.close()


def update_collector_state(database_path, status, *, started_at=None, connected_at=None, error=None):
    db = connect(database_path)
    try:
        with db:
            db.execute('''INSERT INTO liquidation_collector_state
              (id,status,started_at,connected_at,last_event_at,error) VALUES (1,?,?,?,?,?)
              ON CONFLICT(id) DO UPDATE SET status=excluded.status,
              started_at=COALESCE(excluded.started_at,liquidation_collector_state.started_at),
              connected_at=COALESCE(excluded.connected_at,liquidation_collector_state.connected_at),
              error=excluded.error''',
                       (status, started_at, connected_at, None, error))
    finally:
        db.close()


async def collect_force_orders(session, database_path, symbols):
    symbols = frozenset(symbol for symbol in symbols if symbol.endswith('-USD'))
    started_at = datetime.now(timezone.utc).isoformat()
    update_collector_state(database_path, 'connecting', started_at=started_at)
    try:
        while True:
            try:
                async with session.ws_connect(STREAM_URL, heartbeat=30, max_msg_size=4 * 1024 * 1024) as socket:
                    connected_at = datetime.now(timezone.utc).isoformat()
                    update_collector_state(database_path, 'connected', connected_at=connected_at)
                    logging.info('Connected to Binance public liquidation stream')
                    while True:
                        message = await socket.receive()
                        if message.type == WSMsgType.TEXT:
                            try:
                                payload = json.loads(message.data)
                            except json.JSONDecodeError as exc:
                                logging.warning('Ignoring malformed Binance liquidation message: %s', exc)
                                continue
                            events = force_order_rows(payload, symbols)
                            if events:
                                await asyncio.to_thread(persist_force_orders, database_path, events)
                        elif message.type in (WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.CLOSING):
                            raise ConnectionError('Binance liquidation stream closed')
                        elif message.type == WSMsgType.ERROR:
                            raise socket.exception() or ConnectionError('Binance liquidation stream failed')
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.warning('Binance liquidation stream disconnected: %s', exc)
                await asyncio.to_thread(update_collector_state, database_path, 'reconnecting', error=str(exc))
                await asyncio.sleep(5)
    finally:
        await asyncio.to_thread(update_collector_state, database_path, 'stopped')


def read_liquidations(database_path, symbol, start, end, interval):
    if not symbol.endswith('-USD'):
        raise ValueError('Liquidation events are available for configured crypto USD symbols.')
    if interval not in ('1m', '5m', '15m', '1h', '1d', '1w', '1mo'):
        raise ValueError('Interval must be 1m, 5m, 15m, 1h, 1d, 1w, or 1mo.')
    db = connect(database_path)
    try:
        db.row_factory = sqlite3.Row
        rows = db.execute('''SELECT timestamp,side,notional FROM liquidation_events
          WHERE symbol=? AND timestamp>=? AND timestamp<? ORDER BY timestamp''',
                          (symbol, start, end)).fetchall()
        state = db.execute('SELECT status,started_at,connected_at,last_event_at,error FROM liquidation_collector_state WHERE id=1').fetchone()
    finally:
        db.close()
    seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    grouped = {}
    for row in rows:
        instant = datetime.fromisoformat(row['timestamp']).astimezone(timezone.utc)
        if interval in seconds:
            epoch = int(instant.timestamp())
            bucket = datetime.fromtimestamp(epoch - epoch % seconds[interval], timezone.utc)
            key = bucket.isoformat(timespec='seconds')
        elif interval == '1w':
            bucket = instant.date()
            bucket = bucket.fromordinal(bucket.toordinal() - bucket.weekday())
            key = bucket.isoformat()
        elif interval == '1mo':
            key = instant.strftime('%Y-%m-01')
        else:
            key = instant.date().isoformat()
        bar = grouped.setdefault(key, {'date': key, 'longs': 0.0, 'shorts': 0.0, 'count': 0})
        bar['longs' if row['side'] == 'long' else 'shorts'] += row['notional']
        bar['count'] += 1
    result = {
        'venue': 'Binance USD-M Futures',
        'quote_asset': 'USDT',
        'coverage': 'Locally captured public force-order stream; partial and not exchange-wide liquidation totals.',
        'bars': list(grouped.values()),
        'event_count': len(rows),
        'first_event': rows[0]['timestamp'] if rows else None,
        'last_event': rows[-1]['timestamp'] if rows else None,
        'collector': dict(state) if state else {'status': 'not_started'},
    }
    return result
