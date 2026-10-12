"""Bounded, public Ethereum activity lookup; no attribution or risk scoring."""
import asyncio
import re
from datetime import datetime, timezone
from decimal import Decimal, localcontext

from aiohttp import ClientSession, ClientTimeout, web

SOURCE = 'https://github.com/tornadocash/tornado-cli/blob/master/config.js'
POOL_GROUPS = {
    'ETH': ['12D66f87A04A9E220743712cE6d9bB1B5616B8Fc', '47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936', '910Cbd523D972eb0a6f4cAe4618aD62622b39DbF', 'A160cdAB225685dA1d56aa342Ad8841c3b53f291'],
    'DAI': ['D4B88Df4D29F5CedD6857912842cff3b20C8Cfa3', 'FD8610d20aA15b7B2E3Be39B396a1bC3516c7144', '07687e702b410Fa43f4cB4Af7FA097918ffD2730', '23773E65ed146A459791799d01336DB287f25334'],
    'cDAI': ['22aaA7720ddd5388A3c0A3333430953C68f1849b', '03893a7c7463AE47D46bc7f091665f1893656003', '2717c5e28cf931547B621a5dddb772Ab6A35B701', 'D21be7248e0197Ee08E0c20D4a96DEBdaC3D20Af'],
    'USDC': ['d96f2B1c14Db8458374d9Aca76E26c3D18364307', '4736dCf1b7A3d580672CcE6E7c65cd5cc9cFBa9D'],
    'USDT': ['169AD27A470D064DEDE56a2D3ff727986b15D52B', '0836222F2B2B24A3F36f98668Ed8F0B38D1a872f'],
    'WBTC': ['178169B423a011fff22B9e3F3abeA13414dDD0F1', '610B717796ad172B316836AC95a2ffad065CeaB4', 'bB93e510BbCD0B7beb5A853875f9eC60275CF498'],
}
CONTRACTS = {'0x' + address.lower(): f'Tornado Cash classic {token} pool'
             for token, addresses in POOL_GROUPS.items() for address in addresses}
CONTRACTS['0xd90e2f925da726b50c4ed8d0fb90ad053324f31b'] = 'Tornado Cash router'
FEEDS = {'transactions': 'txlist', 'internal': 'txlistinternal', 'tokens': 'tokentx'}


def amount(value, decimals=18):
    try:
        decimals = int(decimals)
        if not 0 <= decimals <= 36 or not re.fullmatch(r'\d{1,100}', str(value)):
            return None
        with localcontext() as context:
            context.prec = 140
            return format(Decimal(value).scaleb(-decimals), 'f')
    except (ValueError, ArithmeticError):
        return None


def normalize(row, feed, address):
    sender, recipient = str(row.get('from', '')).lower(), str(row.get('to', '')).lower()
    counterpart = recipient if sender == address else sender
    failed = str(row.get('isError', '')) == '1' or str(row.get('txreceipt_status', '')) == '0'
    status = 'failed' if failed else ('recorded' if feed == 'tokens' else
             'successful' if str(row.get('isError', '')) == '0' else 'unknown')
    txhash = str(row.get('hash') or row.get('transactionHash') or '')
    if not re.fullmatch(r'0x[0-9a-fA-F]{64}', txhash):
        txhash = ''
    return dict(feed=feed, timestamp=int(row['timeStamp']), sender=sender[:42],
                recipient=recipient[:42], hash=txhash, status=status,
                direction='self' if sender == recipient == address else 'outgoing' if sender == address else 'incoming',
                amount=amount(row.get('value', '0'), row.get('tokenDecimal', 18) if feed == 'tokens' else 18),
                asset=str(row.get('tokenSymbol', 'Unknown token'))[:30] if feed == 'tokens' else 'ETH',
                call_type=str(row.get('callType', row.get('type', '')))[:30] if feed == 'internal' else '',
                match=CONTRACTS.get(counterpart), counterpart=counterpart[:42])


async def fetch_feed(session, action, address, limit):
    try:
        async with session.get('https://eth.blockscout.com/api', params={
                'module': 'account', 'action': action, 'address': address,
                'page': 1, 'offset': limit, 'sort': 'desc'}, allow_redirects=False) as response:
            response.raise_for_status()
            # Reject unexpectedly large provider responses before decoding.
            chunks, size = [], 0
            async for chunk in response.content.iter_chunked(65536):
                size += len(chunk)
                if size > 12 * 1024 * 1024:
                    raise ValueError('Response too large')
                chunks.append(chunk)
            import json
            payload = json.loads(b''.join(chunks))
        if payload.get('status') in ('1', '2'):
            return payload
        if payload.get('result') == [] and 'no transactions' in str(payload.get('message', '')).lower():
            return payload
    except (Exception, asyncio.TimeoutError):
        pass
    return {'error': 'Provider unavailable or rate limited. Retry later.'}


async def scan(address, days, limit):
    now = int(datetime.now(timezone.utc).timestamp())
    cutoff = now - days * 86400
    async with ClientSession(headers={'User-Agent': 'QuantStack/1.0', 'Accept': 'application/json'},
                             timeout=ClientTimeout(total=35)) as session:
        results = await asyncio.gather(*(fetch_feed(session, action, address, limit)
                                       for action in [*FEEDS.values(), 'balance']))
    activity, coverage = [], []
    for (feed, _), payload in zip(FEEDS.items(), results):
        rows = payload.get('result')
        if not isinstance(rows, list):
            coverage.append(dict(feed=feed, state='unavailable', count=0))
            continue
        parsed, invalid = [], False
        for row in rows[:limit]:
            try:
                parsed.append(normalize(row, feed, address))
            except (ValueError, TypeError, KeyError):
                invalid = True
        limited = len(rows) >= limit and (not parsed or min(r['timestamp'] for r in parsed) >= cutoff)
        partial = payload.get('status') == '2' or invalid
        coverage.append(dict(feed=feed, state='partial' if partial else 'capped' if limited else 'window scanned',
                             count=len(parsed), capped=limited,
                             note=str(payload.get('message', ''))[:180] if partial else ''))
        activity.extend(r for r in parsed if cutoff <= r['timestamp'] <= now)
    activity.sort(key=lambda r: r['timestamp'], reverse=True)
    return dict(address=address, chain='Ethereum mainnet', days=days, limit=limit,
                scanned_at=datetime.now(timezone.utc).isoformat(), coverage=coverage,
                balance=amount(results[-1].get('result')) if results[-1].get('status') == '1' else None,
                known_contract=CONTRACTS.get(address), activity=activity,
                registry={'source': SOURCE, 'checked': '2026-10-12', 'contracts': CONTRACTS})


async def scan_handler(request):
    address = request.query.get('address', '').strip()
    try:
        days, limit = int(request.query.get('days', '30')), int(request.query.get('limit', '100'))
        if not re.fullmatch(r'0x[0-9a-fA-F]{40}', address) or days not in (7, 30, 90) or limit not in (100, 250):
            raise ValueError()
    except ValueError:
        return web.json_response({'error': 'Enter a valid Ethereum address and supported scan settings.'}, status=400)
    return web.json_response(await scan(address.lower(), days, limit), headers={'Cache-Control': 'no-store'})
