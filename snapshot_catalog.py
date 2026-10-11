"""Keep configured instruments discoverable in daily-only market snapshots."""
import json


def snapshot_catalog(assets, config_path):
    assets = {asset['symbol']: dict(asset) for asset in assets}
    if config_path.exists():
        for symbol in json.loads(config_path.read_text(encoding='utf-8')).get('symbols', []):
            assets.setdefault(symbol, dict(symbol=symbol, name=symbol, has_daily=False))
    for symbol, asset in assets.items():
        # Older releases classified every crypto except BTC/ETH as stocks.
        if symbol.endswith('-USD'):
            asset['kind'] = 'Crypto'
        if asset.get('has_daily', True):
            continue
        # Intraday-only local quotes have no corresponding history in this snapshot.
        for field in ('date', 'close', 'change', 'quote_timestamp', 'volume',
                      'change_1d_pct', 'return_1w_pct', 'return_1m_pct', 'performance_asof'):
            asset[field] = None
        asset.update(has_data=False, quote_interval='1d',
                     kind='Crypto' if symbol.endswith('-USD') else 'Stocks')
    return sorted(assets.values(), key=lambda asset: asset['symbol'])
