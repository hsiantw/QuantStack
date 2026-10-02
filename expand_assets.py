"""Add international equities and crypto with validated daily price histories.

Uses the existing resumable collector and lock. Does not publish or deploy.
Only successful symbols are added to the recurring collector configuration.
"""
import json
import logging
from contextlib import closing
from datetime import datetime, timezone

from market_data import ROOT, DATA, connect, ingest, process_lock
from expand_stocks import atomic_json, metadata_schema


STOCKS = {
    '2317.TW': 'Hon Hai Precision', '2454.TW': 'MediaTek', '2308.TW': 'Delta Electronics',
    '2881.TW': 'Fubon Financial', '2303.TW': 'United Microelectronics',
    '0700.HK': 'Tencent', '9988.HK': 'Alibaba', '3690.HK': 'Meituan',
    '1810.HK': 'Xiaomi', '1211.HK': 'BYD', '0005.HK': 'HSBC', '1299.HK': 'AIA',
    '7203.T': 'Toyota', '6758.T': 'Sony', '9984.T': 'SoftBank Group',
    '7974.T': 'Nintendo', '8306.T': 'Mitsubishi UFJ', '8035.T': 'Tokyo Electron',
    'ASML.AS': 'ASML', 'SAP.DE': 'SAP', 'SIE.DE': 'Siemens', 'AIR.PA': 'Airbus',
    'MC.PA': 'LVMH', 'OR.PA': "L'Oreal", 'NESN.SW': 'Nestle', 'ROP.SW': 'Roche',
    'NOVN.SW': 'Novartis', 'AZN.L': 'AstraZeneca', 'SHEL.L': 'Shell', 'ULVR.L': 'Unilever',
    'RY.TO': 'Royal Bank of Canada', 'SHOP.TO': 'Shopify', 'ENB.TO': 'Enbridge',
    'BHP.AX': 'BHP', 'CBA.AX': 'Commonwealth Bank', 'CSL.AX': 'CSL',
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services',
    'INFY.NS': 'Infosys', 'HDFCBANK.NS': 'HDFC Bank',
}
CRYPTO = {
    'SOL': 'Solana', 'BNB': 'BNB', 'XRP': 'XRP', 'ADA': 'Cardano', 'DOGE': 'Dogecoin',
    'AVAX': 'Avalanche', 'DOT': 'Polkadot', 'LINK': 'Chainlink', 'LTC': 'Litecoin',
    'BCH': 'Bitcoin Cash', 'XLM': 'Stellar', 'UNI7083': 'Uniswap', 'AAVE': 'Aave',
    'ATOM': 'Cosmos', 'NEAR': 'NEAR Protocol', 'ICP': 'Internet Computer',
    'ETC': 'Ethereum Classic', 'FIL': 'Filecoin', 'HBAR': 'Hedera', 'SUI20947': 'Sui',
    'APT21794': 'Aptos', 'ARB11841': 'Arbitrum', 'OP': 'Optimism', 'INJ': 'Injective', 'TRX': 'TRON',
}


def expand():
    import yfinance as yf
    DATA.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(DATA / 'provider-cache'))
    config_path = ROOT / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    collection = dict(config, attempts=1, request_pause_seconds=0.5)
    targets = {**STOCKS, **{key+'-USD': name for key, name in CRYPTO.items()}}
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'requested': len(targets), 'succeeded': [], 'failed': [], 'added': []}
    report_path = DATA / 'asset-expansion.json'
    with process_lock(DATA / 'ingestion.lock'), closing(connect(DATA / 'market.sqlite')) as db:
        metadata_schema(db)
        consecutive_failures = 0
        for symbol, name in targets.items():
            before = db.execute('SELECT 1 FROM prices WHERE symbol=? LIMIT 1', (symbol,)).fetchone()
            prior_state = db.execute('SELECT 1 FROM state WHERE symbol=?', (symbol,)).fetchone()
            report['current'] = symbol
            atomic_json(report_path, report)
            # An exclusive UTC-day cutoff avoids provisional newest-session bars.
            failed = ingest(db, collection, [symbol], end=datetime.now(timezone.utc).date().isoformat())
            if failed:
                report['failed'].append(symbol)
                consecutive_failures += 1
                error = db.execute('SELECT error FROM state WHERE symbol=?', (symbol,)).fetchone()[0] or ''
                # Keep failed attempts for audit, but do not enroll a never-valid ticker.
                if not prior_state:
                    with db:
                        db.execute('DELETE FROM state WHERE symbol=? AND last_success IS NULL', (symbol,))
                if consecutive_failures >= 5 or any(word in error.lower() for word in ('429', 'rate limit', 'too many requests')):
                    report['status'] = 'paused_provider_errors'
                    break
                continue
            consecutive_failures = 0
            with db:
                db.execute('INSERT INTO stock_metadata(symbol,name) VALUES (?,?) '
                           'ON CONFLICT(symbol) DO UPDATE SET name=COALESCE(stock_metadata.name,excluded.name)',
                           (symbol, name))
            report['succeeded'].append(symbol)
            if not before:
                report['added'].append(symbol)
            # Reload before merging so unrelated configuration edits are preserved.
            current = json.loads(config_path.read_text(encoding='utf-8'))
            current['symbols'] = list(dict.fromkeys(current['symbols'] + [symbol]))
            atomic_json(config_path, current)
        else:
            report['status'] = 'complete_with_errors' if report['failed'] else 'complete'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    raise SystemExit(expand())
