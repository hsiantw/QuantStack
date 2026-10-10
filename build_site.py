"""Build a portable static dashboard from the local database."""
import gzip
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from dashboard import ROOT, DATABASE, catalog, database
from screener import snapshot as screener_snapshot
from snapshot_catalog import snapshot_catalog

OUTPUT = ROOT / 'site'
FIELDS = ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividends', 'splits']


def build():
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'prices').mkdir(exist_ok=True)
    for name in ('index.html', 'interview-prep.html', 'style.css', 'app.js', 'indicators.js', 'drawings.js', 'indicators.css', 'chart-scale.js', 'chart-scale.css', 'screener.js', 'screener.css', 'workspace.js', 'workspace.css', 'terminal.js', 'data-pulls.js', 'watchlists.js', 'watchlists.css', 'terminal.css', 'theme.js', 'theme.css', 'strategy-engine.js', 'strategy.js', 'markov-engine.js', 'markov-worker.js', 'markov.js', 'markov.css', 'brownian-engine.js', 'brownian-worker.js', 'brownian.js', 'risk-engine.js', 'risk.js', 'research-engine.js', 'research-worker.js', 'research.js', 'accounts.js', 'research.css', 'static-data.js'):
        shutil.copyfile(ROOT / 'web' / name, OUTPUT / name)
    index = (OUTPUT / 'index.html').read_text(encoding='utf-8-sig')
    index = index.replace('href="/style.css"', 'href="./style.css"').replace('href="/"', 'href="./"')
    for asset in ('app.js', 'indicators.js', 'drawings.js', 'indicators.css', 'chart-scale.js', 'chart-scale.css', 'screener.js', 'screener.css', 'workspace.js', 'workspace.css', 'terminal.js', 'data-pulls.js', 'watchlists.js', 'watchlists.css', 'terminal.css', 'theme.js', 'theme.css', 'strategy-engine.js', 'strategy.js', 'markov-engine.js', 'markov-worker.js', 'markov.js', 'markov.css', 'brownian-engine.js', 'brownian-worker.js', 'brownian.js'):
        index = index.replace('"/' + asset, '"./' + asset)
    index = index.replace('<script src="./app.js">', '<script src="./static-data.js"></script><script src="./app.js">')
    index = index.replace('Daily updates scheduled for 09:00 local time while logged in.',
                          'Daily snapshots published from the data collector. Check each asset for its latest date.')
    (OUTPUT / 'index.html').write_text(index, encoding='utf-8')
    (OUTPUT / '.nojekyll').touch()
    # Hosted snapshots currently carry daily histories only.
    symbols = snapshot_catalog(catalog(), ROOT / 'config.json')
    # A manifest restricts deployment to files from this build, excluding old symbols.
    files = ['index.html', 'interview-prep.html', 'style.css', 'app.js', 'indicators.js', 'drawings.js', 'indicators.css', 'chart-scale.js', 'chart-scale.css', 'screener.js', 'screener.css', 'workspace.js', 'workspace.css', 'terminal.js', 'data-pulls.js', 'watchlists.js', 'watchlists.css', 'terminal.css', 'theme.js', 'theme.css', 'strategy-engine.js', 'strategy.js', 'markov-engine.js', 'markov-worker.js', 'markov.js', 'markov.css', 'brownian-engine.js', 'brownian-worker.js', 'brownian.js', 'risk-engine.js', 'risk.js', 'research-engine.js', 'research-worker.js', 'research.js', 'accounts.js', 'research.css', 'static-data.js', '.nojekyll', 'symbols.json', 'snapshot.json', 'screener.json']
    (OUTPUT / 'symbols.json').write_text(json.dumps(symbols, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    (OUTPUT / 'screener.json').write_text(json.dumps(screener_snapshot(DATABASE), separators=(',', ':'), allow_nan=False), encoding='utf-8')
    count = 0
    with database() as db:
        for asset in symbols:
            rows = db.execute('SELECT ' + ','.join(FIELDS) + ' FROM prices WHERE symbol=? ORDER BY date', (asset['symbol'],)).fetchall()
            # Arrays avoid repeating field names millions of times. Gzip is decoded in the browser.
            payload = json.dumps({'columns': FIELDS, 'rows': [list(row) for row in rows]}, separators=(',', ':'), allow_nan=False).encode()
            filename = 'prices/' + quote(asset['symbol'], safe='-._') + '.json.gz'
            (OUTPUT / filename).write_bytes(gzip.compress(payload, mtime=0))
            files.append(filename)
            count += len(rows)
    snapshot = {'built_at': datetime.now(timezone.utc).isoformat(), 'symbols': len(symbols), 'records': count}
    (OUTPUT / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf-8')
    archive = ROOT / 'data' / 'site.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for name in files:
            handle.write(OUTPUT / name, name)
    size = sum((OUTPUT / name).stat().st_size for name in files)
    if size > 900_000_000:
        raise RuntimeError('Site exceeds the configured GitHub Pages size budget')
    print(f'Built {len(symbols)} assets, {count:,} records, {size / 1e6:.1f} MB; archive: {archive}', flush=True)
    return archive


if __name__ == '__main__':
    build()
