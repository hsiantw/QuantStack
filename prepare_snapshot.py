"""Install QuantStack's versioned market-data release for same-origin serving."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
import urllib.request
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'market-snapshot.json'
STATIC = ROOT / 'QuantStack-main' / 'static' / 'market-data'
_lock = threading.Lock()


def data_file(name):
    return name in ('symbols.json', 'screener.json', 'snapshot.json') or bool(
        re.fullmatch(r'prices/[A-Za-z0-9%^=._+-]+\.json\.gz', name))


def prepare(manifest_path=MANIFEST, destination=STATIC, archive_path=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    digest = manifest['sha256']
    if not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('Invalid market snapshot checksum.')
    destination = Path(destination)
    target = destination / digest[:16]
    with _lock:
        if (target / '.complete').is_file():
            return target
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=destination) as temporary:
            temporary = Path(temporary)
            archive = Path(archive_path) if archive_path else temporary / 'market-data.zip'
            if archive_path is None:
                request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'QuantStack'})
                with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            with archive.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                    raise ValueError('Market snapshot checksum mismatch; existing data was preserved.')
            staging = temporary / 'data'
            staging.mkdir()
            with zipfile.ZipFile(archive) as source:
                names = source.namelist()
                decoded = [unquote(name) for name in names]
                if (any(not data_file(name) for name in names + decoded)
                        or len(decoded) != len(set(decoded))):
                    raise ValueError('Unexpected files in market data archive.')
                if not {'symbols.json', 'screener.json', 'snapshot.json'}.issubset(names):
                    raise ValueError('Market snapshot is missing its catalogs.')
                if sum(info.file_size for info in source.infolist()) > 900_000_000:
                    raise ValueError('Market snapshot exceeds the size budget.')
                for info in source.infolist():
                    output = staging / unquote(info.filename)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(info) as incoming, output.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            for name in ('symbols.json', 'screener.json', 'snapshot.json'):
                json.loads((staging / name).read_text(encoding='utf-8'))
            (staging / '.complete').write_text(digest, encoding='ascii')
            try:
                staging.rename(target)
            except OSError:
                # Another server process may have installed this identical version.
                if not (target / '.complete').is_file():
                    raise
        return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Install a local archive instead of downloading it.')
    args = parser.parse_args()
    print(f'QuantStack market data ready: {prepare(archive_path=args.archive)}', flush=True)
