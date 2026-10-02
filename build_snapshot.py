"""Package only market data for the QuantStack release, never a second website."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from prepare_snapshot import ROOT, data_file


def build(archive=None):
    if archive is None:
        from build_site import build as build_portable
        archive = build_portable()
    output = ROOT / 'data' / 'market-data.zip'
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as target:
        for name in source.namelist():
            if data_file(name):
                with source.open(name) as incoming, target.open(name, 'w') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        metadata = json.loads(source.read('snapshot.json'))
    with output.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    versioned = output.with_name('market-data-' + digest[:16] + '.zip')
    output.replace(versioned)
    manifest = dict(metadata, sha256=digest,
                    url='https://github.com/hsiantw/QuantStack/releases/download/market-data/market-data.zip')
    return versioned, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-archive', type=Path)
    args = parser.parse_args()
    archive, manifest = build(args.from_archive)
    print(json.dumps(dict(archive=str(archive), **manifest), indent=2))
