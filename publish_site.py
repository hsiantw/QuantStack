"""Publish market data to QuantStack and advance its Render deployment manifest."""
import argparse
import json
import subprocess
from pathlib import Path

from build_snapshot import build
from github_cli import run as gh
from prepare_snapshot import ROOT, MANIFEST

REPOSITORY = 'hsiantw/QuantStack'


def publish(archive=None, push=True):
    config_path = ROOT / 'deployment.json'
    if config_path.exists():
        configured = json.loads(config_path.read_text(encoding='utf-8'))['repository']
        if configured != REPOSITORY:
            raise ValueError(f'Publishing is consolidated in {REPOSITORY}; update deployment.json.')
    archive, manifest = build(archive)
    try:
        gh(['release', 'view', 'market-data', '--repo', REPOSITORY], capture_output=True, text=True)
    except subprocess.CalledProcessError:
        gh(['release', 'create', 'market-data', '--repo', REPOSITORY,
            '--title', 'QuantStack market data', '--notes',
            'Versioned daily market data consumed by the QuantStack application.'])
    gh(['release', 'upload', 'market-data', str(archive), '--clobber', '--repo', REPOSITORY])
    release = json.loads(gh(['api', f'repos/{REPOSITORY}/releases/tags/market-data'],
                            capture_output=True, text=True).stdout)
    uploaded = next(asset for asset in release['assets'] if asset['name'] == archive.name)
    if uploaded.get('digest') != 'sha256:' + manifest['sha256']:
        raise RuntimeError('Uploaded snapshot digest was not verified; deployment manifest was not changed.')
    manifest['url'] = uploaded['browser_download_url']
    manifest['asset_id'] = uploaded['id']
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if push:
        subprocess.run(['git', 'add', 'market-snapshot.json'], cwd=ROOT, check=True)
        changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', 'market-snapshot.json'], cwd=ROOT)
        if changed.returncode == 1:
            subprocess.run(['git', 'commit', '--only', 'market-snapshot.json', '-m', 'Refresh QuantStack market snapshot'], cwd=ROOT, check=True)
            subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
        elif changed.returncode:
            raise RuntimeError('Could not inspect the snapshot manifest change.')
    print(f'QuantStack snapshot uploaded: {manifest["symbols"]} symbols, {manifest["records"]:,} records.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-archive', type=Path, help='Migrate a portable snapshot without rebuilding prices.')
    parser.add_argument('--no-push', action='store_true', help='Prepare the manifest for inclusion in a source commit.')
    args = parser.parse_args()
    publish(args.from_archive, push=not args.no_push)
