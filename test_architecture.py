"""Architecture guide: real routing, portable packaging and browser interactions."""
import asyncio
from contextlib import nullcontext
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer
from playwright.sync_api import sync_playwright, expect

from architecture_assets import ARCHITECTURE_ASSETS
import build_site
from dashboard import Handler
from serve_quantstack import create_app

ROOT = Path(__file__).parent


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


class ArchitectureTests(unittest.TestCase):
    def test_browser_diagrams_files_downloads_and_responsive_layout(self):
        (ROOT / '.test-tmp').mkdir(exist_ok=True)
        server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1050})
                errors = []
                external = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: external.append(request.url) if not request.url.startswith('http://127.0.0.1:') else None)
                page.goto(f'http://127.0.0.1:{server.server_port}/architecture.html')
                expect(page.locator('#mapNav a')).to_have_count(9)
                for name in ['overview','modes','collection','browser','storage','accounts','automation','deployment','merkle']:
                    page.locator(f'[data-diagram="{name}"]').click()
                    expect(page.locator(f'[data-diagram="{name}"]')).to_have_attribute('aria-current', 'true')
                    self.assertGreater(page.locator('.diagram-node').count(), 5)
                    page.locator('.diagram-node').last.focus()
                    page.keyboard.press('Enter')
                    expect(page.locator('.diagram-node').last).to_have_attribute('aria-pressed', 'true')
                    # No label extends beyond its node rectangle.
                    self.assertTrue(page.evaluate('''() => [...document.querySelectorAll('.diagram-node')].every(node => {
                        const box=node.querySelector('rect').getBBox();
                        return [...node.querySelectorAll('text')].every(text => {const r=text.getBBox();return r.x>=box.x && r.x+r.width<=box.x+box.width;});
                    })'''))
                page.locator('#zoomIn').click()
                expect(page.locator('#zoomLabel')).to_have_text('125%')
                page.locator('#zoomFit').click()
                expect(page.locator('#zoomLabel')).to_have_text('100%')
                with page.expect_download() as pending:
                    page.locator('#downloadDiagram').click()
                svg = pending.value.path().read_text(encoding='utf-8')
                self.assertIn('4a66ce787ea62318c6bef0aa1dca373e7cef7d9b', svg)
                self.assertIn('<svg', svg)
                self.assertNotIn('http://cdn', svg)
                page.locator('#fileSearch').fill('serve_quantstack.py')
                expect(page.locator('#fileTree button')).to_have_count(1)
                page.locator('#fileTree button').click()
                expect(page.locator('#fileMetadata')).to_contain_text('16f6ded2f211f9619bdd9e1412e8daa95cbd324f')
                expect(page.locator('#fileSource')).to_have_attribute('href', 'https://github.com/hsiantw/QuantStack/blob/da8f8564c8266f9210e880abf68129ec2eea2134/serve_quantstack.py')
                page.locator('#fileSearch').fill('missing-file-xyz')
                expect(page.locator('.empty')).to_be_visible()
                page.locator('#fileSearch').fill('')
                expect(page.locator('#fileTree button')).to_have_count(261)
                with page.expect_download() as download:
                    page.locator('.download-links a[href="./project-git-objects.tsv"]').click()
                self.assertEqual(len(download.value.path().read_text(encoding='utf-8').splitlines()), 16990)
                page.locator('#mapTheme').select_option('dark')
                page.reload()
                expect(page.locator('html')).to_have_attribute('data-theme', 'dark')
                expect(page.locator('[data-diagram="merkle"]')).to_have_attribute('aria-current', 'true')
                page.locator('#mapTheme').select_option('light')
                page.locator('[data-diagram="overview"]').click()
                page.evaluate('scrollTo(0,0)')
                page.screenshot(path=str(ROOT / '.test-tmp' / 'architecture-desktop.png'), full_page=True)
                for width, height in [(768, 900), (390, 844)]:
                    page.set_viewport_size({'width': width, 'height': height})
                    self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                    page.locator('[data-diagram="deployment"]').click()
                    expect(page.locator('#diagramTitle')).to_contain_text('Data release')
                    page.locator('#zoomFit').click()
                    self.assertTrue(page.evaluate('diagramViewport.scrollWidth <= diagramViewport.clientWidth'))
                page.evaluate('scrollTo(0,0)')
                page.screenshot(path=str(ROOT / '.test-tmp' / 'architecture-mobile.png'), full_page=True)
                self.assertFalse(errors, errors)
                self.assertFalse(external, external)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_portable_archive_contains_guide_and_all_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'web', root / 'web')
            (root / 'data').mkdir()
            (root / 'config.json').write_text(json.dumps({'symbols': []}))
            with patch.object(build_site, 'ROOT', root), patch.object(build_site, 'OUTPUT', root / 'site'), \
                 patch.object(build_site, 'catalog', return_value=[]), \
                 patch.object(build_site, 'screener_snapshot', return_value={}), \
                 patch.object(build_site, 'database', return_value=nullcontext(None)):
                archive = build_site.build()
            with zipfile.ZipFile(archive) as package:
                for name, (source, _) in ARCHITECTURE_ASSETS.items():
                    self.assertEqual(package.read(name), source.read_bytes(), name)


class ArchitectureGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_mode_serves_only_explicit_guide_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            async with TestServer(create_app('http://127.0.0.1:1', snapshot_dir=root,
                                            accounts_path=root / 'accounts.sqlite')) as server:
                async with ClientSession() as client:
                    for name, (source, mime) in ARCHITECTURE_ASSETS.items():
                        async with client.get(server.make_url('/workspace/' + name)) as response:
                            self.assertEqual(response.status, 200, name)
                            self.assertEqual(response.content_type, mime)
                            self.assertEqual(await response.read(), source.read_bytes())
                    for name in ('architecture_assets.py', 'project-map-summary.json', 'users.db', 'docs/project-map.md'):
                        async with client.get(server.make_url('/workspace/' + name)) as response:
                            self.assertEqual(response.status, 404, name)


if __name__ == '__main__':
    unittest.main()
