import json
import time
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from playwright.async_api import async_playwright, expect
from pathlib import Path

from crypto_scanner import CONTRACTS, amount, normalize, scan, scan_handler

ADDRESS = '0x' + 'a' * 40
POOL = next(iter(CONTRACTS))


class ScannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_records_preserve_valid_activity_and_mark_partial_coverage(self):
        async def fake(session, action, address, limit):
            if action == 'balance':
                return {'status': '1', 'result': '0'}
            return {'status': '1', 'result': [None, [], 'invalid', {
                'from': ADDRESS, 'to': POOL, 'timeStamp': str(int(time.time())),
                'isError': '0', 'value': '1'}]}
        with patch('crypto_scanner.fetch_feed', fake):
            result = await scan(ADDRESS, 30, 100)
        self.assertEqual(len(result['activity']), 3)
        self.assertTrue(all(item['state'] == 'partial' for item in result['coverage']))

    def test_evidence_and_precision(self):
        row = dict(timeStamp='100', **{'from': ADDRESS, 'to': POOL.upper(), 'isError': '1'})
        record = normalize(row, 'transactions', ADDRESS)
        self.assertTrue(record['match'])
        self.assertEqual(record['status'], 'failed')
        row.update({'from': POOL, 'to': ADDRESS, 'isError': '0'})
        self.assertEqual(normalize(row, 'internal', ADDRESS)['direction'], 'incoming')
        self.assertEqual(len(CONTRACTS), 20)
        self.assertEqual(amount('123456789012345678901234567890123456789', 18), '123456789012345678901.234567890123456789')

    async def fixture(self):
        async def fake(session, action, address, limit):
            row = {'from': ADDRESS, 'to': POOL, 'isError': '0', 'timeStamp': str(int(time.time())), 'hash': '0x' + 'b' * 64, 'value': '100000000000000000'}
            if action == 'balance':
                return {'status': '1', 'result': '1000000000000000000'}
            if action == 'txlistinternal':
                return {'status': '2', 'message': 'Indexing incomplete', 'result': [row]}
            if action == 'tokentx':
                return {'error': 'Unavailable'}
            return {'status': '1', 'result': [row] * limit}
        with patch('crypto_scanner.fetch_feed', fake):
            return await scan(ADDRESS, 30, 100)

    async def test_coverage(self):
        result = await self.fixture()
        self.assertEqual([c['state'] for c in result['coverage']], ['capped', 'partial', 'unavailable'])
        self.assertEqual(result['balance'], '1.000000000000000000')
        app = web.Application(); app.router.add_get('/scan', scan_handler)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/scan?address=bad')
            self.assertEqual(response.status, 400)

    async def test_browser(self):
        data = await self.fixture()
        app = web.Application()
        root = Path(__file__).parent / 'web'
        async def page(request):
            return web.FileResponse(root / 'crypto-scanner.html')
        async def asset(request):
            return web.FileResponse(root / request.match_info['asset'])
        app.router.add_get('/crypto-scanner', page)
        app.router.add_get('/{asset}', asset)
        async with TestServer(app) as server:
            async with async_playwright() as p:
                browser = await p.chromium.launch(channel='msedge', headless=True)
                page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
                await page.route('**/api/crypto/scan?*', lambda route: route.fulfill(content_type='application/json', body=json.dumps(data)))
                await page.goto(str(server.make_url('/crypto-scanner')))
                await page.locator('#address').fill(ADDRESS)
                await page.locator('#scan').click()
                await expect(page.locator('#finding')).to_have_text('Direct Tornado Cash activity found')
                await expect(page.locator('#coverage')).to_contain_text('unavailable')
                await page.locator('#feed').select_option('tokens')
                await expect(page.locator('#empty')).to_be_visible()
                await page.locator('[data-site-theme]').select_option('forest')
                await expect(page.locator('html')).to_have_attribute('data-palette', 'forest')
                async with page.expect_download() as download:
                    await page.locator('#export').click()
                self.assertTrue((await download.value).suggested_filename.endswith('.json'))
                await page.locator('#feed').select_option('internal')
                await page.screenshot(path=str(root.parent / 'data/crypto-scanner-desktop.png'), full_page=True)
                await page.set_viewport_size({'width': 390, 'height': 844})
                self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                await page.screenshot(path=str(root.parent / 'data/crypto-scanner-mobile.png'), full_page=True)
                self.assertFalse(errors)
                await browser.close()
