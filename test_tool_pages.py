"""Dedicated tool pages, screener workflow, and mobile navigation."""
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect
from serve_quantstack import create_app
from tool_pages import TOOLS


class ToolPageTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_and_workflows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'symbols.json').write_text('[]')
            stocks = [{'symbol': 'AAA', 'name': 'Alpha', 'market_cap': 2000000000, 'close': 25,
                       'currency': 'USD', 'has_history': True, 'price_bars': 100, 'change_1d_pct': 2},
                      {'symbol': 'BBB', 'name': 'Beta', 'market_cap': 1000000000, 'close': 10,
                       'currency': 'USD', 'has_history': True, 'price_bars': 100, 'change_1d_pct': -1}]
            (root / 'screener.json').write_text(json.dumps({'rows': stocks}))
            server = TestServer(create_app('', snapshot_dir=root, accounts_path=root / 'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(channel='msedge', headless=True)
                    page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    for key, (title, _, _) in TOOLS.items():
                        if key == 'crypto-scanner':
                            continue  # Standalone scanner is covered in test_crypto_scanner.
                        await page.goto(str(server.make_url('/' + key)))
                        await expect(page.locator('.tool-page-heading h1')).to_have_text(title)
                        await expect(page.locator('#toolLoading')).to_have_count(0)
                        await expect(page.locator('#toolContent .workspace-dock-pane:visible')).to_have_count(1)
                        await expect(page.locator('body > main')).not_to_be_visible()
                        self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), key)
                        await page.set_viewport_size({'width': 390, 'height': 844})
                        self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), key + ' mobile')
                        await page.set_viewport_size({'width': 1440, 'height': 1000})
                    await page.goto(str(server.make_url('/screener')))
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(2)
                    await expect(page.locator('#refreshScreener')).to_be_visible()
                    await page.locator('#screenerSearch').fill('Alpha')
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(1)
                    await expect(page.locator('#screenerTableBody')).to_contain_text('AAA')
                    await page.reload()
                    await expect(page.locator('#screenerSearch')).to_have_value('Alpha')
                    await page.locator('#screenerSearch').fill('')
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(2)
                    await page.locator('[data-screen-sort="market_cap"]').click()
                    await expect(page.locator('#screenerTableBody tr').first).to_contain_text('BBB')
                    async with page.expect_download() as download:
                        await page.locator('#exportScreener').click()
                    self.assertTrue((await download.value).suggested_filename.endswith('.csv'))
                    await page.screenshot(path=str(Path(__file__).parent / 'data/screener-page-desktop.png'), full_page=True)
                    await page.locator('#toolTheme').select_option('dark')
                    await expect(page.locator('html')).to_have_attribute('data-theme', 'dark')
                    await page.locator('[data-screen-symbol="AAA"]').click()
                    await page.wait_for_url('**/workspace/?symbol=AAA&interval=1d')
                    await page.goto(str(server.make_url('/screener/')))
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(2)
                    await page.set_viewport_size({'width': 390, 'height': 844})
                    await page.locator('#toolMenuToggle').click()
                    await page.locator('#toolFinder').fill('options')
                    await expect(page.locator('#toolNavigationLinks a:visible')).to_have_count(1)
                    await page.locator('#toolNavigationLinks a:visible').click()
                    await page.wait_for_url('**/options')
                    await expect(page.locator('#researchForm')).to_be_visible()
                    await expect(page.locator('#researchExport')).to_be_visible()
                    await page.screenshot(path=str(Path(__file__).parent / 'data/tool-page-mobile.png'), full_page=True)
                    self.assertFalse(errors, errors)
                    await browser.close()
            finally:
                await server.close()
