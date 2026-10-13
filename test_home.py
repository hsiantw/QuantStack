"""Browser checks for home navigation and native analysis deep links."""
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from serve_quantstack import create_app


class HomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_home_navigation_search_and_mobile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'symbols.json').write_text('[]')
            server = TestServer(create_app('', snapshot_dir=root, accounts_path=root / 'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(channel='msedge', headless=True)
                    page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    await page.goto(str(server.make_url('/')))
                    await expect(page.locator('h1')).to_contain_text('A wider view.')
                    response = await page.request.get(str(server.make_url('/home-earth.png')))
                    self.assertEqual(response.status, 200)
                    await page.screenshot(path=str(Path(__file__).parent / 'data/home-desktop-preview.png'), full_page=True)
                    toggle = page.locator('#exploreToggle')
                    await toggle.focus()
                    await page.keyboard.press('ArrowDown')
                    await expect(toggle).to_have_attribute('aria-expanded', 'true')
                    await page.get_by_role('button', name='Products', exact=True).click()
                    await expect(page.locator('#destinationMenu').get_by_role('link', name='Stock screener')).to_be_visible()
                    await page.get_by_role('button', name='Back to destinations').click()
                    await page.get_by_role('button', name='Research', exact=True).click()
                    await expect(page.locator('#destinationMenu').get_by_role('link', name='Idea notebook')).to_be_visible()
                    await page.keyboard.press('Escape')
                    await expect(toggle).to_be_focused()
                    await expect(page.locator('#destinationMenu')).to_be_hidden()
                    await toggle.click()
                    await page.locator('h1').click()
                    await expect(page.locator('#destinationMenu')).to_be_hidden()
                    await page.set_viewport_size({'width': 390, 'height': 844})
                    await toggle.click()
                    await page.get_by_role('button', name='Markets', exact=True).click()
                    self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                    await page.screenshot(path=str(Path(__file__).parent / 'data/home-mobile-preview.png'), full_page=True)
                    await page.keyboard.press('Escape')
                    await page.route('**/workspace/?symbol=*', lambda route: route.fulfill(body='Symbol chart'))
                    await page.locator('#symbolSearch').fill(' btc-usd ')
                    await page.locator('#symbolSearch').press('Enter')
                    await page.wait_for_url('**/workspace/?symbol=BTC-USD')
                    await page.goto(str(server.make_url('/workspace/#tool=portfolio')))
                    await expect(page.locator('#workspaceToolsPanel')).to_be_visible()
                    await expect(page.locator('#workspaceToolsPanel h2')).to_have_text('Portfolio & allocation')
                    await page.evaluate("location.hash = 'tool=screener'")
                    await expect(page.locator('#workspaceScreenerPanel')).to_be_visible()
                    await page.locator('.workspace-brand').click()
                    await expect(page.locator('h1')).to_contain_text('A wider view.')
                    self.assertFalse(errors, errors)
                    await browser.close()
            finally:
                await server.close()
