"""App shortcut editing, persistence, accessible ordering and pointer dragging."""
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from serve_quantstack import create_app

KEY = 'atlas.home.shortcuts.v1'
ROOT = Path(__file__).parent


class HomeShortcutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'symbols.json').write_text('[]')
        self.server = TestServer(create_app('', snapshot_dir=root, accounts_path=root / 'accounts.sqlite'))
        await self.server.start_server()
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(channel='msedge', headless=True)
        self.context = await self.browser.new_context(viewport={'width': 1440, 'height': 1000})
        self.page = await self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.url = str(self.server.make_url('/'))
        await self.page.goto(self.url)

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()
        await self.server.close()
        self.temp.cleanup()

    async def names(self):
        return await self.page.locator('#shortcutGrid strong').all_text_contents()

    async def test_edit_order_drag_storage_and_mobile(self):
        page = self.page
        grid = page.locator('#shortcutGrid')
        await expect(grid.locator('.shortcut-card')).to_have_count(6)
        await expect(grid.get_by_role('link', name='Yahoo Finance (opens in new tab)', exact=True)).to_have_attribute('href', 'https://finance.yahoo.com/')
        await expect(grid.get_by_role('link').first).to_have_attribute('rel', 'noopener noreferrer')
        await page.locator('#shortcutAdd').click()
        await page.locator('#shortcutName').fill('My research')
        await page.locator('#shortcutUrl').fill('javascript:alert(1)')
        await page.get_by_role('button', name='Save shortcut', exact=True).click()
        await expect(page.locator('#shortcutError')).to_contain_text('http or https')
        await page.locator('#shortcutUrl').fill('finviz.com')
        await page.get_by_role('button', name='Save shortcut', exact=True).click()
        await expect(page.locator('#shortcutError')).to_contain_text('already')
        await page.locator('#shortcutUrl').fill('example.com/research')
        await page.locator('#shortcutColor').select_option('violet')
        await page.get_by_role('button', name='Save shortcut', exact=True).click()
        await expect(page.locator('#shortcutDialog')).not_to_be_visible()
        await expect(grid.locator('.shortcut-card')).to_have_count(7)
        await page.locator('#shortcutCustomize').click()
        await page.get_by_role('button', name='Edit My research', exact=True).click()
        await page.locator('#shortcutName').fill('My ideas')
        await page.locator('#shortcutUrl').fill('/ideas')
        await page.get_by_role('button', name='Save shortcut', exact=True).click()
        await expect(grid.get_by_role('link', name='My ideas', exact=True)).to_have_attribute('href', '/ideas')
        self.assertIsNone(await grid.get_by_role('link', name='My ideas', exact=True).get_attribute('target'))
        await page.get_by_role('button', name='Reorder My ideas', exact=True).press('ArrowLeft')
        self.assertEqual((await self.names())[-2:], ['My ideas', 'CoinGecko'])
        await expect(page.get_by_role('button', name='Reorder My ideas', exact=True)).to_be_focused()

        # A real mouse drag moves the first tile to the third position.
        source = await page.get_by_role('button', name='Reorder Finviz', exact=True).bounding_box()
        target = await grid.locator('.shortcut-card').nth(2).bounding_box()
        await page.mouse.move(source['x'] + source['width'] / 2, source['y'] + source['height'] / 2)
        await page.mouse.down()
        await page.mouse.move(target['x'] + target['width'] / 2, target['y'] + target['height'] / 2, steps=12)
        await page.mouse.up()
        self.assertEqual((await self.names())[:3], ['Yahoo Finance', 'TradingView', 'Finviz'])
        await expect(page.locator('.shortcut-ghost')).to_have_count(0)
        expected = await self.names()
        await page.reload()
        self.assertEqual(await self.names(), expected)

        other = await self.context.new_page()
        await other.goto(self.url)
        await page.locator('#shortcutCustomize').click()
        await page.get_by_role('button', name='Remove My ideas', exact=True).click()
        await expect(other.locator('#shortcutGrid strong')).to_have_count(6)
        await page.locator('#shortcutUndo').click()
        await expect(other.locator('#shortcutGrid strong')).to_have_count(7)
        await other.close()
        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.get_by_role('button', name='Move Yahoo Finance later', exact=True).click()
        self.assertEqual((await self.names())[:2], ['TradingView', 'Yahoo Finance'])
        self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        (ROOT / '.test-tmp').mkdir(exist_ok=True)
        await page.locator('.launchpad').screenshot(path=str(ROOT / '.test-tmp/home-shortcuts-mobile.png'))
        await page.locator('#shortcutCustomize').click()
        await page.set_viewport_size({'width': 1440, 'height': 1000})
        await page.locator('[data-site-theme]').select_option('dark')
        await page.locator('.launchpad').screenshot(path=str(ROOT / '.test-tmp/home-shortcuts-desktop.png'))
        # Quota failures must not claim that a reorder was saved.
        await page.evaluate("() => { Storage.prototype.setItem = () => { throw new Error('full'); }; }")
        await page.locator('#shortcutCustomize').click()
        await page.get_by_role('button', name='Move TradingView later', exact=True).click()
        await expect(page.locator('#shortcutStatus')).to_contain_text('Could not save')
        self.assertFalse(self.errors, self.errors)

    async def test_unreadable_storage_and_literal_labels(self):
        page = self.page
        await page.evaluate('(key) => localStorage.setItem(key, "not-json")', KEY)
        await page.reload()
        await expect(page.locator('#shortcutStatus')).to_contain_text('has not been overwritten')
        await page.locator('#shortcutAdd').click()
        await page.locator('#shortcutName').fill('<img src=x onerror=alert(1)>')
        await page.locator('#shortcutUrl').fill('https://example.com/')
        await page.get_by_role('button', name='Save shortcut', exact=True).click()
        await expect(page.locator('#shortcutStatus')).to_contain_text('For this visit only')
        self.assertEqual(await page.evaluate('(key) => localStorage.getItem(key)', KEY), 'not-json')
        await expect(page.locator('#shortcutGrid img')).to_have_count(0)
        await expect(page.locator('#shortcutGrid strong').last).to_have_text('<img src=x onerror=alert(1)>')
        self.assertFalse(self.errors, self.errors)

    async def test_touch_drag_and_cancel(self):
        context = await self.browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        page = await context.new_page()
        await page.goto(self.url)
        await page.locator('#shortcutCustomize').tap()
        source = await page.get_by_role('button', name='Reorder Finviz', exact=True).bounding_box()
        target = await page.locator('#shortcutGrid .shortcut-card').nth(1).bounding_box()
        client = await context.new_cdp_session(page)
        start = {'x': source['x'] + source['width'] / 2, 'y': source['y'] + source['height'] / 2}
        end = {'x': target['x'] + target['width'] / 2, 'y': target['y'] + target['height'] / 2}
        async def drag(finish):
            await client.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [start]})
            for step in range(1, 11):
                point = {axis: start[axis] + (end[axis] - start[axis]) * step / 10 for axis in ('x', 'y')}
                await client.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [point]})
            await client.send('Input.dispatchTouchEvent', {'type': finish, 'touchPoints': []})
        await drag('touchCancel')
        await expect(page.locator('#shortcutGrid strong').first).to_have_text('Finviz')
        await expect(page.locator('.shortcut-ghost')).to_have_count(0)
        await drag('touchEnd')
        await expect(page.locator('#shortcutGrid strong').first).to_have_text('Yahoo Finance')
        await page.reload()
        await expect(page.locator('#shortcutGrid strong').first).to_have_text('Yahoo Finance')
        await context.close()


if __name__ == '__main__':
    unittest.main()
