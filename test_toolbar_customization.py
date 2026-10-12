"""Right-rail choices persist, synchronize, and never hide the customization entry."""
from pathlib import Path
import tempfile
import unittest
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect
from serve_quantstack import create_app


class ToolbarTests(unittest.IsolatedAsyncioTestCase):
    async def test_choices_restore_sync_and_mobile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'symbols.json').write_text('[]')
            (root/'screener.json').write_text('{"rows":[]}')
            async with TestServer(create_app('',snapshot_dir=root,accounts_path=root/'accounts.sqlite')) as server:
                async with async_playwright() as p:
                    browser=await p.chromium.launch(channel='msedge',headless=True)
                    context=await browser.new_context(viewport={'width':1440,'height':900})
                    page=await context.new_page();errors=[]
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    await page.goto(str(server.make_url('/workspace/')))
                    rail=page.locator('.workspace-right-rail')
                    await expect(rail.locator('[data-research-open="screener"]')).to_be_hidden()
                    await page.get_by_role('button',name='Customize right toolbar').click()
                    dialog=page.locator('#rightToolbarDialog')
                    await dialog.get_by_label('Watchlist',exact=True).uncheck()
                    await dialog.get_by_label('Stock screener',exact=True).check()
                    await dialog.get_by_role('button',name='Done',exact=True).click()
                    await expect(rail.locator('[data-side-panel="watchlist"]')).to_be_hidden()
                    await rail.locator('[data-research-open="screener"]').click()
                    await expect(page.locator('#screenerContent')).to_be_visible()
                    await page.reload()
                    await expect(rail.locator('[data-research-open="screener"]')).to_be_visible()
                    await expect(rail.locator('[data-side-panel="watchlist"]')).to_be_hidden()
                    other=await context.new_page();await other.goto(str(server.make_url('/workspace/')))
                    await page.get_by_role('button',name='Customize right toolbar').click()
                    await dialog.get_by_role('button',name='Hide all',exact=True).click()
                    await expect(other.locator('.workspace-right-rail > button:visible')).to_have_count(1)
                    await dialog.get_by_label('Find a function').fill('portfolio')
                    await expect(dialog.locator('#rightToolbarChoices label:visible')).to_have_count(1)
                    await dialog.get_by_label('Find a function').fill('not-a-function')
                    await expect(page.locator('#rightToolbarEmpty')).to_be_visible()
                    await dialog.get_by_label('Find a function').fill('')
                    await dialog.get_by_role('button',name='Restore defaults').click()
                    await expect(dialog.get_by_label('Watchlist',exact=True)).to_be_checked()
                    await expect(dialog.get_by_label('Stock screener',exact=True)).not_to_be_checked()
                    await page.set_viewport_size({'width':390,'height':844})
                    self.assertTrue(await page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
                    await page.screenshot(path=str(Path(__file__).parent/'data/toolbar-customization-mobile.png'))
                    await page.keyboard.press('Escape')
                    await expect(dialog).not_to_be_visible()
                    await expect(page.get_by_role('button',name='Customize right toolbar')).to_be_focused()
                    self.assertFalse(errors,errors)
                    await browser.close()
