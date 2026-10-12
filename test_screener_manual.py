"""Manual filter drafts must combine correctly and commit only on Apply."""
import json
from pathlib import Path
import tempfile
import unittest
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect
from serve_quantstack import create_app


class ManualScreenerTests(unittest.IsolatedAsyncioTestCase):
    async def test_quick_filters_atomic_manual_conditions_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'symbols.json').write_text('[]')
            stocks = [dict(symbol=symbol, name=symbol + ' company', close=price, pe=pe, rsi14=rsi,
                           market_cap=2e9, currency=currency, has_history=True, sector='Technology')
                      for symbol, price, pe, rsi, currency in [('AAA',150,10,20,'USD'),('BBB',50,30,25,'USD'),
                                                             ('CCC',250,None,75,'USD'),('DDD',150,10,20,'TWD')]]
            (root / 'screener.json').write_text(json.dumps({'rows':stocks}))
            server = TestServer(create_app('', snapshot_dir=root, accounts_path=root / 'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(channel='msedge', headless=True)
                    page = await browser.new_page(viewport={'width':1440,'height':1000})
                    errors=[]
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    await page.goto(str(server.make_url('/screener')))
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(4)
                    await page.locator('[data-quick="close"]').click()
                    await page.get_by_role('button',name='Above 100',exact=True).click()
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(2)
                    await page.locator('#screenManual').click()
                    dialog=page.locator('#screenManualDialog')
                    await dialog.get_by_label('P/E condition',exact=True).select_option('custom')
                    await dialog.get_by_label('P/E comparison',exact=True).select_option('lte')
                    await dialog.get_by_label('P/E value',exact=True).fill('20')
                    await dialog.get_by_role('button',name='Technical',exact=True).click()
                    await dialog.get_by_label('RSI 14 condition',exact=True).select_option('0')
                    await expect(page.locator('#screenManualPreview')).to_contain_text('1 matching stocks')
                    # Edits remain drafts; cancel leaves the two price matches intact.
                    await page.locator('#screenManualCancel').click()
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(2)
                    await page.locator('#screenManual').click()
                    await expect(dialog.get_by_label('P/E condition',exact=True)).to_have_value('')
                    await dialog.get_by_label('P/E condition',exact=True).select_option('custom')
                    await dialog.get_by_label('P/E comparison',exact=True).select_option('lte')
                    await dialog.get_by_label('P/E value',exact=True).fill('20')
                    await dialog.get_by_label('RSI 14 condition',exact=True).select_option('0')
                    await page.screenshot(path=str(Path(__file__).parent/'data/manual-filter-desktop.png'))
                    await page.locator('#screenManualApply').click()
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(1)
                    await expect(page.locator('#screenerTableBody')).to_contain_text('AAA')
                    await page.reload()
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(1)
                    await page.locator('#screenManual').click()
                    await dialog.get_by_label('Price condition',exact=True).select_option('custom')
                    await dialog.get_by_label('Price comparison',exact=True).select_option('between')
                    await dialog.get_by_label('Price value',exact=True).fill('200')
                    await dialog.get_by_label('Price maximum',exact=True).fill('100')
                    await expect(page.locator('#screenManualApply')).to_be_disabled()
                    await expect(page.locator('#screenManualPreview')).to_contain_text('minimum must not exceed maximum')
                    await dialog.get_by_label('Price maximum',exact=True).fill('300')
                    await expect(page.locator('#screenManualApply')).to_be_enabled()
                    await page.locator('#screenManualMode').select_option('any')
                    await expect(page.locator('#screenManualPreview')).to_contain_text('4 matching stocks')
                    await page.set_viewport_size({'width':390,'height':844})
                    self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                    self.assertTrue(await dialog.evaluate('(d) => d.scrollWidth <= d.clientWidth'))
                    await page.screenshot(path=str(Path(__file__).parent/'data/manual-filter-mobile.png'))
                    await page.keyboard.press('Escape')
                    await expect(dialog).not_to_be_visible()
                    await page.locator('#screenReset').click()
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(4)
                    await page.locator('[data-quick="close"]').click()
                    await page.get_by_role('button',name='Above 100',exact=True).click()
                    await page.locator('#screenManual').click()
                    await dialog.locator('[data-manual-field="close"]').get_by_role('button',name='+ Another condition').click()
                    await expect(dialog.get_by_label('Price condition',exact=True)).to_have_count(2)
                    await dialog.get_by_label('Price comparison',exact=True).nth(1).select_option('lt')
                    await dialog.get_by_label('Price value',exact=True).nth(1).fill('200')
                    await expect(page.locator('#screenManualPreview')).to_contain_text('1 matching stocks')
                    await page.locator('#screenManualApply').click()
                    await expect(page.locator('#screenerTableBody tr')).to_have_count(1)
                    await expect(page.locator('[data-quick="close"]')).to_contain_text('2 rules')
                    await page.locator('#screenReset').click()
                    await page.get_by_role('button',name='Add filter',exact=True).click()
                    await page.get_by_label('Search all filters').fill('ADX')
                    await page.locator('#screenFilterCatalog').get_by_role('button',name='ADX 14').click()
                    await expect(page.locator('#screenFilterPopover')).to_contain_text('ADX 14')
                    await page.keyboard.press('Escape')
                    self.assertFalse(errors,errors)
                    await browser.close()
            finally:
                await server.close()
