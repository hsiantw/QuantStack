"""Browser checks for ranking, missing data, watchlists and refresh updates."""
from pathlib import Path
import unittest

from playwright.async_api import async_playwright, expect


class MarketOverviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_rankings_watchlist_refresh_and_chart(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel='msedge', headless=True)
            page = await browser.new_page(viewport={'width': 1280, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.route('http://overview.test/**', lambda route: route.fulfill(body='<div id="workspaceToolsPanel"></div><button id="workspaceCloseDock">Close</button>'))
            await page.goto('http://overview.test/')
            await page.evaluate('''() => {
              window.$ = id => document.getElementById(id);
              window.fmt = n => n == null ? '—' : Number(n).toFixed(2);
              window.pct = n => (n >= 0 ? '+' : '') + fmt(n) + '%';
              window.esc = s => String(s ?? '').replace(/[&<>"']/g, c => '&#' + c.charCodeAt(0) + ';');
              window.assets = [];
              window.select = symbol => window.openedSymbol = symbol;
              window.data = [
                {symbol:'AAA',name:'Alpha',kind:'Stocks',close:10,change_1d_pct:5,volume:100,performance_asof:'2026-10-09'},
                {symbol:'BBB',name:'Beta',kind:'Stocks',close:20,change_1d_pct:-3,volume:900,performance_asof:'2026-10-09'},
                {symbol:'CCC',name:'Missing',kind:'Stocks',close:null,change_1d_pct:null,volume:null},
                {symbol:'SPY',name:'S&P tracker',kind:'Stocks',close:30,change_1d_pct:1,volume:300},
                {symbol:'BTC-USD',kind:'Crypto',close:100,change_1d_pct:9,volume:2000}
              ];
              window.api = async () => structuredClone(window.data);
              localStorage.setItem('atlas.saved', JSON.stringify(['AAA','CCC']));
              localStorage.setItem('atlas.watchlists.v1', JSON.stringify({lists:[{id:'list-red',name:'Red flags',color:'red',symbols:[]}],labels:{BBB:'red'}}));
            }''')
            await page.add_style_tag(path=str(Path(__file__).parent / 'web/market-overview.css'))
            await page.add_script_tag(path=str(Path(__file__).parent / 'web/market-overview.js'))
            await page.evaluate("() => { MarketOverview.mount(document.getElementById('workspaceToolsPanel')); document.getElementById('workspaceCloseDock').onclick = () => document.getElementById('workspaceToolsPanel').hidden = true; }")
            await expect(page.locator('#marketStatus')).to_contain_text('Last checked')
            await expect(page.locator('#marketGainers tbody tr').first).to_contain_text('AAA')
            await expect(page.locator('#marketLosers tbody tr').first).to_contain_text('BBB')
            await expect(page.locator('#marketVolume tbody tr').first).to_contain_text('BBB')
            await expect(page.locator('#marketIndexes')).to_contain_text('ETF tracker')
            await expect(page.locator('#marketIndexes')).to_contain_text('No stored data')
            await expect(page.locator('#marketBreadth')).to_contain_text('1 unavailable')
            await page.evaluate('window.data[0].close = 12')
            await page.locator('#marketRefresh').click()
            await expect(page.locator('#marketWatchlist tbody tr').first).to_contain_text('Updated')
            await page.locator('#marketUniverse').select_option('crypto')
            await expect(page.locator('#marketVolume tbody tr').first).to_contain_text('BTC-USD')
            await page.locator('#marketUniverse').select_option('watchlist')
            await page.locator('#marketList').select_option('list-red')
            await expect(page.locator('#marketWatchlist')).to_contain_text('BBB')
            await expect(page.locator('#marketGainers')).to_contain_text('No matching data')
            await page.set_viewport_size({'width':390,'height':844})
            self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            await page.locator('#marketWatchlist [data-market-symbol="BBB"]').click()
            self.assertEqual(await page.evaluate('window.openedSymbol'), 'BBB')
            await expect(page.locator('#marketOverview')).not_to_be_visible()
            self.assertFalse(errors, errors)
            await browser.close()
