"""Exercise watchlist persistence, labels and contextual actions in a real browser."""
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from dashboard import Handler


def run():
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='msedge',headless=True)
            for width in (1440,390):
                context=browser.new_context(viewport={'width':width,'height':900})
                instruments=[dict(symbol=symbol,name=name,kind=kind,close=100,change=1,date='2026-10-02',currency='USD',has_daily=True) for symbol,name,kind in [('AAPL','Apple','Stocks'),('MSFT','Microsoft','Stocks'),('BTC-USD','Bitcoin','Crypto')]]
                for asset,cap,weekly,monthly in zip(instruments,[1e9,2e9,None],[2,-3,None],[12,4,None]):
                    asset.update(market_cap=cap,return_1w_pct=weekly,return_1m_pct=monthly,change_1d_pct=0,performance_asof='2026-10-02',performance_basis='adjusted',volume=1000)
                bars=[dict(date=f'2026-09-{day:02}',open=100,high=103,low=99,close=101,adjusted_close=101,volume=1000,dividends=0,splits=0) for day in range(1,29)]
                context.route('**/api/symbols',lambda route:route.fulfill(json=instruments))
                context.route('**/api/history?*',lambda route:route.fulfill(json=bars))
                context.add_init_script("if(!localStorage.getItem('watchlist-test-seeded')){localStorage.setItem('atlas.saved','[\"MSFT\"]');localStorage.setItem('watchlist-test-seeded','yes');}")
                page=context.new_page();errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                url=f'http://127.0.0.1:{server.server_port}'
                page.goto(url+'/?symbol=AAPL')
                page.wait_for_function("selected?.symbol==='AAPL' && rows.length>0")
                def show_list():
                    if not page.locator('#workspaceSideWatchlist').is_visible():page.locator('#workspaceSymbol').click()
                def row(symbol):
                    page.locator('#search').fill(symbol)
                    return page.locator(f'#assets .asset[data-symbol="{symbol}"]')
                def options(symbol):
                    show_list();button=row(symbol)
                    if width<600:page.locator(f'[data-watchlist-menu="{symbol}"]').click()
                    else:button.click(button='right')
                    expect(page.locator('#watchlistMenu')).to_be_visible()
                    box=page.locator('#watchlistMenu').bounding_box()
                    assert box['x']>=0 and box['x']+box['width']<=width and box['y']>=0 and box['y']+box['height']<=900
                show_list()
                # Sort real numeric values, keep unavailable values last, and save columns.
                page.locator('#watchlistSort').select_option('market_cap')
                expect(page.locator('#assets .asset').first).to_have_attribute('data-symbol','MSFT')
                expect(page.locator('#assets .asset').last).to_have_attribute('data-symbol','BTC-USD')
                page.locator('#watchlistDirection').click()
                expect(page.locator('#assets .asset').first).to_have_attribute('data-symbol','AAPL')
                expect(page.locator('#assets .asset').last).to_have_attribute('data-symbol','BTC-USD')
                page.locator('#watchlistColumns').click()
                for column in ['market_cap','return_1w_pct','return_1m_pct']:
                    page.locator(f'[data-watchlist-column="{column}"]').check()
                page.locator('[data-watchlist-column="close"]').uncheck()
                page.locator('#watchlistColumnsDone').click()
                expect(page.locator('[data-watchlist-cell="close"]')).to_have_count(0)
                expect(page.locator('.asset[data-symbol="AAPL"] [data-watchlist-cell="market_cap"]')).to_have_text('1.00B')
                expect(page.locator('.asset[data-symbol="MSFT"] [data-watchlist-cell="return_1w_pct"]')).to_have_text('-3.00%')
                expect(page.locator('.asset[data-symbol="BTC-USD"] [data-watchlist-cell="market_cap"]')).to_have_text('—')
                page.reload();page.wait_for_function('rows.length>0');show_list()
                expect(page.locator('#watchlistSort')).to_have_value('market_cap')
                expect(page.locator('[data-watchlist-cell="close"]')).to_have_count(0)
                page.locator('[data-watchlist-sort="return_1m_pct"]').click()
                expect(page.locator('#watchlistSort')).to_have_value('return_1m_pct')
                expect(page.locator('#assets .asset').first).to_have_attribute('data-symbol','AAPL')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=f'data/watchlist-columns-{width}.png')
                page.locator('#watchlistColumns').click();page.locator('#watchlistColumnsReset').click();page.locator('#watchlistColumnsDone').click()
                page.locator('#watchlistSort').select_option('symbol')
                page.locator('#watchlistSelect').select_option('saved')
                expect(page.locator('#assets .asset[data-symbol="MSFT"]')).to_be_visible()
                page.locator('#watchlistSelect').select_option('all')
                options('MSFT')
                assert page.evaluate('selected.symbol')=='AAPL', 'Context click must not switch the chart'
                page.locator('[data-color="red"]').click()
                expect(page.locator('.watchlist-flag[aria-label="Red label"]')).to_be_visible()
                expect(page.locator('#watchlistStatus')).to_contain_text('Saved in this browser')
                page.locator('[data-label-filter="red"]').click()
                page.locator('#watchlistSaveColor').click()
                expect(page.locator('#watchlistName')).to_have_value('Red flags')
                page.locator('#watchlistNameSubmit').click()
                color_list=page.locator('#watchlistSelect').input_value()
                expect(page.locator('[data-label-filter="red"]')).to_be_disabled()
                expect(page.locator('#assets .asset')).to_have_count(1)
                page.locator('#watchlistAddSymbol').click()
                expect(page.locator('#watchlistColorHelp')).to_contain_text('replacing any previous label')
                page.locator('#watchlistAddSearch').fill('AAPL')
                page.locator('[data-add-symbol="AAPL"]').click()
                page.locator('#watchlistAddClose').click()
                expect(page.locator('#assets .asset')).to_have_count(2)
                page.reload();page.wait_for_function('rows.length>0');show_list()
                expect(page.locator('#watchlistSelect')).to_have_value(color_list)
                expect(page.locator('.watchlist-flag[aria-label="Red label"]')).to_have_count(2)
                options('AAPL');page.locator('[data-color="blue"]').click()
                page.locator('#search').fill('')
                expect(page.locator('#assets .asset')).to_have_count(1)
                expect(page.locator('#assets .asset[data-symbol="MSFT"]')).to_be_visible()
                page.locator('#watchlistManage').click();page.locator('[data-action="delete"]').click()
                page.locator('#watchlistNameSubmit').click()
                expect(page.locator('[data-label-filter="red"]')).to_be_enabled()
                expect(page.locator('.watchlist-flag[aria-label="Red label"]')).to_have_count(1)
                # Deleting a color watchlist keeps labels; clear the extra test flag.
                options('AAPL');page.locator('[data-action="clear"]').click()
                options('MSFT');page.locator('#watchlistMenu [data-action="create"]').click()
                page.locator('#watchlistName').fill('Tech focus');page.locator('#watchlistNameSubmit').click()
                expect(page.locator('#watchlistSelect option:checked')).to_have_text('Tech focus')
                tech=page.locator('#watchlistSelect').input_value()
                expect(page.locator('#assets .asset')).to_have_count(1)
                page.locator('#watchlistAddSymbol').click();page.locator('#watchlistAddSearch').fill('AAPL')
                page.locator('[data-add-symbol="AAPL"]').click();page.locator('#watchlistAddClose').click()
                expect(page.locator('#assets .asset')).to_have_count(2)
                page.locator('[data-label-filter="red"]').click()
                expect(page.locator('#assets .asset')).to_have_count(1)
                page.reload();page.wait_for_function('rows.length>0');show_list()
                expect(page.locator('#watchlistSelect')).to_have_value(tech)
                expect(page.locator('[data-label-filter="red"]')).to_have_attribute('aria-pressed','true')
                expect(page.locator('.watchlist-flag')).to_have_attribute('aria-label','Red label')
                page.locator('[data-label-filter="all"]').click()
                page.locator('#watchlistManage').click();page.locator('[data-action="rename"]').click()
                page.locator('#watchlistName').fill('Tech & growth');page.locator('#watchlistNameSubmit').click()
                expect(page.locator('#watchlistSelect option:checked')).to_have_text('Tech & growth')
                # Duplicate names are rejected, including the current list's name.
                page.locator('#watchlistManage').click();page.locator('[data-action="create"]').click()
                page.locator('#watchlistName').fill('Tech & growth');page.locator('#watchlistNameSubmit').click()
                expect(page.locator('#watchlistNameError')).to_contain_text('unique')
                page.locator('#watchlistName').fill('Second list');page.locator('#watchlistNameSubmit').click()
                second=page.locator('#watchlistSelect').input_value()
                page.locator('#watchlistSelect').select_option(tech)
                options('MSFT');page.locator('[data-action="lists"]').click()
                page.locator(f'[data-list="{second}"]').click()
                expect(page.locator(f'[data-list="{second}"]')).to_have_attribute('aria-checked','true')
                page.keyboard.press('Escape')
                page.locator('#watchlistSelect').select_option(second)
                expect(row('MSFT')).to_be_visible()
                # Same symbol's flag is shared across lists and browser tabs.
                other=context.new_page();other.goto(url);other.wait_for_function('rows.length>0')
                options('MSFT');page.locator('[data-color="blue"]').click()
                other.wait_for_function("JSON.parse(localStorage.getItem('atlas.watchlists.v1')).labels.MSFT==='blue'")
                expect(other.locator('.watchlist-flag')).to_have_attribute('aria-label','Blue label')
                other.close()
                options('MSFT');page.locator('[data-action="compare"]').click()
                expect(page.locator('#comparisonLegend')).to_contain_text('MSFT')
                options('MSFT');page.locator('[data-action="note"]').click()
                expect(page.locator('#workspaceNotesSymbol')).to_have_text('MSFT')
                page.locator('#workspaceNotes').fill('Review earnings')
                assert page.evaluate("localStorage.getItem('atlas.notes.MSFT')")=='Review earnings'
                show_list();options('MSFT');page.locator('[data-action="clear"]').click()
                expect(page.locator('.watchlist-flag')).to_have_count(0)
                # Keyboard context menu and dismiss restore focus.
                row('MSFT').focus();page.keyboard.press('Shift+F10')
                expect(page.locator('#watchlistMenu')).to_be_visible();page.keyboard.press('ArrowDown');page.keyboard.press('Escape')
                expect(page.locator('#assets .asset[data-symbol="MSFT"]')).to_be_focused()
                options('MSFT');page.locator('[data-action="remove"]').click()
                expect(page.locator('#assets .asset')).to_have_count(0)
                page.locator('#watchlistManage').click();page.locator('[data-action="delete"]').click()
                page.locator('#watchlistNameSubmit').click()
                expect(page.locator('#watchlistSelect')).to_have_value('all')
                # Theme + menu appearance and bounding box at both sizes.
                page.evaluate("atlasTheme.apply('dark',true)");options('MSFT')
                assert page.locator('#watchlistMenu').evaluate('(el)=>getComputedStyle(el).color')=='rgb(220, 226, 237)'
                Path('data').mkdir(exist_ok=True)
                page.screenshot(path=f'data/watchlist-menu-{width}.png')
                # A failed browser-storage write must not silently apply unsaved labels.
                page.evaluate("() => {window.originalStorageWrite=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){if(k==='atlas.watchlists.v1')throw new DOMException('Full','QuotaExceededError');return window.originalStorageWrite.call(this,k,v);};}")
                page.locator('[data-color="red"]').click()
                expect(page.locator('#watchlistStatus')).to_contain_text('not saved')
                expect(page.locator('.watchlist-flag')).to_have_count(0)
                page.evaluate('() => {Storage.prototype.setItem=window.originalStorageWrite;}')
                assert not errors,errors
                context.close()
            browser.close()
        print('PASS: named watchlists, existing favorites, color labels, filters, reload/tab sync, compare, notes, keyboard, desktop/mobile and dark theme')
    finally:
        server.shutdown();server.server_close()


if __name__=='__main__':
    run()
