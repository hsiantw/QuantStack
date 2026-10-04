"""Verify native research panels with deterministic prices and browser isolation."""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from dashboard import Handler
from test_research import bars


def run():
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='msedge',headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            def fixture(route):
                url=urlparse(route.request.url)
                if url.path=='/api/symbols':data=[dict(symbol=s,name=s,kind='Stocks',currency='USD',close=120,date='2025-09-17',change=1,has_daily=True) for s in ['AAPL','MSFT']]
                elif url.path=='/api/history':data=bars(scale=1 if parse_qs(url.query)['symbol'][0]=='AAPL' else 2)
                elif url.path=='/api/screener':data={'rows':[dict(symbol='AAPL',name='Apple',sector='Technology',currency='USD',metadata_date='2025-09-17',metadata_source='Test fixture')]}
                else:route.continue_();return
                route.fulfill(status=200,content_type='application/json',body=json.dumps(data))
            page.route('**/api/**',fixture)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('rows.length===260')
            assert page.locator('iframe').count()==0
            for key in ['portfolio','pairs','options','fundamentals','liquidity','forecast','compare','markets','journal']:
                page.locator(f'.workspace-right-rail [data-research-open={key}]').click()
                expect(page.locator('#workspaceToolsPanel')).to_be_visible()
                page.locator('#researchRun').click()
                expect(page.locator('#researchStatus')).to_contain_text('Complete',timeout=15000)
                expect(page.locator('#researchExport')).to_be_enabled()
                assert page.locator('#researchOutput').inner_text()
            page.locator('.workspace-right-rail [data-research-open=portfolio]').click()
            page.locator('#rt-name').fill('Test allocation')
            page.locator('#rt-save').click()
            expect(page.locator('#researchStatus')).to_contain_text('saved')
            page.locator('#researchRun').click()
            expect(page.locator('#researchStatus')).to_contain_text('Complete')
            page.screenshot(path='data/native-research-desktop.png')
            page.evaluate("select('MSFT')")
            expect(page.locator('#researchExport')).to_be_disabled()
            expect(page.locator('#researchOutput')).to_be_empty()
            page.locator('[data-side-panel=tools]').click()
            page.locator('#researchSearch').fill('Monte Carlo')
            page.locator('#researchLibraryList [data-research-open=brownian]').click()
            expect(page.locator('#workspaceBrownianPanel')).to_be_visible()
            page.set_viewport_size({'width':390,'height':844})
            page.locator('.workspace-right-rail [data-research-open=options]').click()
            page.locator('#researchRun').click()
            expect(page.locator('#researchStatus')).to_contain_text('Complete')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path='data/native-research-mobile.png')
            page.evaluate("document.documentElement.dataset.theme='dark'")
            page.screenshot(path='data/native-research-dark.png')
            assert not errors,errors
            browser.close()
        print('PASS: nine native tools, library navigation, saved portfolios, stale results, desktop/mobile and theme.')
    finally:
        server.shutdown();server.server_close()


if __name__=='__main__':run()
