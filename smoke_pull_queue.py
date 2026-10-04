"""Browser checks without submitting real provider downloads."""
import json
import threading
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright, expect
from dashboard import Handler


def run():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=True)
            for width in (1440, 390):
                page = browser.new_page(viewport={'width':width,'height':900})
                errors=[]; sent=[]
                page.on('pageerror', lambda error: errors.append(str(error)))
                def api(route):
                    if route.request.method == 'POST':
                        sent.append(route.request.post_data_json)
                        route.fulfill(json={'id':1,'status':'queued'})
                    else:
                        route.fulfill(json={'requests':[]})
                page.route('**/api/pull-queue', api)
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function('typeof selected !== "undefined" && selected && rows.length > 0')
                expect(page.locator('.chart-home-menu')).to_have_count(0)
                page.locator('#pullData').click()
                expect(page.locator('#pullSymbol')).to_have_value(page.evaluate('selected.symbol'))
                expect(page.locator('#pullSubmit')).to_be_enabled()
                page.locator('#pullSubmit').click()
                expect(page.locator('#pullMessage')).to_contain_text('queued ahead')
                assert sent[0]['days']==7 and sent[0]['interval']=='60m'
                page.locator('#pullDays').fill('14')
                page.locator('#pullName').fill('Two weeks hourly')
                page.locator('#pullSave').click()
                page.locator('#pullClose').click()
                page.reload()
                page.wait_for_function('selected && rows.length > 0')
                page.locator('#pullData').click()
                page.locator('#pullFavorites').select_option(label='Two weeks hourly')
                page.locator('#pullApply').click()
                expect(page.locator('#pullDays')).to_have_value('14')
                page.locator('#pullRange').select_option('custom')
                expect(page.locator('#pullStart')).to_be_visible()
                expect(page.locator('#pullDays')).to_be_hidden()
                box=page.locator('#pullDialog').bounding_box()
                assert box['x']>=0 and box['x']+box['width']<=width
                page.locator('#pullDelete').click()
                page.locator('#pullClose').click()
                page.locator('#terminalSettings').click()
                page.locator('#chartShare').click()
                expect(page.locator('#chartShareUrl')).to_be_visible()
                page.locator('#chartHomeClose').click()
                page.locator('#chartShortcuts').click()
                expect(page.locator('#chartHomeTitle')).to_have_text('Keyboard shortcuts')
                assert not errors, errors
                page.close()
            browser.close()
        print('PASS: priority pull UI, presets, mobile layout, no Workspace menu, relocated utilities')
    finally:
        server.shutdown();server.server_close()


if __name__ == '__main__':
    run()
