"""Check local enrollment controls in desktop and mobile layouts."""
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
                page = browser.new_page(viewport={'width': width, 'height': 900})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function("typeof rows !== 'undefined' && rows.length > 0")
                if width < 600:
                    page.locator('#workspaceSymbol').click()
                page.locator('#watchlistAddSymbol').click()
                page.locator('#schedulerButton').click()
                expect(page.locator('#schedulerSubmit')).to_be_enabled()
                expect(page.locator('#schedulerDialog')).to_be_visible()
                page.locator('#schedulerKind').select_option('crypto')
                page.locator('#schedulerSymbols').fill('BTC ETH')
                expect(page.locator('#schedulerPreview')).to_contain_text('BTC-USD')
                page.locator('#schedulerSymbols').fill('../invalid')
                expect(page.locator('#schedulerSubmit')).to_be_disabled()
                page.locator('#schedulerSymbols').fill('BTC ETH')
                expect(page.locator('#schedulerSubmit')).to_be_enabled()
                expect(page.locator('#schedulerProgress')).to_contain_text('hourly:')
                def save(route):
                    payload = route.request.post_data_json
                    assert payload == {'symbols': 'BTC ETH', 'kind': 'crypto', 'background_usage': 'low'}
                    route.fulfill(status=200, content_type='application/json', body=json.dumps(
                        {'symbols': ['BTC-USD', 'ETH-USD'], 'added': ['ETH-USD']}))
                page.route('**/api/local-scheduler', save)
                page.locator('#schedulerSubmit').click()
                expect(page.locator('#schedulerStatus')).to_contain_text('1 new symbols queued')
                page.locator('#schedulerSymbols').fill('BTC ETH BTC')
                expect(page.locator('#schedulerPreview')).to_contain_text('0 new · 2 already configured')
                box = page.locator('#schedulerDialog').bounding_box()
                assert box['x'] >= 0 and box['x'] + box['width'] <= width
                page.locator('#schedulerClose').click()
                expect(page.locator('#schedulerDialog')).not_to_be_visible()
                assert not errors, errors
                page.close()
            browser.close()
        print('Scheduler browser checks passed: desktop and mobile.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    run()
