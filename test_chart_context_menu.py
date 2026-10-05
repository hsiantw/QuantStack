"""Exercise chart context actions through actual browser input."""
import threading
import unittest
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright, expect
from dashboard import Handler
from test_research import bars


class ChartContextMenuTests(unittest.TestCase):
    def test_menu_actions_keyboard_and_viewport(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 900})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                history = bars()
                def api(route):
                    path = route.request.url.split('/api/')[1].split('?')[0]
                    value = ([dict(symbol='TEST', name='Test asset', asset_type='Stocks',
                                   close=history[-1]['close'], change=1, currency='USD')]
                             if path == 'symbols' else history if path == 'history'
                             else [] if path == 'indicators' else {})
                    route.fulfill(json=value)
                page.route('**/api/**', api)
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function('rows.length > 0')
                chart, menu = page.locator('#chart'), page.locator('#chartContextMenu')

                def open_menu():
                    chart.click(button='right', position={'x': 120, 'y': 100})
                    expect(menu).to_be_visible()

                open_menu()
                menu.get_by_role('menuitem', name='Chart settings').click()
                expect(page.locator('#terminalSettingsDialog')).to_be_visible()
                page.keyboard.press('Escape')
                open_menu()
                menu.get_by_role('menuitemcheckbox', name='Logarithmic scale').click()
                self.assertEqual(page.evaluate('chartScale.mode'), 'log')
                open_menu()
                expect(menu.get_by_role('menuitemcheckbox', name='Logarithmic scale')).to_have_attribute('aria-checked', 'true')
                menu.get_by_role('menuitem', name='Draw horizontal line').click()
                chart.click(position={'x': 150, 'y': 130})
                self.assertEqual(page.evaluate('drawings.length'), 1)
                open_menu()
                menu.get_by_role('menuitem', name='Remove all drawings').click()
                self.assertEqual(page.evaluate('drawings.length'), 0)
                open_menu()
                menu.get_by_role('menuitem', name='Undo drawing').click()
                self.assertEqual(page.evaluate('drawings.length'), 1)
                open_menu()
                menu.get_by_role('menuitem', name='Indicators').click()
                expect(page.locator('#indicatorDialog')).to_be_visible()
                page.keyboard.press('Escape')
                chart.focus()
                page.keyboard.press('Shift+F10')
                expect(menu).to_be_visible()
                page.keyboard.press('End')
                expect(menu.get_by_role('menuitem', name='Full screen', exact=True)).to_be_focused()
                page.keyboard.press('Home')
                expect(menu.get_by_role('menuitem', name='Chart settings')).to_be_focused()
                page.keyboard.press('Escape')
                expect(menu).to_be_hidden()
                expect(chart).to_be_focused()
                open_menu()
                page.locator('#symbol').click()
                expect(menu).to_be_hidden()
                for width, height in [(1440, 900), (390, 640)]:
                    page.set_viewport_size({'width': width, 'height': height})
                    chart.scroll_into_view_if_needed()
                    page.wait_for_timeout(250)
                    page.evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                    chart.dispatch_event('contextmenu', {'clientX': width - 2, 'clientY': height - 2})
                    box = menu.bounding_box()
                    self.assertGreaterEqual(box['x'], 0)
                    self.assertGreaterEqual(box['y'], 0)
                    self.assertLessEqual(box['x'] + box['width'], width)
                    self.assertLessEqual(box['y'] + box['height'], height)
                    page.keyboard.press('Escape')
                page.evaluate("rows = []; draw()")
                open_menu()
                expect(menu.get_by_role('menuitem', name='Reset chart view')).to_be_disabled()
                expect(menu.get_by_role('menuitem', name='Chart settings')).to_be_enabled()
                self.assertFalse(errors, errors)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
