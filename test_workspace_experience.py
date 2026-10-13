"""Exercise chart focus, symbol discovery and compact layouts through browser input."""
import re
import threading
import unittest
from http.server import ThreadingHTTPServer

from playwright.sync_api import expect, sync_playwright

from dashboard import Handler
from test_research import bars


class QuietHandler(Handler):
    def log_message(self, *_args):
        pass


class WorkspaceExperienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel='msedge', headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.context = self.browser.new_context(
            viewport={'width': 1440, 'height': 900}, color_scheme='light')
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        history = bars()
        # Deliberately unsorted: exact tickers must beat company-name matches.
        catalog = [
            ('AAA', 'AAPL Holdings', 'Stocks'),
            ('AAPLX', 'Apple Suppliers', 'Stocks'),
            ('MSFT', 'Microsoft Corporation', 'Stocks'),
            ('AAPL', 'Apple Inc.', 'Stocks'),
            ('BTC-USD', 'Bitcoin', 'Crypto'),
        ]
        self.catalog = [dict(symbol=symbol, name=name, kind=kind,
                             asset_type=kind, close=history[-1]['close'],
                             change=1.2, currency='USD', exchange='TEST',
                             date=history[-1]['date'], has_daily=True)
                        for symbol, name, kind in catalog]

        def api(route):
            path = route.request.url.split('/api/')[1].split('?')[0]
            value = (self.catalog if path == 'symbols' else history if path == 'history'
                     else [] if path == 'indicators' else {})
            route.fulfill(json=value)

        self.page.route('**/api/**', api)
        self.page.goto(f'http://127.0.0.1:{self.server.server_port}')
        self.page.wait_for_function('rows.length > 0')
        expect(self.page.locator('body')).to_have_class(re.compile(r'\bworkspace-enhanced\b'))

    def tearDown(self):
        self.context.close()
        self.assertFalse(self.errors, self.errors)

    def settle_layout(self):
        self.page.evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')

    def assert_no_overflow(self):
        self.settle_layout()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        chart = self.page.locator('#chart').bounding_box()
        self.assertGreater(chart['width'], 180)
        self.assertGreaterEqual(chart['height'], 100)

    def test_focus_restores_panels_sizes_and_shortcut_help(self):
        page = self.page
        page.locator('[data-side-panel="details"]').click()
        separator = page.locator('#workspaceSidebarResize')
        separator.focus()
        page.keyboard.press('ArrowLeft')
        page.keyboard.press('ArrowLeft')
        sidebar_width = page.locator('#workspaceSidebar').bounding_box()['width']
        page.locator('#workspaceOpenData').click()
        page.locator('#workspaceDockResize').focus()
        page.keyboard.press('ArrowUp')
        dock_height = page.locator('#workspaceDock').bounding_box()['height']
        original_chart = page.locator('#chart').bounding_box()

        page.locator('#chart').focus()
        page.keyboard.press('Alt+f')
        expect(page.locator('#workspaceFocus')).to_have_attribute('aria-pressed', 'true')
        for selector in ['.workspace-drawing-rail', '#workspaceSidebar', '.workspace-right-rail',
                         '.workspace-bottom-tabs', '#workspaceDock']:
            expect(page.locator(selector)).to_be_hidden()
            self.assertTrue(page.locator(selector).evaluate('(element) => element.inert'))
        self.settle_layout()
        focused_chart = page.locator('#chart').bounding_box()
        self.assertGreater(focused_chart['width'], original_chart['width'] + 200)
        self.assertGreater(focused_chart['height'], original_chart['height'] + 200)
        page.keyboard.press('Escape')
        expect(page.locator('#workspaceFocus')).to_have_attribute('aria-pressed', 'false')
        expect(page.locator('#workspaceSidedetails')).to_be_visible()
        expect(page.locator('#workspaceDataPanel')).to_be_visible()
        self.settle_layout()
        self.assertAlmostEqual(page.locator('#workspaceSidebar').bounding_box()['width'], sidebar_width, delta=1)
        self.assertAlmostEqual(page.locator('#workspaceDock').bounding_box()['height'], dock_height, delta=1)
        self.assertFalse(page.locator('#workspaceSidebar').evaluate('(element) => element.inert'))

        help_button = page.locator('#workspaceHelp')
        help_button.click()
        expect(page.locator('#chartHomeDialog')).to_be_visible()
        expect(page.locator('#chartHomeContent')).to_contain_text('Alt + F')
        page.keyboard.press('Control+k')
        expect(page.locator('#workspaceSymbolDialog')).to_be_hidden()
        page.keyboard.press('Escape')
        expect(page.locator('#chartHomeDialog')).to_be_hidden()
        expect(page.locator('#terminalSettingsDialog')).to_be_hidden()
        expect(help_button).to_be_focused()

    def test_symbol_search_full_catalog_ranking_keyboard_and_editing_guards(self):
        page = self.page
        page.locator('#filters [data-filter="Crypto"]').click()
        expect(page.locator('#assets button[data-symbol]')).to_have_count(1)
        page.locator('#workspaceSymbol').click()
        dialog = page.locator('#workspaceSymbolDialog')
        search = page.locator('#workspaceSymbolSearchInput')
        options = page.locator('#workspaceSymbolSearchResults [role="option"]')
        expect(dialog).to_be_visible()
        expect(search).to_be_focused()
        search.fill('aapl')
        expect(options).to_have_count(3)
        expect(options.nth(0)).to_contain_text('AAPL')
        self.assertEqual(options.nth(0).locator('strong').inner_text(), 'AAPL')
        self.assertEqual(options.nth(1).locator('strong').inner_text(), 'AAPLX')
        self.assertEqual(options.nth(2).locator('strong').inner_text(), 'AAA')
        search.fill('micro')
        expect(options).to_have_count(1)
        expect(options.first).to_contain_text('Microsoft Corporation')
        page.keyboard.press('Enter')
        expect(dialog).to_be_hidden()
        expect(page.locator('#symbol')).to_have_text('MSFT')
        expect(page.locator('#filters [data-filter="Crypto"]')).to_have_class(re.compile(r'\bactive\b'))

        page.locator('#chart').focus()
        page.keyboard.press('Control+k')
        expect(dialog).to_be_visible()
        dialog.get_by_role('button', name='Crypto', exact=True).click()
        expect(options).to_have_count(1)
        expect(options.first).to_contain_text('BTC-USD')
        search.fill('missing company')
        expect(page.locator('#workspaceSymbolSearchEmpty')).to_be_visible()
        page.keyboard.press('Enter')
        expect(dialog).to_be_visible()
        page.keyboard.press('Escape')
        expect(page.locator('#chart')).to_be_focused()

        page.keyboard.press('/')
        expect(dialog).to_be_visible()
        expect(search).to_have_value('')
        first_active = search.get_attribute('aria-activedescendant')
        page.keyboard.press('ArrowDown')
        self.assertNotEqual(search.get_attribute('aria-activedescendant'), first_active)
        page.keyboard.press('ArrowUp')
        self.assertEqual(search.get_attribute('aria-activedescendant'), first_active)
        page.keyboard.press('Escape')

        page.locator('[data-side-panel="notes"]').click()
        notes = page.locator('#workspaceNotes')
        notes.focus()
        page.keyboard.press('/')
        expect(notes).to_have_value('/')
        page.keyboard.press('Control+k')
        page.keyboard.press('Alt+f')
        expect(dialog).to_be_hidden()
        expect(page.locator('#workspaceFocus')).to_have_attribute('aria-pressed', 'false')

    def test_responsive_palettes_and_mobile_panel_dismissal(self):
        page = self.page
        for width, height in [(1440, 900), (768, 900), (390, 844), (320, 740)]:
            with self.subTest(width=width):
                page.set_viewport_size({'width': width, 'height': height})
                self.assert_no_overflow()
                expect(page.get_by_role('button', name='Keyboard shortcuts', exact=True)).to_be_visible()
                expect(page.locator('#workspaceFocus')).to_be_visible()
                page.locator('#workspaceSymbol').click()
                expect(page.locator('#workspaceSymbolSearchInput')).to_be_focused()
                dialog_box = page.locator('#workspaceSymbolDialog').bounding_box()
                self.assertGreaterEqual(dialog_box['x'], 0)
                self.assertLessEqual(dialog_box['x'] + dialog_box['width'], width)
                page.keyboard.press('Escape')
                if width <= 900:
                    page.locator('[data-side-panel="notes"]').click()
                    expect(page.locator('#workspaceSidenotes')).to_be_visible()
                    page.locator('#workspacePanelBackdrop').click(position={'x': 4, 'y': 15})
                    expect(page.locator('#workspaceSidebar')).to_be_hidden()
                    expect(page.locator('#workspaceWatchlistToggle')).to_be_focused()
                    page.locator('[data-side-panel="notes"]').click()
                    page.keyboard.press('Alt+f')
                    expect(page.locator('#workspaceSidebar')).to_be_hidden()
                    page.keyboard.press('Escape')
                    expect(page.locator('#workspaceSidenotes')).to_be_visible()
                    self.assertFalse(page.locator('#workspaceSidebar').evaluate('(element) => element.inert'))
                    page.locator('#workspacePanelBackdrop').click(position={'x': 4, 'y': 15})

        for palette in ['light', 'dark', 'midnight', 'forest', 'paper']:
            with self.subTest(palette=palette):
                page.locator('#workspaceTheme').select_option(palette)
                expect(page.locator('html')).to_have_attribute('data-palette', palette)
                page.locator('#workspaceFocus').click()
                expect(page.locator('#workspaceFocus')).to_have_accessible_name('Exit chart focus')
                self.assert_no_overflow()
                page.locator('#workspaceSymbol').click()
                expect(page.locator('#workspaceSymbolSearchInput')).to_be_focused()
                page.keyboard.press('Escape')
                page.keyboard.press('Escape')
                expect(page.locator('#workspaceFocus')).to_have_attribute('aria-pressed', 'false')


if __name__ == '__main__':
    unittest.main()
