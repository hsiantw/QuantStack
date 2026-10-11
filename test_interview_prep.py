"""Interview guide themes, note tabs, search and timed practice in a browser."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import unittest

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).parent


class InterviewPrepTests(unittest.TestCase):
    def test_workspace_desktop_mobile_and_storage_failures(self):
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT / 'web')))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width':1440, 'height':1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/interview-prep.html')
                expect(page.locator('#prepGuide > section')).to_have_count(12)
                page.locator('#prepTheme').select_option('dark')
                expect(page.locator('html')).to_have_attribute('data-prep-theme', 'dark')
                page.locator('#prepSearch').fill('unfindable-test-string')
                expect(page.locator('#prepNoResults')).to_be_visible()
                page.locator('#prepSearch').fill('authorise')
                expect(page.locator('#section-6')).to_be_visible()
                page.locator('aside a[href="#section-1"]').click()
                expect(page.locator('#prepSearch')).to_have_value('')
                page.locator('#section-1 .review-check input').check()
                expect(page.locator('#prepProgressText')).to_have_text('1 of 12 sections reviewed')
                page.locator('#section-1 .section-notes').click()
                expect(page.locator('#prepNotes')).to_be_visible()
                page.locator('#prepNote').fill('My real example <not HTML>.')
                expect(page.locator('#noteSaved')).to_have_text('Saved on this device')
                page.locator('#note-tab-general').click()
                page.locator('#prepNote').fill('General feedback')
                page.reload()
                expect(page.locator('html')).to_have_attribute('data-prep-theme', 'dark')
                expect(page.locator('#prepProgressText')).to_have_text('1 of 12 sections reviewed')
                page.locator('#tab-notes').click()
                expect(page.locator('#prepNote')).to_have_value('General feedback')
                page.locator('#note-tab-section-1').click()
                expect(page.locator('#prepNote')).to_have_value('My real example <not HTML>.')
                page.locator('#note-tab-section-1').press('ArrowRight')
                expect(page.locator('#note-tab-section-2')).to_be_focused()
                expect(page.locator('#note-tab-section-2')).to_have_attribute('aria-selected', 'true')
                with page.expect_download() as download:
                    page.locator('#prepExport').click()
                text = download.value.path().read_text(encoding='utf-8')
                self.assertIn('General feedback', text)
                self.assertIn('My real example <not HTML>.', text)
                page.locator('#tab-practice').click()
                expect(page.locator('#practiceAnswer')).not_to_be_visible()
                page.locator('#practiceReveal').click()
                expect(page.locator('#practiceAnswer')).to_be_visible()
                page.clock.install()
                page.locator('#practiceTimer').click()
                page.clock.fast_forward(30000)
                expect(page.locator('#practiceClock')).to_have_text('1:00')
                page.locator('#practiceTimer').click()
                page.clock.fast_forward(5000)
                expect(page.locator('#practiceClock')).to_have_text('1:00')
                page.locator('#practiceTimer').click()
                page.clock.fast_forward(60000)
                expect(page.locator('#practiceStatus')).to_contain_text('90 seconds complete')
                page.locator('#practiceNext').click()
                expect(page.locator('#practiceClock')).to_have_text('1:30')
                expect(page.locator('#practiceAnswer')).not_to_be_visible()
                page.locator('#practiceNote').click()
                expect(page.locator('#prepNotes')).to_be_visible()
                page.set_viewport_size({'width':390,'height':844})
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                page.locator('#prepNotes').scroll_into_view_if_needed()
                page.screenshot(path=str(ROOT / 'data' / 'interview-dark-mobile.png'))
                page.emulate_media(media='print')
                expect(page.locator('#prepGuide')).to_be_visible()
                expect(page.locator('#prepNotes')).not_to_be_visible()
                page.emulate_media(media='screen')
                page.evaluate("() => {Storage.prototype.setItem = function() {throw new Error('Storage full')}}")
                page.locator('#prepNote').fill('Unsaved but exportable feedback')
                expect(page.locator('#noteSaved')).to_contain_text('Could not save')
                self.assertFalse(errors, errors)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
