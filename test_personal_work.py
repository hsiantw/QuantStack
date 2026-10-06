"""Verify personal notes, drawings and portable backups in an isolated browser."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright, expect
from dashboard import Handler
from test_research import bars


class PersonalWorkTests(unittest.TestCase):
    def test_save_reload_backup_restore_and_invalid_file(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
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
                def notes():
                    if not page.locator('#workspaceNotes').is_visible():
                        page.locator('[data-side-panel=notes]').click()
                notes()
                page.locator('#workspaceNotes').fill('Watch the breakout <keep literal>.')
                expect(page.locator('#workspaceNotesStatus')).to_contain_text('Saved')
                page.get_by_role('button', name='Open idea notebook', exact=True).click()
                page.locator('#ideaTitle').fill('A market idea')
                page.locator('#ideaBody').fill('Check volume before entry.')
                page.locator('#ideaSave').click()
                expect(page.locator('#ideaStatus')).to_contain_text('Saved')
                # Use the drawing editor persistence path with deterministic geometry.
                page.evaluate("""() => {
                    drawings = [{id:'test-drawing',type:'horizontal',price:120,style:{...drawingDefaults}}];
                    saveDrawings();
                }""")
                page.reload()
                page.wait_for_function('rows.length > 0 && drawings.length === 1')
                notes()
                expect(page.locator('#workspaceNotes')).to_have_value('Watch the breakout <keep literal>.')
                with page.expect_download() as download:
                    page.locator('#personalWorkExport').click()
                backup = json.loads(download.value.path().read_text(encoding='utf-8'))
                self.assertEqual(len(backup['entries']), 3)
                self.assertIn('atlas.ideaNotes.v1', backup['entries'])
                def upload(data):
                    page.locator('#personalWorkFile').set_input_files(dict(
                        name='backup.json', mimeType='application/json', buffer=json.dumps(data).encode()))
                bad = {**backup, 'entries': {**backup['entries'], 'atlas.terminal': '{}'}}
                upload(bad)
                expect(page.locator('#personalWorkStatus')).to_contain_text('Unsupported')
                expect(page.locator('#workspaceNotes')).to_have_value('Watch the breakout <keep literal>.')
                # A rejected confirmation must preserve current edits.
                page.locator('#workspaceNotes').fill('Current work')
                page.once('dialog', lambda dialog: dialog.dismiss())
                upload(backup)
                expect(page.locator('#personalWorkStatus')).to_contain_text('canceled')
                expect(page.locator('#workspaceNotes')).to_have_value('Current work')
                # Storage failure rolls back earlier writes in the same import.
                page.evaluate("""() => {
                    window.originalSetItem = Storage.prototype.setItem;
                    Storage.prototype.setItem = function(key, value) {
                        if (key.startsWith('atlas.drawings.')) throw new DOMException('Full', 'QuotaExceededError');
                        return window.originalSetItem.call(this, key, value);
                    };
                }""")
                ordered = {**backup, 'entries': dict(sorted(backup['entries'].items(), key=lambda pair: pair[0].startswith('atlas.drawings.')))}
                page.once('dialog', lambda dialog: dialog.accept())
                upload(ordered)
                expect(page.locator('#personalWorkStatus')).to_contain_text('previous work was preserved')
                self.assertEqual(page.evaluate("localStorage.getItem('atlas.notes.TEST')"), 'Current work')
                page.evaluate('() => { Storage.prototype.setItem = window.originalSetItem; }')
                page.evaluate("localStorage.setItem('atlas.notes.OTHER', 'Keep me')")
                page.once('dialog', lambda dialog: dialog.accept())
                with page.expect_navigation():
                    upload(backup)
                page.wait_for_function('rows.length > 0 && drawings.length === 1')
                notes()
                expect(page.locator('#workspaceNotes')).to_have_value('Watch the breakout <keep literal>.')
                self.assertEqual(page.evaluate("localStorage.getItem('atlas.notes.OTHER')"), 'Keep me')
                page.get_by_role('button', name='Open idea notebook', exact=True).click()
                expect(page.locator('#ideaTitle')).to_have_value('A market idea')
                page.set_viewport_size({'width': 390, 'height': 844})
                notes()
                expect(page.locator('#personalWorkExport')).to_be_visible()
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                self.assertFalse(errors, errors)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
