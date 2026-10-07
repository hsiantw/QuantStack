"""Notebook autosave, organisation, safe failures and legacy data compatibility."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright, expect
from dashboard import Handler
from test_research import bars


class NotesBrowserTests(unittest.TestCase):
    def test_notebook_persistence_and_controls(self):
        class QuietHandler(Handler):
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                errors = []
                history = bars()

                def api(route):
                    name = route.request.url.split('/api/')[1].split('?')[0]
                    data = ([dict(symbol='TEST', name='Test', close=120, change=1, currency='USD', asset_type='Stocks')]
                            if name == 'symbols' else history if name == 'history' else [])
                    route.fulfill(json=data)

                context.route('**/api/**', api)
                page = context.new_page()
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function('rows.length > 0')
                legacy = dict(id='legacy', title='Existing note', body='Keep old data', category='Idea', updatedAt='2026-10-01T00:00:00Z')
                page.evaluate('(n) => localStorage.setItem("atlas.ideaNotes.v1", JSON.stringify([n]))', legacy)
                page.evaluate("openChartResearch('ideas')")
                expect(page.locator('#ideaTitle')).to_have_value('Existing note')
                page.locator('.idea-category-manager summary').click()
                page.locator('#ideaCategoryName').fill('Earnings')
                page.locator('#ideaCategoryCreate').click()
                expect(page.locator('[data-category="Earnings"]')).to_have_attribute('aria-pressed', 'true')
                page.locator('#ideaNewChecklist').click()
                expect(page.locator('#ideaCategory')).to_have_value('Earnings')
                page.locator('#ideaTitle').fill('Results checklist')
                page.get_by_role('textbox', name='Checklist item 1', exact=True).fill('Check earnings date')
                expect(page.locator('#ideaStatus')).to_contain_text('Saved')
                expect(page.locator('[data-category="Earnings"] small')).to_have_text('1')
                page.locator('#ideaCategoryName').fill('Quarterly reviews')
                page.locator('#ideaCategoryRename').click()
                expect(page.locator('#ideaCategory')).to_have_value('Quarterly reviews')
                page.locator('#ideaCategoryName').fill('Research')
                page.locator('#ideaCategoryCreate').click()
                expect(page.locator('#ideaStatus')).to_contain_text('already exists')
                page.once('dialog', lambda dialog: dialog.accept())
                page.locator('#ideaCategoryRemove').click()
                expect(page.locator('#ideaCategory')).to_have_value('Idea')
                expect(page.locator('#ideaTitle')).to_have_value('Results checklist')
                page.locator('[data-category=""]').click()
                page.locator('#ideaNew').click()
                page.locator('#ideaTitle').fill('Earnings <literal>')
                page.locator('#ideaBody').fill('Check guidance')
                page.locator('#ideaLabels').fill('earnings, follow up')
                page.locator('#ideaColor').select_option('yellow')
                page.locator('#ideaAddItem').click()
                page.get_by_role('textbox', name='Checklist item 1', exact=True).fill('Read filing')
                page.get_by_role('checkbox', name='Complete item 1', exact=True).check()
                expect(page.locator('#ideaStatus')).to_contain_text('Saved')
                page.locator('#ideaPin').click()
                expect(page.locator('#ideaPin')).to_have_attribute('aria-pressed', 'true')
                saved = page.evaluate('JSON.parse(localStorage.getItem("atlas.ideaNotes.v1"))')
                self.assertEqual(saved[0]['items'], [dict(text='Read filing', done=True)])
                self.assertEqual(saved[0]['color'], 'yellow')
                self.assertEqual(len(saved), 3)
                # Reload without pressing Save note.
                page.reload()
                page.wait_for_function('rows.length > 0')
                page.evaluate("openChartResearch('ideas')")
                expect(page.locator('#ideaTitle')).to_have_value('Earnings <literal>')
                expect(page.get_by_role('checkbox', name='Complete item 1', exact=True)).to_be_checked()
                page.locator('#ideaArchive').click()
                expect(page.locator('#ideaNotesList')).not_to_contain_text('Earnings <literal>')
                page.locator('#ideaView').select_option('archive')
                expect(page.locator('#ideaNotesList')).to_contain_text('Earnings <literal>')
                page.locator('#ideaDelete').click()
                page.locator('#ideaView').select_option('trash')
                expect(page.locator('#ideaRestore')).to_be_visible()
                page.locator('#ideaRestore').click()
                expect(page.locator('#ideaView')).to_have_value('notes')
                page.locator('#ideaSearch').fill('follow up')
                expect(page.locator('[data-idea-open]')).to_have_count(1)
                page.locator('#ideaSearch').fill('')
                page.locator('#ideaView').select_option('pinned')
                expect(page.locator('[data-idea-open]')).to_have_count(1)
                page.locator('#ideaView').select_option('notes')
                page.locator('#ideaDuplicate').click()
                expect(page.locator('#ideaTitle')).to_have_value('Earnings <literal> (copy)')
                # Navigating immediately must flush pending edits.
                page.locator('#ideaBody').fill('Navigation flush')
                page.evaluate("openChartResearch('options'); openChartResearch('ideas')")
                expect(page.locator('#ideaBody')).to_have_value('Navigation flush')
                # Pending edits are included in the portable backup.
                page.locator('#ideaBody').fill('Backup flush')
                with page.expect_download() as download:
                    page.locator('#ideaBackup').click()
                backup = json.loads(download.value.path().read_text(encoding='utf-8'))
                self.assertIn('Backup flush', backup['entries']['atlas.ideaNotes.v1'])
                # A stale editor preserves the other tab's version and creates a copy.
                page.evaluate("""() => {
                    const input = document.getElementById('ideaBody');
                    input.value = 'Local conflict edit'; input.dispatchEvent(new Event('input', {bubbles:true}));
                    const notes = JSON.parse(localStorage.getItem('atlas.ideaNotes.v1'));
                    notes[0].body = 'Other tab version';
                    localStorage.setItem('atlas.ideaNotes.v1', JSON.stringify(notes));
                    QuantNotes.flush();
                }""")
                expect(page.locator('#ideaStatus')).to_contain_text('conflict copy')
                notes = page.evaluate('JSON.parse(localStorage.getItem("atlas.ideaNotes.v1"))')
                self.assertTrue(any(n['body'] == 'Other tab version' for n in notes))
                self.assertTrue(any(n['body'] == 'Local conflict edit' for n in notes))
                # A full browser store must leave edits in the editor and allow retry.
                page.evaluate("""() => {
                    window.originalSetItem = Storage.prototype.setItem;
                    Storage.prototype.setItem = function(k,v) {
                        if (k === 'atlas.ideaNotes.v1') throw Error('Storage full');
                        return originalSetItem.call(this,k,v);
                    };
                }""")
                page.locator('#ideaBody').fill('Recoverable draft')
                expect(page.locator('#ideaStatus')).to_contain_text('Storage full')
                page.evaluate("openChartResearch('options')")
                expect(page.locator('#ideaBody')).to_have_value('Recoverable draft')
                page.evaluate('() => { Storage.prototype.setItem = originalSetItem; }')
                page.locator('#ideaSave').click()
                expect(page.locator('#ideaStatus')).to_contain_text('Saved')
                # Fresh writes from another tab become visible without reopening.
                other = context.new_page()
                other.goto(f'http://127.0.0.1:{server.server_port}')
                other.evaluate("""() => {
                    const notes = JSON.parse(localStorage.getItem('atlas.ideaNotes.v1'));
                    notes[0].body = 'From second tab';
                    localStorage.setItem('atlas.ideaNotes.v1', JSON.stringify(notes));
                }""")
                expect(page.locator('#ideaBody')).to_have_value('From second tab')
                other.close()
                # The notes grid stays bounded for a large existing notebook.
                page.evaluate("""() => {
                    QuantNotes.unmount();
                    const notes = Array.from({length:150},(_,i)=>({id:'bulk'+i,title:'Note '+i,body:'Body',category:'Idea',updatedAt:'2026-10-07T00:00:00Z'}));
                    localStorage.setItem('atlas.ideaNotes.v1',JSON.stringify(notes));
                    openChartResearch('options'); openChartResearch('ideas');
                }""")
                expect(page.locator('[data-idea-open]')).to_have_count(60)
                page.locator('#ideaMore').click()
                expect(page.locator('[data-idea-open]')).to_have_count(120)
                Path('data').mkdir(exist_ok=True)
                page.locator('#ideaExpand').scroll_into_view_if_needed()
                page.locator('#ideaExpand').click()
                page.locator('#ideaTitle').scroll_into_view_if_needed()
                page.screenshot(path='data/notebook-desktop.png')
                page.set_viewport_size({'width': 390, 'height': 844})
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                page.screenshot(path='data/notebook-mobile.png')
                self.assertFalse(errors, errors)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
