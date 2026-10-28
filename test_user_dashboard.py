"""Personal dashboard reads saved browser/account work without replacing either copy."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from serve_quantstack import create_app


def note(identifier, title, **extra):
    return dict(id=identifier, title=title, body='Research saved in this browser.',
                category='Idea', updatedAt='2026-10-12T12:00:00Z', **extra)


class UserDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'symbols.json').write_text('[]', encoding='utf-8')
        self.server = TestServer(create_app('', snapshot_dir=root,
                                            accounts_path=root / 'accounts.sqlite'))
        await self.server.start_server()
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(channel='msedge', headless=True)
        self.context = await self.browser.new_context(viewport={'width': 1440, 'height': 1000})
        self.page = await self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        await self.page.goto(str(self.server.make_url('/dashboard')))

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()
        await self.server.close()
        self.temp.cleanup()

    async def seed(self, values, page=None):
        page = page or self.page
        entries = {key: json.dumps(value) for key, value in values.items()}
        await page.evaluate("""entries => {
            for (const [key, value] of Object.entries(entries)) localStorage.setItem(key, value);
        }""", entries)

    async def storage(self):
        # The backup journal drains asynchronously; it is not saved user work.
        return await self.page.evaluate('''Object.fromEntries(Object.entries(localStorage)
            .filter(([key]) => !key.startsWith('quantstack.localPreferences.pending.')))''')

    async def account(self, action, payload):
        result = await self.page.evaluate("""async ({action, payload}) => {
            const response = await fetch('/workspace/api/account/' + action, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                credentials: 'same-origin', body: JSON.stringify(payload)
            });
            return {status: response.status, data: await response.json()};
        }""", {'action': action, 'payload': payload})
        self.assertEqual(result['status'], 200, result['data'])
        return result['data']

    async def test_browser_summary_navigation_cross_tab_and_mobile(self):
        page = self.page
        for path in ('/dashboard', '/dashboard/', '/user-dashboard.js', '/user-dashboard.css'):
            response = await page.request.get(str(self.server.make_url(path)))
            self.assertEqual(response.status, 200, path)
        await expect(page.locator('#dashboardSource')).to_have_value('browser')
        await expect(page.locator('#dashboardSource option[value=account]')).to_have_js_property('disabled', True)
        for identifier in ('dashboardSavedCount', 'dashboardWatchlistCount', 'dashboardNotesCount'):
            await expect(page.locator('#' + identifier)).to_have_text('0')
        await expect(page.locator('#dashboardPrepCount')).to_have_text(re.compile(r'0\s*/\s*13'))
        for identifier in ('dashboardRecentCharts', 'dashboardWatchlists', 'dashboardNotes'):
            self.assertTrue((await page.locator('#' + identifier).inner_text()).strip(), identifier)

        literal_title = '<img src=x onerror="window.dashboardInjected=true"> literal note'
        await self.seed({
            'atlas.saved': ['AAPL', 'AAPL', 'BTC-USD', 'javascript:alert(1)'],
            'atlas.watchlists.v1': {
                'lists': [
                    {'id': 'list-research', 'name': 'Research', 'symbols': ['AAPL']},
                    {'id': 'list-red', 'name': 'Red signals', 'symbols': ['STALE'], 'color': 'red'}
                ],
                'labels': {'AAPL': 'red', 'MSFT': 'red', 'BTC-USD': 'blue'}
            },
            'atlas.ideaNotes.v1': [
                note('11111111-1111-4111-8111-111111111111', literal_title, symbol='AAPL'),
                note('22222222-2222-4222-8222-222222222222', 'Second active note'),
                note('archived-note', 'Archived private title', archived=True),
                note('trashed-note', 'Deleted private title', trashed=True)
            ],
            'atlas.chart.session': {'symbol': 'AAPL', 'interval': '1w', 'period': 'CUSTOM',
                                    'start': '2026-01-01', 'end': '2026-10-01'},
            'atlas.interview-prep.v1': {'notes': {'general': 'Interview feedback'},
                                       'reviewed': ['section-1', 'section-1', 'algorithms', 'invented'],
                                       'note': 'general'}
        })
        before = await self.storage()
        await page.reload()
        for identifier in ('dashboardSavedCount', 'dashboardWatchlistCount', 'dashboardNotesCount'):
            await expect(page.locator('#' + identifier)).to_have_text('2')
        await expect(page.locator('#dashboardPrepCount')).to_have_text(re.compile(r'2\s*/\s*13'))
        await expect(page.locator('#dashboardPrepProgress')).to_have_attribute('value', '2')
        await expect(page.locator('#dashboardNotes')).to_contain_text(literal_title)
        await expect(page.locator('#dashboardNotes')).not_to_contain_text('Archived private title')
        await expect(page.locator('#dashboardNotes')).not_to_contain_text('Deleted private title')
        self.assertFalse(await page.evaluate('Boolean(window.dashboardInjected)'))
        await expect(page.locator('#dashboardNotes img')).to_have_count(0)
        note_link = page.locator('#dashboardNotes a').filter(has_text=literal_title)
        self.assertEqual(parse_qs(urlsplit(await note_link.get_attribute('href')).query)['note'],
                         ['11111111-1111-4111-8111-111111111111'])
        self.assertEqual(urlsplit(await note_link.get_attribute('href')).path, '/ideas')
        chart = urlsplit(await page.locator('#dashboardContinue').get_attribute('href'))
        self.assertEqual(chart.path, '/workspace/')
        self.assertEqual(parse_qs(chart.query), {
            'symbol': ['AAPL'], 'interval': ['1w'], 'period': ['CUSTOM'],
            'start': ['2026-01-01'], 'end': ['2026-10-01']})
        await expect(page.locator('#dashboardWatchlists')).to_contain_text('Red signals')
        await expect(page.locator('#dashboardWatchlists')).not_to_contain_text('STALE')
        await page.locator('#dashboardNoteSearch').fill('Second active')
        await expect(page.locator('#dashboardNotes')).to_contain_text('Second active note')
        await expect(page.locator('#dashboardNotes')).not_to_contain_text(literal_title)
        await page.locator('#dashboardNoteSearch').fill('')
        self.assertEqual(await self.storage(), before, 'Reading/searching dashboard must preserve local work')

        # A real second tab updates the same origin; the dashboard should refresh without reloading.
        other = await self.context.new_page()
        await other.goto(str(self.server.make_url('/dashboard')))
        await self.seed({'atlas.saved': ['AAPL', 'MSFT', 'BTC-USD'],
                         'atlas.chart.session': {'symbol': 'javascript:alert(1)'}}, other)
        await expect(page.locator('#dashboardSavedCount')).to_have_text('3')
        await expect(page.locator('#dashboardContinue')).not_to_have_attribute('href', re.compile('javascript'))
        await other.close()
        await page.set_viewport_size({'width': 390, 'height': 844})
        self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        await page.locator('[data-site-theme]').select_option('dark')
        await expect(page.locator('html')).to_have_attribute('data-palette', 'dark')
        await page.locator('[data-site-theme]').select_option('paper')
        await expect(page.locator('html')).to_have_attribute('data-palette', 'paper')
        self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.assertFalse(self.errors, self.errors)

    async def test_account_source_preserves_browser_and_clears_on_signout(self):
        page = self.page
        registration = await self.account('register', {'username': 'dashboard-user', 'password': 'test-dashboard-password'})
        user = registration['user']
        # Authentication selects a separate local preference profile. Seed this
        # profile after bootstrap, so this tests preview isolation rather than
        # expecting guest favorites to be imported into a different account.
        await page.reload()
        await self.seed({'atlas.saved': ['AAPL'], 'atlas.ideaNotes.v1': [note('local', 'Local notebook')]})
        await self.account('workspace', {'account_id': user['id'], 'revision': 0, 'entries': {
            'atlas.saved': json.dumps(['MSFT', 'BTC-USD']),
            'atlas.ideaNotes.v1': json.dumps([note('private', 'Account-only private notebook')])
        }})
        before = await self.storage()
        await page.reload()
        await expect(page.locator('#dashboardSource option[value=account]')).to_have_js_property('disabled', False)
        await expect(page.locator('#dashboardSavedCount')).to_have_text('1')
        await expect(page.locator('#dashboardNotes')).to_contain_text('Local notebook')
        await page.locator('#dashboardSource').select_option('account')
        await expect(page.locator('#dashboardSavedCount')).to_have_text('2')
        await expect(page.locator('#dashboardNotes')).to_contain_text('Account-only private notebook')
        await expect(page.locator('#dashboardNotes')).not_to_contain_text('Local notebook')
        self.assertEqual(await self.storage(), before, 'Account preview must not import or overwrite browser work')
        await page.locator('#dashboardSource').select_option('browser')
        await expect(page.locator('#dashboardNotes')).to_contain_text('Local notebook')
        await page.locator('#dashboardSource').select_option('account')
        await expect(page.locator('#dashboardNotes')).to_contain_text('Account-only private notebook')

        # Signing out from another tab invalidates the private preview immediately.
        other = await self.context.new_page()
        await other.goto(str(self.server.make_url('/dashboard')))
        await self.account('logout', {'account_id': user['id']})
        await other.evaluate("localStorage.setItem('quantstack.accountChanged', 'signed-out-test')")
        await expect(page.locator('#dashboardSource')).to_have_value('browser')
        await expect(page.locator('#dashboardNotes')).not_to_contain_text('Account-only private notebook')
        await expect(page.locator('#dashboardNotes')).to_contain_text('Local notebook')
        await expect(page.locator('#dashboardSource option[value=account]')).to_have_js_property('disabled', True)
        remaining = await self.storage()
        remaining.pop('quantstack.accountChanged', None)
        self.assertEqual(remaining, before)
        await other.close()
        self.assertFalse(self.errors, self.errors)

    async def test_unreadable_storage_is_reported_and_never_overwritten(self):
        page = self.page
        await page.evaluate("""() => {
            localStorage.setItem('atlas.ideaNotes.v1', '{broken JSON');
            localStorage.setItem('atlas.saved', '["AAPL"]');
        }""")
        before = await self.storage()
        await page.reload()
        await expect(page.locator('#dashboardSavedCount')).to_have_text('1')
        await expect(page.locator('main')).to_contain_text(re.compile(r'could not|unreadable|unable|unavailable', re.I))
        self.assertEqual(await self.storage(), before)

        await page.add_init_script("""Storage.prototype.getItem = function() {
            throw new DOMException('Storage denied', 'SecurityError');
        };""")
        await page.reload()
        await expect(page.locator('main')).to_contain_text(re.compile(r'could not|unreadable|unable|unavailable', re.I))
        self.assertEqual(await self.storage(), before)
        self.assertFalse(self.errors, self.errors)

    async def test_note_link_opens_requested_older_note(self):
        await self.seed({'atlas.ideaNotes.v1': [
            note('older', 'Older idea to resume'),
            dict(note('newer', 'Newest idea'), updatedAt='2026-10-13T12:00:00Z')
        ]})
        await self.page.reload()
        await self.page.locator('#dashboardNotes a').filter(has_text='Older idea to resume').click()
        await expect(self.page.locator('#ideaTitle')).to_have_value('Older idea to resume')
        self.assertFalse(self.errors, self.errors)


if __name__ == '__main__':
    unittest.main()
