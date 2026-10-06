"""Exercise registration and saved work across two independent browser devices."""
import asyncio
from pathlib import Path
import tempfile
import unittest

from aiohttp import web
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from serve_quantstack import create_app
from test_research import bars

ROOT = Path(__file__).parent


class AccountBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_devices_notes_drawings_conflicts_and_logout(self):
        history = bars()
        async def backend(request):
            if request.path == '/api/symbols':
                return web.json_response([dict(symbol='TEST', name='Test asset', asset_type='Stocks', close=120, change=1, currency='USD')])
            if request.path == '/api/history':
                return web.json_response(history)
            if request.path.startswith('/api/'):
                return web.json_response({})
            name = request.path.lstrip('/') or 'index.html'
            path = ROOT / 'web' / name
            if path.parent != ROOT / 'web' or not path.is_file():
                raise web.HTTPNotFound()
            return web.FileResponse(path)
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', backend)
        upstream = TestServer(app)
        await upstream.start_server()
        with tempfile.TemporaryDirectory() as directory:
            server = TestServer(create_app(str(upstream.make_url('')).rstrip('/'), accounts_path=Path(directory) / 'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(channel='msedge', headless=True)
                    a = await browser.new_context(viewport={'width':1440,'height':1000})
                    b = await browser.new_context(viewport={'width':1440,'height':1000})
                    first, second = await a.new_page(), await b.new_page()
                    errors = []
                    for page in (first, second):
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        page.on('dialog', lambda dialog: dialog.accept())
                        await page.goto(str(server.make_url('/workspace/')))
                        await page.wait_for_function('rows.length > 0')
                    await first.locator('[data-side-panel=notes]').click()
                    await first.locator('#workspaceNotes').fill('Account research')
                    await first.evaluate("""() => {
                        drawings=[{id:'account-line',type:'horizontal',price:120,style:{...drawingDefaults}}];
                        saveDrawings();
                    }""")
                    await first.locator('#accountButton').click()
                    await first.locator('#accountUsername').fill('researcher')
                    await first.locator('#accountPassword').fill('long-test-password')
                    await first.locator('#accountRegister').click()
                    await expect(first.locator('#accountSave')).to_be_enabled()
                    await first.locator('#accountSave').click()
                    await expect(first.locator('#accountStatus')).to_contain_text('Saved to your account')
                    await second.locator('#accountButton').click()
                    await second.locator('#accountUsername').fill('researcher')
                    await second.locator('#accountPassword').fill('long-test-password')
                    await second.locator('#accountLogin').click()
                    await expect(second.locator('#accountLoad')).to_be_enabled()
                    await expect(second.locator('#accountSave')).to_be_disabled()
                    await second.locator('#accountLoad').click()
                    await second.wait_for_function('rows.length > 0 && drawings.length === 1')
                    if not await second.locator('#workspaceNotes').is_visible():
                        await second.locator('[data-side-panel=notes]').click()
                    await expect(second.locator('#workspaceNotes')).to_have_value('Account research')
                    await second.locator('#workspaceNotes').fill('Updated on second device')
                    await second.locator('#accountButton').click()
                    await expect(second.locator('#accountSave')).to_be_enabled()
                    await second.locator('#accountSave').click()
                    await expect(second.locator('#accountStatus')).to_contain_text('Saved to your account')
                    await first.locator('#accountSave').click()
                    await expect(first.locator('#accountStatus')).to_contain_text('newer workspace')
                    await second.locator('#accountLogout').click()
                    await second.wait_for_function('rows.length > 0 && drawings.length === 0')
                    self.assertIsNone(await second.evaluate("localStorage.getItem('atlas.notes.TEST')"))
                    await second.locator('#accountButton').click()
                    await expect(second.locator('#accountForm')).to_be_visible()
                    await second.locator('#accountUsername').fill('otheruser')
                    await second.locator('#accountPassword').fill('another-long-password')
                    await second.locator('#accountRegister').click()
                    await expect(second.locator('#accountIdentity')).to_contain_text('No workspace saved')
                    await second.set_viewport_size({'width':390,'height':844})
                    self.assertTrue(await second.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                    await expect(second.locator('#accountLogout')).to_be_visible()
                    self.assertFalse(errors, errors)
                    await browser.close()
            finally:
                await server.close()
                await upstream.close()


if __name__ == '__main__':
    unittest.main()
