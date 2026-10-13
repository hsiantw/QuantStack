"""Terminal entry page: real account requests and chart navigation."""
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from serve_quantstack import create_app


class LoginTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_login_session_and_mobile(self):
        with tempfile.TemporaryDirectory() as directory:
            server = TestServer(create_app('', snapshot_dir=Path(directory),
                                           accounts_path=Path(directory) / 'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(channel='msedge', headless=True)
                    context = await browser.new_context(viewport={'width': 1440, 'height': 1050})
                    page = await context.new_page()
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    # Isolate the entry flow from market-data/chart initialization.
                    await page.route('**/workspace/', lambda route: route.fulfill(body='Supercharts workspace'))
                    await page.goto(str(server.make_url('/login')))
                    await expect(page.locator('h1')).to_contain_text('One terminal.')
                    await page.screenshot(path=str(Path(__file__).parent / 'login-preview.png'), full_page=True)
                    await page.locator('#registerTab').click()
                    await page.locator('#username').fill('terminaltester')
                    await page.locator('#password').fill('terminal-test-password')
                    await page.locator('#showPassword').click()
                    await expect(page.locator('#password')).to_have_attribute('type', 'text')
                    await page.locator('#submit').click()
                    await page.wait_for_url('**/workspace/')
                    await page.goto(str(server.make_url('/login')))
                    await expect(page.locator('#identity')).to_contain_text('terminaltester')
                    await page.locator('#switchAccount').click()
                    await page.locator('#username').fill('terminaltester')
                    await page.locator('#password').fill('incorrect-password')
                    await page.locator('#submit').click()
                    await expect(page.locator('#status')).to_contain_text('Invalid username or password')
                    await page.locator('#password').fill('terminal-test-password')
                    await page.locator('#submit').click()
                    await page.wait_for_url('**/workspace/')
                    await context.clear_cookies()
                    await page.set_viewport_size({'width': 390, 'height': 844})
                    await page.goto(str(server.make_url('/login')))
                    await expect(page.locator('#loginForm')).to_be_visible()
                    self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                    await page.locator('.guest').click()
                    await page.wait_for_url('**/workspace/')
                    self.assertFalse(errors, errors)
                    await browser.close()
            finally:
                await server.close()
