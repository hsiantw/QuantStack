"""A single palette preference follows navigation and synchronizes across tabs."""
from pathlib import Path
import tempfile
import unittest
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect
from serve_quantstack import create_app


class SiteThemeTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_theme_navigation_sync_and_mobile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'symbols.json').write_text('[]')
            (root/'screener.json').write_text('{"rows":[]}')
            server=TestServer(create_app('',snapshot_dir=root,accounts_path=root/'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser=await p.chromium.launch(channel='msedge',headless=True)
                    context=await browser.new_context(viewport={'width':1440,'height':1000},color_scheme='light')
                    page=await context.new_page()
                    errors=[]
                    page.on('pageerror',lambda error:errors.append(str(error)))
                    await page.goto(str(server.make_url('/')))
                    await page.get_by_label('Site theme').select_option('forest')
                    for path, control in [('/','[data-site-theme]'),('/login','[data-site-theme]'),('/account','[data-site-theme]'),('/screener','#toolTheme'),('/portfolio','#toolTheme'),('/workspace/','#workspaceTheme'),('/workspace/architecture.html','#mapTheme'),('/workspace/interview-prep.html','#prepTheme')]:
                        await page.goto(str(server.make_url(path)))
                        await expect(page.locator('html')).to_have_attribute('data-palette','forest')
                        await expect(page.locator(control)).to_have_value('forest')
                        self.assertEqual(await page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--atlas-surface').trim()"),'#101d1b')
                        await page.set_viewport_size({'width':390,'height':844})
                        if path!='/workspace/':
                            await expect(page.locator(control)).to_be_visible()
                            self.assertTrue(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'),path)
                        await page.set_viewport_size({'width':1440,'height':1000})
                    await page.goto(str(server.make_url('/login')))
                    for palette in ['light','dark','midnight','forest','paper']:
                        await page.get_by_label('Site theme').select_option(palette)
                        await expect(page.locator('html')).to_have_attribute('data-palette',palette)
                    other=await context.new_page()
                    await other.goto(str(server.make_url('/screener')))
                    await expect(other.locator('#toolTheme')).to_have_value('paper')
                    await page.get_by_label('Site theme').select_option('dark')
                    await expect(other.locator('#toolTheme')).to_have_value('dark')
                    await expect(other.locator('html')).to_have_attribute('data-palette','dark')
                    await page.get_by_label('Site theme').select_option('system')
                    await page.emulate_media(color_scheme='dark')
                    await expect(page.locator('html')).to_have_attribute('data-palette','dark')
                    await page.emulate_media(color_scheme='light')
                    await expect(page.locator('html')).to_have_attribute('data-palette','light')
                    await page.goto(str(server.make_url('/')))
                    await page.screenshot(path=str(Path(__file__).parent/'data/home-theme-light.png'))
                    await page.get_by_label('Site theme').select_option('dark')
                    await page.screenshot(path=str(Path(__file__).parent/'data/home-theme-dark.png'))
                    await page.goto(str(server.make_url('/login')))
                    await page.get_by_label('Site theme').select_option('light')
                    await page.screenshot(path=str(Path(__file__).parent/'data/login-theme-light.png'))
                    self.assertFalse(errors,errors)
                    await browser.close()
            finally:
                await server.close()
