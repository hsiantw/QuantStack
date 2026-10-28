"""Polling lifecycle checks with fake APIs: never submit provider downloads."""
from pathlib import Path
import unittest

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parent


class PollingUITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(channel='msedge', headless=True)
        self.page = await self.browser.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        await self.page.route('http://polling.test/**', lambda route: route.fulfill(
            body='<div class="terminal-actions"></div>'))
        await self.page.goto('http://polling.test/')
        await self.page.clock.install()
        await self.page.evaluate('''() => {
          window.selected = {symbol:'BTC-USD'};
          window.interval = '1m';
          window.testHidden = false;
          Object.defineProperty(document, 'hidden', {get: () => testHidden});
          window.visibility = hidden => {
            testHidden = hidden;
            document.dispatchEvent(new Event('visibilitychange'));
          };
          window.calls = [];
          window.inflight = 0;
          window.peak = 0;
          window.hold = false;
          window.fetch = async (path, options = {}) => {
            calls.push({path:String(path), method:options.method || 'GET'});
            peak = Math.max(peak, ++inflight);
            if (hold) await new Promise(resolve => window.release = resolve);
            --inflight;
            const value = String(path).includes('pull-queue') ? {requests:[]} :
              String(path).includes('?id=') ? {status:'running', message:'Working'} : {available:true};
            return {ok:true, json:async()=>value, text:async()=>JSON.stringify(value)};
          };
        }''')

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()
        self.assertEqual(self.errors, [])

    async def test_queue_coalesces_refresh_and_pauses_hidden_or_closed(self):
        page = self.page
        await page.add_script_tag(path=str(ROOT / 'web/data-pulls.js'))
        await page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
        await page.locator('#pullData').click()
        await expect(page.locator('#pullSubmit')).to_be_enabled()
        self.assertEqual(await page.evaluate('calls.length'), 1)
        await page.evaluate('''() => {
          hold = true;
          document.getElementById('pullRefresh').click();
          document.getElementById('pullRefresh').click();
        }''')
        self.assertEqual(await page.evaluate('calls.length'), 2)
        await page.clock.run_for(8000)
        self.assertEqual(await page.evaluate('calls.length'), 2)
        await page.evaluate('visibility(true); hold=false; release()')
        await page.clock.run_for(12000)
        self.assertEqual(await page.evaluate('calls.length'), 2)
        await page.evaluate('visibility(false)')
        await page.wait_for_function('calls.length === 3 && inflight === 0')
        await page.clock.run_for(4000)
        self.assertEqual(await page.evaluate('calls.length'), 4)
        await page.locator('#pullClose').click()
        await page.clock.run_for(8000)
        self.assertEqual(await page.evaluate('calls.length'), 4)
        self.assertEqual(await page.evaluate('peak'), 1)

    async def test_research_pauses_status_polling_without_cancelling_job(self):
        page = self.page
        await page.evaluate("localStorage.setItem('quantstack.analysis.lastJob', 'job-1')")
        await page.add_script_tag(path=str(ROOT / 'web/analysis.js'))
        await page.evaluate("openQuantResearch('strategy')")
        await expect(page.locator('#analysisStatus')).to_have_text('Running: Working')
        self.assertEqual(await page.evaluate('calls.length'), 2)
        await page.evaluate('visibility(true)')
        await page.clock.run_for(12000)
        self.assertEqual(await page.evaluate('calls.length'), 2)
        await page.evaluate('visibility(false)')
        await page.wait_for_function('calls.length === 3 && inflight === 0')
        await page.clock.run_for(1200)
        self.assertEqual(await page.evaluate('calls.length'), 4)
        await page.locator('#analysisClose').click()
        await page.clock.run_for(12000)
        self.assertEqual(await page.evaluate('calls.length'), 4)
        self.assertTrue(await page.evaluate("calls.every(call => call.method === 'GET')"))


if __name__ == '__main__':
    unittest.main()
