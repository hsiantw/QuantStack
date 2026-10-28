"""Real chart -> gateway -> worker smoke test, isolated fixtures and optional public CCXT pull."""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from analysis_jobs import Jobs
from dashboard import Handler
from serve_quantstack import create_app
from test_analysis import fixture

ROOT = Path(__file__).resolve().parent


async def main(live_exchange=False, database_only=False):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        database=root/'market.sqlite'
        fixture(database)
        jobs=Jobs(ROOT,database=database)
        jobs.directory=root/'jobs'
        jobs.records={}
        backend=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=backend.serve_forever,daemon=True)
        with patch('dashboard.DATABASE',database), patch('analysis_jobs._manager',jobs):
            thread.start()
            server=TestServer(create_app(f'http://127.0.0.1:{backend.server_port}',accounts_path=root/'accounts.sqlite'))
            await server.start_server()
            try:
                async with async_playwright() as p:
                    browser=await p.chromium.launch(channel='msedge',headless=True)
                    page=await browser.new_page(viewport={'width':1440,'height':1000})
                    errors=[]
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    await page.goto(str(server.make_url('/workspace/')))
                    await page.wait_for_function("typeof rows !== 'undefined' && rows.length > 100")
                    await page.locator('#workspaceChartOptions > summary').click()
                    await page.locator('#databaseOpen').click()
                    await expect(page.locator('#databaseTable')).to_have_value('prices')
                    await page.locator('#databaseClear').click()
                    await expect(page.locator('#databaseStatusLine')).to_contain_text('1,300 matching rows')
                    await page.locator('#databaseSymbol').fill('AAA')
                    await page.locator('#databaseApply').click()
                    await expect(page.locator('#databaseStatusLine')).to_contain_text('260 matching rows')
                    await page.locator('#databaseNext').click()
                    await expect(page.locator('#databasePage')).to_contain_text('51–100')
                    async with page.expect_download() as csv_download:
                        await page.locator('#databaseExport').click()
                    assert (await csv_download.value).suggested_filename=='prices-rows-51.csv'
                    await page.locator('#databaseDialog summary').click()
                    await expect(page.locator('#databaseSchema')).to_contain_text('adjusted_close')
                    shots=ROOT/'data'/'analysis'
                    shots.mkdir(parents=True,exist_ok=True)
                    await page.screenshot(path=str(shots/'database-desktop.png'))
                    await page.locator('#databaseTable').select_option('intraday_state')
                    await expect(page.locator('#databaseRows')).to_contain_text('No rows match')
                    await page.set_viewport_size({'width':390,'height':844})
                    await page.evaluate("window.atlasTheme.apply('dark')")
                    await page.screenshot(path=str(shots/'database-mobile.png'))
                    assert await page.locator('#databaseDialog').evaluate('(el)=>el.scrollWidth<=el.clientWidth+2')
                    response=await page.request.get(str(server.make_url('/workspace/api/database')),headers={'Origin':'https://other.example'})
                    assert response.status==403
                    await page.locator('#databaseClose').click()
                    await page.locator('#analysisOpen').click()
                    await expect(page.locator('#analysisDialog')).to_be_visible()
                    if database_only:
                        await page.screenshot(path=str(shots/'ui-mobile-dark.png'))
                        assert not errors,errors
                        await browser.close()
                        print('Database UI passed: schema, filters, pagination, CSV, empty table, local access and mobile layout.')
                        return
                    await page.set_viewport_size({'width':1440,'height':1000})
                    await page.evaluate("window.atlasTheme.apply('light')")
                    async def fill(symbols):
                        await page.locator('#analysisSymbols').fill(symbols)
                        await page.locator('#analysisSource').select_option('Yahoo Finance')
                        await page.locator('#analysisInterval').select_option('1d',force=True)
                        await page.locator('#analysisStart').fill('2024-01-01')
                        await page.locator('#analysisEnd').fill('2024-09-16')
                    async def complete():
                        await page.locator('#analysisRun').click()
                        await expect(page.locator('#analysisStatus')).to_contain_text('Complete:',timeout=120000)
                        await expect(page.locator('#analysisError')).to_be_hidden()
                        await expect(page.locator('#analysisDownload')).to_be_visible()
                    await fill('AAA')
                    await complete()
                    await expect(page.locator('#analysisResult')).to_contain_text('vectorbt')
                    async with page.expect_download() as download:
                        await page.locator('#analysisDownload').click()
                    assert (await download.value).suggested_filename.endswith('.json')
                    await page.locator('[data-analysis-kind=portfolio]').click()
                    await page.locator('#analysisSymbols').fill('AAA, BBB')
                    await page.locator('#analysisWeights').fill('60, 40')
                    await complete()
                    await expect(page.locator('#analysisResult')).to_contain_text('quantstats')
                    await page.locator('[data-analysis-kind=factor]').click()
                    await page.locator('#analysisSymbols').fill('AAA BBB CCC DDD EEE')
                    await complete()
                    await expect(page.locator('#analysisResult')).to_contain_text('alphalens-reloaded')
                    shots=ROOT/'data'/'analysis'
                    shots.mkdir(parents=True,exist_ok=True)
                    await page.screenshot(path=str(shots/'ui-desktop.png'))
                    # Remote requests cannot start jobs or read results through the gateway.
                    response=await page.request.post(str(server.make_url('/workspace/api/analysis')),headers={'Origin':'https://other.example'},data={})
                    assert response.status==403
                    response=await page.request.get(str(server.make_url('/workspace/api/analysis')),headers={'Host':'remote.example'})
                    assert response.status==403
                    # Cancellation is a real worker lifecycle action, not just a UI flag.
                    await page.locator('[data-analysis-kind=strategy]').click()
                    await page.locator('#analysisSymbols').fill('AAA')
                    await page.locator('#analysisRun').click()
                    await expect(page.locator('#analysisCancel')).to_be_visible()
                    await page.locator('#analysisCancel').click()
                    await expect(page.locator('#analysisStatus')).to_contain_text('Cancelled:',timeout=10000)
                    if live_exchange:
                        await page.locator('[data-analysis-kind=exchange]').click()
                        await page.locator('#analysisSymbols').fill('BTC-USD')
                        await page.locator('#analysisSource').select_option('Bitstamp')
                        await page.locator('#analysisInterval').select_option('60m')
                        day=(datetime.now(timezone.utc)-timedelta(days=1)).date().isoformat()
                        await page.locator('#analysisStart').fill(day)
                        await page.locator('#analysisEnd').fill(day)
                        await complete()
                        await expect(page.locator('#analysisResult')).to_contain_text('ccxt')
                        await page.locator('#analysisViewChart').click()
                        await page.wait_for_function("selected?.symbol==='BTC-USD' && rows.length > 0 && rows.every(r=>r.source==='Bitstamp')")
                        await expect(page.locator('#analysisChartSource')).to_have_value('Bitstamp')
                        await page.locator('#analysisOpen').click()
                    await page.set_viewport_size({'width':390,'height':844})
                    await page.evaluate("window.atlasTheme.apply('dark')")
                    await page.wait_for_timeout(100)
                    await page.screenshot(path=str(shots/'ui-mobile-dark.png'))
                    assert await page.locator('#analysisDialog').evaluate('(el)=>el.scrollWidth<=el.clientWidth+2')
                    await page.locator('#analysisClose').click()
                    await expect(page.locator('#analysisDialog')).to_be_hidden()
                    assert not errors,errors
                    await browser.close()
                    print('Research UI passed: real strategy/portfolio/factor jobs, downloads, origin checks, cancellation, responsive layout'+(', live CCXT pull and main chart source' if live_exchange else ''))
            finally:
                for item in list(jobs.records.values()):
                    jobs.cancel(item['id'])
                if jobs.thread:
                    await asyncio.to_thread(jobs.thread.join,15)
                await server.close()
                await asyncio.to_thread(backend.shutdown)
                backend.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live-exchange',action='store_true')
    parser.add_argument('--database-only',action='store_true')
    args=parser.parse_args()
    asyncio.run(main(args.live_exchange,args.database_only))
