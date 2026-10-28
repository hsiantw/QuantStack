"""Exercise strategy interactions against fixture prices and an isolated local DB.

Run with .venv-integration\\Scripts\\python.exe smoke_strategy_ui.py.
No live user preferences or market database are changed.
"""
import argparse
import asyncio
import math
from datetime import date, timedelta
from pathlib import Path
import tempfile

from aiohttp import web
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from serve_quantstack import create_app
from smoke_strategy import run as legacy_checks

ROOT = Path(__file__).resolve().parent


def fixture_prices():
    result = []
    for i in range(900):
        value = 100 + math.sin(i / 35) * 25 + math.sin(i / 4) * 12 + i * .03 - (25 if i % 31 == 0 else 0)
        result.append(dict(date=(date(2024, 1, 1) + timedelta(days=i)).isoformat(),
                           open=value, high=value + 1, low=value - 1, close=value,
                           adjusted_close=value, volume=1000 + i))
    return result


async def interactions(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='msedge', headless=True)
        context = await browser.new_context(viewport={'width': 1512, 'height': 1100})
        page = await context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(url)
        await page.wait_for_function("typeof rows !== 'undefined' && rows.length > 40")
        await page.locator('#workspaceOpenStrategy').click()
        await page.locator('#workspaceMaximizeDock').click()

        # Selecting a family and navigating with the keyboard updates real controls.
        await page.locator('[data-strategy-family="Reversion"]').click()
        await expect(page.locator('#st-kind')).to_have_value('rsi')
        await page.locator('#strategySignalTab').focus()
        await page.keyboard.press('ArrowRight')
        await expect(page.locator('#strategyRiskTab')).to_be_focused()
        await expect(page.locator('#strategyRiskConfig')).to_be_visible()
        await expect(page.locator('#strategySignalConfig')).not_to_be_visible()
        await page.locator('[data-allocation="50"]').click()
        await expect(page.locator('#st-allocation')).to_have_value('50')
        await expect(page.locator('#strategyAllocationRange')).to_have_value('50')
        await page.locator('#strategyAllocationRange').fill('75')
        await expect(page.locator('#st-allocation')).to_have_value('75')
        await page.locator('#st-allocation').fill('60')
        await expect(page.locator('#strategyAllocationRange')).to_have_value('60')

        await page.locator('#strategySignalTab').click()
        await page.locator('#st-kind').select_option('sma')
        await page.locator('#st-fast').fill('3')
        await page.locator('#st-slow').fill('7')
        await page.locator('#strategyRun').click()
        await expect(page.locator('#strategyResults')).to_be_visible()
        await expect(page.locator('#strategyError')).not_to_be_visible()

        # The inspection slider and keyboard support expose equivalent chart values.
        await page.locator('#strategyInspect').fill('0')
        initial = await page.locator('#strategyCurveValue').inner_text()
        await page.locator('#strategyCurve').focus()
        await page.keyboard.press('End')
        await expect(page.locator('#strategyCurveValue')).not_to_have_text(initial)
        await page.locator('#strategyChartMode').select_option('return')
        await page.locator('#strategyBenchmark').uncheck()
        await expect(page.locator('#strategyBenchmark')).not_to_be_checked()
        await page.locator('#strategyBenchmark').check()

        # Filtering/sorting works on the run snapshot and opens trade details.
        await page.locator('#strategyTradesTab').click()
        await page.locator('#strategyTradeFilter').select_option('winners')
        await expect(page.locator('#strategyTradeRows [data-trade-index]').first).to_be_visible()
        winning_indices = await page.locator('#strategyTradeRows [data-trade-index]').evaluate_all("nodes => nodes.map(n => Number(n.dataset.tradeIndex))")
        assert winning_indices, 'Fixture must contain winning trades'
        await page.locator('#strategyTradeRows [data-trade-index]').first.click()
        await expect(page.locator('#strategyTradeDetail')).to_be_visible()
        await page.locator('#strategyTradeFilter').select_option('losers')
        await expect(page.locator('#strategyTradeRows [data-trade-index]').first).to_be_visible()
        losing_indices = await page.locator('#strategyTradeRows [data-trade-index]').evaluate_all("nodes => nodes.map(n => Number(n.dataset.tradeIndex))")
        assert losing_indices and not set(winning_indices).intersection(losing_indices)
        await page.locator('#strategyTradeFilter').select_option('all')
        await page.locator('#strategyTradeSort').select_option('newest')
        indices = await page.locator('#strategyTradeRows [data-trade-index]').evaluate_all("nodes => nodes.map(n => Number(n.dataset.tradeIndex))")
        assert indices == sorted(indices, reverse=True), indices
        await page.locator('#strategyTradeSort').select_option('oldest')
        await page.locator('#strategyTradeRows [data-trade-index]').first.focus()
        await page.keyboard.press('Enter')
        await expect(page.locator('#strategyTradeDetail')).to_be_visible()
        await page.locator('#strategyOverviewTab').click()
        await page.locator('#strategyOverviewTab').focus()
        await page.keyboard.press('End')
        await expect(page.locator('#strategyPerformanceTab')).to_be_focused()
        await expect(page.locator('#strategyPerformance')).to_be_visible()

        # Editing parameters must make old results unavailable until a new run.
        await page.locator('#strategyRiskTab').click()
        await page.locator('#st-capital').fill('20000')
        await expect(page.locator('#strategyResults')).not_to_be_visible()
        await expect(page.locator('#strategyExport')).to_be_disabled()
        await page.locator('#strategyRun').click()
        await expect(page.locator('#strategyResults')).to_be_visible()
        await page.locator('#strategyOverviewTab').click()
        await page.locator('#strategySignalTab').click()

        await page.locator('#strategySavedTab').click()
        await page.locator('#strategySetupName').fill('Modern UI setup')
        await page.locator('#strategySaveSetup').click()
        await expect(page.locator('#strategySetupNotice')).to_contain_text('Saved')
        await page.evaluate('QuantLocalPreferences.flush()')
        saved = await page.evaluate("async () => (await fetch('/workspace/api/account/local-preferences')).json()")
        assert 'Modern UI setup' in saved['values']['atlas.strategy.setups.v1']
        assert '20000' in saved['values']['atlas.strategy.v1']
        await page.locator('#strategySignalTab').click()

        for theme, width in [('dark', 1512), ('light', 390)]:
            await page.locator('#workspaceTheme').select_option(theme)
            await page.set_viewport_size({'width': width, 'height': 1100})
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), theme
            await page.screenshot(path=str(ROOT / f'data/strategy-modern-{theme}.png'), full_page=True)
            for name in ('Signal', 'Risk', 'Saved'):
                await page.locator(f'#strategy{name}Tab').click()
                await expect(page.locator(f'#strategy{name}Config')).to_be_visible()
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (theme, name)

        # Hold worker startup so cancellation can be exercised deterministically.
        worker_requested, worker_release = asyncio.Event(), asyncio.Event()

        async def hold_worker(route):
            worker_requested.set()
            await worker_release.wait()
            try:
                await route.continue_()
            except Exception:
                pass  # Cancelling the worker can close its initial script request.

        await page.route('**/strategy-worker.js', hold_worker)
        await page.locator('#strategyRun').click()
        await asyncio.wait_for(worker_requested.wait(), timeout=10)
        await expect(page.locator('#strategyCancel')).to_be_visible()
        await page.locator('#strategyCancel').click()
        await expect(page.locator('#strategyCancel')).not_to_be_visible()
        await expect(page.locator('#strategyRun')).to_be_enabled()
        await expect(page.locator('#strategyResults')).not_to_be_visible()
        worker_release.set()
        await page.unroute('**/strategy-worker.js', hold_worker)

        assert not errors, errors
        await context.close()

        # A fresh browser profile restores the setup from the isolated disk backup.
        restored = await browser.new_context()
        fresh = await restored.new_page()
        await fresh.goto(url)
        await fresh.wait_for_function("typeof rows !== 'undefined' && rows.length > 40")
        await fresh.locator('#workspaceOpenStrategy').click()
        await fresh.locator('#strategySavedTab').click()
        await fresh.locator('#strategySavedSetup').select_option('Modern UI setup')
        await expect(fresh.locator('#st-capital')).to_have_value('20000')
        await expect(fresh.locator('#st-allocation')).to_have_value('60')
        await restored.close()
        await browser.close()
    print('PASS: strategy families, keyboard tabs, allocation controls, chart inspection, trade filtering/details, invalidation, themes and mobile layout')


async def run(include_legacy=True):
    prices = fixture_prices()

    async def backend(request):
        if request.path == '/api/symbols':
            return web.json_response([dict(symbol='TEST', name='Strategy test asset', asset_type='Stocks', close=prices[-1]['close'], change=1, currency='USD')])
        if request.path == '/api/history':
            return web.json_response(prices)
        if request.path.startswith('/api/'):
            return web.json_response({})
        path = ROOT / 'web' / (request.path.lstrip('/') or 'index.html')
        if path.parent != ROOT / 'web' or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app = web.Application()
    app.router.add_route('*', '/{path:.*}', backend)
    async with TestServer(app) as upstream:
        with tempfile.TemporaryDirectory() as directory:
            gateway = create_app(str(upstream.make_url('')).rstrip('/'), accounts_path=Path(directory) / 'accounts.sqlite')
            async with TestServer(gateway) as server:
                url = str(server.make_url('/workspace/'))
                if include_legacy:
                    await asyncio.to_thread(legacy_checks, url)
                await interactions(url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ui-only', action='store_true', help='Skip the existing engine and custom-rule regression suite.')
    asyncio.run(run(not parser.parse_args().ui_only))
