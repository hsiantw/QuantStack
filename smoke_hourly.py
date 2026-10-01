"""Verify native hourly chart controls and expanded symbols in the local dashboard."""
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width':1440,'height':900})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8765')
    page.wait_for_function("typeof rows !== 'undefined' && rows.length > 0")
    assert page.locator('[data-interval]').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.interval)') == ['1d','1h']
    page.locator('[data-interval="1h"]').click()
    page.wait_for_function("interval === '1h' && rows.length > 0 && !document.getElementById('refresh').disabled")
    expect(page.locator('#rangeSummary')).to_contain_text('1h bars')
    page.evaluate("select('AARD')")
    page.wait_for_function("selected.symbol === 'AARD' && interval === '1h' && rows.length > 500")
    assert page.evaluate('selected.has_daily') is False
    assert page.evaluate('selected.quote_interval') == '1h'
    assert page.evaluate("rows.every(row=>row.source==='Yahoo Finance')")
    page.screenshot(path='data/hourly-chart-verified.png')
    page.evaluate("select('BTC-USD')")
    page.wait_for_function("selected.symbol === 'BTC-USD' && rows.length > 500")
    # The native BTC candles are stored directly, without minute rollups.
    assert page.evaluate("rows.every(row=>row.source==='Yahoo Finance')")
    assert not errors, errors
    print('Hourly chart smoke passed: 1D/1H controls, hourly-only symbol selection, native stock and BTC candles, no JavaScript errors.')
    browser.close()
