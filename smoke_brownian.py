"""Verify Brownian worker, exports, invalidation, themes, and static packaging."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def run(url, static=False):
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('rows.length > 20')
        page.locator('#workspaceOpenBrownian').click()
        page.locator('#workspaceMaximizeDock').click()
        expect(page.locator('#bm-drift')).to_be_disabled()
        page.locator('#brownianRun').click()
        expect(page.locator('#brownianResults')).to_be_visible(timeout=30000)
        expect(page.locator('#brownianStatus')).to_contain_text('Completed 2,000 paths')
        expect(page.locator('#brownianFan svg')).to_be_visible()
        page.locator('#brownianShowPaths').check()
        expect(page.locator('.brownian-sample')).to_have_count(12)
        page.locator('#brownianHorizon').fill('10')
        expect(page.locator('#brownianReadout tbody td').first).to_have_text('10')
        with page.expect_download() as download:
            page.locator('#brownianExport').click()
        with open(download.value.path(),encoding='utf-8') as source:
            result=json.load(source)
        assert result['model']=='Geometric Brownian motion'
        assert result['context']['symbol']=='AAPL'
        assert len(result['fan'])==61
        assert result['fan'][0]['median']==result['data']['price']
        with page.expect_download() as download:
            page.locator('#brownianCSV').click()
        assert Path(download.value.path()).read_text().startswith('bar,p05,p25,median,p75,p95,mean,lossProbability')

        page.locator('#bm-mode').select_option('manual')
        expect(page.locator('#brownianResults')).not_to_be_visible()
        expect(page.locator('#brownianExport')).to_be_disabled()
        page.locator('#bm-drift').fill('0')
        page.locator('#bm-volatility').fill('0')
        page.locator('#brownianRun').click()
        expect(page.locator('#brownianResults')).to_be_visible()
        expect(page.locator('#brownianMetrics')).to_contain_text('0.00%')
        page.locator('#workspaceTheme').select_option('midnight')
        page.screenshot(path=f'data/brownian-{static}-desktop.png')
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        expect(page.locator('#brownianRun')).to_be_visible()
        page.screenshot(path=f'data/brownian-{static}-mobile.png')
        page.set_viewport_size({'width':1440,'height':1000})
        page.locator('#workspaceMaximizeDock').click()
        page.locator('[data-period="6M"]').click()
        expect(page.locator('#brownianResults')).not_to_be_visible()
        page.wait_for_function('rows.length > 20')
        page.evaluate("() => {document.getElementById('brownianForm').requestSubmit();document.getElementById('brownianCancel').click();}")
        expect(page.locator('#brownianStatus')).to_have_text('Simulation canceled.')
        page.evaluate("() => {window.brownianRows=rows;rows=rows.slice(0,10);document.getElementById('brownianForm').requestSubmit();}")
        expect(page.locator('#brownianError')).to_contain_text('21 price bars')
        page.evaluate('rows=window.brownianRows')
        page.locator('#brownianRun').click()
        expect(page.locator('#brownianResults')).to_be_visible()
        page.reload()
        page.wait_for_function('rows.length > 20')
        page.locator('#workspaceOpenBrownian').click()
        expect(page.locator('#bm-mode')).to_have_value('manual')
        expect(page.locator('#brownianResults')).not_to_be_visible()
        assert not errors,errors
        browser.close()
        print(f'PASS: Brownian {"static" if static else "local"} worker, exports, controls, cancellation, stale results, recovery and mobile.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8502/workspace/')
    parser.add_argument('--static',action='store_true')
    args=parser.parse_args()
    if args.static:
        from smoke_markov import static_fixture
        with static_fixture() as url:run(url,True)
    else:run(args.url)
