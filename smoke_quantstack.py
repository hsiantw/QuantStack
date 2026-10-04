"""Verify the unified chart app and retired research route with stored data."""
import argparse
from playwright.sync_api import sync_playwright, expect


def run(url='http://127.0.0.1:8501'):
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1600,'height':1000})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(url.rstrip('/')+'/?research=1&symbol=AAPL&period=1Y')
        assert '/workspace/' in page.url and 'research=1' not in page.url
        assert page.locator('iframe').count()==0
        page.wait_for_function("selected?.symbol=='AAPL' && rows.length>100")
        for key in ['pairs','options','liquidity','forecast','fundamentals']:
            page.locator(f'.workspace-right-rail [data-research-open={key}]').click()
            page.locator('#researchRun').click()
            expect(page.locator('#researchStatus')).to_contain_text('Complete',timeout=60000)
        page.locator('#workspaceOpenStrategy').click()
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyResults')).to_be_visible(timeout=30000)
        page.locator('#workspaceOpenMarkov').click()
        page.locator('#markovRun').click()
        expect(page.locator('#markovResults')).to_be_visible(timeout=30000)
        page.locator('.chart-home-menu summary').click()
        page.locator('#chartResearch').click()
        expect(page.locator('#researchSearch')).to_be_visible()
        assert page.request.get(url+'/_stcore/health').status==404
        assert not errors,errors
        page.screenshot(path='data/quantstack-native-integration.png')
        browser.close()
        print('PASS: native app, retired research route, real-data research, strategy, Markov and library.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8501')
    run(parser.parse_args().url)
