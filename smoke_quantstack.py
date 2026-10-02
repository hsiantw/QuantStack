"""Verify the Streamlit page and its same-origin research workspace."""
import argparse
from pathlib import Path
import re
from playwright.sync_api import sync_playwright, expect


def run(url, standalone=False):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1200})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url.rstrip('/') + '/market_workspace')
        expect(page.get_by_role('heading', name='Market workspace', exact=True)).to_be_visible(timeout=30000)
        frame = page.frame_locator('iframe[srcdoc]' if standalone else 'iframe[src="/workspace/"]')
        expect(frame.locator('#symbol')).to_have_text('AAPL', timeout=30000)
        expect(frame.locator('#rangeSummary')).to_contain_text(re.compile(r'\d+ 1d bars'), timeout=30000)
        frame.locator('#workspaceOpenStrategy').click()
        frame.locator('#strategyRun').click()
        expect(frame.locator('#strategyResults')).to_be_visible(timeout=30000)
        frame.locator('#workspaceOpenMarkov').click()
        expect(frame.locator('#markovForm')).to_be_visible()
        frame.locator('#markovRun').click()
        expect(frame.locator('#markovResults')).to_be_visible(timeout=30000)
        if standalone:
            expect(frame.locator('[data-interval="1h"]')).to_be_disabled()
            frame.locator('#workspaceOpenData').click()
            with page.expect_download() as download:
                frame.locator('#download').click()
            assert download.value.suggested_filename == 'AAPL-prices.csv'
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
        Path('data').mkdir(exist_ok=True)
        page.screenshot(path='data/quantstack-integration.png', full_page=True)
        assert not errors, errors
        browser.close()
        print('PASS: Streamlit navigation, websocket session, embedded charts, strategy test and Markov panel.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--standalone', action='store_true')
    args = parser.parse_args()
    run(args.url, args.standalone)
