"""Check landing-chart persistence and sharing in a clean browser profile."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def run(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url + '?symbol=MSFT&interval=1d&period=6M')
        page.wait_for_function("selected?.symbol === 'MSFT' && rows.length > 0")
        assert page.locator('#assets .asset').count() <= 100
        if page.locator('#watchlistMore').count():
            page.locator('#watchlistMore').click()
            assert page.locator('#assets .asset').count() == 200
        page.locator('#search').fill('MSFT')
        expect(page.locator('#assets [data-symbol="MSFT"]')).to_be_visible()
        page.locator('#search').fill('')
        assert page.locator('#assets .asset').count() <= 100
        expect(page.locator('[data-period="6M"]')).to_have_class('active')
        page.locator('#refresh').click()
        expect(page.locator('#refresh')).to_be_enabled()
        expect(page.locator('#symbol')).to_have_text('MSFT')
        page.goto(url)
        page.wait_for_function("selected?.symbol === 'MSFT' && rows.length > 0")
        assert page.evaluate('period') == '6M'
        page.locator('#terminalSettings').click()
        page.locator('#chartShare').click()
        expect(page.locator('#chartHomeDialog')).to_be_visible()
        link = page.locator('#chartShareUrl').input_value()
        assert 'symbol=MSFT' in link and 'period=6M' in link
        page.locator('#chartHomeClose').click()

        # Custom ranges survive a reload; links override browser preferences.
        page.goto(url + '?symbol=AAPL&interval=1d&period=CUSTOM&start=2025-01-02&end=2025-03-31')
        page.wait_for_function("selected?.symbol === 'AAPL' && rows.length > 0")
        assert page.evaluate('period') == 'CUSTOM'
        expect(page.locator('#start')).to_have_value('2025-01-02')
        page.goto(url)
        page.wait_for_function('rows.length > 0')
        expect(page.locator('#end')).to_have_value('2025-03-31')

        page.goto(url + '?symbol=AAPL&interval=1d&period=CUSTOM&start=&end=2025-03-31')
        page.wait_for_function('rows.length > 0')
        assert page.evaluate('period') == 'CUSTOM'
        expect(page.locator('#start')).to_have_value('')

        page.goto(url + '?symbol=AAPL&period=CUSTOM&start=broken&end=2025-03-31')
        page.wait_for_function('rows.length > 0')
        assert page.evaluate('period') == '1Y'
        page.locator('#terminalSettings').click()
        page.locator('#chartShortcuts').click()
        expect(page.locator('#chartHomeTitle')).to_have_text('Keyboard shortcuts')
        page.keyboard.press('Escape')
        expect(page.locator('#chartHomeDialog')).not_to_be_visible()
        Path('data').mkdir(exist_ok=True)
        page.screenshot(path='data/chart-home-desktop.png')
        page.locator('#terminalCancelSettings').click()
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#terminalSettings').click()
        page.locator('#chartShare').click()
        expect(page.locator('#chartShareUrl')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='data/chart-home-mobile.png')
        assert not errors, errors
        browser.close()
        print('PASS: chart links, refresh, session and custom range restoration, shortcuts, mobile sharing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765/')
    run(parser.parse_args().url)
