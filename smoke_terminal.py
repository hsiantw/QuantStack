"""Exercise analysis tools in an isolated browser with the local dataset."""
import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def run(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('rows.length > 0')
        for style in ['bars', 'area', 'line', 'candles']:
            page.locator('#terminalStyle').select_option(style)
            assert page.evaluate('chartType') == style
        page.locator('#terminalSettings').click()
        page.locator('#terminalGrid').uncheck()
        page.locator('#terminalUp').fill('#1267bb')
        page.locator('#terminalSettingsForm button[type=submit], #terminalSettingsForm .terminal-primary').click()
        page.reload()
        page.wait_for_function('rows.length > 0')
        assert page.evaluate('chartAppearance.grid') is False
        assert page.evaluate('chartAppearance.up') == '#1267bb'
        page.locator('#terminalSettings').click()
        page.locator('#terminalDefaults').click()
        page.locator('#terminalSettingsForm .terminal-primary').click()
        assert page.evaluate('chartAppearance.grid') is True

        page.locator('#terminalCompare').click()
        page.locator('#terminalCompareSearch').fill('MSFT')
        page.locator('[data-compare-symbol="MSFT"]').click()
        page.locator('[data-close="terminalCompareDialog"]').click()
        expect(page.locator('#comparisonBaseline')).to_contain_text('0% at')
        expect(page.locator('#comparisonLegend')).not_to_contain_text('Loading')
        page.mouse.move(10, 10)
        # Verify plotted return labels against independently fetched closing data.
        expected = page.evaluate('''async () => {
            const q = query(); q.set('symbol', 'MSFT');
            const data = await api('/api/history?' + q), lookup = new Map(data.map(r => [r.date, r.close]));
            const g = geometry(), first = g.data.find((r, i) => g.x(i) >= g.left && lookup.has(r.date));
            const last = g.data.at(-1);
            return pct((lookup.get(last.date) / lookup.get(first.date) - 1) * 100);
        }''')
        expect(page.locator('#comparisonLegend')).to_contain_text(expected)
        page.locator('#terminalGo').click()
        target = page.evaluate('rows[Math.floor(rows.length / 3)].date.slice(0,10)')
        page.locator('#terminalGoDate').fill(target)
        page.locator('#terminalGoForm .terminal-primary').click()
        assert page.evaluate('(date) => visible().some(r => r.date.startsWith(date))', target)
        assert page.evaluate('viewCount') <= 100
        expect(page.locator('#terminalNotice')).to_contain_text('Showing')
        with page.expect_download() as pending:
            page.locator('#terminalSnapshot').click()
        destination = Path('data/terminal-snapshot.png')
        pending.value.save_as(destination)
        assert destination.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert destination.stat().st_size > 10000
        page.screenshot(path='data/terminal-desktop.png')
        page.reload()
        page.wait_for_function('rows.length > 0')
        expect(page.locator('#comparisonBaseline')).to_contain_text('0% at')
        # Sparse data must remain gaps, with no invented matching baseline.
        page.route('**/api/history?**', lambda route: route.fulfill(json=[]) if 'symbol=MSFT' in route.request.url else route.continue_())
        page.locator('[data-period="6M"]').click()
        expect(page.locator('#comparisonLegend')).to_contain_text('No data for this range')
        page.unroute('**/api/history?**')
        page.locator('[data-period="1Y"]').click()
        expect(page.locator('#comparisonBaseline')).to_contain_text('0% at')
        for width, height in [(1024, 768), (768, 900), (390, 844)]:
            page.set_viewport_size({'width': width, 'height': height})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
            assert page.locator('#chart').bounding_box()['height'] >= 100
            page.locator('#terminalSettings').click()
            expect(page.locator('#terminalSettingsForm .terminal-primary')).to_be_visible()
            page.locator('[data-close="terminalSettingsDialog"]').click()
            page.screenshot(path=f'data/terminal-{width}.png')
        page.locator('[data-remove-comparison="MSFT"]').first.click()
        expect(page.locator('#comparisonPane')).not_to_be_visible()
        assert not errors, errors
        browser.close()
        print('Terminal checks passed: chart styles, preferences, comparison math and gaps, navigation, PNG export, persistence, responsive layouts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    run(parser.parse_args().url)
