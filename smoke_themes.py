"""Validate theme rendering, draft settings, persistence and mobile controls."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def pixel(page, selector='#chart'):
    return page.locator(selector).evaluate("c => Array.from(c.getContext('2d').getImageData(0,0,1,1).data).slice(0,3)")


def rgb(color):
    return [int(color[i:i+2], 16) for i in (1, 3, 5)]


def run(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 900}, color_scheme='light')
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('rows.length > 0')
        for name, color in [('midnight', '#0b1020'), ('forest', '#101d1b'), ('paper', '#faf6ed'), ('dark', '#131722'), ('light', '#ffffff')]:
            page.locator('#workspaceTheme').select_option(name)
            page.wait_for_function('(name) => document.documentElement.dataset.palette === name', arg=name)
            page.evaluate('draw()')
            assert pixel(page) == rgb(color), name
            assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--surface').trim()") == color

        page.locator('#terminalSettings').click()
        saved = page.evaluate("localStorage.getItem('atlas.terminal')")
        page.locator('[data-appearance-theme="midnight"]').click()
        assert pixel(page, '#terminalAppearancePreview') == rgb('#0b1020')
        assert pixel(page) == rgb('#ffffff'), 'Draft must not recolor the chart'
        page.locator('#terminalCancelSettings').click()
        assert page.evaluate("localStorage.getItem('atlas.terminal')") == saved

        page.locator('#terminalSettings').click()
        page.locator('[data-appearance-theme="midnight"]').click()
        page.locator('#terminalWicks').uncheck()
        page.locator('#terminalBorders').check()
        page.locator('#terminalWatermark').uncheck()
        page.locator('#terminalLineWidth').select_option('4')
        page.locator('#terminalFontSize').select_option('14')
        page.locator('#terminalGridStyle').select_option('solid')
        page.locator('#terminalCustomColors summary').click()
        for key, color in [('background', '#172133'), ('textColor', '#e5edf8'), ('lineColor', '#f0b56b')]:
            page.locator('#follow-'+key).uncheck()
            page.locator('#appearance-'+key).fill(color)
        Path('data').mkdir(exist_ok=True)
        page.screenshot(path='data/theme-settings-desktop.png')
        page.locator('#terminalSettingsForm .terminal-primary').click()
        assert pixel(page) == rgb('#172133')
        page.reload()
        page.wait_for_function('rows.length > 0')
        assert pixel(page) == rgb('#172133')
        assert page.evaluate('chartAppearance.lineWidth') == 4
        assert page.evaluate('chartAppearance.fontSize') == 14
        assert page.evaluate('chartAppearance.wicks') is False
        assert page.evaluate('chartAppearance.borders') is True
        assert page.evaluate('chartAppearance.watermark') is False
        assert page.evaluate('chartAppearance.gridStyle') == 'solid'
        expect(page.locator('#workspaceTheme')).to_have_value('midnight')
        page.locator('#workspaceTheme').select_option('forest')
        page.evaluate('draw()')
        assert pixel(page) == rgb('#172133'), 'Explicit colors survive theme changes'
        page.locator('#terminalCompare').click()
        page.locator('#terminalCompareSearch').fill('MSFT')
        page.locator('[data-compare-symbol="MSFT"]').click()
        page.locator('[data-close="terminalCompareDialog"]').click()
        expect(page.locator('#comparisonBaseline')).to_contain_text('0% at')
        assert pixel(page, '#comparisonCanvas') == rgb('#172133')
        page.locator('#terminalStyle').select_option('area')
        with page.expect_download() as download:
            page.locator('#terminalSnapshot').click()
        download.value.save_as('data/theme-snapshot.png')
        from PIL import Image
        assert list(Image.open('data/theme-snapshot.png').convert('RGB').getpixel((0, 0))) == rgb('#172133')
        page.screenshot(path='data/theme-chart-desktop.png')

        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#terminalSettings').click()
        page.screenshot(path='data/theme-settings-mobile.png')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert page.locator('#terminalSettingsDialog').evaluate('e=>e.scrollWidth <= e.clientWidth')
        page.locator('#terminalDefaults').click()
        page.locator('#terminalSettingsForm .terminal-primary').click()
        page.evaluate('draw()')
        assert pixel(page) == rgb('#ffffff')
        assert page.evaluate('chartAppearance.lineWidth') == 2
        page.emulate_media(color_scheme='dark')
        page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
        page.evaluate('draw()')
        assert pixel(page) == rgb('#131722')
        assert not errors, errors
        browser.close()
        print('PASS: five themes, draft/cancel, custom colors, persistence, PNG colors, mobile settings and system theme.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8501/workspace/')
    run(parser.parse_args().url)
