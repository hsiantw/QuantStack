"""Exercise appearance profiles and validate imported settings in the browser."""
import argparse
import json
from playwright.sync_api import sync_playwright, expect


def run(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('rows.length > 0')
        page.locator('#terminalSettings').click()
        page.locator('[data-appearance-theme="midnight"]').click()
        page.locator('#terminalAppearanceStyle').select_option('area')
        page.locator('#terminalAreaOpacity').fill('65')
        page.locator('#terminalGridDirection').select_option('vertical')
        page.locator('#terminalCustomColors summary').click()
        page.locator('#follow-wickUp').uncheck()
        page.locator('#appearance-wickUp').fill('#abcdef')
        page.locator('#appearanceProfileName').fill('Night <analysis>')
        page.locator('#appearanceProfileName').press('Enter')
        expect(page.locator('#appearanceProfileStatus')).to_contain_text('Saved')
        expect(page.locator('#terminalSettingsDialog')).to_be_visible()
        assert page.evaluate('chartType') == 'candles'
        page.locator('#terminalCancelSettings').click()
        page.reload()
        page.wait_for_function('rows.length > 0')
        assert page.evaluate('chartType') == 'candles'
        page.locator('#terminalSettings').click()
        page.locator('#appearanceProfileList').select_option('Night <analysis>')
        page.locator('#appearanceProfileLoad').click()
        expect(page.locator('#terminalAppearanceStyle')).to_have_value('area')
        expect(page.locator('#terminalAreaOpacity')).to_have_value('65')
        expect(page.locator('#terminalGridDirection')).to_have_value('vertical')
        with page.expect_download() as download:
            page.locator('#appearanceProfileExport').click()
        data = json.loads(open(download.value.path(), encoding='utf-8').read())
        assert data['theme'] == 'midnight'
        assert data['appearance']['wickUp'] == '#abcdef'
        assert data['appearance']['areaOpacity'] == 65
        assert 'comparisons' not in data and 'symbol' not in data
        page.locator('#terminalSettingsForm .terminal-primary').click()
        page.reload()
        page.wait_for_function('rows.length > 0')
        assert page.evaluate('chartType') == 'area'
        assert page.evaluate('chartAppearance.areaOpacity') == 65
        assert page.evaluate('chartAppearance.gridDirection') == 'vertical'
        assert page.evaluate('chartAppearance.wickUp') == '#abcdef'

        page.locator('#terminalSettings').click()
        data['theme'] = 'paper'
        data['appearance']['areaOpacity'] = 0
        page.locator('#appearanceProfileFile').set_input_files({'name': 'preset.json', 'mimeType': 'application/json', 'buffer': json.dumps(data).encode()})
        expect(page.locator('#appearanceProfileStatus')).to_contain_text('Imported into the preview')
        assert page.evaluate('window.atlasTheme.preference') == 'midnight'
        expect(page.locator('#terminalAreaOpacity')).to_have_value('0')
        page.locator('#terminalSettingsForm .terminal-primary').click()
        assert page.evaluate('window.atlasTheme.preference') == 'paper'
        assert page.evaluate('chartAppearance.areaOpacity') == 0

        page.locator('#terminalSettings').click()
        before = page.evaluate("localStorage.getItem('atlas.terminal')")
        for payload in [b'{invalid', json.dumps({**data, 'version': 99}).encode(), json.dumps({**data, 'appearance': {'areaOpacity': 999}}).encode(), b' ' * 65537]:
            page.locator('#appearanceProfileFile').set_input_files({'name': 'invalid.json', 'mimeType': 'application/json', 'buffer': payload})
            expect(page.locator('#appearanceProfileStatus')).not_to_contain_text('Imported')
            expect(page.locator('#terminalAreaOpacity')).to_have_value('0')
            assert page.evaluate("localStorage.getItem('atlas.terminal')") == before
        # Replacing a preset uses its name without creating duplicate entries.
        page.locator('#appearanceProfileName').fill('Night <analysis>')
        page.locator('#appearanceProfileSave').click()
        assert page.locator('#appearanceProfileList option').count() == 2
        page.locator('#appearanceProfileDelete').click()
        assert page.locator('#appearanceProfileList option').count() == 1
        page.set_viewport_size({'width': 390, 'height': 844})
        page.get_by_role('button', name='My presets', exact=True).click()
        expect(page.locator('#appearanceProfileName')).to_be_in_viewport()
        assert page.locator('#terminalSettingsDialog').evaluate('e=>e.scrollWidth <= e.clientWidth')
        page.screenshot(path='data/appearance-presets-mobile.png')
        page.locator('#terminalCancelSettings').click()
        assert not errors, errors
        browser.close()
        print('PASS: save/load/replace/delete, reload, export/import, draft isolation, invalid files and mobile presets.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8501/workspace/')
    run(parser.parse_args().url)
