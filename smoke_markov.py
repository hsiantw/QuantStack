"""Exercise Markov workers, exports, stale results, keyboard controls and mobile UI.

Use --static to build and serve an isolated snapshot fixture under a URL prefix.
"""
import argparse
import csv
import io
import json
import math
import shutil
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from datetime import date, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from playwright.sync_api import sync_playwright, expect


@contextmanager
def static_fixture():
    import build_site
    import dashboard
    from market_data import connect, persist
    with tempfile.TemporaryDirectory(prefix='atlas-markov-') as directory:
        root = Path(directory)
        (root / 'data').mkdir()
        shutil.copytree('web', root / 'web')
        db_path = root / 'data' / 'market.sqlite'
        db = connect(db_path)
        bars = []
        price = 100
        for i in range(501):
            price *= math.exp(math.sin(i * .7) * .025 + math.cos(i * .13) * .012)
            day = (date.today() - timedelta(days=501-i)).isoformat()
            bars.append(('AAPL', day, price, price*1.01, price*.99, price, price,
                         1000000, 0, 0, 'USD', 'NMS', 'America/New_York', 'fixture'))
        persist(db, 'AAPL', bars, True, 'fixture', 0)
        db.close()
        output = root / 'nested' / 'site'
        output.parent.mkdir()
        with patch.object(dashboard, 'DATABASE', db_path), patch.object(build_site, 'DATABASE', db_path), \
             patch.object(build_site, 'ROOT', root), patch.object(build_site, 'OUTPUT', output):
            archive = build_site.build()
        with zipfile.ZipFile(archive) as bundle:
            for name in ('markov-engine.js', 'markov-worker.js', 'markov.js', 'markov.css'):
                assert name in bundle.namelist(), name
        html = (output / 'index.html').read_text(encoding='utf-8')
        assert 'src="./markov.js"' in html and 'href="./markov.css"' in html
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/nested/site/'
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def run(url, static=False):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1440, 'height':1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("typeof rows !== 'undefined' && rows.length >= 61")
        page.locator('#workspaceOpenMarkov').click()
        page.locator('#workspaceMaximizeDock').click()
        expect(page.locator('#workspaceMarkovPanel')).to_be_visible()
        expect(page.locator('#workspaceStrategyPanel')).not_to_be_visible()
        page.locator('#mk-paths').fill('500')
        page.locator('#markovRun').click()
        expect(page.locator('#markovResults')).to_be_visible(timeout=20000)
        expect(page.locator('#markovStatus')).to_contain_text('Analysis complete')
        expect(page.locator('#markovError')).not_to_be_visible()
        expect(page.locator('#markovOverview tbody tr')).to_have_count(3)
        page.locator('#markovTabTransitions').click()
        page.locator('#markovOrigin').select_option('0')
        expect(page.locator('#markovRowDetails')).to_contain_text('Wilson')
        page.locator('[data-markov-origin="2"]').click()
        expect(page.locator('#markovOrigin')).to_have_value('2')
        page.locator('#markovTabTransitions').focus()
        page.keyboard.press('ArrowRight')
        expect(page.locator('#markovForecasts')).to_be_visible()
        expect(page.locator('#markovFan svg')).to_be_visible()
        page.locator('#markovInspectHorizon').fill('5')
        expect(page.locator('#markovFanReadout')).to_contain_text('5 bars')
        page.screenshot(path=f'data/markov-{"static" if static else "local"}-forecast.png')
        page.locator('#markovTabValidation').click()
        expect(page.locator('#markovValidation')).to_contain_text('Historical frequencies')
        expect(page.locator('#markovValidation')).to_contain_text('Confusion matrix')
        with page.expect_download() as downloaded:
            page.locator('#markovExport').click()
        report = json.loads(Path(downloaded.value.path()).read_text(encoding='utf-8'))
        assert report['context']['symbol'] and report['context']['interval'] == '1d'
        assert len(report['validation']['predictions']) == report['validation']['count']
        assert len(report['history']) == report['data']['returns']
        assert report['parameters']['paths'] == 500
        with page.expect_download() as downloaded:
            page.locator('#markovCSV').click()
        records = list(csv.DictReader(io.StringIO(Path(downloaded.value.path()).read_text(encoding='utf-8-sig'))))
        assert len(records) == 9
        for i in range(3):
            assert abs(sum(float(row['model_probability']) for row in records[i*3:(i+1)*3])-1) < 1e-12

        page.locator('#mk-states').select_option('5')
        expect(page.locator('#markovResults')).not_to_be_visible()
        expect(page.locator('#markovExport')).to_be_disabled()
        page.locator('#markovRun').click()
        expect(page.locator('#markovResults')).to_be_visible()
        page.locator('#markovTabOverview').click()
        expect(page.locator('#markovOverview tbody tr')).to_have_count(5)
        page.locator('#mk-mode').select_option('fixed')
        expect(page.locator('#mk-states')).not_to_be_visible()
        expect(page.locator('#mk-threshold')).to_be_visible()
        page.locator('#mk-validation').select_option('frozen')
        page.locator('#markovRun').click()
        expect(page.locator('#markovResults')).to_be_visible()
        page.locator('#markovTabValidation').click()
        expect(page.locator('#markovValidation')).to_contain_text('remain frozen')
        page.reload()
        page.wait_for_function('rows.length >= 61')
        page.locator('#workspaceOpenMarkov').click()
        page.locator('#workspaceMaximizeDock').click()
        expect(page.locator('#mk-mode')).to_have_value('fixed')
        expect(page.locator('#mk-validation')).to_have_value('frozen')
        expect(page.locator('#markovResults')).not_to_be_visible()

        page.locator('#mk-mode').select_option('quantile')
        page.locator('#mk-states').select_option('3')
        page.locator('#markovRun').click()
        expect(page.locator('#markovResults')).to_be_visible()
        page.locator('#workspaceTheme').select_option('dark')
        page.locator('#markovTabTransitions').click()
        page.screenshot(path=f'data/markov-{"static" if static else "local"}-dark.png')
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile document overflow'
        expect(page.locator('#markovRun')).to_be_visible()
        page.locator('#markovTabForecasts').click()
        page.screenshot(path=f'data/markov-{"static" if static else "local"}-mobile.png')
        page.set_viewport_size({'width':1440,'height':1000})
        page.locator('#workspaceTheme').select_option('light')

        # Change the chart range to invalidate all result/export state.
        page.locator('#workspaceMaximizeDock').click()
        page.locator('[data-period="6M"]').click()
        expect(page.locator('#markovResults')).not_to_be_visible()
        expect(page.locator('#markovExport')).to_be_disabled()
        page.wait_for_function('rows.length >= 61')
        # Start and immediately invalidate synchronously to exercise cancellation races.
        page.evaluate("""() => {document.getElementById('markovForm').requestSubmit();document.getElementById('markovCancel').click();}""")
        expect(page.locator('#markovStatus')).to_have_text('Analysis canceled.')
        expect(page.locator('#markovRun')).to_be_enabled()
        expect(page.locator('#markovResults')).not_to_be_visible()

        # Worker errors should be readable and recoverable.
        page.evaluate("""() => {window.markovSavedRows=rows;rows=rows.slice(0,10);document.getElementById('markovForm').requestSubmit();}""")
        expect(page.locator('#markovError')).to_contain_text('61 price bars')
        page.evaluate('rows=window.markovSavedRows')
        page.locator('#markovRun').click()
        expect(page.locator('#markovResults')).to_be_visible()
        assert not errors, errors
        browser.close()
        print(f'Markov {"static" if static else "local"} checks passed: worker, 3/5/fixed states, validation, exports, persistence, keyboard, themes, mobile, cancellation and recovery.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--static', action='store_true')
    args = parser.parse_args()
    if args.static:
        with static_fixture() as fixture_url:
            run(fixture_url, static=True)
    else:
        run(args.url)
