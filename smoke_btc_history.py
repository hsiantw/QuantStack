"""Exercise BTC archive paging in a fresh browser against the local chart."""
from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8501/workspace/?symbol=BTC-USD&interval=1m&period=MAX')
        page.wait_for_function("rows.length === 10000 && interval === '1m'")
        latest = page.evaluate('rows.at(-1).date')
        first = page.evaluate('rows[0].date')
        # Prefetch before the edge and keep the same candles under the pointer.
        page.evaluate('viewStart=1500; viewCount=300; chartScale.offset=0; draw()')
        anchor = page.evaluate("({date:rows[1530].date,x:geometry().x(30),count:viewCount})")
        point = page.evaluate("({x:geometry().box.x+geometry().left+20,y:geometry().box.y+100})")
        page.mouse.move(point['x'], point['y'])
        page.mouse.down()
        page.mouse.move(point['x'] + 80, point['y'], steps=5)
        page.mouse.up()
        page.wait_for_function('first => rows[0].date < first', arg=first)
        page.wait_for_function("document.getElementById('chart').getAttribute('aria-busy')==='false'")
        state = page.evaluate("date => ({count:viewCount,bars:rows.length,x:geometry().x(rows.findIndex(r=>r.date===date)-viewStart),first:rows[0].date,offset:chartScale.offset})", anchor['date'])
        assert state['bars'] == 10000 and state['count'] == anchor['count'], state
        assert abs(state['x'] - anchor['x'] - 80) < 2, state
        assert state['offset'] < 1, state  # Earlier candles now fill the left side.
        print('Older page and viewport continuity:', state)
        page.screenshot(path='data/btc-history-fixed.png')
        # Sliding forward returns the original newest page without retaining the archive in RAM.
        page.evaluate('moveChartTime(rows.length-400,300); draw()')
        page.wait_for_function('latest => rows.at(-1).date === latest', arg=latest)
        page.wait_for_function("document.getElementById('chart').getAttribute('aria-busy')==='false'")
        assert page.evaluate('rows.length') == 10000
        # A failed page is recoverable on the next pan.
        page.route('**/api/history?**before=**', lambda route: route.abort(), times=1)
        page.evaluate('moveChartTime(0,300); draw()')
        page.wait_for_function("document.getElementById('chart').getAttribute('aria-busy')==='false'")
        assert page.evaluate('rows[0].date') == first
        page.evaluate('moveChartTime(0,300); draw()')
        page.wait_for_function('first => rows[0].date < first', arg=first)
        page.wait_for_function("document.getElementById('chart').getAttribute('aria-busy')==='false'")
        # A new range must discard the previous page cursor.
        page.locator('[data-period="1D"]').click()
        page.wait_for_function("period==='1D' && rows.length>0 && !query().has('before') && !query().has('after')")
        # A late archive response must not replace a newly selected date range.
        page.locator('[data-period="MAX"]').click()
        page.wait_for_function("period==='MAX' && rows.length===10000")
        held = []
        page.route('**/api/history?**before=**', lambda route: held.append(route), times=1)
        page.evaluate('moveChartTime(0,300); draw()')
        page.wait_for_timeout(150)
        assert held
        response = held[0].fetch()
        page.locator('[data-period="1D"]').click()
        page.wait_for_function("period==='1D' && rows.length>0 && !query().has('before')")
        expected = page.evaluate('[generation,rows[0].date,rows.at(-1).date,rows.length]')
        held[0].fulfill(response=response)
        page.wait_for_timeout(150)
        assert page.evaluate('[generation,rows[0].date,rows.at(-1).date,rows.length]') == expected
        assert not errors, errors
        print('Forward paging, retry, range reset, stale responses and browser errors: OK')
        browser.close()


if __name__ == '__main__':
    run()
