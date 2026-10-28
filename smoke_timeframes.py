"""Verify timeframe favorites, selection, persistence, and archive paging."""
from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':900})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8501/workspace/?symbol=BTC-USD&interval=1m&period=MAX')
        page.wait_for_function("rows.length===10000 && interval==='1m'")
        page.locator('#timeframePicker').click()
        assert page.locator('[data-timeframe]').count()==11
        page.locator('[data-timeframe-star="4h"]').click()
        page.locator('[data-timeframe-star="3m"]').click()
        page.locator('[data-timeframe-star="5m"]').click()
        page.screenshot(path='data/timeframes-picker.png')
        page.locator('[data-timeframe="30m"]').click()
        page.wait_for_function("interval==='30m' && rows.length===1000 && document.getElementById('chart').getAttribute('aria-busy')==='false'")
        assert page.evaluate("rows.every(r=>Date.parse(r.date)%1800000===0)")
        first=page.evaluate('rows[0].date')
        page.evaluate('moveChartTime(100,300);draw()')
        page.wait_for_function('first=>rows[0].date<first',arg=first)
        page.wait_for_function("document.getElementById('chart').getAttribute('aria-busy')==='false'")
        assert page.evaluate('rows.length')==1000
        assert page.evaluate('viewStart-chartScale.offset>=0 && viewStart-chartScale.offset+viewCount<=rows.length')
        page.reload()
        page.wait_for_function("interval==='30m' && rows.length===1000")
        assert page.locator('#intervals [data-interval="4h"]').count()==1
        assert page.locator('#intervals [data-interval="3m"]').count()==1
        assert page.locator('#intervals [data-interval="5m"]').count()==0
        for interval in ['1m','3m','5m','15m','30m','1h','2h','4h','1d','1w','1mo']:
            page.locator('#timeframePicker').click()
            page.locator(f'[data-timeframe="{interval}"]').click()
            page.wait_for_function("id=>interval===id && rows.length>0 && document.getElementById('chart').getAttribute('aria-busy')==='false'",arg=interval)
        page.set_viewport_size({'width':600,'height':800})
        page.locator('#timeframePicker').click()
        box=page.locator('#timeframeDialog').bounding_box()
        assert box['x']>=0 and box['x']+box['width']<=600,box
        page.keyboard.press('Escape')
        assert not page.locator('#timeframeDialog').is_visible()
        assert not errors,errors
        print('PASS: all 11 intervals, 30m archive paging, favorites persisted after reload, compact layout, keyboard dismissal')
        browser.close()


if __name__=='__main__':
    run()
