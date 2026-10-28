"""Stress full-canvas archive dragging, delayed pages, and frame-coalesced paint."""
from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8501/workspace/?symbol=BTC-USD&interval=1m&period=MAX')
        page.wait_for_function("rows.length===10000 && interval==='1m'")
        held = []
        page.route('**/api/history?**before=**', lambda route: held.append(route), times=1)
        page.evaluate('viewStart=10; viewCount=5000; chartScale.offset=0; draw()')
        point = page.evaluate("({x:geometry().box.x+geometry().left+40,y:geometry().box.y+100,pw:geometry().pw})")
        page.mouse.move(point['x'], point['y'])
        page.mouse.down()
        page.mouse.move(point['x']+200, point['y'], steps=12)
        page.wait_for_timeout(100)
        assert held
        filled = """() => {
            const origin=viewStart-chartScale.offset,g=geometry();
            return origin>=0 && origin+viewCount<=rows.length && g.data.length===viewCount;
        }"""
        assert page.evaluate(filled), 'Delayed page exposed empty canvas'
        anchor = page.evaluate("({date:rows[100].date,x:geometry().x(100-viewStart),count:viewCount})")
        response = held[0].fetch()
        held[0].fulfill(response=response)
        page.wait_for_function("query().has('before') && document.getElementById('chart').getAttribute('aria-busy')==='false'")
        assert page.evaluate(filled), 'Broad viewport was truncated by the replacement page'
        x = page.evaluate('date => geometry().x(rows.findIndex(r=>r.date===date)-viewStart)', anchor['date'])
        assert abs(x-anchor['x']) < .01, (x, anchor)
        origin = page.evaluate('viewStart-chartScale.offset')
        page.mouse.move(point['x']+210, point['y'])
        page.mouse.up()
        actual = page.evaluate('viewStart-chartScale.offset')
        assert abs(actual-(origin-10/point['pw']*5000)) < .01, 'Drag jumped after loading'
        # Excessive zoom and edge drags stay filled, including the latest edge.
        page.evaluate('moveChartTime(1e9,1e6); draw()')
        page.wait_for_function("document.getElementById('chart').getAttribute('aria-busy')==='false'")
        assert page.evaluate(filled)
        assert page.evaluate('viewCount') == 5000
        page.evaluate('viewStart=4000;viewCount=300;chartScale.offset=0;draw()')
        # A burst of pointer events schedules one paint and does not reset canvas buffers.
        stats = page.evaluate("""async () => {
            await new Promise(requestAnimationFrame);
            const canvas=$('chart'), g=geometry(), original=draw;
            let paints=0, resets=0;
            const descriptors={};
            for(const name of ['width','height']) {
                const descriptor=Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype,name);
                descriptors[name]=descriptor;
                Object.defineProperty(canvas,name,{configurable:true,get(){return descriptor.get.call(this)},set(value){resets++;descriptor.set.call(this,value)}});
            }
            draw=function(){paints++;return original()};
            const x=g.box.x+g.left+50,y=g.box.y+100;
            // Use the real handlers without pointer capture for this synthetic burst.
            chartNavigationGesture={kind:'pan',id:77,x,y,g,start:viewStart,count:viewCount,wasAuto:true};
            for(let i=0;i<120;i++) canvas.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,pointerId:77,clientX:x+i/4,clientY:y}));
            chartNavigationGesture=null;
            await new Promise(requestAnimationFrame);
            draw=original;
            for(const name of ['width','height']) delete canvas[name];
            return {paints,resets};
        }""")
        assert stats['paints'] == 1 and stats['resets'] == 0, stats
        assert page.evaluate(filled)
        page.screenshot(path='data/btc-drag-filled.png')
        assert not errors, errors
        print('PASS: full canvas during delayed loading, 5000-bar page overlap, no post-load jump, bounded zoom,', stats)
        browser.close()


if __name__ == '__main__':
    run()
