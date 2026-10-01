"""Check custom strategy rules, signal timing, saved setups and responsive controls."""
import argparse
import csv
import io
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def run(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("typeof rows !== 'undefined' && rows.length > 40")
        print(page.evaluate('''() => {
            let count = 0;
            const assert = (ok, label) => { count++; if (!ok) throw Error(label); };
            const makeBars = values => values.map((v,i) => ({date:new Date(Date.UTC(2025,0,i+1)).toISOString().slice(0,10),open:v,high:v+1,low:v-1,close:v,adjusted_close:v}));
            const bars = makeBars([10,10,11,12,9,8,11,10]);
            const condition = (op, value) => ({left:{type:'close'},operator:op,right:{type:'constant',value}});
            const groups = (entry, exit, entryMode='all', exitMode='any') => ({entry:{mode:entryMode,rules:entry},exit:{mode:exitMode,rules:exit}});
            const base = {kind:'custom',capital:1000,commission:0,slippage:0,adjusted:false};
            const test = (conditions, data=bars) => AtlasBacktest.run(data,{...base,conditions});
            const entry = condition('crossUp',10), exit = condition('crossDown',10);
            const crossed = test(groups([entry],[exit]));
            assert(crossed.trades[0].entry===bars[3].date && crossed.trades[0].entryPrice===12,'Cross above equality fills next open');
            assert(crossed.trades[0].exit===bars[5].date && crossed.trades[0].exitPrice===8,'Cross below fills next open');
            assert(crossed.trades[0].reason==='Signal','Custom exit reason');
            assert(test(groups([entry,condition('gt',100)],[exit])).trades.length===0,'Entry AND requires every condition');
            assert(test(groups([entry,condition('gt',100)],[exit],'any')).trades.length===2,'Entry OR accepts one condition');
            assert(test(groups([entry],[exit,condition('gt',100)],'all','all')).trades[0].reason==='End of test','Exit AND independently combines rules');
            assert(test(groups([entry],[exit,condition('gt',100)])).trades[0].exit===bars[5].date,'Exit OR independently combines rules');
            const trend=makeBars([11,12,13,14,15,16,17,18]);
            assert(test(groups([entry],[exit]),trend).trades.length===0,'Crossing must not treat already-above as a fresh cross');
            assert(test(groups([condition('gt',10)],[exit]),trend).trades.length===1,'Above conditions can enter an existing trend');
            assert(test(groups([condition('gte',10)],[condition('lt',0)])).trades[0].entry===bars[1].date,'At or above includes equality');
            assert(test(groups([condition('lte',10)],[condition('lt',0)])).trades[0].entry===bars[1].date,'At or below includes equality');
            const ma={left:{type:'sma',period:2},operator:'gt',right:{type:'ema',period:3}};
            assert(test(groups([ma],[condition('lt',0)]),trend).warmup===3,'Independent indicator periods control warmup');
            const longExit={left:{type:'close'},operator:'lt',right:{type:'sma',period:5}};
            assert(test(groups([condition('gt',0)],[longExit]),trend).trades[0].entry===trend[5].date,'Exit indicator warmup also completes before trading');
            const crossingMa={...ma,operator:'crossUp'};
            assert(test(groups([crossingMa],[condition('lt',0)]),trend).warmup===4,'Crossings need an extra previous value');
            const rejects = conditions => {try {test(conditions);return false;}catch{return true;}};
            assert(rejects(groups([],[exit])),'Reject empty entry rules');
            assert(rejects(groups([entry],[])),'Reject empty exit rules');
            assert(rejects(groups([entry],[exit],'invalid')),'Reject invalid combination');
            assert(rejects(groups([{...entry,operator:'eval'}],[exit])),'Reject unknown operators');
            assert(rejects(groups([{...entry,left:{type:'unknown'}}],[exit])),'Reject unknown indicators');
            assert(rejects(groups([condition('gt',null)],[exit])),'Reject blank thresholds');
            assert(rejects(groups([{...entry,left:{type:'sma',period:1.5}}],[exit])),'Reject fractional periods');
            assert(rejects(groups(Array(13).fill(entry),[exit])),'Enforce rule count limit');
            const calibration=makeBars([10,11,12,13,14,15]);
            for(const [type,period,expected] of [['stochastic',3,75],['williams',3,-25],['cci',3,100],['roc',2,20],['channelMid',2,10.5]]) {
                const lower={left:{type,period},operator:'gt',right:{type:'constant',value:expected-.01}};
                const upper={left:{type,period},operator:'lt',right:{type:'constant',value:expected+.01}};
                const calculated=test(groups([lower,upper],[condition('lt',0)]),calibration);
                assert(calculated.trades[0]?.entry===calibration[3].date,type+' agrees with hand-calculated first eligible value');
            }
            const flat=calibration.map(b=>({...b,open:10,high:10,low:10,close:10,adjusted_close:10}));
            for(const type of ['stochastic','williams']) {
                const check={left:{type,period:3},operator:'gt',right:{type:'constant',value:-1000}};
                assert(test(groups([check],[condition('lt',0)]),flat).trades.length===0,type+' undefined zero range must not pass a condition');
            }
            const flatCci={left:{type:'cci',period:3},operator:'gte',right:{type:'constant',value:0}};
            assert(test(groups([flatCci],[condition('lt',0)]),flat).trades.length===1,'Flat CCI uses neutral zero');
            const fixture=makeBars(Array.from({length:480},(_,i)=>100+Math.sin(i/35)*25+Math.sin(i/4)*12+i*.03-(i%31===0?25:0)));
            assert(Object.keys(AtlasBacktest.catalog).length===20,'Twenty built-in strategy presets');
            for(const kind of Object.keys(AtlasBacktest.catalog)) {
                const result=AtlasBacktest.run(fixture,{...base,kind,fast:3,slow:7,lookback:14});
                assert(Number.isFinite(result.finalEquity),kind+' finite accounting');
                assert(result.trades.length>0,kind+' produces fixture trades');
                const custom=AtlasBacktest.run(fixture,{...base,conditions:result.parameters.conditions});
                assert(JSON.stringify(custom.trades)===JSON.stringify(result.trades),kind+' customization preserves fills');
                const changed=structuredClone(fixture); changed[350]={...changed[350],open:200,high:205,low:195,close:200};
                const later=AtlasBacktest.run(changed,{...base,kind,fast:3,slow:7,lookback:14});
                assert(JSON.stringify(result.curve.filter(r=>r.date<changed[350].date))===JSON.stringify(later.curve.filter(r=>r.date<changed[350].date)),kind+' future data does not alter earlier equity');
            }
            const changes=structuredClone(fixture); changes[150]={...changes[150],open:200,high:205,low:195,close:200};
            const opts={...base,conditions:AtlasBacktest.rulesFor({...AtlasBacktest.defaults,kind:'macd'})};
            const original=AtlasBacktest.run(fixture,opts), changed=AtlasBacktest.run(changes,opts);
            assert(JSON.stringify(original.curve.filter(r=>r.date<changes[150].date))===JSON.stringify(changed.curve.filter(r=>r.date<changes[150].date)),'Future changes do not affect prior custom MACD equity');
            const editable=groups([entry],[exit]), snap=test(editable); editable.entry.rules[0].right.value=999;
            assert(snap.parameters.conditions.entry.rules[0].right.value===10,'Result keeps an immutable rule snapshot');
            return `${count} custom strategy engine checks passed`;
        }'''))

        page.locator('#workspaceOpenStrategy').click()
        page.locator('#workspaceMaximizeDock').click()
        presets = page.evaluate('Object.keys(AtlasBacktest.catalog)')
        assert len(presets) == 20
        for kind in presets:
            page.locator('#st-kind').select_option(kind)
            expect(page.locator('#strategyRulePreview')).to_contain_text('Enter')
            page.locator('#strategyRun').click()
            expect(page.locator('#strategyResults')).to_be_visible()
            expect(page.locator('#strategyError')).not_to_be_visible()

        page.locator('#st-kind').select_option('sma')
        page.locator('#strategyCustomize').click()
        expect(page.locator('#st-kind')).to_have_value('custom')
        expect(page.locator('#strategyRuleBuilder')).to_be_visible()
        expect(page.locator('#strategyResults')).not_to_be_visible()
        entry = page.locator('#st-entry-rules .strategy-condition').first
        entry.locator('[data-part="left"][data-property="type"]').select_option('close')
        entry.locator('[data-property="operator"]').select_option('gt')
        entry.locator('[data-part="right"][data-property="type"]').select_option('constant')
        entry.locator('[data-property="value"]').fill('0')
        page.locator('[data-add-condition="entry"]').click()
        page.locator('#st-entry-mode').select_option('any')
        page.locator('[data-add-condition="exit"]').click()
        page.locator('#st-exit-mode').select_option('all')
        page.locator('#st-exit-rules [data-remove-condition]').last.click()
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyResults')).to_be_visible()
        with page.expect_download() as download:
            page.locator('#strategyExport').click()
        exported = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
        assert exported and json.loads(exported[0]['conditions'])['entry']['mode'] == 'any'
        page.locator('#strategySetupName').fill('Custom <trend>')
        expect(page.locator('#strategyResults')).to_be_visible()
        page.locator('#strategySaveSetup').click()
        expect(page.locator('#strategySetupNotice')).to_contain_text('Saved')
        page.locator('#st-kind').select_option('macd')
        page.locator('#strategySavedSetup').select_option('Custom <trend>')
        expect(page.locator('#st-kind')).to_have_value('custom')
        expect(page.locator('#st-entry-mode')).to_have_value('any')
        expect(page.locator('#st-exit-mode')).to_have_value('all')
        expect(page.locator('#st-entry-rules .strategy-condition')).to_have_count(2)
        expect(page.locator('#strategyResults')).not_to_be_visible()

        page.reload()
        page.wait_for_function('rows.length > 40')
        page.locator('#workspaceOpenStrategy').click()
        page.locator('#workspaceMaximizeDock').click()
        expect(page.locator('#st-kind')).to_have_value('custom')
        page.locator('#strategySavedSetup').select_option('Custom <trend>')
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyResults')).to_be_visible()
        for theme, width in [('dark', 1440), ('light', 390)]:
            page.locator('#workspaceTheme').select_option(theme)
            page.set_viewport_size({'width': width, 'height': 1000})
            page.locator('#strategyRuleBuilder').scroll_into_view_if_needed()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=f'data/strategy-builder-{theme}.png')
        page.locator('#st-entry-rules [data-remove-condition]').first.click()
        page.locator('#st-entry-rules [data-remove-condition]').first.click()
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyError')).to_contain_text('entry conditions')
        expect(page.locator('#strategyResults')).not_to_be_visible()
        page.locator('#strategyDeleteSetup').click()
        expect(page.locator('#strategySavedSetup option')).to_have_count(1)
        assert not errors, errors
        browser.close()
        print('PASS: strategy presets, custom entry/exit, saved setups, CSV rules, validation, themes and mobile layout.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    run(parser.parse_args().url)
