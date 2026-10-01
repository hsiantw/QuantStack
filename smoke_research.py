"""Verify strategy execution math and research controls in an isolated Edge profile."""
from pathlib import Path
import argparse
from playwright.sync_api import sync_playwright, expect


def run(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("typeof rows !== 'undefined' && rows.length > 0")
        # Known prices make execution timing, accounting, and ambiguity testable.
        checks = page.evaluate("""() => {
          const assert = (ok, text) => { if (!ok) throw Error(text); };
          const near = (a,b) => Math.abs(a-b) < 1e-7;
          const bars = [10,11,12,13,14,15,16,17].map((v,i) => ({date:`2026-01-${String(i+1).padStart(2,'0')}`,open:v,high:v+1,low:v-1,close:v,adjusted_close:v}));
          const base = {fast:2,slow:3,capital:1000,commission:0,slippage:0,adjusted:false};
          const a = AtlasBacktest.run(bars,base);
          assert(a.trades[0].entry === bars[3].date && a.trades[0].entryPrice === 13,'Next open execution after warmup');
          assert(a.trades.length === 1 && near(a.finalEquity,1000/13*17),'Long position accounting');
          assert(near(a.returnPct,a.benchmarkPct),'Equal benchmark window and costs');
          assert(near(a.performance.averageTrade,a.netProfit) && a.performance.winners===1,'Trade statistics match net accounting');
          assert(a.monthly.length===1 && near(a.monthly[0].returnPct,a.returnPct),'First month includes initial capital');
          const months = bars.map((b,i)=>({...b,date:`2026-${i<5?'01':'02'}-${String(i+1).padStart(2,'0')}`}));
          const monthly = AtlasBacktest.run(months,base);
          assert(monthly.monthly.length===2 && near(monthly.monthly.reduce((v,m)=>v*(1+m.returnPct/100),1),monthly.finalEquity/base.capital),'Monthly returns compound to total equity');
          assert(a.trades[0].reason === 'End of test','Final liquidation');
          const fee = AtlasBacktest.run(bars,{...base,commission:1,slippage:1});
          const quantity = 1000/(13*1.01*1.01), proceeds = quantity*17*.99;
          assert(near(fee.finalEquity,proceeds*.99),'Fees and adverse slippage on both fills');
          assert(near(fee.fees,quantity*13*1.01*.01+proceeds*.01),'Fee totals');
          const sized = AtlasBacktest.run(bars,{...base,allocation:50});
          assert(near(sized.finalEquity,500+500/13*17),'Cash remains uninvested');
          const stopBars = structuredClone(bars); stopBars[3].low=8; stopBars[3].high=18;
          const stop = AtlasBacktest.run(stopBars,{...base,stop:10,target:10});
          assert(stop.trades[0].reason==='Stop loss' && near(stop.trades[0].exitPrice,11.7),'Conservative same-bar stop priority');
          const gapBars = structuredClone(bars); Object.assign(gapBars[4],{open:8,low:7,high:15});
          const gap = AtlasBacktest.run(gapBars,{...base,stop:10});
          assert(gap.trades[0].reason==='Stop gap' && gap.trades[0].exitPrice===8,'Gap stop uses worse open');
          const flat=bars.map(b=>({...b,open:10,high:10,low:10,close:10,adjusted_close:10}));
          assert(AtlasBacktest.run(flat,{...base}).trades.length===0,'Flat averages do not trade');
          assert(AtlasBacktest.run(flat,{...base,kind:'rsi',lookback:2}).trades.length===0,'Flat RSI equals 50');
          assert(AtlasBacktest.run(flat,{...base,kind:'rsi',lookback:2,fast:0,slow:0}).trades.length===0,'Hidden MA settings do not block RSI');
          assert(AtlasBacktest.run(flat,{...base,lower:90,upper:10}).trades.length===0,'Hidden RSI settings do not block MA');
          for (const kind of ['ema','breakout','rsi']) assert(Number.isFinite(AtlasBacktest.run(bars,{...base,kind,lookback:2}).finalEquity),kind+' produces finite accounting');
          const adjusted = bars.map(b=>({...b,adjusted_close:b.close/2}));
          assert(near(AtlasBacktest.run(adjusted,{...base,adjusted:true}).returnPct,a.returnPct),'OHLC consistently adjusted');
          const rejects = (data,opts) => { try { AtlasBacktest.run(data,opts); return false; } catch {return true;} };
          assert(rejects(bars,{...base,fast:4}),'Reject invalid periods');
          assert(rejects(bars.slice(0,3),base),'Reject insufficient warmup');
          assert(rejects(bars.map((b,i)=>i===3?{...b,open:null}:b),base),'Reject incomplete OHLC');
          assert(rejects(bars.map((b,i)=>i===3?{...b,adjusted_close:null}:b),{...base,adjusted:true}),'Reject missing adjusted prices');
          assert(rejects([...bars].reverse(),base),'Reject unsorted data');
          const changed=structuredClone(bars); changed[6].close=20; changed[6].high=21;
          assert(AtlasBacktest.run(changed,base).curve.slice(0,3).every((r,i)=>near(r.equity,a.curve[i].equity)),'Future prices do not change past equity');
          return 'Strategy execution, trade statistics, and monthly compounding checks passed';
        }""")
        print(checks)
        page.locator('#workspaceTheme').select_option('dark')
        expect(page.locator('html')).to_have_attribute('data-theme', 'dark')
        assert page.locator('body').evaluate("e=>getComputedStyle(e).backgroundColor") == 'rgb(19, 23, 34)'
        page.locator('#workspaceOpenStrategy').click()
        page.locator('#workspaceMaximizeDock').click()
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyResults')).to_be_visible()
        expect(page.locator('#strategyError')).not_to_be_visible()
        page.locator('#strategyPerformanceTab').click()
        expect(page.locator('#strategyPerformance')).to_contain_text('Monthly returns')
        page.locator('#strategyPerformanceTab').focus()
        page.keyboard.press('Home')
        expect(page.locator('#strategyOverviewTab')).to_be_focused()
        expect(page.locator('#strategyOverview')).to_be_visible()
        page.locator('#strategyTradesTab').click()
        expect(page.locator('#strategyTradeRows tr')).not_to_have_count(0)
        with page.expect_download() as download:
            page.locator('#strategyExport').click()
        assert 'strategy-trades' in download.value.suggested_filename
        page.locator('#st-fast').fill('8')
        expect(page.locator('#strategyResults')).not_to_be_visible()
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyResults')).to_be_visible()
        page.locator('#strategyOverviewTab').click()
        Path('data').mkdir(exist_ok=True)
        page.screenshot(path='data/research-dark.png')
        page.locator('#workspaceOpenScreener').click()
        page.wait_for_function("document.querySelector('#screenerContent').getAttribute('aria-busy')==='false'")
        page.locator('#screenerCompact').check()
        page.locator('#screenerHeat').check()
        page.locator('[data-screen-preset="gainers"]').click()
        expect(page.locator('#screenerPrimarySort')).to_have_value('change_1d_pct')
        expect(page.locator('#screenerCompact')).to_be_checked()
        expect(page.locator('#screenerHeat')).to_be_checked()
        page.wait_for_function("[...document.querySelectorAll('#screenerTableBody tr')].every(r=>!r.querySelector('.negative'))")
        page.locator('[data-screen-preset="losers"]').click()
        expect(page.locator('#screenerPrimaryDirection')).to_have_value('asc')
        page.locator('[data-clear-filter^="rule:"]').click()
        page.locator('#screenerPrimarySort').select_option('market_cap')
        page.locator('#screenerPrimaryDirection').select_option('desc')
        page.locator('#screenerSecondarySort').select_option('volume')
        expect(page.locator('#screenerSortNote')).to_contain_text('then Volume')
        page.locator('#screenerColumnsDetails summary').click()
        page.locator('#screenerColumnSearch').fill('RSI')
        expect(page.locator('[data-custom-column="rsi14"]')).to_be_visible()
        expect(page.locator('[data-custom-column="pe"]')).not_to_be_visible()
        page.locator('#screenerMoveColumn').select_option('close')
        page.locator('#screenerColumnLeft').click()
        page.wait_for_function("document.querySelectorAll('#screenerTableHead th')[2].textContent.includes('Price')")
        page.locator('#screenerColumnsDetails summary').click()
        page.locator('#screenerSearch').fill('AAPL')
        page.wait_for_function("document.querySelectorAll('#screenerTableBody [data-screen-symbol]').length===1")
        old_saved = page.evaluate('[...saved]')
        page.locator('#screenerSaveMatches').click()
        page.wait_for_function("saved.has('AAPL')")
        page.locator('#screenerUndoSave').click()
        assert page.evaluate('[...saved]') == old_saved
        page.locator('#screenerSearch').fill('')
        page.wait_for_function("document.querySelectorAll('#screenerTableBody [data-screen-symbol]').length>1")
        first = page.locator('[data-screen-symbol]:not([disabled])').first
        symbol = first.get_attribute('data-screen-symbol')
        first.focus(); first.press('ArrowDown')
        assert page.evaluate('document.activeElement.dataset.screenSymbol') != symbol
        page.screenshot(path='data/screener-dark.png')
        page.reload()
        page.wait_for_function("typeof rows !== 'undefined' && rows.length > 0")
        expect(page.locator('html')).to_have_attribute('data-theme', 'dark')
        page.locator('#workspaceOpenScreener').click()
        expect(page.locator('#screenerCompact')).to_be_checked()
        page.locator('#workspaceTheme').select_option('system')
        page.emulate_media(color_scheme='light')
        expect(page.locator('html')).to_have_attribute('data-theme','light')
        page.emulate_media(color_scheme='dark')
        expect(page.locator('html')).to_have_attribute('data-theme','dark')
        page.locator('#workspaceTheme').select_option('light')
        page.set_viewport_size({'width':390,'height':844})
        page.locator('#workspaceOpenStrategy').click()
        page.locator('#workspaceMaximizeDock').click()
        page.locator('#strategyRun').click()
        expect(page.locator('#strategyResults')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='data/research-mobile.png')
        page.set_viewport_size({'width':1440,'height':1000})
        page.screenshot(path='data/research-light.png')
        page.locator('[data-period="6M"]').click()
        expect(page.locator('#strategyResults')).not_to_be_visible()
        assert not errors, errors
        browser.close()
        print('Research checks passed: themes, persistence, screener controls, strategy UI, export, stale results, mobile; no JavaScript errors.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    run(parser.parse_args().url)
