// Market overview uses the same catalog and personal lists as Supercharts.
(() => {
  const dialog = document.createElement('section');
  dialog.id = 'marketOverview'; dialog.setAttribute('aria-labelledby', 'marketOverviewTitle');
  const field = id => dialog.querySelector('#' + id);
  const visible = () => dialog.isConnected && dialog.getClientRects().length > 0;
  dialog.innerHTML = `<div class="market-heading"><div><small>QUANTSTACK / MARKET MONITOR</small><h2 id="marketOverviewTitle">Market overview</h2></div></div>
    <p class="market-caption">Latest stored prices and completed daily performance. Not a live feed. Select an instrument to open its chart.</p>
    <div class="market-controls"><label>Ranking universe <select id="marketUniverse"><option value="stocks">Stocks &amp; ETFs</option><option value="crypto">Crypto</option><option value="watchlist">Selected watchlist</option></select></label><label>Watchlist <select id="marketList"></select></label><button id="marketRefresh" type="button">Refresh data</button></div>
    <p id="marketStatus" role="status" aria-live="polite"></p><section><h3>Major indexes &amp; trackers</h3><div id="marketIndexes" class="market-indexes"></div></section>
    <section><h3>Watchlist updates</h3><p id="marketBreadth" class="market-caption"></p><div id="marketWatchlist" class="market-table"></div></section>
    <div class="market-rankings"><section><h3>Top gainers <small>DAILY %</small></h3><div id="marketGainers" class="market-table"></div></section><section><h3>Top losers <small>DAILY %</small></h3><div id="marketLosers" class="market-table"></div></section><section><h3>Volume ranking <small>DAILY VOLUME</small></h3><p class="market-caption">Raw units, not traded value. Crypto volumes use different base assets and are not directly comparable.</p><div id="marketVolume" class="market-table"></div></section></div>`;
  let catalog = [], lists = [], previous = new Map(), checked = '', busy = false;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const num = value => finite(value) ? fmt(value) : '—';
  const compact = value => finite(value) ? new Intl.NumberFormat(undefined, {notation:'compact', maximumFractionDigits:2}).format(value) : '—';
  const color = value => finite(value) ? value > 0 ? 'positive' : value < 0 ? 'negative' : '' : '';
  const stamp = item => item.performance_asof || item.date || 'Unavailable';
  function readLists() {
    let state = {}, favorites = [];
    try { state = JSON.parse(localStorage.getItem('atlas.watchlists.v1') || '{}') || {}; } catch {}
    try { favorites = JSON.parse(localStorage.getItem('atlas.saved') || '[]'); } catch {}
    lists = [{id:'saved', name:'Saved favorites', symbols:Array.isArray(favorites) ? favorites : []}];
    for (const list of Array.isArray(state.lists) ? state.lists : []) {
      if (!list || typeof list.id !== 'string' || typeof list.name !== 'string') continue;
      lists.push({...list, symbols:list.color ? Object.keys(state.labels || {}).filter(s => state.labels[s] === list.color) : Array.isArray(list.symbols) ? list.symbols : []});
    }
    const chosen = field('marketList').value || state.active;
    field('marketList').replaceChildren(...lists.map(list => new Option(list.name, list.id)));
    field('marketList').value = lists.some(list => list.id === chosen) ? chosen : 'saved';
  }
  function table(items, volume = false, tracking = false) {
    if (!items.length) return '<p class="market-empty">No matching data available.</p>';
    return `<table><thead><tr><th>Instrument</th><th>${volume ? 'Volume' : 'Last'}</th><th>Day %</th><th>Daily bar</th>${tracking ? '<th>Update</th>' : ''}</tr></thead><tbody>${items.map(item => {
      const before = previous.get(item.symbol);
      const changed = before && (before.close !== item.close || before.quote_timestamp !== item.quote_timestamp || before.volume !== item.volume);
      return `<tr><td><button type="button" data-market-symbol="${esc(item.symbol)}">${esc(item.symbol)}</button><small>${esc(item.name || '')}</small></td><td>${volume ? compact(item.volume) : num(item.close)}<small>${esc(volume ? 'units' : item.currency || '')}</small></td><td class="${color(item.change_1d_pct)}">${finite(item.change_1d_pct) ? pct(item.change_1d_pct) : '—'}</td><td>${esc(stamp(item))}<small>Price: ${esc(item.quote_timestamp || item.date || 'unavailable')}</small></td>${tracking ? `<td>${!before ? 'First check' : changed ? 'Updated' : 'Unchanged'}</td>` : ''}</tr>`;
    }).join('')}</tbody></table>`;
  }
  function render() {
    const symbols = new Set(lists.find(list => list.id === field('marketList').value)?.symbols || []);
    const watch = [...symbols].map(symbol => catalog.find(item => item.symbol === symbol) || {symbol, name:'No stored data'});
    const choice = field('marketUniverse').value;
    const universe = choice === 'watchlist' ? watch : catalog.filter(item => !item.symbol.startsWith('^') && (choice === 'crypto' ? item.kind === 'Crypto' || item.symbol.endsWith('-USD') : item.kind !== 'Crypto' && !item.symbol.endsWith('-USD')));
    const performance = universe.filter(item => finite(item.change_1d_pct));
    const indices = [['^GSPC','S&P 500','SPY'],['^DJI','Dow Jones','DIA'],['^IXIC','Nasdaq Composite',null],['^NDX','Nasdaq 100','QQQ'],['^RUT','Russell 2000','IWM'],['^VIX','Volatility index',null]];
    field('marketIndexes').innerHTML = indices.map(([symbol, name, tracker]) => {
      const index = catalog.find(item => item.symbol === symbol && finite(item.close));
      const item = index || catalog.find(item => item.symbol === tracker && finite(item.close));
      return `<article><small>${esc(name)}</small>${item ? `<button type="button" data-market-symbol="${esc(item.symbol)}">${esc(item.symbol)}${index ? '' : ' · ETF tracker'}</button><strong>${num(item.close)} <small>${esc(item.currency || '')}</small></strong><span class="${color(item.change_1d_pct)}">${finite(item.change_1d_pct) ? pct(item.change_1d_pct) : '—'} daily</span><small>Price as of ${esc(item.quote_timestamp || item.date || 'unavailable')}</small>` : '<strong>—</strong><small>No stored data</small>'}</article>`;
    }).join('');
    field('marketWatchlist').innerHTML = watch.length ? table(watch, false, true) : '<p class="market-empty">Your watchlist is empty. Add instruments using the chart watchlist controls.</p>';
    field('marketBreadth').textContent = `${watch.length} instruments · ${watch.filter(a => finite(a.change_1d_pct) && a.change_1d_pct > 0).length} advancing · ${watch.filter(a => finite(a.change_1d_pct) && a.change_1d_pct < 0).length} declining · ${watch.filter(a => a.change_1d_pct === 0).length} flat · ${watch.filter(a => !finite(a.change_1d_pct)).length} unavailable`;
    field('marketGainers').innerHTML = table(performance.filter(a => a.change_1d_pct > 0).sort((a,b) => b.change_1d_pct-a.change_1d_pct).slice(0,10));
    field('marketLosers').innerHTML = table(performance.filter(a => a.change_1d_pct < 0).sort((a,b) => a.change_1d_pct-b.change_1d_pct).slice(0,10));
    field('marketVolume').innerHTML = table(universe.filter(a => finite(a.volume) && a.volume >= 0).sort((a,b) => b.volume-a.volume).slice(0,10), true);
    field('marketStatus').textContent = `${checked ? 'Last checked ' + checked + '. ' : ''}Rankings cover ${universe.length} instruments in this dataset; dates may differ. Refresh checks stored data, not the market provider.`;
  }
  async function refresh() {
    if (busy) return;
    busy = true; field('marketRefresh').disabled = true; field('marketStatus').textContent = 'Checking stored market data…';
    try {
      const next = await api('/api/symbols');
      if (!Array.isArray(next)) throw Error('Invalid market data response.');
      previous = new Map(catalog.map(item => [item.symbol, item])); catalog = next;
      checked = new Date().toLocaleTimeString(); readLists(); render();
    } catch (error) { field('marketStatus').textContent = `Unable to refresh: ${error.message} Previously displayed data is retained.`; }
    finally { busy = false; field('marketRefresh').disabled = false; }
  }
  window.MarketOverview = {mount(pane) { pane.replaceChildren(dialog); readLists(); render(); refresh(); }};
  field('marketRefresh').onclick = refresh;
  field('marketUniverse').onchange = render; field('marketList').onchange = render;
  dialog.addEventListener('click', event => {
    const target = event.target.closest('[data-market-symbol]'); if (!target) return;
    const item = catalog.find(a => a.symbol === target.dataset.marketSymbol);
    if (!item || !finite(item.close)) { field('marketStatus').textContent = 'No stored chart data for this instrument.'; return; }
    if (document.body.dataset.toolPage) { location.assign('/workspace/?symbol=' + encodeURIComponent(item.symbol)); return; }
    if (!assets.some(a => a.symbol === item.symbol)) assets.push(item);
    document.getElementById('workspaceCloseDock').click(); select(item.symbol);
  });
  window.addEventListener('storage', event => { if (visible() && ['atlas.saved','atlas.watchlists.v1'].includes(event.key)) { readLists(); render(); } });
  setInterval(() => { if (visible() && !document.hidden) refresh(); }, 60000);
})();
