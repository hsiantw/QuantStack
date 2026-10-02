// Analysis tools share the chart's viewport and use the existing history API.
(() => {
  'use strict';
  const colors = ['#9333ea', '#e07816', '#089981'];
  let comparisons = [], requestVersion = 0;
  try {
    const prefs = JSON.parse(localStorage.getItem('atlas.terminal') || '{}');
    for (const key of ['grid', 'lastPrice', 'crosshair']) if (typeof prefs[key] === 'boolean') chartAppearance[key] = prefs[key];
    for (const key of ['up', 'down']) if (/^#[0-9a-f]{6}$/i.test(prefs[key])) chartAppearance[key] = prefs[key];
    if (['candles', 'bars', 'line', 'area'].includes(prefs.style)) chartType = prefs.style;
    comparisons = [...new Set((Array.isArray(prefs.comparisons) ? prefs.comparisons : []).filter(s => typeof s === 'string'))].slice(0, 3).map(symbol => ({symbol, data: new Map(), status: 'Loading'}));
  } catch {}
  function save() {
    try { localStorage.setItem('atlas.terminal', JSON.stringify({...chartAppearance, style: chartType, comparisons: comparisons.map(item => item.symbol)})); } catch {}
  }
  const chartTypes = $('chartTypes');
  chartTypes.insertAdjacentHTML('beforeend', `<select id="terminalStyle" aria-label="Chart style"><option value="candles">Candles</option><option value="bars">OHLC bars</option><option value="line">Line</option><option value="area">Area</option></select>`);
  $('terminalStyle').value = chartType;
  $('terminalStyle').onchange = event => { chartType = event.target.value; save(); render(); };
  const actions = document.createElement('div');
  actions.className = 'terminal-actions';
  actions.innerHTML = `<button id="terminalCompare" title="Compare symbols (Alt+C)"><span aria-hidden="true">⊕</span> Compare</button><button id="terminalGo" title="Go to date (Alt+G)">Go to date</button><button id="terminalSettings" title="Chart appearance">⚙ <span>Settings</span></button><button id="terminalSnapshot" title="Download chart and analysis panes as PNG">↧ <span>Snapshot</span></button><button id="terminalFullscreen" title="Full screen" aria-label="Full screen">⛶</button>`;
  const actionIcons = {
    terminalCompare: ['Compare', '<circle cx="12" cy="12" r="8"/><path d="M8 12h8m-4-4v8"/>'],
    terminalGo: ['Go to date', '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4m8-4v4M4 10h16m-12 5h4"/>'],
    terminalSettings: ['Settings', '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" fill="white"/><circle cx="15" cy="12" r="2" fill="white"/><circle cx="9" cy="18" r="2" fill="white"/>'],
    terminalSnapshot: ['Snapshot', '<path d="M4 7h4l2-3h4l2 3h4v13H4z"/><circle cx="12" cy="13" r="4"/>'],
    terminalFullscreen: ['', '<path d="M8 4H4v4m12-4h4v4M4 16v4h4m12-4v4h-4"/>'],
  };
  for (const [id, [label, path]] of Object.entries(actionIcons)) {
    const button = actions.querySelector('#' + id);
    button.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${path}</svg>${label ? `<span>${label}</span>` : ''}`;
    button.setAttribute('aria-label', label || 'Full screen');
  }
  document.querySelector('.chart-top').after(actions);
  const notice = document.createElement('span'); notice.id = 'terminalNotice'; notice.setAttribute('role', 'status'); actions.append(notice);
  function notify(message) { notice.textContent = message; }
  const pane = document.createElement('section');
  pane.id = 'comparisonPane'; pane.hidden = true;
  pane.setAttribute('aria-label', 'Relative price performance');
  pane.innerHTML = `<div class="comparison-heading"><strong>Performance</strong><span id="comparisonBaseline"></span><button id="comparisonClose" aria-label="Remove all comparisons" title="Remove all comparisons">×</button></div><div id="comparisonLegend"></div><canvas id="comparisonCanvas" aria-label="Percentage change in close prices on matching dates"></canvas>`;
  $('indicatorPanels').before(pane);
  document.body.insertAdjacentHTML('beforeend', `
    <dialog id="terminalCompareDialog" class="terminal-dialog" aria-labelledby="terminalCompareTitle"><div class="terminal-dialog-head"><div><small>RELATIVE PERFORMANCE</small><h2 id="terminalCompareTitle">Compare symbols</h2></div><button data-close="terminalCompareDialog" aria-label="Close comparisons">×</button></div><p>Add up to three symbols. The pane follows your chart as you pan and zoom. Returns use closing prices, without dividend adjustments.</p><input id="terminalCompareSearch" type="search" placeholder="Search symbol or company" aria-label="Search comparison symbols"><div id="terminalCompareActive"></div><div id="terminalCompareResults"></div></dialog>
    <dialog id="terminalGoDialog" class="terminal-dialog" aria-labelledby="terminalGoTitle"><div class="terminal-dialog-head"><div><small>CHART NAVIGATION</small><h2 id="terminalGoTitle">Go to date</h2></div><button data-close="terminalGoDialog" aria-label="Close go to date">×</button></div><p id="terminalGoRange"></p><form id="terminalGoForm"><label for="terminalGoDate">Date</label><input id="terminalGoDate" type="date" required><p id="terminalGoError" role="alert"></p><button class="terminal-primary">Go to bar</button></form></dialog>
    <dialog id="terminalSettingsDialog" class="terminal-dialog" aria-labelledby="terminalSettingsTitle"><div class="terminal-dialog-head"><div><small>YOUR WORKSPACE</small><h2 id="terminalSettingsTitle">Chart appearance</h2></div><button data-close="terminalSettingsDialog" aria-label="Close chart appearance">×</button></div><p>Preferences are saved in this browser.</p><form id="terminalSettingsForm"><label class="terminal-setting">Grid lines<input id="terminalGrid" type="checkbox"></label><label class="terminal-setting">Last price line<input id="terminalLastPrice" type="checkbox"></label><label class="terminal-setting">Crosshair<input id="terminalCrosshair" type="checkbox"></label><label class="terminal-setting">Rising candle / bar<input id="terminalUp" type="color"></label><label class="terminal-setting">Falling candle / bar<input id="terminalDown" type="color"></label><div class="terminal-dialog-foot"><button id="terminalDefaults" type="button">Reset defaults</button><button class="terminal-primary">Apply</button></div></form></dialog>`);
  document.querySelectorAll('[data-close]').forEach(button => { button.onclick = () => $(button.dataset.close).close(); });
  document.querySelectorAll('.terminal-dialog').forEach(dialog => dialog.addEventListener('click', event => { if (event.target === dialog) { const box = dialog.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close(); } }));

  function renderSearch() {
    const q = $('terminalCompareSearch').value.trim().toLowerCase();
    $('terminalCompareActive').innerHTML = comparisons.map((item, index) => `<button data-remove-comparison="${esc(item.symbol)}"><i style="background:${colors[index]}"></i>${esc(item.symbol)} <span aria-hidden="true">×</span><span class="terminal-sr-only">Remove comparison</span></button>`).join('');
    const matches = assets.filter(asset => asset.symbol !== selected?.symbol && !comparisons.some(item => item.symbol === asset.symbol) && `${asset.symbol} ${asset.name}`.toLowerCase().includes(q)).slice(0, 35);
    $('terminalCompareResults').innerHTML = comparisons.length >= 3 ? '<p>Three comparisons added. Remove one to add another.</p>' : matches.map(asset => `<button data-compare-symbol="${esc(asset.symbol)}"><strong>${esc(asset.symbol)}</strong><span>${esc(asset.name)}</span><b aria-hidden="true">+</b></button>`).join('') || '<p>No matching symbols.</p>';
  }
  $('terminalCompare').onclick = () => { renderSearch(); $('terminalCompareDialog').showModal(); $('terminalCompareSearch').focus(); };
  $('terminalCompareSearch').oninput = renderSearch;
  function removeComparison(symbol) { comparisons = comparisons.filter(item => item.symbol !== symbol); save(); renderSearch(); refreshComparisons(); }
  $('terminalCompareActive').onclick = event => { const button = event.target.closest('[data-remove-comparison]'); if (button) removeComparison(button.dataset.removeComparison); };
  $('comparisonLegend').onclick = event => { const button = event.target.closest('[data-remove-comparison]'); if (button) removeComparison(button.dataset.removeComparison); };
  $('terminalCompareResults').onclick = event => {
    const button = event.target.closest('[data-compare-symbol]'); if (!button || comparisons.length >= 3) return;
    comparisons.push({symbol: button.dataset.compareSymbol, data: new Map(), status: 'Loading'});
    save(); renderSearch(); refreshComparisons();
  };
  $('comparisonClose').onclick = () => { comparisons = []; requestVersion++; save(); drawComparison(); };
  async function refreshComparisons() {
    const version = ++requestVersion, chartGeneration = generation;
    for (const item of comparisons) { item.data = new Map(); item.status = item.symbol === selected?.symbol ? 'Main symbol' : 'Loading'; }
    drawComparison();
    if (!selected || !rows.length) { for (const item of comparisons) item.status = 'No chart data'; drawComparison(); return; }
    const params = query();
    await Promise.all(comparisons.map(async item => {
      if (item.symbol === selected.symbol) return;
      const q = new URLSearchParams(params); q.set('symbol', item.symbol);
      try {
        const result = await api('/api/history?' + q);
        if (version !== requestVersion || chartGeneration !== generation) return;
        item.data = new Map(result.filter(row => Number.isFinite(row.close) && row.close > 0).map(row => [row.date, row.close]));
        item.status = item.data.size ? '' : 'No data for this range';
      } catch { if (version === requestVersion && chartGeneration === generation) item.status = 'Could not load · remove and retry'; }
    }));
    if (version === requestVersion && chartGeneration === generation) drawComparison();
  }
  const originalLoadHistory = loadHistory;
  loadHistory = async function() {
    requestVersion++;
    for (const item of comparisons) { item.data = new Map(); item.status = 'Loading'; }
    const task = originalLoadHistory(), token = generation;
    drawComparison();
    await task;
    if (token === generation) await refreshComparisons();
  };

  function drawComparison() {
    pane.hidden = !comparisons.length;
    $('terminalCompare').classList.toggle('active', !!comparisons.length);
    if (!comparisons.length) return;
    const g = geometry(), data = g.data;
    const active = comparisons.filter(item => !item.status && item.symbol !== selected?.symbol);
    const first = data.findIndex((row, index) => g.x(index) >= g.left && g.x(index) <= g.left + g.pw && row.close > 0 && active.every(item => item.data.has(row.date)));
    const baseline = data[first];
    const canvas = $('comparisonCanvas'), box = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(box.width * dpr)); canvas.height = Math.max(1, Math.round(box.height * dpr));
    const c = canvas.getContext('2d'); c.scale(dpr, dpr); c.fillStyle = window.atlasTheme.palette.surface; c.fillRect(0, 0, box.width, box.height);
    const series = baseline && active.length ? [{symbol: selected.symbol, color: '#2962ff', values: data.map(row => Number.isFinite(row.close) ? (row.close / baseline.close - 1) * 100 : null)}, ...active.map(item => ({symbol: item.symbol, color: colors[comparisons.indexOf(item)], values: data.map(row => item.data.has(row.date) ? (item.data.get(row.date) / item.data.get(baseline.date) - 1) * 100 : null)}))] : [];
    const lastVisible = Math.min(data.length - 1, g.index(g.left + g.pw));
    const inspected = hover >= first && hover <= lastVisible ? hover : lastVisible;
    $('comparisonBaseline').textContent = baseline && active.length ? `0% at ${labelTime(baseline.date)} · matching timestamps` : active.length ? 'No shared timestamps in view' : 'Choose symbols with data for this interval';
    const legend = [{symbol: selected?.symbol || '', color: '#2962ff'}, ...comparisons.map((item, index) => ({...item, color: colors[index], removable: true}))];
    const markup = legend.map(item => {
      const values = series.find(s => s.symbol === item.symbol)?.values, value = values?.[inspected];
      const label = item.status || (Number.isFinite(value) ? pct(value) : 'No matching bar');
      return `<span style="--series-color:${item.color}" title="${esc(data[inspected]?.date || '')}"><i></i><strong>${esc(item.symbol)}</strong> ${esc(label)}${item.removable ? `<button data-remove-comparison="${esc(item.symbol)}" aria-label="Remove ${esc(item.symbol)} comparison">×</button>` : ''}</span>`;
    }).join('');
    if ($('comparisonLegend').innerHTML !== markup) $('comparisonLegend').innerHTML = markup;
    if (!series.length || box.width < 100) { c.fillStyle = window.atlasTheme.palette.muted; c.font = '12px Segoe UI'; c.fillText(active.length ? 'No common baseline. Pan or expand the loaded date range.' : 'Comparison data will appear here when available.', 14, 45); return; }
    let low = 0, high = 0;
    for (const item of series) item.values.forEach((value, index) => { if (index >= first && index <= lastVisible && Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); } });
    const pad = (high - low || 1) * .14; low -= pad; high += pad;
    const top = 12, height = box.height - 25, y = value => top + (high - value) / (high - low) * height;
    c.font = '10px Segoe UI';
    for (let i = 0; i < 3; i++) {
      const value = low + (high - low) * i / 2, py = y(value);
      c.strokeStyle = window.atlasTheme.palette.grid; c.beginPath(); c.moveTo(g.left, py); c.lineTo(g.left + g.pw, py); c.stroke(); c.fillStyle = window.atlasTheme.palette.muted; c.fillText(`${fmt(value)}%`, g.left + g.pw + 8, py + 3);
    }
    c.save(); c.beginPath(); c.rect(g.left, 0, g.pw, box.height); c.clip();
    c.strokeStyle = '#a7afbd'; c.setLineDash([3, 4]); c.beginPath(); c.moveTo(g.left, y(0)); c.lineTo(g.left + g.pw, y(0)); c.stroke(); c.setLineDash([]);
    for (const item of series) {
      c.beginPath(); let started = false;
      item.values.forEach((value, index) => {
        if (index < first || !Number.isFinite(value)) { started = false; return; }
        if (started) c.lineTo(g.x(index), y(value)); else c.moveTo(g.x(index), y(value)); started = true;
      });
      c.lineWidth = 1.7; c.strokeStyle = item.color; c.stroke();
    }
    if (chartAppearance.crosshair && hover >= first && hover <= lastVisible) { c.setLineDash([3, 4]); c.strokeStyle = '#9598a1'; c.lineWidth = 1; c.beginPath(); c.moveTo(g.x(hover), 0); c.lineTo(g.x(hover), box.height); c.stroke(); }
    c.restore();
  }
  const originalDraw = draw;
  draw = function() { originalDraw(); drawComparison(); };
  new ResizeObserver(() => drawComparison()).observe($('comparisonCanvas'));

  $('terminalGo').onclick = () => {
    if (!rows.length) { notify('Load price bars first.'); return; }
    const input = $('terminalGoDate'); input.min = rows[0].date.slice(0, 10); input.max = rows.at(-1).date.slice(0, 10); input.value = (visible()[0]?.date || rows[0].date).slice(0, 10);
    $('terminalGoRange').textContent = `Loaded: ${input.min} to ${input.max}. Non-trading dates go to the next available bar. Load a wider range using All or the date range picker.`;
    $('terminalGoError').textContent = ''; $('terminalGoDialog').showModal(); input.focus();
  };
  $('terminalGoForm').onsubmit = event => {
    event.preventDefault(); const target = $('terminalGoDate').value;
    const index = rows.findIndex(row => row.date.slice(0, 10) >= target);
    if (index < 0 || target < rows[0]?.date.slice(0, 10)) { $('terminalGoError').textContent = 'Choose a date within the loaded range.'; return; }
    moveChartTime(index - Math.floor(Math.min(viewCount, 100) / 2), Math.min(viewCount, 100));
    chartScale.auto = true; chartScale.bounds = null; hover = -1; hoverY = null; draw(); $('terminalGoDialog').close(); notify(`Showing ${labelTime(rows[index].date)}`);
  };
  function fillSettings(prefs) {
    for (const [id, key] of [['terminalGrid', 'grid'], ['terminalLastPrice', 'lastPrice'], ['terminalCrosshair', 'crosshair']]) $(id).checked = prefs[key];
    $('terminalUp').value = prefs.up; $('terminalDown').value = prefs.down;
  }
  $('terminalSettings').onclick = () => { fillSettings(chartAppearance); $('terminalSettingsDialog').showModal(); };
  $('terminalDefaults').onclick = () => fillSettings({grid: true, lastPrice: true, crosshair: true, up: '#089981', down: '#f23645'});
  $('terminalSettingsForm').onsubmit = event => {
    event.preventDefault(); Object.assign(chartAppearance, {grid: $('terminalGrid').checked, lastPrice: $('terminalLastPrice').checked, crosshair: $('terminalCrosshair').checked, up: $('terminalUp').value, down: $('terminalDown').value}); save(); draw(); $('terminalSettingsDialog').close();
  };
  $('terminalFullscreen').onclick = async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { notify('Full screen is unavailable in this browser.'); }
  };
  document.addEventListener('fullscreenchange', () => { $('terminalFullscreen').setAttribute('aria-label', document.fullscreenElement ? 'Exit full screen' : 'Full screen'); });
  $('terminalSnapshot').onclick = () => {
    if (!rows.length) { notify('Load price bars before taking a snapshot.'); return; }
    draw();
    const canvases = [$('chart'), ...(pane.hidden ? [] : [$('comparisonCanvas')]), ...document.querySelectorAll('#indicatorPanels canvas')].filter(canvas => canvas.width && canvas.getBoundingClientRect().height);
    const width = $('chart').getBoundingClientRect().width, scale = 2;
    const output = document.createElement('canvas'); output.width = width * scale; output.height = (100 + canvases.reduce((sum, canvas) => sum + canvas.getBoundingClientRect().height + 30, 0)) * scale;
    const c = output.getContext('2d'); c.scale(scale, scale); c.fillStyle = window.atlasTheme.palette.surface; c.fillRect(0, 0, output.width / scale, output.height / scale);
    c.fillStyle = window.atlasTheme.palette.text; c.font = '600 17px Segoe UI'; c.fillText(`${selected.symbol} · ${interval} · ${$('terminalStyle').selectedOptions[0].text}`, 14, 28);
    c.fillStyle = window.atlasTheme.palette.muted; c.font = '11px Segoe UI'; c.fillText('QUANTSTACK · Stored market data · ' + new Date().toISOString().slice(0, 10), 14, 50);
    let y = 70;
    for (const canvas of canvases) {
      const height = canvas.getBoundingClientRect().height;
      const title = canvas.id === 'chart' ? `${visible()[0]?.date || ''} — ${visible().at(-1)?.date || ''}` : canvas.id === 'comparisonCanvas' ? `${$('comparisonBaseline').textContent} | ${[selected.symbol, ...comparisons.map(item => item.symbol)].join(' / ')}` : canvas.parentElement.querySelector('.indicator-panel-label')?.textContent || 'Study';
      c.fillStyle = window.atlasTheme.palette.muted; c.font = '10px Segoe UI'; c.fillText(title, 14, y + 12, width - 28); y += 26;
      c.drawImage(canvas, 0, y, width, height); y += height + 4;
    }
    const filename = `QuantStack-${selected.symbol.replace(/[^a-z0-9_-]/gi, '_')}-${interval}.png`;
    output.toBlob(blob => {
      if (!blob) { notify('Snapshot could not be created.'); return; }
      const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000); notify('Chart snapshot downloaded.');
    }, 'image/png');
  };
  document.addEventListener('keydown', event => {
    if (!event.altKey || event.ctrlKey || event.metaKey || ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName) || event.target.isContentEditable || document.querySelector('dialog[open]')) return;
    const button = {c: 'terminalCompare', g: 'terminalGo'}[event.key.toLowerCase()]; if (button) { event.preventDefault(); $(button).click(); }
  });
})();
