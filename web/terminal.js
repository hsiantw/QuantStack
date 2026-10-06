// Right-click actions reuse the workspace controls and their saved preferences.
(() => {
  const chart = $('chart'), menu = document.createElement('div');
  menu.id = 'chartContextMenu'; menu.className = 'chart-context-menu';
  menu.hidden = true; menu.setAttribute('role', 'menu'); menu.setAttribute('aria-label', 'Chart actions');
  chart.setAttribute('aria-haspopup', 'menu'); chart.setAttribute('aria-controls', menu.id);
  chart.setAttribute('aria-expanded', 'false');
  const items = [
    ['settings', 'Chart settings…', 'terminalSettings'],
    ['reset', 'Reset chart view', 'chartResetView', 'Alt R'],
    ['auto', 'Auto scale', 'chartAutoScale'],
    ['log', 'Logarithmic scale', 'chartLogScale'],
    ['scale', 'More scale settings…', 'chartScaleMenuButton'],
    null,
    ['indicators', 'Indicators…', 'indicatorLibrary'],
    ['compare', 'Compare symbols…', 'terminalCompare', 'Alt C'],
    ['date', 'Go to date…', 'terminalGo', 'Alt G'],
    null,
    ['horizontal', 'Draw horizontal line', null],
    ['trend', 'Draw trend line', null],
    ['measure', 'Measure price and bars', null],
    ['undo', 'Undo drawing', 'undoDrawing', 'Ctrl Z'],
    ['clear', 'Remove all drawings', 'clearDrawings'],
    null,
    ['snapshot', 'Download chart image', 'terminalSnapshot'],
    ['fullscreen', 'Full screen', 'terminalFullscreen'],
  ];
  for (const item of items) {
    if (!item) { const divider = document.createElement('hr'); divider.setAttribute('role', 'separator'); menu.append(divider); continue; }
    const [key, label, , shortcut = ''] = item, button = document.createElement('button');
    button.type = 'button'; button.dataset.action = key; button.tabIndex = -1;
    button.setAttribute('role', ['auto', 'log'].includes(key) ? 'menuitemcheckbox' : 'menuitem');
    button.innerHTML = `<span class="context-check" aria-hidden="true"></span><span>${label}</span><kbd>${shortcut}</kbd>`;
    menu.append(button);
  }
  document.body.append(menu);
  function close(restore = false) {
    menu.hidden = true; chart.setAttribute('aria-expanded', 'false');
    if (restore) chart.focus({preventScroll: true});
  }
  function open(x, y) {
    setTool('cursor'); closeScaleMenu(); $('tooltip').hidden = true;
    for (const button of menu.querySelectorAll('button')) {
      const key = button.dataset.action, target = items.find(item => item?.[0] === key)?.[2];
      button.disabled = (['reset', 'auto', 'log', 'date', 'horizontal', 'trend', 'measure', 'snapshot'].includes(key) && !rows.length)
        || (key === 'undo' && !drawingUndo.length) || (key === 'clear' && !drawings.length)
        || !!(target && $(target)?.disabled);
      if (['auto', 'log'].includes(key)) {
        const checked = key === 'auto' ? chartScale.auto : chartScale.mode === 'log';
        button.setAttribute('aria-checked', String(checked));
        button.querySelector('.context-check').textContent = checked ? '✓' : '';
      }
      if (key === 'fullscreen') button.children[1].textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen';
    }
    menu.hidden = false; menu.scrollTop = 0;
    const box = menu.getBoundingClientRect();
    menu.style.left = `${Math.max(8, Math.min(x, innerWidth - box.width - 8))}px`;
    menu.style.top = `${Math.max(8, Math.min(y, innerHeight - box.height - 8))}px`;
    chart.setAttribute('aria-expanded', 'true'); menu.querySelector('button').focus({preventScroll: true});
  }
  chart.addEventListener('contextmenu', event => {
    event.preventDefault();
    const box = chart.getBoundingClientRect();
    open(event.clientX || box.left + box.width / 2, event.clientY || box.top + box.height / 2);
  });
  chart.addEventListener('keydown', event => {
    if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
      event.preventDefault(); event.stopPropagation();
      const box = chart.getBoundingClientRect(); open(box.left + box.width / 2, box.top + box.height / 2);
    }
  });
  menu.addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button || button.disabled) return;
    const [key, , target] = items.find(item => item?.[0] === button.dataset.action);
    close(true);
    if (target) $(target).click(); else setTool(key);
  });
  menu.addEventListener('keydown', event => {
    event.stopPropagation();
    if (event.key === 'Escape') { event.preventDefault(); close(true); }
    else if (event.key === 'Tab') close(true);
    else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const buttons = [...menu.querySelectorAll('button:not(:disabled)')], index = buttons.indexOf(document.activeElement);
      buttons[event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length].focus();
    }
  });
  document.addEventListener('pointerdown', event => { if (!menu.hidden && !menu.contains(event.target)) close(); });
  window.addEventListener('resize', () => close());
  window.addEventListener('blur', () => close());
})();

// Analysis tools share the chart's viewport and use the existing history API.
(() => {
  'use strict';
  const colors = ['#9333ea', '#e07816', '#089981'];
  let comparisons = [], requestVersion = 0;
  try {
    const prefs = JSON.parse(localStorage.getItem('atlas.terminal') || '{}');
    Object.assign(chartAppearance,normalizeAppearance(prefs));
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
  $('workspaceSettingsDock').append($('terminalSettings'));
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
  window.addChartComparison = symbol => {
    if(symbol===selected?.symbol)return 'This instrument is already the main chart.';
    if(comparisons.some(item=>item.symbol===symbol))return 'This instrument is already in the comparison.';
    if(comparisons.length>=3)return 'Three comparisons are already added. Remove one in Compare first.';
    if(!assets.some(asset=>asset.symbol===symbol))return 'No stored data for this instrument.';
    comparisons.push({symbol, data:new Map(), status:'Loading'});
    save();renderSearch();refreshComparisons();return null;
  };
  $('terminalCompareResults').onclick = event => {
    const button=event.target.closest('[data-compare-symbol]');if(!button)return;
    const error=window.addChartComparison(button.dataset.compareSymbol);if(error)notify(error);
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
    const c = canvas.getContext('2d'); c.scale(dpr, dpr); c.fillStyle = chartPalette().surface; c.fillRect(0, 0, box.width, box.height);
    const series = baseline && active.length ? [{symbol: selected.symbol, color: chartPalette().accent, values: data.map(row => Number.isFinite(row.close) ? (row.close / baseline.close - 1) * 100 : null)}, ...active.map(item => ({symbol: item.symbol, color: colors[comparisons.indexOf(item)], values: data.map(row => item.data.has(row.date) ? (item.data.get(row.date) / item.data.get(baseline.date) - 1) * 100 : null)}))] : [];
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
    if (!series.length || box.width < 100) { c.fillStyle = chartPalette().muted; c.font = '12px Segoe UI'; c.fillText(active.length ? 'No common baseline. Pan or expand the loaded date range.' : 'Comparison data will appear here when available.', 14, 45); return; }
    let low = 0, high = 0;
    for (const item of series) item.values.forEach((value, index) => { if (index >= first && index <= lastVisible && Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); } });
    const pad = (high - low || 1) * .14; low -= pad; high += pad;
    const top = 12, height = box.height - 25, y = value => top + (high - value) / (high - low) * height;
    c.font = '10px Segoe UI';
    for (let i = 0; i < 3; i++) {
      const value = low + (high - low) * i / 2, py = y(value);
      c.strokeStyle = chartPalette().grid; c.beginPath(); c.moveTo(g.left, py); c.lineTo(g.left + g.pw, py); c.stroke(); c.fillStyle = chartPalette().muted; c.fillText(`${fmt(value)}%`, g.left + g.pw + 8, py + 3);
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
  const checks = {terminalGrid:'grid',terminalLastPrice:'lastPrice',terminalCrosshair:'crosshair',terminalWicks:'wicks',terminalBorders:'borders',terminalWatermark:'watermark'};
  const customColors = {background:['Background','surface'],gridColor:['Grid','grid'],textColor:['Axis labels','text'],lineColor:['Line / area','accent'],crosshairColor:['Crosshair','crosshair'],wickUp:['Rising wick','wickUp'],wickDown:['Falling wick','wickDown'],borderColor:['Candle borders','borderColor']};
  const settings = $('terminalSettingsForm');
  settings.innerHTML = `<div class="appearance-layout"><div class="appearance-controls">
    <fieldset><legend>Workspace theme</legend><div class="appearance-presets">${[['system','System'],...Object.entries(window.atlasTheme.palettes).map(([id,p])=>[id,p.name])].map(([id,name])=>`<button type="button" data-appearance-theme="${id}"><i style="background:${window.atlasTheme.palettes[id]?.surface || '#8795ae'};border-color:${window.atlasTheme.palettes[id]?.accent || '#fff'}"></i>${name}</button>`).join('')}</div><p>The theme colors the entire workspace.</p></fieldset>
    <fieldset><legend>Price series</legend><div class="appearance-fields">
    <label class="terminal-setting">Chart type<select id="terminalAppearanceStyle"><option value="candles">Candles</option><option value="bars">OHLC bars</option><option value="line">Line</option><option value="area">Area</option></select></label>
    <label class="terminal-setting">Area opacity<div class="appearance-range"><input id="terminalAreaOpacity" type="range" min="0" max="100" step="1"><output id="terminalAreaOpacityValue" for="terminalAreaOpacity"></output></div></label>
    <label class="terminal-setting">Rising candle / bar<input id="terminalUp" type="color"></label><label class="terminal-setting">Falling candle / bar<input id="terminalDown" type="color"></label>
    <label class="terminal-setting">Candle wicks<input id="terminalWicks" type="checkbox"></label><label class="terminal-setting">Candle borders<input id="terminalBorders" type="checkbox"></label>
    <label class="terminal-setting">Line width<select id="terminalLineWidth">${[1,2,3,4].map(n=>`<option value="${n}">${n} px</option>`).join('')}</select></label></div></fieldset>
    <fieldset><legend>Canvas &amp; labels</legend><div class="appearance-fields">
    ${[['terminalGrid','Grid lines'],['terminalLastPrice','Last price line'],['terminalCrosshair','Crosshair'],['terminalWatermark','Watermark']].map(([id,name])=>`<label class="terminal-setting">${name}<input id="${id}" type="checkbox"></label>`).join('')}
    <label class="terminal-setting">Grid direction<select id="terminalGridDirection"><option value="both">Both</option><option value="horizontal">Horizontal</option><option value="vertical">Vertical</option></select></label><label class="terminal-setting">Grid style<select id="terminalGridStyle"><option value="dotted">Dotted</option><option value="dashed">Dashed</option><option value="solid">Solid</option></select></label>
    <label class="terminal-setting">Axis label size<select id="terminalFontSize">${[10,11,12,13,14].map(n=>`<option value="${n}">${n} px</option>`).join('')}</select></label></div>
    <details id="terminalCustomColors"><summary>Custom chart colors</summary><p>Turn off “Theme” for any color to keep your own choice across themes.</p>
    ${Object.entries(customColors).map(([key,[label]])=>`<div class="terminal-setting"><label for="appearance-${key}">${label}</label><div class="appearance-color"><input type="color" id="appearance-${key}"><label><input type="checkbox" id="follow-${key}" checked> ${key.startsWith('wick') ? 'Candle' : 'Theme'}</label></div></div>`).join('')}</details></fieldset>
    </div><div class="appearance-preview"><canvas id="terminalAppearancePreview" width="360" height="220" aria-label="Sample chart appearance preview"></canvas><strong>Appearance preview</strong><p>Sample prices in your selected chart style. Apply to update your chart. Your settings are saved in this browser.</p></div></div>
    <div class="terminal-dialog-foot"><div><button id="terminalDefaults" type="button">Reset defaults</button><button id="workspaceEditOverview" type="button">Edit overview</button></div><div><button id="terminalCancelSettings" type="button">Cancel</button><button class="terminal-primary" type="submit">Apply</button></div></div>
    <p class="muted">Your preferences are saved on this device in this browser's local storage, including chart appearance, watchlists, indicators, drawings and layout. Use the same browser and site address to restore them. Clearing site data or ending a private browsing session removes saved preferences.</p>
    <div class="terminal-settings-extras"><a id="terminalInterviewPrep" href="./interview-prep.html" target="_blank" rel="noopener" aria-label="Interview prep (opens in a new tab)">Interview prep</a></div>`;
  let draftTheme = window.atlasTheme.preference;
  function draftPalette() { return window.atlasTheme.palettes[draftTheme] || window.atlasTheme.palettes[matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light']; }
  function readSettings() {
    const prefs = {...chartAppearanceDefaults};
    for (const [id,key] of Object.entries(checks)) prefs[key] = $(id).checked;
    prefs.up = $('terminalUp').value; prefs.down = $('terminalDown').value;
    for (const key of Object.keys(customColors)) prefs[key] = $('follow-'+key).checked ? null : $('appearance-'+key).value;
    prefs.areaOpacity = Number($('terminalAreaOpacity').value); prefs.gridDirection = $('terminalGridDirection').value;
    prefs.lineWidth = Number($('terminalLineWidth').value); prefs.fontSize = Number($('terminalFontSize').value); prefs.gridStyle = $('terminalGridStyle').value;
    return prefs;
  }
  function previewSettings() {
    const prefs = readSettings(), palette = chartPalette(prefs,draftPalette());
    settings.querySelectorAll('[data-appearance-theme]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.appearanceTheme === draftTheme)));
    for (const [key,[,field]] of Object.entries(customColors)) {
      const control = $('appearance-'+key), follows = $('follow-'+key).checked;
      control.disabled = follows; if (follows) control.value = palette[field];
    }
    $('terminalAreaOpacityValue').textContent = prefs.areaOpacity + '%';
    const style = $('terminalAppearanceStyle').value;
    const c = $('terminalAppearancePreview').getContext('2d'); c.fillStyle = palette.surface; c.fillRect(0,0,360,220);
    if (prefs.grid) {
      c.strokeStyle=palette.grid;c.lineWidth=1;c.setLineDash(prefs.gridStyle === 'solid' ? [] : prefs.gridStyle === 'dashed' ? [6,4] : [1,4]);
      if(prefs.gridDirection !== 'vertical') for(let y=35;y<200;y+=40){c.beginPath();c.moveTo(10,y);c.lineTo(310,y);c.stroke();}
      if(prefs.gridDirection !== 'horizontal') for(let x=35;x<310;x+=45){c.beginPath();c.moveTo(x,12);c.lineTo(x,195);c.stroke();}
      c.setLineDash([]);
    }
    const points=Array.from({length:12},(_,i)=>[22+i*23,140-i*6+(i%3)*12]);
    if(style==='candles' || style==='bars') {
      points.forEach(([x,y],i)=>{
        const color=i%3 ? prefs.up : prefs.down;
        c.fillStyle=color;c.strokeStyle=style==='bars' ? color : i%3 ? palette.wickUp : palette.wickDown;
        if(prefs.wicks || style==='bars'){c.beginPath();c.moveTo(x,y-15);c.lineTo(x,y+27);c.stroke();}
        if(style==='bars'){c.beginPath();c.moveTo(x-5,y);c.lineTo(x,y);c.moveTo(x,y+15);c.lineTo(x+5,y+15);c.stroke();}
        else {c.fillRect(x-5,y,10,15);if(prefs.borders){c.strokeStyle=palette.borderColor;c.strokeRect(x-5,y,10,15);}}
      });
    } else {
      if(style==='area') {
        const gradient=c.createLinearGradient(0,12,0,195);
        gradient.addColorStop(0,palette.accent+Math.round(prefs.areaOpacity*2.55).toString(16).padStart(2,'0'));gradient.addColorStop(1,palette.accent+'00');
        c.fillStyle=gradient;c.beginPath();c.moveTo(points[0][0],195);points.forEach(([x,y])=>c.lineTo(x,y));c.lineTo(points.at(-1)[0],195);c.closePath();c.fill();
      }
      c.strokeStyle=palette.accent;c.lineWidth=prefs.lineWidth;c.beginPath();points.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.stroke();c.lineWidth=1;
    }
    c.fillStyle=palette.text;c.font=`${prefs.fontSize}px Segoe UI`;c.fillText('125.00',312,80);c.fillText('100.00',312,160);
    if(prefs.crosshair){c.strokeStyle=palette.crosshair;c.setLineDash([4,4]);c.beginPath();c.moveTo(210,12);c.lineTo(210,195);c.moveTo(10,110);c.lineTo(310,110);c.stroke();c.setLineDash([]);}
    if(prefs.lastPrice){c.strokeStyle=prefs.up;c.setLineDash([2,3]);c.beginPath();c.moveTo(10,95);c.lineTo(310,95);c.stroke();c.setLineDash([]);}
    if(prefs.watermark){c.fillStyle=palette.muted;c.font='10px Segoe UI';c.fillText('QUANTSTACK',12,207);}
  }
  function fillSettings(prefs,theme=window.atlasTheme.preference,style=chartType) {
    $('terminalAppearanceStyle').value=style;
    $('terminalAreaOpacity').value=prefs.areaOpacity; $('terminalGridDirection').value=prefs.gridDirection;
    draftTheme=theme;
    for(const [id,key] of Object.entries(checks)) $(id).checked=prefs[key];
    $('terminalUp').value=prefs.up; $('terminalDown').value=prefs.down;
    $('terminalLineWidth').value=prefs.lineWidth; $('terminalFontSize').value=prefs.fontSize; $('terminalGridStyle').value=prefs.gridStyle;
    for(const [key,[,field]] of Object.entries(customColors)){ $('follow-'+key).checked=!prefs[key]; $('appearance-'+key).value=prefs[key] || chartPalette(prefs,draftPalette())[field]; }
    previewSettings();
  }
  settings.oninput = previewSettings;
  settings.querySelectorAll('[data-appearance-theme]').forEach(button=>button.onclick=()=>{draftTheme=button.dataset.appearanceTheme;previewSettings();});
  $('terminalSettings').onclick=()=>{fillSettings(chartAppearance);$('terminalSettingsDialog').showModal();$('terminalSettingsDialog').scrollTop=0;};
  $('terminalDefaults').onclick=()=>fillSettings(chartAppearanceDefaults,'system','candles');
  $('terminalCancelSettings').onclick=()=>$('terminalSettingsDialog').close();
  const profileKey = 'atlas.appearance.presets.v1';
  const profileFormat = 'quantstack-appearance';
  function parseProfile(value) {
    if (!value || value.format !== profileFormat || value.version !== 1) throw Error('Choose a QuantStack appearance JSON file (version 1).');
    if (value.theme !== 'system' && !Object.hasOwn(window.atlasTheme.palettes,value.theme)) throw Error('Unknown workspace theme.');
    if (!['candles','bars','line','area'].includes(value.style)) throw Error('Unknown chart style.');
    return {format:profileFormat,version:1,name:typeof value.name === 'string' ? value.name.trim().slice(0,60) : '',theme:value.theme,style:value.style,appearance:normalizeAppearance(value.appearance,true)};
  }
  let profiles = [];
  try {
    const stored = JSON.parse(localStorage.getItem(profileKey) || '[]');
    if (Array.isArray(stored)) for (const entry of stored.slice(0,20)) {
      try { const profile = parseProfile(entry); if (profile.name && !profiles.some(p=>p.name===profile.name)) profiles.push(profile); } catch {}
    }
  } catch {}
  const manager = document.createElement('fieldset');
  manager.className = 'appearance-profile-manager';
  manager.innerHTML = `<legend>My appearance presets</legend><p>Save the current draft as a reusable preset. Preset saves and deletions are immediate; chart changes take effect with Apply.</p>
    <label for="appearanceProfileName">Preset name</label><div class="appearance-profile-row"><input id="appearanceProfileName" type="text" maxlength="60" placeholder="e.g. Evening analysis"><button id="appearanceProfileSave" type="button">Save / replace</button></div>
    <label for="appearanceProfileList">Saved presets</label><div class="appearance-profile-row"><select id="appearanceProfileList"></select><button id="appearanceProfileLoad" type="button">Load</button><button id="appearanceProfileDelete" type="button">Delete</button></div>
    <div class="appearance-profile-row"><button id="appearanceProfileExport" type="button">Export JSON</button><button id="appearanceProfileImport" type="button">Import JSON</button><input id="appearanceProfileFile" type="file" accept="application/json,.json" hidden></div><p id="appearanceProfileStatus" role="status" aria-live="polite"></p>`;
  settings.querySelector('.appearance-controls').append(manager);
  const sections=document.createElement('nav');sections.className='appearance-sections';sections.setAttribute('aria-label','Chart settings sections');
  const sectionFields=[...settings.querySelectorAll('.appearance-controls>fieldset')];
  ['Themes','Price series','Canvas','My presets'].forEach((label,index)=>{
    const button=document.createElement('button');button.type='button';button.textContent=label;
    const field=sectionFields[index];field.id='appearanceSection'+index;field.tabIndex=-1;button.setAttribute('aria-controls',field.id);
    button.onclick=()=>{field.focus({preventScroll:true});field.scrollIntoView({block:'start'});};sections.append(button);
  });
  settings.prepend(sections);
  const profileStatus = message => { $('appearanceProfileStatus').textContent = message; };
  function currentProfile() { return {format:profileFormat,version:1,name:$('appearanceProfileName').value.trim(),theme:draftTheme,style:$('terminalAppearanceStyle').value,appearance:readSettings()}; }
  function renderProfiles(selectedName = $('appearanceProfileList').value) {
    const list = $('appearanceProfileList'); list.replaceChildren(new Option('Choose a saved preset',''));
    for (const profile of profiles) list.add(new Option(profile.name,profile.name));
    list.value=selectedName;
    $('appearanceProfileLoad').disabled=$('appearanceProfileDelete').disabled=!list.value;
  }
  function storeProfiles(next) {
    try { localStorage.setItem(profileKey,JSON.stringify(next)); profiles=next; return true; }
    catch { profileStatus('Could not save presets. Browser storage may be full or unavailable.'); return false; }
  }
  $('appearanceProfileList').onchange=()=>renderProfiles();
  $('appearanceProfileName').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();$('appearanceProfileSave').click();}};
  $('appearanceProfileSave').onclick=()=>{
    const profile=currentProfile();
    if (!profile.name) { profileStatus('Enter a name for this preset.'); $('appearanceProfileName').focus(); return; }
    const index=profiles.findIndex(item=>item.name===profile.name),next=[...profiles];
    if(index<0 && next.length>=20){profileStatus('You can save up to 20 presets. Delete or replace one first.');return;}
    if(index<0)next.push(profile);else next[index]=profile;
    if(storeProfiles(next)){renderProfiles(profile.name);profileStatus(`Saved “${profile.name}”.`);}
  };
  $('appearanceProfileLoad').onclick=()=>{
    const profile=profiles.find(p=>p.name===$('appearanceProfileList').value);if(!profile)return;
    fillSettings(profile.appearance,profile.theme,profile.style);$('appearanceProfileName').value=profile.name;profileStatus(`Loaded “${profile.name}” into the preview. Choose Apply to use it.`);
  };
  $('appearanceProfileDelete').onclick=()=>{
    const name=$('appearanceProfileList').value;if(!name)return;
    if(storeProfiles(profiles.filter(p=>p.name!==name))){renderProfiles('');profileStatus(`Deleted “${name}”. The chart is unchanged.`);}
  };
  $('appearanceProfileExport').onclick=()=>{
    const profile=currentProfile(),blob=new Blob([JSON.stringify(profile,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
    link.href=url;link.download='QuantStack-appearance.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);profileStatus('Exported the current appearance draft.');
  };
  $('appearanceProfileImport').onclick=()=>$('appearanceProfileFile').click();
  let importVersion=0;
  $('terminalSettingsDialog').addEventListener('close',()=>{importVersion++;});
  $('appearanceProfileFile').onchange=async event=>{
    const file=event.target.files[0],version=++importVersion;event.target.value='';if(!file)return;
    try {
      if(file.size>65536)throw Error('Appearance files must be smaller than 64 KB.');
      const profile=parseProfile(JSON.parse(await file.text()));
      if(version!==importVersion)return;
      fillSettings(profile.appearance,profile.theme,profile.style);$('appearanceProfileName').value=profile.name;
      profileStatus('Imported into the preview. Apply to use it, or Save / replace to keep a named preset.');
    } catch(error) { if(version===importVersion)profileStatus(error instanceof SyntaxError ? 'This file is not valid JSON.' : error.message); }
  };
  renderProfiles();
  settings.onsubmit=event=>{event.preventDefault();Object.assign(chartAppearance,readSettings());chartType=$('terminalAppearanceStyle').value;$('terminalStyle').value=chartType;window.atlasTheme.apply(draftTheme,true);save();draw();$('terminalSettingsDialog').close();notify('Chart appearance saved.');};
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
    const c = output.getContext('2d'); c.scale(scale, scale); c.fillStyle = chartPalette().surface; c.fillRect(0, 0, output.width / scale, output.height / scale);
    c.fillStyle = chartPalette().text; c.font = '600 17px Segoe UI'; c.fillText(`${selected.symbol} · ${interval} · ${$('terminalStyle').selectedOptions[0].text}`, 14, 28);
    c.fillStyle = chartPalette().muted; c.font = '11px Segoe UI'; c.fillText('QUANTSTACK · Stored market data · ' + new Date().toISOString().slice(0, 10), 14, 50);
    let y = 70;
    for (const canvas of canvases) {
      const height = canvas.getBoundingClientRect().height;
      const title = canvas.id === 'chart' ? `${visible()[0]?.date || ''} — ${visible().at(-1)?.date || ''}` : canvas.id === 'comparisonCanvas' ? `${$('comparisonBaseline').textContent} | ${[selected.symbol, ...comparisons.map(item => item.symbol)].join(' / ')}` : canvas.parentElement.querySelector('.indicator-panel-label')?.textContent || 'Study';
      c.fillStyle = chartPalette().muted; c.font = '10px Segoe UI'; c.fillText(title, 14, y + 12, width - 28); y += 26;
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

// Keep the chart useful as the application's landing page.
(() => {
  const key = 'atlas.chart.session';
  const periods = ['1D', '5D', '1M', '6M', '1Y', '5Y', 'MAX', 'CUSTOM'];
  const validDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) &&
    Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
  let restored = false;
  window.restoreChartSession = () => {
    if (restored) return selected?.symbol;
    restored = true;
    let prefs = {};
    try { prefs = JSON.parse(localStorage.getItem(key) || '{}') || {}; } catch {}
    const params = new URLSearchParams(location.search);
    if (params.has('symbol')) prefs = Object.fromEntries(params);
    const intervals = window.ATLAS_STATIC ? ['1d', '1w', '1mo'] : ['1d', '1h', '1w', '1mo'];
    interval = intervals.includes(prefs.interval) ? prefs.interval : '1d';
    period = periods.includes(prefs.period) ? prefs.period : '1Y';
    if (period === 'CUSTOM') {
      if ((prefs.start === '' || validDate(prefs.start)) && (prefs.end === '' || validDate(prefs.end)) &&
          (!prefs.start || !prefs.end || prefs.start <= prefs.end)) {
        $('start').value = prefs.start; $('end').value = prefs.end;
      } else period = '1Y';
    }
    document.querySelectorAll('[data-interval]').forEach(button => button.classList.toggle('active', button.dataset.interval === interval));
    document.querySelectorAll('[data-period]').forEach(button => button.classList.toggle('active', button.dataset.period === period));
    return typeof prefs.symbol === 'string' ? prefs.symbol.toUpperCase() : null;
  };
  const state = () => ({symbol: selected.symbol, interval, period, start: $('start').value, end: $('end').value});
  const load = loadHistory;
  let revision = 0;
  loadHistory = async function() {
    const current = ++revision;
    if (selected) {
      try { localStorage.setItem(key, JSON.stringify(state())); } catch {}
      // Keep an opened share link in sync when the user changes the chart.
      if (new URLSearchParams(location.search).has('symbol')) {
        try { history.replaceState(null, '', location.pathname + '?' + new URLSearchParams(state()) + location.hash); } catch {}
      }
    }
    $('chart').setAttribute('aria-busy', 'true');
    try { await load(); }
    finally { if (current === revision) $('chart').setAttribute('aria-busy', 'false'); }
  };

  const chartUtilities = document.createElement('div');
  chartUtilities.className = 'chart-utilities';
  chartUtilities.innerHTML = '<button type="button" id="chartShare">Share chart link</button><button type="button" id="chartShortcuts">Keyboard shortcuts</button>';
  $('terminalSettingsDialog').querySelector('.terminal-dialog-foot').before(chartUtilities);
  let utilityOpener;
  const dialog = document.createElement('dialog');
  dialog.id = 'chartHomeDialog';
  dialog.innerHTML = `<div class="dialog-head"><h2 id="chartHomeTitle">Chart workspace</h2><button id="chartHomeClose" aria-label="Close">×</button></div><div id="chartHomeContent"></div>`;
  dialog.setAttribute('aria-labelledby', 'chartHomeTitle');
  document.body.append(dialog);
  dialog.addEventListener('close', () => { $('terminalSettingsDialog').showModal(); utilityOpener?.focus(); });
  $('chartHomeClose').onclick = () => dialog.close();
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  $('chartShortcuts').onclick = () => {
    utilityOpener = document.activeElement; $('terminalSettingsDialog').close();
    $('chartHomeTitle').textContent = 'Keyboard shortcuts';
    $('chartHomeContent').innerHTML = `<dl><dt>Search symbols</dt><dd><kbd>/</kbd></dd><dt>Compare symbols</dt><dd><kbd>Alt + C</kbd></dd><dt>Go to date</dt><dd><kbd>Alt + G</kbd></dd><dt>Undo drawing</dt><dd><kbd>Ctrl + Z</kbd></dd><dt>Redo drawing</dt><dd><kbd>Ctrl + Y</kbd></dd><dt>Cancel drawing / close dialog</dt><dd><kbd>Esc</kbd></dd></dl>`;
    dialog.showModal();
  };
  $('chartShare').onclick = () => {
    if (!selected) return;
    utilityOpener = document.activeElement; $('terminalSettingsDialog').close();
    const url = new URL(location.href);
    url.search = new URLSearchParams(state()).toString();
    $('chartHomeTitle').textContent = 'Share chart';
    $('chartHomeContent').innerHTML = `<p>Open this symbol, interval and date range. Drawings and private notes stay in your browser.</p><label for="chartShareUrl">Chart link</label><input id="chartShareUrl" readonly><button id="chartCopyLink">Copy link</button><p id="chartCopyStatus" role="status"></p>`;
    $('chartShareUrl').value = url.href;
    $('chartCopyLink').onclick = async () => {
      try { await navigator.clipboard.writeText(url.href); $('chartCopyStatus').textContent = 'Link copied.'; }
      catch { $('chartShareUrl').focus(); $('chartShareUrl').select(); $('chartCopyStatus').textContent = 'Press Ctrl+C (or Cmd+C) to copy the selected link.'; }
    };
    dialog.showModal();
    $('chartShareUrl').select();
  };
})();
// A labeled, read-only view of the actual workspace for describing UI changes.
(() => {
  'use strict';
  const sections = [
    ['top-toolbar', 'Top toolbar', 'Symbol search, chart interval, chart type and indicator controls.', 'main > header'],
    ['chart-actions', 'Chart action bar', 'Pull data, Compare, Go to date, Snapshot and full-screen controls.', '.terminal-actions'],
    ['left-ribbon', 'Left ribbon — drawing tools', 'Cursor, trend lines, levels, shapes, measurements and drawing settings.', '.workspace-drawing-rail'],
    ['chart-canvas', 'Main chart', 'Price candles, indicators, crosshair, time axis and price scale.', '.canvas-wrap'],
    ['right-panel', 'Right side panel', 'The expanded watchlist, symbol details, notes, objects or analysis library. This is separate from the narrow right ribbon.', '#workspaceSidebar'],
    ['right-top', 'Right ribbon — top', 'Watchlist and other panel buttons, followed by the analysis icons for portfolio, pairs, options and the other tools.', '.workspace-right-rail'],
    ['right-bottom', 'Right ribbon — bottom', 'Chart appearance settings and the workspace theme selector.', '#workspaceSettingsDock'],
    ['quote-card', 'Right panel — quote summary', 'Selected symbol, stored price, daily change and data timestamp at the bottom of the watchlist.', '.overview'],
    ['range-bar', 'Bottom chart bar', 'Date-range shortcuts and chart scale controls, including log and auto scale.', '.range-toolbar'],
    ['analysis-tabs', 'Bottom analysis tabs', 'Stock screener, Strategy tester, Markov, Brownian, Returns & risk, Price bars and Analysis tools. The dock controls sit at the right end.', '.workspace-bottom-tabs'],
    ['analysis-dock', 'Bottom analysis panel', 'The resizable results and controls underneath the chart. It appears when an analysis tab or right-ribbon tool is open.', '#workspaceDock']
  ];
  const button = document.getElementById('workspaceEditOverview');
  const appearance = document.getElementById('terminalSettingsDialog');
  const dialog = document.createElement('dialog');
  dialog.id = 'workspaceOverview';
  dialog.setAttribute('aria-labelledby', 'workspaceOverviewTitle');
  dialog.innerHTML = `<div id="workspaceOverviewRegions"></div><section class="overview-guide"><div class="overview-guide-head"><div><small>WORKSPACE SECTION GUIDE</small><h2 id="workspaceOverviewTitle">Edit overview</h2></div><button id="workspaceOverviewClose" type="button" aria-label="Back to chart appearance">Close</button></div><p>Select a labeled area or choose a section below. Use its name when describing a change.</p><label for="workspaceOverviewSelect">Workspace section</label><select id="workspaceOverviewSelect"></select><h3 id="workspaceOverviewName"></h3><p id="workspaceOverviewDescription"></p><p id="workspaceOverviewVisibility" role="status"></p><label for="workspaceOverviewReference">Name to use in your request</label><input id="workspaceOverviewReference" readonly><button id="workspaceOverviewCopy" type="button">Copy section name</button><p id="workspaceOverviewCopyStatus" role="status"></p><small>This guide labels the workspace. It does not change your layout or apply appearance settings. Close or press Esc to return.</small></section>`;
  document.body.append(dialog);
  const get = id => document.getElementById(id), regions = get('workspaceOverviewRegions');
  let active = 'right-top', frame = 0, savedScroll = 0;
  for (const [key, name, description] of sections) {
    get('workspaceOverviewSelect').add(new Option(name, key));
    const region = document.createElement('button');
    region.type = 'button'; region.className = 'overview-region'; region.dataset.section = key;
    region.setAttribute('aria-label', name + ': ' + description);
    region.setAttribute('aria-pressed', 'false');
    const label = document.createElement('span'); label.textContent = name;
    region.append(label); region.onclick = () => choose(key); regions.append(region);
  }
  function bounds(key, selector) {
    const target = document.querySelector(selector);
    if (!target || target.hidden || !target.getClientRects().length || getComputedStyle(target).visibility === 'hidden') return null;
    if (key === 'right-panel' && target.inert) return null;
    if (key === 'quote-card' && document.getElementById('workspaceSidebar').inert) return null;
    const box = target.getBoundingClientRect();
    let left = Math.max(0, box.left), top = Math.max(0, box.top), right = Math.min(innerWidth, box.right), bottom = Math.min(innerHeight, box.bottom);
    if (key === 'right-top') bottom = Math.min(bottom, get('workspaceSettingsDock').getBoundingClientRect().top);
    return right > left && bottom > top ? {left, top, width: right-left, height: bottom-top} : null;
  }
  function paint() {
    frame = 0;
    if (!dialog.open) return;
    for (const [key,, ,selector] of sections) {
      const region = regions.querySelector(`[data-section="${key}"]`), box = bounds(key, selector);
      region.hidden = !box;
      region.classList.toggle('selected', key === active);
      region.setAttribute('aria-pressed', String(key === active));
      if (box) {
        Object.assign(region.style, {left: box.left+'px', top: box.top+'px', width: box.width+'px', height: box.height+'px'});
        region.classList.toggle('overview-label-left', key.startsWith('right-') || key === 'quote-card');
      }
      if (key === active) get('workspaceOverviewVisibility').textContent = box ? 'Highlighted on your current workspace.' : 'This section is currently collapsed or hidden in this layout. Open it in the workspace to see its position.';
    }
  }
  function choose(key) {
    active = key;const section = sections.find(s => s[0] === key);
    get('workspaceOverviewSelect').value = key;
    get('workspaceOverviewName').textContent = section[1];
    get('workspaceOverviewDescription').textContent = section[2];
    get('workspaceOverviewReference').value = section[1];
    get('workspaceOverviewCopyStatus').textContent = '';
    paint();
  }
  get('workspaceOverviewSelect').onchange = event => choose(event.target.value);
  get('workspaceOverviewCopy').onclick = async () => {
    const name = get('workspaceOverviewReference').value;
    try { await navigator.clipboard.writeText(name); get('workspaceOverviewCopyStatus').textContent = 'Section name copied.'; }
    catch { get('workspaceOverviewReference').focus();get('workspaceOverviewReference').select();get('workspaceOverviewCopyStatus').textContent = 'Press Ctrl+C or use your device’s Copy command to copy the selected name.'; }
  };
  button.onclick = () => { savedScroll = appearance.scrollTop;appearance.close();dialog.showModal();choose(active);get('workspaceOverviewSelect').focus(); };
  get('workspaceOverviewClose').onclick = () => dialog.close();
  dialog.addEventListener('close', () => { if(frame)cancelAnimationFrame(frame);frame=0;appearance.showModal();appearance.scrollTop=savedScroll;button.focus({preventScroll:true}); });
  window.addEventListener('resize', () => { if(dialog.open&&!frame)frame=requestAnimationFrame(paint); });
  choose(active);
})();
