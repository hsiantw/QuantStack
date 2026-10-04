// Arrange the existing chart controls into a compact, responsive workspace.
// Collector controls are loaded only on demand; no background polling.
document.addEventListener('DOMContentLoaded', () => {
  const button = document.createElement('button');
  button.id = 'schedulerButton';
  button.className = 'secondary scheduler-add';
  button.textContent = '+ Add symbols';
  button.title = 'Add stocks and crypto to the local collection schedule';
  document.querySelector('.sidebar-heading').append(button);
  const dialog = document.createElement('dialog');
  dialog.id = 'schedulerDialog';
  dialog.setAttribute('aria-labelledby', 'schedulerTitle');
  dialog.innerHTML = `<form id="schedulerForm">
    <div class="dialog-head"><div><span class="eyebrow">LOCAL DATA COLLECTION</span>
    <h2 id="schedulerTitle">Add symbols</h2></div><button type="button" id="schedulerClose" class="secondary" aria-label="Close scheduler">Close</button></div>
    <p class="muted">Queue stocks and crypto for the next scheduled collection. Charts appear after a successful download.</p>
    <label>Asset type<select id="schedulerKind"><option value="stocks">Stocks / ETFs</option><option value="crypto">Crypto · USD pairs</option></select></label>
    <label>Symbols<textarea id="schedulerSymbols" rows="3" maxlength="4000" placeholder="AAPL, MSFT, 2330.TW" aria-describedby="schedulerHelp"></textarea></label>
    <p id="schedulerHelp" class="muted">Separate provider tickers with commas or spaces. Up to 100 per save. Crypto symbols such as BTC become BTC-USD.</p>
    <p id="schedulerPreview" class="scheduler-preview" role="status" aria-live="polite"></p>
    <div class="scheduler-profile"><strong id="schedulerProfile">Low background usage</strong><span id="schedulerProfileDetails">1 download worker · 2-second pauses · 10-minute collection budgets</span></div>
    <p class="muted">When installed, Windows tasks collect daily at 09:00 and hourly while logged in and on AC power. Large universes take multiple cycles.</p>
    <div class="scheduler-progress"><strong>Latest collection reports</strong><button type="button" id="schedulerRefresh" class="secondary">Refresh status</button><p id="schedulerProgress" class="muted">No reports yet.</p></div>
    <details><summary id="schedulerCount">Configured symbols</summary><p id="schedulerList" class="scheduler-list"></p></details>
    <p id="schedulerStatus" role="status" aria-live="polite"></p>
    <div class="dialog-actions"><button id="schedulerSubmit" type="submit" class="secondary">Save to local schedule</button></div>
    </form>`;
  document.body.append(dialog);
  const field = id => document.getElementById(id);
  let configured = new Set(), available = false, saving = false;
  function preview() {
    const raw = field('schedulerSymbols').value.trim().toUpperCase();
    const tokens = raw ? raw.split(/[\s,;]+/) : [];
    const symbols = [...new Set(tokens.map(symbol => field('schedulerKind').value === 'crypto' && !symbol.endsWith('-USD') ? symbol + '-USD' : symbol))];
    const invalid = symbols.filter(symbol => !/^[A-Z0-9^][A-Z0-9.^=-]{0,29}$/.test(symbol));
    const added = symbols.filter(symbol => !configured.has(symbol));
    const error = invalid.length ? 'Check ticker format: ' + invalid.join(', ') : new Set(tokens).size > 100 ? 'Add at most 100 symbols per save.' : '';
    field('schedulerPreview').textContent = error || (symbols.length ? `${added.length} new · ${symbols.length - added.length} already configured — ${symbols.join(', ')}` : 'Paste symbols to preview your additions, or save to apply low usage.');
    field('schedulerSymbols').setCustomValidity(error);
    field('schedulerSubmit').disabled = !available || saving || Boolean(error);
  }
  const showConfig = data => {
    configured = new Set(data.symbols);
    field('schedulerCount').textContent = `${data.symbols.length} explicitly configured symbols`;
    field('schedulerList').textContent = data.symbols.join(', ');
    field('schedulerProfile').textContent = data.background_usage === 'low' ? 'Low background usage active' : 'Save to apply low background usage';
    field('schedulerProfileDetails').textContent = data.hourly_workers == null ? '1 download worker · 2-second pauses · 10-minute collection budgets' : `${data.hourly_workers} hourly worker · ${data.hourly_pause_seconds}s pauses · ${data.hourly_budget_minutes}m hourly / ${data.daily_budget_minutes ?? 10}m daily budget`;
    field('schedulerProgress').textContent = ['hourly', 'daily'].map(kind => {
      const report = data.collection?.[kind];
      if (!report) return `${kind}: no collection report yet`;
      const stamp = report.updated_at ? new Date(report.updated_at).toLocaleString() : 'unknown time';
      return `${kind}: ${(report.status || 'unknown').replaceAll('_', ' ')} · ${report.succeeded ?? 0} succeeded · ${report.pending ?? 0} pending · ${report.failed ?? 0} failed · ${stamp}`;
    }).join('\n');
    preview();
  };
  const endpoint = new URL('./api/local-scheduler', location.href);
  async function request(options) {
    const response = await fetch(endpoint, options);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || 'Open the local QuantStack app to manage collection. Hosted snapshots are read-only.');
    }
    return response.json();
  }
  button.onclick = async () => {
    dialog.showModal();
    available = false;
    field('schedulerSubmit').disabled = true;
    field('schedulerStatus').textContent = 'Loading collector settings…';
    try {
      showConfig(await request());
      available = true;
      field('schedulerStatus').textContent = 'Saving also applies the low background profile. Existing universe collection is preserved.';
      preview();
      field('schedulerSymbols').focus();
    } catch (error) { field('schedulerStatus').textContent = error.message; }
  };
  field('schedulerClose').onclick = () => dialog.close();
  field('schedulerSymbols').oninput = preview;
  field('schedulerRefresh').onclick = async () => {
    field('schedulerRefresh').disabled = true;
    try { showConfig(await request()); }
    catch (error) { field('schedulerStatus').textContent = error.message; }
    finally { field('schedulerRefresh').disabled = false; }
  };
  field('schedulerKind').onchange = () => {
    field('schedulerSymbols').placeholder = field('schedulerKind').value === 'crypto' ? 'BTC, ETH, SOL' : 'AAPL, MSFT, 2330.TW';
    preview();
  };
  field('schedulerForm').onsubmit = async event => {
    event.preventDefault();
    saving = true;
    field('schedulerSubmit').disabled = true;
    try {
      const data = await request({method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symbols: field('schedulerSymbols').value, kind: field('schedulerKind').value, background_usage: 'low'})});
      showConfig(data);
      field('schedulerSymbols').value = '';
      field('schedulerStatus').textContent = `${data.added.length} new symbols queued. Low usage saved. Tickers will be checked by the provider during collection; unsupported tickers may fail.`;
    } catch (error) { field('schedulerStatus').textContent = error.message; }
    finally { saving = false; preview(); }
  };
});

(() => {
  'use strict';
  const find = selector => document.querySelector(selector);
  const byId = id => document.getElementById(id);
  const main = find('main'), header = find('main > header'), sidebar = find('body > aside');
  const chartPanel = find('.chart-panel'), toolbar = find('.terminal-toolbar');
  const ranges = find('.range-toolbar'), actions = find('.header-actions');
  document.body.classList.add('workspace');
  sidebar.classList.add('workspace-sidebar');
  sidebar.setAttribute('aria-label', 'Watchlist and selected symbol');

  const brand = find('.brand');
  brand.classList.add('workspace-brand');
  brand.setAttribute('aria-label', 'QuantStack home');
  brand.title = 'QuantStack';
  brand.innerHTML = '<span class="logo">Q<span>↗</span></span>';
  header.prepend(brand);
  header.children[1].classList.add('workspace-original-heading');
  const symbolButton = document.createElement('button');
  symbolButton.id = 'workspaceSymbol';
  symbolButton.className = 'workspace-symbol';
  symbolButton.innerHTML = '<span aria-hidden="true">⌕</span><strong>Symbol</strong>';
  symbolButton.title = 'Find a symbol in the watchlist';
  symbolButton.setAttribute('aria-label', 'Search for a symbol');
  brand.after(symbolButton);

  const rail = document.createElement('nav');
  rail.className = 'workspace-drawing-rail';
  rail.setAttribute('aria-label', 'Chart drawing tools');
  rail.append(byId('drawingTools'));
  main.append(rail);
  for (const button of rail.querySelectorAll('button')) {
    button.setAttribute('aria-label', button.title || button.textContent);
  }
  byId('magnet').textContent = '∩';
  byId('magnet').classList.add('workspace-magnet');

  const icons = {
    cursor: '<path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="3"/>',
    trend: '<path d="m5 19 14-14"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="5" r="2"/>',
    ray: '<path d="m5 19 16-16M15 3h6v6"/><circle cx="5" cy="19" r="2"/>',
    horizontal: '<path d="M3 12h18"/><circle cx="12" cy="12" r="2"/>',
    vertical: '<path d="M12 3v18"/><circle cx="12" cy="12" r="2"/>',
    rectangle: '<rect x="4" y="5" width="16" height="14" rx="1"/>',
    fib: '<path d="M4 5h16M4 10h12M4 14h16M4 19h9"/>',
    measure: '<path d="m3 16 13-13 5 5L8 21zM7 12l3 3M11 8l3 3M15 4l3 3"/>',
    magnet: '<path d="M5 4v10a7 7 0 0 0 14 0V4h-4v10a3 3 0 0 1-6 0V4zM5 8h4m6 0h4"/>',
    undoDrawing: '<path d="m8 4-5 5 5 5M3 9h11a6 6 0 0 1 0 12"/>',
    redoDrawing: '<path d="m16 4 5 5-5 5m5-5H10a6 6 0 0 0 0 12"/>',
    clearDrawings: '<path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/>',
    watchlist: '<rect x="4" y="3" width="16" height="18" rx="1"/><path d="M8 8h8M8 12h8M8 16h5"/>',
    details: '<path d="M4 20V10h4v10m4 0V4h4v16m4 0V8"/>',
    notes: '<path d="M14 3H4v18h16V11M10 14l1-5 8-8 4 4-8 8z"/>',
    objects: '<path d="m12 3 10 5-10 5L2 8zm-10 9 10 5 10-5M2 17l10 5 10-5"/>',
    tools: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><path d="M14 17h7m-3.5-3.5v7"/>',
    settings: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" fill="currentColor"/><circle cx="15" cy="12" r="2" fill="currentColor"/><circle cx="9" cy="18" r="2" fill="currentColor"/>',
    expand: '<path d="m9 5 7 7-7 7"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
  };
  const icon = name => `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.details}</svg>`;
  for (const button of byId('chartTypes').querySelectorAll('button')) {
    const candles = button.dataset.type === 'candles';
    button.setAttribute('aria-label', candles ? 'Candles' : 'Line chart');
    button.title = button.getAttribute('aria-label');
    button.innerHTML = `${icon(candles ? 'details' : 'trend')}<span>${candles ? 'Candles' : 'Line'}</span>`;
    // The original chart-type handler uses the clicked element's dataset.
    button.querySelectorAll('*').forEach(child => { child.style.pointerEvents = 'none'; });
  }
  for (const button of rail.querySelectorAll('button')) {
    const label = button.title.replace(/ \(.*\)/, '');
    button.innerHTML = `${icon(button.dataset.tool || button.id)}<span class="workspace-tool-label">${label}</span>`;
  }
  const expandTools = document.createElement('button');
  expandTools.id = 'workspaceExpandTools';
  expandTools.className = 'workspace-expand-tools';
  expandTools.innerHTML = `${icon('expand')}<span class="workspace-tool-label">Drawing tools</span>`;
  expandTools.setAttribute('aria-controls', 'drawingTools');
  rail.prepend(expandTools);
  function setToolsExpanded(expanded) {
    document.body.classList.toggle('workspace-tools-expanded', expanded);
    expandTools.setAttribute('aria-expanded', String(expanded));
    expandTools.setAttribute('aria-label', expanded ? 'Collapse drawing tools' : 'Expand drawing tools');
    expandTools.title = expandTools.getAttribute('aria-label');
    try { localStorage.setItem('atlas.workspace.toolsExpanded', String(expanded)); } catch {}
  }
  let toolsExpanded = false;
  try { toolsExpanded = localStorage.getItem('atlas.workspace.toolsExpanded') === 'true'; } catch {}
  setToolsExpanded(toolsExpanded);
  expandTools.onclick = () => setToolsExpanded(expandTools.getAttribute('aria-expanded') !== 'true');

  const datePicker = document.createElement('details');
  datePicker.className = 'workspace-date-picker';
  datePicker.innerHTML = '<summary title="Choose a custom date range" aria-label="Custom date range">▦</summary>';
  const dates = find('.dates');
  byId('start').setAttribute('aria-label', 'Chart start date');
  byId('end').setAttribute('aria-label', 'Chart end date');
  datePicker.append(dates);
  byId('periods').after(datePicker);
  byId('apply').addEventListener('click', () => {
    if (!byId('start').value || !byId('end').value || byId('start').value <= byId('end').value) datePicker.open = false;
  });

  toolbar.classList.add('workspace-toolbar');
  for (const separator of toolbar.querySelectorAll('.tool-separator')) separator.remove();
  header.insertBefore(toolbar, actions);
  byId('studySettingsButton').textContent = '⚙';
  byId('studySettingsButton').setAttribute('aria-label', 'Study settings');
  byId('indicatorLibrary').textContent = 'ƒ Indicators';
  const quickStudies = document.createElement('details');
  quickStudies.className = 'workspace-quick-studies';
  quickStudies.innerHTML = '<summary title="Quick indicators">Studies <span aria-hidden="true">⌄</span></summary><div></div>';
  byId('indicatorTools').prepend(quickStudies);
  for (const button of byId('indicatorTools').querySelectorAll('[data-indicator]')) quickStudies.lastElementChild.append(button);

  const sidebarToggle = document.createElement('button');
  sidebarToggle.id = 'workspaceWatchlistToggle';
  sidebarToggle.className = 'secondary workspace-watchlist-toggle';
  sidebarToggle.textContent = '☷';
  sidebarToggle.title = 'Show or hide watchlist';
  sidebarToggle.setAttribute('aria-label', 'Show or hide watchlist');
  sidebarToggle.setAttribute('aria-controls', 'workspaceSidebar');
  actions.append(sidebarToggle);
  const compact = matchMedia('(max-width: 900px)');
  function watchlistVisible() {
    return compact.matches ? document.body.classList.contains('workspace-watchlist-open') : !document.body.classList.contains('workspace-watchlist-hidden');
  }
  function updateWatchlist() {
    sidebarToggle.setAttribute('aria-expanded', String(watchlistVisible()));
    sidebar.inert = !watchlistVisible();
    for (const button of document.querySelectorAll('[data-side-panel]')) {
      button.classList.toggle('active', watchlistVisible() && button.dataset.sidePanel === activeSidePanel);
      button.setAttribute('aria-expanded', String(watchlistVisible() && button.dataset.sidePanel === activeSidePanel));
    }
    requestAnimationFrame(() => typeof draw === 'function' && draw());
  }
  sidebarToggle.addEventListener('click', () => {
    document.body.classList.toggle(compact.matches ? 'workspace-watchlist-open' : 'workspace-watchlist-hidden');
    updateWatchlist();
  });
  compact.addEventListener('change', updateWatchlist);
  symbolButton.addEventListener('click', () => {
    selectSidePanel('watchlist');
    document.body.classList.remove('workspace-watchlist-hidden');
    document.body.classList.toggle('workspace-watchlist-open', compact.matches);
    updateWatchlist();
    byId('search').focus(); byId('search').select();
  });
  find('.sidebar-heading').textContent = 'Watchlist';
  byId('search').placeholder = 'Search symbols';
  byId('search').setAttribute('aria-label', 'Search watchlist symbols');
  find('.search > span').setAttribute('aria-hidden', 'true');
  find('.list-heading').lastElementChild.textContent = 'LAST / CHANGE';
  const quote = find('.overview');
  quote.classList.add('workspace-quote-card');
  find('.sidebar-footer').before(quote);
  byId('assets').addEventListener('click', event => {
    if (compact.matches && event.target.closest('[data-symbol]')) {
      document.body.classList.remove('workspace-watchlist-open'); updateWatchlist();
    }
  });

  // Keep a narrow selector rail visible even when the right panel is closed.
  sidebar.id = 'workspaceSidebar';
  const watchlistPane = document.createElement('div');
  watchlistPane.className = 'workspace-side-pane workspace-watchlist-pane';
  watchlistPane.id = 'workspaceSideWatchlist';
  watchlistPane.append(...Array.from(sidebar.children));
  sidebar.append(watchlistPane);
  const sidePanes = {watchlist: watchlistPane};
  let activeSidePanel = 'watchlist';
  for (const [key, title] of [['details', 'Symbol details'], ['notes', 'Symbol notes'], ['objects', 'Object tree'], ['tools', 'Analysis library']]) {
    const pane = document.createElement('section');
    pane.className = 'workspace-side-pane';
    pane.id = `workspaceSide${key}`;
    pane.hidden = true;
    pane.innerHTML = `<div class="sidebar-heading">${title}</div>`;
    sidePanes[key] = pane;
    sidebar.append(pane);
  }
  sidePanes.details.innerHTML += '<p class="workspace-panel-help">Price and performance for the selected symbol. Select a symbol in your watchlist to inspect it.</p>';
  sidePanes.notes.innerHTML += '<label class="workspace-notes-label" for="workspaceNotes">Notes for <strong id="workspaceNotesSymbol">this symbol</strong></label><textarea id="workspaceNotes" placeholder="Write your analysis, levels to watch, or a trading idea…"></textarea><p id="workspaceNotesStatus" class="workspace-panel-help" role="status">Notes are saved in this browser, separately for each symbol.</p>';
  sidePanes.objects.innerHTML += '<p class="workspace-panel-help">Select an object to edit its style, lock it, or remove it.</p>';
  const rightRail = document.createElement('nav');
  rightRail.className = 'workspace-right-rail';
  rightRail.setAttribute('aria-label', 'Workspace panels');
  for (const [key, label] of [['watchlist', 'Watchlist'], ['details', 'Symbol details'], ['notes', 'Symbol notes'], ['objects', 'Object tree'], ['tools', 'Analysis library']]) {
    const button = document.createElement('button');
    button.dataset.sidePanel = key;
    button.innerHTML = icon(key);
    button.title = label;
    button.setAttribute('aria-label', label);
    button.setAttribute('aria-controls', sidePanes[key].id);
    button.onclick = () => {
      if (activeSidePanel === key && watchlistVisible()) sidebarToggle.click();
      else {
        selectSidePanel(key);
        document.body.classList.remove('workspace-watchlist-hidden');
        document.body.classList.toggle('workspace-watchlist-open', compact.matches);
        updateWatchlist();
      }
    };
    rightRail.append(button);
  }
  const settingsSlot = document.createElement('div');
  settingsSlot.id = 'workspaceSettingsDock';
  rightRail.append(settingsSlot);
  document.body.append(rightRail);
  function selectSidePanel(key) {
    activeSidePanel = key;
    for (const [name, pane] of Object.entries(sidePanes)) pane.hidden = name !== key;
    if (key === 'details') sidePanes.details.querySelector('.sidebar-heading').after(quote);
    else watchlistPane.querySelector('.sidebar-footer').before(quote);
    updateWatchlist();
  }
  for (const pane of Object.values(sidePanes)) {
    const close = document.createElement('button');
    close.className = 'workspace-panel-close';
    close.innerHTML = icon('close');
    close.setAttribute('aria-label', 'Close side panel');
    close.onclick = () => { sidebarToggle.click(); rightRail.querySelector(`[data-side-panel="${activeSidePanel}"]`).focus(); };
    pane.querySelector('.sidebar-heading').append(close);
  }
  let notesSymbol = '';
  function loadNotes() {
    notesSymbol = byId('symbol').textContent;
    byId('workspaceNotesSymbol').textContent = notesSymbol;
    try { byId('workspaceNotes').value = localStorage.getItem(`atlas.notes.${notesSymbol}`) || ''; } catch { byId('workspaceNotes').value = ''; }
    byId('workspaceNotesStatus').textContent = 'Notes are saved in this browser, separately for each symbol.';
  }
  byId('workspaceNotes').oninput = () => {
    try {
      localStorage.setItem(`atlas.notes.${notesSymbol}`, byId('workspaceNotes').value);
      byId('workspaceNotesStatus').textContent = 'Saved in this browser';
    } catch { byId('workspaceNotesStatus').textContent = 'Browser storage is unavailable. Keep a copy of your notes.'; }
  };
  loadNotes();

  const resizer = document.createElement('div');
  resizer.id = 'workspaceSidebarResize';
  resizer.className = 'workspace-sidebar-resize';
  resizer.tabIndex = 0;
  resizer.setAttribute('role', 'separator');
  resizer.setAttribute('aria-label', 'Watchlist width');
  resizer.setAttribute('aria-orientation', 'vertical');
  resizer.setAttribute('aria-valuemin', '220');
  resizer.setAttribute('aria-valuemax', '420');
  sidebar.prepend(resizer);
  let sidebarWidth = 286, resizing = false;
  try { sidebarWidth = Number(localStorage.getItem('atlas.workspace.watchlistWidth')) || sidebarWidth; } catch {}
  function setSidebarWidth(value, persist = false) {
    sidebarWidth = Math.max(220, Math.min(420, window.innerWidth - 360, value));
    document.body.style.setProperty('--workspace-sidebar-width', `${sidebarWidth}px`);
    resizer.setAttribute('aria-valuenow', String(Math.round(sidebarWidth)));
    if (persist) try { localStorage.setItem('atlas.workspace.watchlistWidth', String(sidebarWidth)); } catch {}
  }
  setSidebarWidth(sidebarWidth);
  resizer.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    event.preventDefault(); resizing = true; resizer.setPointerCapture(event.pointerId);
    document.body.classList.add('workspace-resizing');
  });
  resizer.addEventListener('pointermove', event => { if (resizing) setSidebarWidth(window.innerWidth - 46 - event.clientX); });
  const finishResize = () => {
    if (!resizing) return;
    resizing = false; document.body.classList.remove('workspace-resizing'); setSidebarWidth(sidebarWidth, true);
  };
  resizer.addEventListener('pointerup', finishResize);
  resizer.addEventListener('pointercancel', finishResize);
  resizer.addEventListener('lostpointercapture', finishResize);
  resizer.addEventListener('dblclick', () => setSidebarWidth(286, true));
  resizer.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
    event.preventDefault(); setSidebarWidth(event.key === 'Home' ? 286 : sidebarWidth + (event.key === 'ArrowLeft' ? 10 : -10), true);
  });

  const titleRow = find('.chart-top > div');
  titleRow.append(byId('ohlc'));
  chartPanel.append(ranges);
  if (!byId('chartScaleControls')) {
    const scaleSlot = document.createElement('div'); scaleSlot.id = 'chartScaleControls'; ranges.append(scaleSlot);
  }
  const caption = find('.chart-caption');
  caption.classList.add('workspace-chart-caption');

  const drawingPopover = document.createElement('details');
  drawingPopover.id = 'workspaceDrawingSettings';
  drawingPopover.className = 'workspace-drawing-settings';
  drawingPopover.innerHTML = '<summary title="Drawing settings and objects" aria-label="Drawing settings and objects">⚙</summary>';
  drawingPopover.append(find('.drawing-editor'));
  rail.append(drawingPopover);
  // Keep the date-range controls clickable while editing a selected drawing.
  function positionDrawingSettings() {
    const rangeTop = ranges.getBoundingClientRect().top;
    const editor = drawingPopover.querySelector('.drawing-editor');
    editor.style.bottom = `${Math.max(43, innerHeight - rangeTop + 8)}px`;
    editor.style.maxHeight = `${Math.max(100, rangeTop - 62)}px`;
  }
  drawingPopover.addEventListener('toggle', positionDrawingSettings);
  new ResizeObserver(positionDrawingSettings).observe(chartPanel);
  window.addEventListener('resize', positionDrawingSettings);
  new MutationObserver(() => {
    if (byId('drawingSelection').textContent !== 'New drawing') drawingPopover.open = true;
  }).observe(byId('drawingSelection'), {childList: true});

  const bottom = document.createElement('div');
  bottom.className = 'workspace-bottom-tabs';
  bottom.innerHTML = '<div class="workspace-tab-list" role="tablist" aria-label="Analysis panels"><button id="workspaceOpenScreener" role="tab" aria-controls="workspaceScreenerPanel">Stock screener</button><button id="workspaceOpenStrategy" role="tab" aria-controls="workspaceStrategyPanel">Strategy tester</button><button id="workspaceOpenMarkov" role="tab" aria-controls="workspaceMarkovPanel">Markov analysis</button><button id="workspaceOpenBrownian" role="tab" aria-controls="workspaceBrownianPanel">Brownian motion</button><button id="workspaceOpenRisk" role="tab" aria-controls="workspaceRiskPanel">Returns &amp; risk</button><button id="workspaceOpenData" role="tab" aria-controls="workspaceDataPanel">Price bars</button></div><button id="workspaceOpenObjects">Drawing objects</button><span class="workspace-data-status"><i></i> Local market data</span><div class="workspace-dock-actions"><button id="workspaceMaximizeDock" title="Maximize panel" aria-label="Maximize panel" aria-pressed="false">⛶</button><button id="workspaceCloseDock" title="Collapse panel" aria-label="Collapse bottom panel">⌄</button></div>';
  main.append(bottom);
  byId('workspaceOpenObjects').addEventListener('click', () => {
    rightRail.querySelector('[data-side-panel="objects"]').click();
  });
  sidePanes.objects.append(byId('drawingObjects'));
  byId('drawingObjects').open = true;

  const dock = document.createElement('section');
  dock.id = 'workspaceDock';
  dock.className = 'workspace-dock';
  dock.hidden = true;
  dock.innerHTML = '<div id="workspaceDockResize" class="workspace-dock-resize" role="separator" tabindex="0" aria-label="Analysis panel height" aria-orientation="horizontal" aria-valuemin="160" aria-valuemax="600"></div><div id="workspaceScreenerPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenScreener"></div><div id="workspaceStrategyPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenStrategy" hidden></div><div id="workspaceMarkovPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenMarkov" hidden></div><div id="workspaceBrownianPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenBrownian" hidden></div><div id="workspaceRiskPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenRisk" hidden></div><div id="workspaceDataPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenData" hidden><div class="workspace-data-body"></div></div>';
  main.append(dock);
  dock.insertAdjacentHTML('beforeend', '<div id="workspaceToolsPanel" class="workspace-dock-pane" role="tabpanel" aria-labelledby="workspaceOpenTools" hidden></div>');
  bottom.querySelector('[role="tablist"]').insertAdjacentHTML('beforeend', '<button id="workspaceOpenTools" role="tab" aria-controls="workspaceToolsPanel">Analysis tools</button>');
  byId('workspaceDataPanel').firstElementChild.append(find('.stats'), find('.table-panel'), find('main > footer'));
  const usage = document.createElement('button');
  usage.id = 'workspaceUsage'; usage.className = 'secondary'; usage.textContent = 'API usage';
  find('#workspaceDataPanel .table-top').append(usage);
  byId('workspaceUsage').onclick = () => byId('usageButton').click();

  // Reuse the screener's filters and loading behavior inside a non-modal dock.
  const screener = byId('stockScreener'), screenerToggle = byId('toggleScreener');
  const wasExpanded = screenerToggle.getAttribute('aria-expanded') === 'true';
  if (screenerToggle.getAttribute('aria-expanded') === 'true') screenerToggle.click();
  byId('workspaceScreenerPanel').append(screener);
  const screenFilters = document.createElement('details');
  screenFilters.id = 'workspaceScreenerFilters';
  screenFilters.innerHTML = '<summary>Filters &amp; saved screens</summary>';
  const screenSearch = find('.screener-search');
  const screenToolbar = document.createElement('div');
  screenToolbar.className = 'workspace-screen-toolbar';
  screenToolbar.append(screenSearch);
  byId('screenerContent').prepend(screenToolbar);
  screenToolbar.after(screenFilters);
  screenToolbar.after(find('.screener-presets'));
  for (const selector of ['.screener-coverage', '.screener-filters', '.screener-rule-box', '.screener-saved']) screenFilters.append(find(selector));
  const filtersToggle = document.createElement('button');
  filtersToggle.className = 'secondary';
  filtersToggle.textContent = 'Filters';
  filtersToggle.setAttribute('aria-controls', screenFilters.id);
  filtersToggle.setAttribute('aria-expanded', 'false');
  filtersToggle.onclick = () => { screenFilters.open = !screenFilters.open; };
  screenFilters.addEventListener('toggle', () => filtersToggle.setAttribute('aria-expanded', String(screenFilters.open)));
  screenToolbar.append(filtersToggle);
  let activeDock = null;
  const dockTabs = {screener: byId('workspaceOpenScreener'), strategy: byId('workspaceOpenStrategy'), markov: byId('workspaceOpenMarkov'), brownian: byId('workspaceOpenBrownian'), risk: byId('workspaceOpenRisk'), data: byId('workspaceOpenData'), tools: byId('workspaceOpenTools')};
  function showDock(name) {
    const hadFocus = dock.contains(document.activeElement);
    activeDock = name;
    dock.hidden = !name;
    document.body.classList.toggle('workspace-dock-open', !!name);
    if (!name) {
      document.body.classList.remove('workspace-dock-maximized');
      byId('workspaceMaximizeDock').setAttribute('aria-pressed', 'false');
      byId('workspaceMaximizeDock').setAttribute('aria-label', 'Maximize panel');
      byId('workspaceMaximizeDock').title = 'Maximize panel';
    }
    byId('workspaceScreenerPanel').hidden = name !== 'screener';
    byId('workspaceDataPanel').hidden = name !== 'data';
    byId('workspaceStrategyPanel').hidden = name !== 'strategy';
    byId('workspaceMarkovPanel').hidden = name !== 'markov';
    byId('workspaceBrownianPanel').hidden = name !== 'brownian';
    byId('workspaceRiskPanel').hidden = name !== 'risk';
    byId('workspaceToolsPanel').hidden = name !== 'tools';
    for (const [key, button] of Object.entries(dockTabs)) {
      button.setAttribute('aria-selected', String(key === name));
      button.classList.toggle('active', key === name);
      button.tabIndex = key === (name || 'screener') ? 0 : -1;
    }
    byId('workspaceCloseDock').disabled = byId('workspaceMaximizeDock').disabled = !name;
    const expanded = screenerToggle.getAttribute('aria-expanded') === 'true';
    if (expanded !== (name === 'screener')) screenerToggle.click();
    if (hadFocus) (dockTabs[name] || dockTabs.screener).focus({preventScroll: true});
    window.dispatchEvent(new CustomEvent('workspace:dock', {detail: name}));
  }
  window.ChartWorkspace = {openDock(name) {
    if (!(name in dockTabs)) return;
    document.body.classList.remove('workspace-watchlist-open'); updateWatchlist(); showDock(name);
  }};
  function syncScreener() {
    const expanded = screenerToggle.getAttribute('aria-expanded') === 'true';
    if (expanded && activeDock !== 'screener') showDock('screener');
    if (!expanded && activeDock === 'screener') showDock(null);
    byId('collapseScreener').textContent = expanded ? 'Collapse' : 'Open';
  }
  new MutationObserver(syncScreener).observe(screenerToggle, {attributes: true, attributeFilter: ['aria-expanded']});
  for (const [key, button] of Object.entries(dockTabs)) button.onclick = () => showDock(activeDock === key ? null : key);
  bottom.querySelector('[role="tablist"]').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const keys = Object.keys(dockTabs), index = keys.findIndex(key => dockTabs[key] === document.activeElement);
    const next = event.key === 'Home' ? keys[0] : event.key === 'End' ? keys.at(-1) : keys[(index + (event.key === 'ArrowRight' ? 1 : keys.length - 1)) % keys.length];
    showDock(next); dockTabs[next].focus();
  });
  byId('workspaceCloseDock').onclick = () => { showDock(null); dockTabs.screener.focus(); };
  byId('workspaceMaximizeDock').onclick = () => {
    const maximized = document.body.classList.toggle('workspace-dock-maximized');
    byId('workspaceMaximizeDock').setAttribute('aria-pressed', String(maximized));
    byId('workspaceMaximizeDock').setAttribute('aria-label', maximized ? 'Restore panel size' : 'Maximize panel');
    byId('workspaceMaximizeDock').title = maximized ? 'Restore panel size' : 'Maximize panel';
  };
  let dockHeight = 320;
  try { dockHeight = Number(localStorage.getItem('atlas.workspace.dockHeight')) || dockHeight; } catch {}
  const dockResize = byId('workspaceDockResize');
  function setDockHeight(height, persist = false) {
    const maximum = Math.max(160, window.innerHeight - 250);
    dockHeight = Math.round(Math.max(160, Math.min(maximum, height)));
    document.body.style.setProperty('--workspace-dock-height', `${dockHeight}px`);
    dockResize.setAttribute('aria-valuenow', String(dockHeight));
    dockResize.setAttribute('aria-valuemax', String(maximum));
    if (persist) try { localStorage.setItem('atlas.workspace.dockHeight', String(dockHeight)); } catch {}
  }
  let resizingDock = false;
  dockResize.onpointerdown = event => {
    if (event.button !== 0) return;
    event.preventDefault();
    resizingDock = true; dockResize.setPointerCapture(event.pointerId);
    document.body.classList.add('workspace-dock-resizing');
  };
  dockResize.onpointermove = event => {
    if (resizingDock) setDockHeight(window.innerHeight - event.clientY);
  };
  const finishDockResize = () => {
    if (!resizingDock) return;
    resizingDock = false; document.body.classList.remove('workspace-dock-resizing'); setDockHeight(dockHeight, true);
  };
  dockResize.onpointerup = dockResize.onpointercancel = dockResize.onlostpointercapture = finishDockResize;
  dockResize.ondblclick = () => setDockHeight(320, true);
  dockResize.onkeydown = event => {
    if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); setDockHeight(event.key === 'Home' ? 160 : event.key === 'End' ? window.innerHeight : dockHeight + (event.key === 'ArrowUp' ? 20 : -20), true);
  };
  setDockHeight(dockHeight);
  window.addEventListener('resize', () => { setDockHeight(dockHeight); setSidebarWidth(sidebarWidth); });
  showDock(wasExpanded ? 'screener' : null);
  new MutationObserver(() => {
    symbolButton.querySelector('strong').textContent = byId('symbol').textContent;
    symbolButton.title = `${byId('symbol').textContent} · Search symbols`;
    loadNotes();
  }).observe(byId('symbol'), {childList: true});

  document.addEventListener('pointerdown', event => {
    for (const detail of [datePicker, quickStudies, drawingPopover]) {
      if (detail.open && !detail.contains(event.target) && !event.target.closest('#workspaceOpenObjects')) detail.open = false;
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    datePicker.open = quickStudies.open = drawingPopover.open = false;
    if (dock.contains(event.target) && !find('dialog[open]')) showDock(null);
    if (compact.matches) { document.body.classList.remove('workspace-watchlist-open'); updateWatchlist(); }
  });
  document.addEventListener('keydown', event => {
    if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey || find('dialog[open]')) return;
    const active = document.activeElement;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(active.tagName) || active.isContentEditable) return;
    event.preventDefault(); event.stopImmediatePropagation(); symbolButton.click();
  }, true);
  updateWatchlist();
})();
