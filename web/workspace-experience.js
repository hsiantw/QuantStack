// Chart-first chrome. Existing controls keep their original data and handlers.
document.addEventListener('DOMContentLoaded', () => {
  'use strict';
  if (document.body.dataset.toolPage || !document.body.classList.contains('workspace')) return;
  const el = id => document.getElementById(id);
  const body = document.body;
  body.classList.add('workspace-enhanced');
  const toolbar = document.querySelector('.workspace-toolbar');
  const actions = document.querySelector('.terminal-actions');
  const headerActions = document.querySelector('.header-actions');
  const chartHeading = document.querySelector('.chart-top');
  const svg = path => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${path}</svg>`;

  // Primary chart actions belong beside the symbol and resolution controls.
  toolbar.append(el('terminalCompare'));
  headerActions.prepend(el('accountButton'));
  actions.prepend(el('usageButton'));
  actions.append(el('toggleScreener'));
  chartHeading.append(actions);
  const quoteLine = document.createElement('span');
  quoteLine.className = 'workspace-quote-inline';
  quoteLine.innerHTML = '<span class="workspace-stored-label">Stored</span><strong></strong><span></span>';
  el('chartTitle').after(quoteLine);
  function syncQuote() {
    quoteLine.querySelector('strong').textContent = `${el('price').textContent} ${el('currency').textContent}`.trim();
    const change = quoteLine.lastElementChild;
    change.textContent = el('change').textContent;
    change.className = el('change').classList.contains('negative') ? 'negative' : el('change').classList.contains('positive') ? 'positive' : '';
    quoteLine.title = el('asof').textContent || 'Latest stored quote in the dataset';
  }
  const quoteObserver = new MutationObserver(syncQuote);
  for (const id of ['price', 'currency', 'change', 'asof']) quoteObserver.observe(el(id), {childList:true, characterData:true, subtree:true});
  syncQuote();
  const watchlist = el('workspaceSideWatchlist');
  const collection = watchlist.querySelector('.watchlist-collection');
  if (collection) watchlist.querySelector('.sidebar-footer').before(collection);
  const addSymbol = el('watchlistAddSymbol');
  addSymbol.title = 'Add a symbol to your watchlist';
  addSymbol.setAttribute('aria-label', 'Add symbol to watchlist');
  watchlist.querySelector('.sidebar-heading').insertBefore(addSymbol, el('watchlistManage'));
  // Keep filtering within reach without permanently occupying the watchlist.
  const filters = document.createElement('details');
  filters.id = 'workspaceWatchlistFilters';
  filters.className = 'workspace-watchlist-filters';
  const filterSummary = document.createElement('summary');
  filterSummary.textContent = 'Filter & sort';
  filters.append(filterSummary);
  watchlist.querySelector('.search').after(filters);
  for (const selector of ['#filters', '.watchlist-controls', '.watchlist-display-tools']) filters.append(watchlist.querySelector(selector));
  try { filters.open = localStorage.getItem('atlas.workspace.watchlistFilters.v1') === 'true'; } catch {}
  filters.addEventListener('toggle', () => {
    try { localStorage.setItem('atlas.workspace.watchlistFilters.v1', String(filters.open)); } catch {}
  });
  function describeFilters() {
    const kind = el('filters').querySelector('.active')?.textContent.trim();
    const color = el('watchlistColor').querySelector('[aria-pressed="true"]')?.getAttribute('aria-label');
    const criteria = [kind !== 'All' ? kind : '', color !== 'All labels' ? color : ''].filter(Boolean);
    filterSummary.textContent = 'Filter & sort' + (criteria.length ? ' · ' + criteria.join(' · ') : '');
  }
  const filterObserver = new MutationObserver(describeFilters);
  filterObserver.observe(el('filters'), {subtree:true,attributes:true,attributeFilter:['class']});
  filterObserver.observe(el('watchlistColor'), {subtree:true,attributes:true,attributeFilter:['aria-pressed']});
  describeFilters();

  const quoteActions = document.createElement('div');
  quoteActions.className = 'workspace-quote-actions';
  for (const [panel, label, path] of [
    ['notes', 'Add a note', '<path d="M4 4h12v5M4 4v16h16v-8M10 14l1-4 7-7 3 3-7 7-4 1Z"/>'],
    ['details', 'Symbol details', '<path d="M4 20V10h4v10m4 0V4h4v16m4 0V8h3v12"/>']
  ]) {
    const button = document.createElement('button');
    button.type = 'button'; button.dataset.quotePanel = panel;
    button.innerHTML = svg(path) + '<span>' + label + '</span>';
    button.onclick = () => document.querySelector(`[data-side-panel="${panel}"]`).click();
    quoteActions.append(button);
  }
  document.querySelector('.workspace-quote-card').append(quoteActions);
  const dashboard = document.createElement('a');
  dashboard.href = '/dashboard'; dashboard.className = 'workspace-dashboard-link';
  dashboard.title = 'Your dashboard'; dashboard.setAttribute('aria-label', 'Your dashboard');
  dashboard.innerHTML = svg('<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>');
  headerActions.prepend(dashboard);
  el('workspaceWatchlistToggle').innerHTML = svg('<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M14 3v18M7 8h4M7 12h4"/>');
  el('terminalCompare').title = 'Compare symbols (Alt+C)';
  for (const id of ['terminalGo', 'terminalSnapshot', 'terminalFullscreen']) {
    el(id).classList.add('workspace-icon-action');
  }
  el('intervals').setAttribute('role', 'group');
  el('intervals').setAttribute('aria-label', 'Chart interval');
  el('periods').setAttribute('role', 'group');
  el('periods').setAttribute('aria-label', 'Visible date range');
  for (const id of ['intervals', 'periods']) {
    const group = el(id);
    const sync = () => group.querySelectorAll('button').forEach(button => {
      button.setAttribute('aria-pressed', String(button.classList.contains('active')));
    });
    new MutationObserver(sync).observe(group, {subtree: true, attributes: true, attributeFilter: ['class']});
    sync();
  }

  const focus = document.createElement('button');
  focus.id = 'workspaceFocus';
  focus.className = 'workspace-icon-action';
  focus.type = 'button';
  focus.innerHTML = svg('<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><rect x="8" y="8" width="8" height="8" rx="1"/>') + '<span>Focus chart</span>';
  actions.insertBefore(focus, el('terminalFullscreen'));
  let focused = false, inertBefore = [];
  const regions = ['.workspace-drawing-rail', '.workspace-sidebar', '.workspace-right-rail', '.workspace-bottom-tabs', '.workspace-dock'];
  function focusChart(enabled) {
    if (focused === enabled) return;
    focused = enabled;
    if (enabled) {
      inertBefore = regions.map(selector => document.querySelector(selector)).filter(Boolean).map(node => [node, node.inert]);
      inertBefore.forEach(([node]) => { node.inert = true; });
      document.querySelectorAll('details[open]').forEach(detail => { if (!detail.closest('dialog')) detail.open = false; });
    } else {
      inertBefore.forEach(([node, inert]) => { node.inert = inert; });
      inertBefore = [];
    }
    body.classList.toggle('workspace-focus', enabled);
    focus.setAttribute('aria-pressed', String(enabled));
    focus.setAttribute('aria-label', enabled ? 'Exit chart focus' : 'Focus chart');
    focus.title = enabled ? 'Exit chart focus (Alt+F or Escape)' : 'Focus chart (Alt+F)';
    focus.querySelector('span').textContent = enabled ? 'Exit focus' : 'Focus chart';
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  }
  focus.setAttribute('aria-pressed', 'false');
  focus.setAttribute('aria-label', 'Focus chart');
  focus.title = 'Focus chart (Alt+F)';
  focus.onclick = () => focusChart(!focused);

  // Keep the chart's frequent actions visible and group occasional controls.
  const options = document.createElement('details');
  options.id = 'workspaceChartOptions'; options.className = 'workspace-chart-options';
  options.innerHTML = '<summary aria-label="Chart options">Chart options <span aria-hidden="true">⌄</span></summary><div class="workspace-chart-options-body"></div>';
  const optionsBody = options.querySelector('div');
  const source = actions.querySelector('.analysis-source');
  if (source) optionsBody.append(source);
  for (const [id, label] of [['terminalGo','Go to date'],['terminalSnapshot','Save chart image'],['terminalFullscreen','Full screen'],['usageButton','Collection activity']]) {
    const button = el(id); if (!button) continue;
    button.classList.remove('workspace-icon-action');
    button.setAttribute('aria-label', label);
    if (id === 'usageButton') button.textContent = label;
    optionsBody.append(button);
  }
  actions.append(options);
  optionsBody.addEventListener('click', event => { if (event.target.closest('button')) options.open = false; });
  document.addEventListener('pointerdown', event => { if (!options.contains(event.target)) options.open = false; });
  options.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); options.open=false; options.querySelector('summary').focus(); }
  });
  const sourceButton = document.createElement('button');
  sourceButton.type='button'; sourceButton.id='workspaceCandleSource'; sourceButton.title='Change candle source';
  quoteLine.append(sourceButton);
  const updateSource=()=>{sourceButton.textContent=el('analysisChartSource')?.value || 'Automatic source';};
  sourceButton.onclick=()=>{options.open=true;el('analysisChartSource')?.focus();};
  el('analysisChartSource')?.addEventListener('change', updateSource);updateSource();
  window.addEventListener('chart-history-loaded',updateSource);

  // Empty charts give the next useful action, rather than a dead end.
  const empty = el('chartEmpty');
  empty.addEventListener('click', event => {
    const action = event.target.closest('[data-empty-action]')?.dataset.emptyAction;
    if(action==='symbol') el('workspaceSymbol')?.click();
    else if(action==='database') window.openMarketDatabase?.();
    else if(action==='retry') loadHistory();
  });
  // Opening an analysis panel should always reveal it, including from a deep link.
  window.addEventListener('workspace:dock', event => { if (event.detail && focused) focusChart(false); });

  const help = document.createElement('button');
  help.id = 'workspaceHelp';
  help.type = 'button';
  help.textContent = '?';
  help.setAttribute('aria-label', 'Keyboard shortcuts');
  help.title = 'Keyboard shortcuts (?)';
  headerActions.insertBefore(help, el('workspaceWatchlistToggle'));
  help.onclick = () => { help.focus(); el('chartShortcuts').click(); };
  document.addEventListener('keydown', event => {
    const editing = event.target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])');
    if (editing || document.querySelector('dialog[open]') || event.ctrlKey || event.metaKey || event.repeat) return;
    if (event.altKey && event.key.toLowerCase() === 'f') {
      event.preventDefault(); focusChart(!focused); focus.focus();
    } else if (!event.altKey && event.key === '?') {
      event.preventDefault(); help.click();
    } else if (event.key === 'Escape' && focused && !document.querySelector('details[open], #chartScaleMenu:not([hidden]), #chartContextMenu:not([hidden])')) {
      event.preventDefault(); event.stopImmediatePropagation(); focusChart(false); focus.focus();
    }
  }, true);

  // A small-screen side panel behaves as a dismissible sheet.
  const backdrop = document.createElement('button');
  backdrop.id = 'workspacePanelBackdrop';
  backdrop.type = 'button';
  backdrop.tabIndex = -1;
  backdrop.setAttribute('aria-label', 'Close side panel');
  body.append(backdrop);
  backdrop.onclick = () => { el('workspaceWatchlistToggle').click(); el('workspaceWatchlistToggle').focus(); };

  // Navigate the currently filtered/sorted watchlist without changing its order.
  el('assets').addEventListener('keydown', event => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key) || event.ctrlKey || event.altKey || event.metaKey) return;
    const current = event.target.closest('button[data-symbol]');
    if (!current) return;
    const entries = [...el('assets').querySelectorAll('button[data-symbol]')];
    const index = entries.indexOf(current);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? entries.length - 1 : Math.max(0, Math.min(entries.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)));
    event.preventDefault();
    entries[next]?.focus();
  });
});
