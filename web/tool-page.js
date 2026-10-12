(() => {
  'use strict';
  const key = document.body.dataset.toolPage;
  const registry = JSON.parse(document.getElementById('toolRegistry').textContent);
  const [title, group, description] = registry[key];
  const el = id => document.getElementById(id);
  document.body.classList.add('tool-page');
  const shell = document.createElement('div');
  shell.id = 'toolShell';
  shell.innerHTML = `<a class="tool-skip" href="#toolContent">Skip to tool</a>
    <header class="tool-header"><a href="/" class="tool-brand">Q<span>↗</span> <b>QuantStack</b></a><nav aria-label="Primary"><a href="/workspace/">Charts</a><a href="/screener">Screener</a><a href="/markets">Markets</a><a href="/portfolio">Portfolio</a></nav><div class="tool-header-actions"><label class="tool-theme-label">Theme <select id="toolTheme" aria-label="Theme"><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option><option value="midnight">Midnight</option><option value="forest">Forest</option><option value="paper">Paper</option></select></label><a href="/account">Account</a><button id="toolMenuToggle" aria-expanded="false" aria-controls="toolNavigation">All tools ☰</button></div></header>
    <aside id="toolNavigation" class="tool-navigation"><a class="tool-home" href="/">← Home</a><input id="toolFinder" type="search" aria-label="Find a tool" placeholder="Find a tool…"><div id="toolNavigationLinks"></div><p id="toolNoMatch" hidden>No matching tools.</p><a class="tool-chart-link" href="/workspace/">Open chart workspace ↗</a></aside>
    <section class="tool-page-heading"><p class="tool-breadcrumb">${group.toUpperCase()} <span>/</span> ${title}</p><div><h1>${title}</h1><a id="toolOpenChart" href="/workspace/">Open in charts ↗</a></div><p>${description}</p></section>
    <section class="tool-context" aria-label="Analysis context"><form id="toolSymbolForm"><label for="toolSymbol">Symbol</label><input id="toolSymbol" name="symbol" list="toolSymbols" placeholder="AAPL" required autocomplete="off" maxlength="40"><datalist id="toolSymbols"></datalist><button>Load</button></form><div id="toolIntervals"></div><div id="toolDates"></div><p id="toolContextStatus" role="status">Loading stored data…</p></section>
    <div id="toolContent" tabindex="-1"><p id="toolLoading" role="status">Loading ${title.toLowerCase()}…</p></div>`;
  document.body.prepend(shell);
  const grouped = [...new Set(Object.values(registry).map(value => value[1]))];
  for (const name of grouped) {
    const section = document.createElement('section');
    const heading = document.createElement('h2'); heading.textContent = name; section.append(heading);
    for (const [id, [label, category]] of Object.entries(registry)) {
      if (name !== category) continue;
      const link = document.createElement('a'); link.href = '/' + id; link.textContent = label;
      if (id === key) link.setAttribute('aria-current', 'page'); section.append(link);
    }
    el('toolNavigationLinks').append(section);
  }
  shell.querySelectorAll('.tool-header nav a').forEach(link => {
    if (link.getAttribute('href') === '/' + key) link.setAttribute('aria-current', 'page');
  });
  const toggle = el('toolMenuToggle');
  function closeMenu(focus = false) { document.body.classList.remove('tool-navigation-open'); toggle.setAttribute('aria-expanded', 'false'); if (focus) toggle.focus(); }
  toggle.onclick = () => { const open = document.body.classList.toggle('tool-navigation-open'); toggle.setAttribute('aria-expanded', String(open)); if (open) el('toolFinder').focus(); };
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(true); });
  document.addEventListener('click', event => { if (!event.target.closest('#toolNavigation, #toolMenuToggle')) closeMenu(); });
  document.addEventListener('focusin', event => { if (!event.target.closest('#toolNavigation, #toolMenuToggle')) closeMenu(); });
  el('toolFinder').oninput = event => {
    let count = 0;
    for (const section of el('toolNavigationLinks').children) {
      let matches = 0;
      for (const link of section.querySelectorAll('a')) { link.hidden = !link.textContent.toLowerCase().includes(event.target.value.toLowerCase().trim()); if (!link.hidden) matches++; }
      section.hidden = !matches; count += matches;
    }
    el('toolNoMatch').hidden = count > 0;
  };
  window.atlasTheme.bind(el('toolTheme'));
  const universal = new Set(['screener', 'markets', 'rotation', 'ideas']);
  function updateContext() {
    el('toolContextStatus').textContent = selected ? `${selected.symbol} · ${rows.length.toLocaleString()} stored bars · ${interval}` : 'No stored price history. Load data from the chart workspace.';
    const params = new URLSearchParams({symbol: selected?.symbol || '', interval, period, start: el('start').value, end: el('end').value});
    el('toolOpenChart').href = '/workspace/?' + params + '#tool=' + key;
  }
  const originalLoad = loadHistory;
  loadHistory = async function() { try { return await originalLoad(); } finally { updateContext(); } };
  if (universal.has(key)) shell.querySelector('.tool-context').hidden = true;
  else {
    el('toolIntervals').append(el('intervals'));
    el('toolDates').append(document.querySelector('.dates'));
  }
  el('toolSymbolForm').onsubmit = event => {
    event.preventDefault();
    const symbol = el('toolSymbol').value.trim().toUpperCase();
    if (!assets.some(asset => asset.symbol === symbol)) { el('toolContextStatus').textContent = 'Symbol is not in your stored universe. Choose an available symbol.'; return; }
    const params = new URLSearchParams(location.search); params.set('symbol', symbol);
    params.set('interval', interval); params.set('period', period);
    location.assign('/' + key + '?' + params);
  };
  window.initializeToolPage = () => {
    el('toolLoading').remove();
    const dock = el('workspaceDock');
    el('toolContent').append(el('error'), dock);
    window.openChartResearch(key);
    const pane = [...dock.querySelectorAll('.workspace-dock-pane')].find(node => !node.hidden);
    if (pane) { pane.removeAttribute('role'); pane.removeAttribute('aria-labelledby'); }
    if (!universal.has(key)) {
      const options=document.createDocumentFragment();
      for (const asset of assets) { const option = document.createElement('option'); option.value = asset.symbol; option.label = asset.name || asset.symbol; options.append(option); }
      el('toolSymbols').append(options);
    }
    el('toolSymbol').value = selected?.symbol || '';
    updateContext();
    if (key === 'screener') {
      el('screenerSearch').setAttribute('aria-label', 'Search stocks');
      el('workspaceScreenerFilters').open = false;
      document.querySelector('.workspace-screen-toolbar').append(el('refreshScreener'));
      const tools = document.createElement('details'); tools.className = 'tool-advanced';
      tools.innerHTML = '<summary>Advanced table options</summary>';
      const enhancements = document.querySelector('#screenerContent > .screener-enhancements');
      if (enhancements) { enhancements.before(tools); tools.append(enhancements); }
      window.initializeScreenerPage?.();
    }
  };
  window.startToolPage = async () => {
    if (universal.has(key)) {
      // Universe tools do not need an unrelated symbol's full chart history.
      try { assets = await api('/api/symbols'); renderAssets(); }
      catch (error) { showError(error.message); }
    } else await init();
    window.initializeToolPage();
  };
})();
