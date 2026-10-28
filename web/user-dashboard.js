(() => {
  'use strict';
  const el = id => document.getElementById('dashboard' + id);
  const keys = ['atlas.saved', 'atlas.watchlists.v1', 'atlas.ideaNotes.v1', 'atlas.chart.session', 'atlas.workspace.recentSymbols.v1', 'atlas.interview-prep.v1'];
  const sections = ['assessment-reflection', ...Array.from({length:8}, (_,i) => 'section-' + (i+1)), 'algorithms', 'section-9', 'assessment-brief', 'assessment-strategy'];
  const colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink'];
  const intervals = {'1m':'1 minute','3m':'3 minutes','5m':'5 minutes','15m':'15 minutes','30m':'30 minutes','1h':'1 hour','2h':'2 hours','4h':'4 hours','1d':'Daily','1w':'Weekly','1mo':'Monthly'};
  const object = value => value && typeof value === 'object' && !Array.isArray(value);
  const symbol = value => typeof value === 'string' && /^[A-Z0-9^][A-Z0-9.^=-]{0,29}$/.test(value);
  const symbols = value => [...new Set(Array.isArray(value) ? value.filter(symbol) : [])];
  let user = null, notes = [], epoch = 0, controller;
  function node(tag, className, text) {
    const result = document.createElement(tag); result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
  }
  function chartURL(ticker, session = {}) {
    if (!symbol(ticker)) return '/workspace/';
    const params = new URLSearchParams({symbol:ticker});
    if (Object.hasOwn(intervals, session.interval)) params.set('interval', session.interval);
    if (['1D','5D','1M','6M','1Y','5Y','MAX'].includes(session.period)) params.set('period', session.period);
    const date = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0,10) === value;
    if (session.period === 'CUSTOM' && date(session.start) && date(session.end) && session.start <= session.end) {
      params.set('period', 'CUSTOM'); params.set('start', session.start); params.set('end', session.end);
    }
    return '/workspace/?' + params;
  }
  function empty(id, message, action) {
    const panel=node('div','dashboard-empty');panel.append(node('p','',message));
    if (action) {
      const control=node(action.href?'a':'button','dashboard-empty-action',action.label);
      if(action.href)control.href=action.href;else{control.type='button';control.onclick=action.run;}
      panel.append(control);
    }
    el(id).replaceChildren(panel);
  }
  function item(title, meta, href) {
    const row = node(href ? 'a' : 'article', 'dashboard-item');
    if (href) row.href = href;
    const copy = node('div', ''); copy.append(node('span', 'dashboard-item-title', title), node('span', 'dashboard-item-meta', meta));
    row.append(copy); return row;
  }
  function parse(entries, key, fallback, valid, errors) {
    if (entries[key] == null) return fallback;
    try { const value = JSON.parse(entries[key]); if (!valid(value)) throw new Error(); return value; }
    catch { errors.push(key); return fallback; }
  }
  function renderNotes() {
    const query = el('NoteSearch').value.trim().toLowerCase();
    const found = notes.filter(n => [n.title,n.body,n.symbol].join(' ').toLowerCase().includes(query));
    el('Notes').replaceChildren();
    for (const note of found.slice(0,6)) {
      const date = Number.isFinite(Date.parse(note.updatedAt)) ? new Date(note.updatedAt).toLocaleDateString(undefined, {month:'short',day:'numeric'}) : '';
      const remaining = Array.isArray(note.items) ? note.items.filter(i => object(i) && !i.done).length : 0;
      const meta = [note.symbol, date, remaining ? `${remaining} tasks remaining` : '', typeof note.body === 'string' ? note.body.slice(0,100) : ''].filter(Boolean).join(' · ');
      const href = el('Source').value === 'account' ? '/account#notebooks' : '/ideas?' + new URLSearchParams({note:note.id});
      el('Notes').append(item(note.title || 'Untitled note', meta, href));
    }
    if (!found.length) empty('Notes', query ? 'No notes match your search.' : 'Save an idea in your notebook to see it here.', query ? {label:'Clear search',run:()=>{el('NoteSearch').value='';renderNotes();el('NoteSearch').focus();}} : {label:'Write your first idea',href:'/ideas'});
  }
  function render(entries, unavailable = false) {
    const errors = [];
    const favorites = symbols(parse(entries, keys[0], [], Array.isArray, errors));
    const watch = parse(entries, keys[1], {lists:[]}, v => object(v) && Array.isArray(v.lists), errors);
    const lists = watch.lists.filter(l => object(l) && typeof l.id === 'string' && typeof l.name === 'string' && Array.isArray(l.symbols));
    notes = parse(entries, keys[2], [], Array.isArray, errors).filter(n => object(n) && typeof n.id === 'string' && typeof n.title === 'string' && !n.archived && !n.trashed)
      .sort((a,b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
    const session = parse(entries, keys[3], {}, object, errors);
    const recent = symbols([session.symbol, ...symbols(parse(entries, keys[4], [], Array.isArray, errors))]);
    const prep = parse(entries, keys[5], {}, object, errors);
    const reviewed = new Set(Array.isArray(prep.reviewed) ? prep.reviewed.filter(s => sections.includes(s)) : []);
    for (const [id, count, key] of [['SavedCount',favorites.length,keys[0]],['WatchlistCount',lists.length,keys[1]],['NotesCount',notes.length,keys[2]],['PrepCount',`${reviewed.size} / 13`,keys[5]]]) {
      el(id).textContent = unavailable || errors.includes(key) ? '—' : count;
    }
    el('Continue').href = chartURL(session.symbol, session);
    el('ContinueSymbol').textContent = symbol(session.symbol) ? 'Continue ' + session.symbol : 'Open charts';
    el('ContinueMeta').textContent = symbol(session.symbol) ? [session.symbol, intervals[session.interval] || '', 'Last chart in this workspace'].filter(Boolean).join(' · ') : 'Search a symbol and start exploring your next idea.';
    el('RecentCharts').replaceChildren();
    for (const ticker of recent.slice(0,6)) el('RecentCharts').append(item(ticker, ticker === session.symbol ? 'Last chart' : 'Recently explored', chartURL(ticker, ticker === session.symbol ? session : {})));
    if (!recent.length) empty('RecentCharts', 'Open a chart to start your recent history.', {label:'Find a symbol',href:'/workspace/'});
    if (favorites.length) {
      const group = item('Saved symbols', `${favorites.length} saved`);
      const chips = node('div', 'dashboard-chips');
      for (const ticker of favorites.slice(0,12)) { const a = node('a', '', ticker); a.href = chartURL(ticker); chips.append(a); }
      group.firstElementChild.append(chips); el('RecentCharts').append(group);
    }
    el('Watchlists').replaceChildren();
    for (const list of lists.slice(0,8)) {
      const members = colors.includes(list.color) ? symbols(Object.entries(object(watch.labels) ? watch.labels : {}).filter(([,color]) => color === list.color).map(([ticker]) => ticker)) : symbols(list.symbols);
      const row = item(list.name, `${members.length} symbols${list.color ? ' · ' + list.color + ' labels' : ''}`);
      const chips = node('div', 'dashboard-chips');
      for (const ticker of members.slice(0,6)) { const a = node('a', '', ticker); a.href = chartURL(ticker); chips.append(a); }
      if (members.length > 6) chips.append(node('span', '', '+' + (members.length-6)));
      row.firstElementChild.append(chips); el('Watchlists').append(row);
    }
    if (!lists.length) empty('Watchlists', 'Create a watchlist in Charts to organize your symbols.', {label:'Open chart watchlists',href:'/workspace/'});
    renderNotes();
    el('PrepProgress').value = reviewed.size;
    el('PrepProgress').textContent = `${reviewed.size} of 13`;
    el('PrepSummary').textContent = `${reviewed.size} of 13 sections reviewed. ${reviewed.size === 13 ? 'Ready for another practice round.' : 'Continue with your next unreviewed section.'}`;
    el('PrepLink').href = '/workspace/interview-prep.html' + (reviewed.size < 13 ? '#' + sections.find(s => !reviewed.has(s)) : '');
    if (unavailable) {
      for (const id of ['RecentCharts','Watchlists','Notes']) empty(id, 'Workspace unavailable. Your saved work has not been changed.', {label:'Try again',run:refresh});
      el('PrepSummary').textContent = 'Progress unavailable.';
    } else {
      for (const [key, id] of [[keys[1],'Watchlists'],[keys[2],'Notes']]) if (errors.includes(key)) empty(id, 'Saved data could not be read. Your original data has been preserved.');
      if (errors.includes(keys[5])) el('PrepSummary').textContent = 'Saved progress could not be read.';
    }
    return errors.length ? 'Some saved data could not be read. Your original data has been preserved.' : '';
  }
  function browserWorkspace() {
    const entries = {};
    try { keys.forEach(key => { entries[key] = localStorage.getItem(key); }); return render(entries); }
    catch { render({}, true); return 'Browser storage is unavailable. Enable storage and try Refresh.'; }
  }
  function identity(value) {
    user = value;
    el('Greeting').textContent = user ? `Welcome back, ${user.username}` : 'Your dashboard';
    el('Identity').textContent = user ? 'Your charts, research and preparation, together.' : 'Your personal workspace in this browser. Sign in to view your saved account too.';
    el('Source').querySelector('[value="account"]').disabled = !user;
    el('Account').textContent = user ? 'Your account' : 'Sign in';
    el('Account').href = user ? '/account' : '/login';
    el('AccountSummary').textContent = user ? `Signed in as ${user.username}. Your saved account and this browser can contain different work.` : 'Sign in to view the workspace saved to your account.';
  }
  async function refresh() {
    const run = ++epoch;
    controller?.abort(); controller = new AbortController();
    const signal = controller.signal, requested = el('Source').value, previousId = user?.id;
    let message = '';
    if (requested === 'browser') message = browserWorkspace(); else render({}, true);
    el('SourceHint').textContent = requested === 'browser' ? 'Showing the workspace saved in this browser.' : 'Previewing your saved account. To edit this copy, load it from Account in Charts first.';
    el('Status').textContent = requested === 'account' ? 'Loading saved account…' : message;
    el('Refresh').setAttribute('aria-busy', 'true');
    async function get(path) {
      const response = await fetch('/workspace/api/account/' + path, {credentials:'same-origin',cache:'no-store',signal});
      if (!response.ok) throw new Error('Account unavailable');
      return response.json();
    }
    try {
      const session = await get('session');
      if (run !== epoch) return;
      const current = object(session.user) && typeof session.user.username === 'string' ? session.user : null;
      identity(current);
      if (requested === 'account' && (!current || (previousId && previousId !== current.id))) {
        el('Source').value = 'browser';
        message = browserWorkspace();
        el('SourceHint').textContent = 'Showing the workspace saved in this browser.';
        el('Status').textContent = message || 'Your account session changed. Showing this browser’s work.';
        return;
      }
      if (requested === 'account') {
        const saved = await get('workspace');
        if (run !== epoch) return;
        if (!saved.user || saved.user.id !== current.id || !object(saved.entries)) throw new Error('Account changed');
        message = render(saved.entries);
      }
      el('Status').textContent = message;
    } catch (error) {
      if (run !== epoch || error.name === 'AbortError') return;
      if (requested === 'account') render({}, true);
      identity(null);
      el('Status').textContent = [message, 'Account is unavailable. You can still use your browser workspace.'].filter(Boolean).join(' ');
    } finally { if (run === epoch) el('Refresh').removeAttribute('aria-busy'); }
  }
  function resetAccount() {
    identity(null); el('Source').value = 'browser'; el('NoteSearch').value = ''; refresh();
  }
  el('Source').addEventListener('change', () => { el('NoteSearch').value = ''; refresh(); });
  el('Refresh').addEventListener('click', refresh);
  el('NoteSearch').addEventListener('input', renderNotes);
  window.addEventListener('storage', event => {
    if (event.key === 'quantstack.accountChanged' || event.key === null) resetAccount();
    else if ((keys.includes(event.key) && el('Source').value === 'browser') || event.key === 'quantstack.accountWorkspaceChanged') refresh();
  });
  window.addEventListener('focus', refresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  refresh();
})();
