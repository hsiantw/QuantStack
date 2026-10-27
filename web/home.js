(() => {
  'use strict';
  const toggle = document.getElementById('exploreToggle');
  const menu = document.getElementById('destinationMenu');
  const chart = (key) => key ? '/' + key : '/workspace/';
  const groups = {
    Products: [
      ['↗', 'Charts', chart(), 'Your complete chart workspace'],
      ['◎', 'Stock screener', chart('screener')],
      ['⌁', 'Strategy tester', chart('strategy')],
      ['▥', 'Portfolios', chart('portfolio')],
      ['◇', 'Options lab', chart('options')],
      ['◴', 'Returns & risk', chart('risk')]
    ],
    Markets: [
      ['◉', 'Market overview', chart('markets')],
      ['↻', 'Sector rotation', chart('rotation')],
      ['▤', 'Company fundamentals', chart('fundamentals')],
      ['▥', 'Volume & liquidity', chart('liquidity')],
      ['≋', 'Stored market data', chart('data')]
    ],
    Research: [
      ['◎', 'Crypto address scanner', chart('crypto-scanner')],
      ['✎', 'Idea notebook', chart('ideas')],
      ['▤', 'Research journal', chart('journal')],
      ['⌁', 'Pairs analysis', chart('pairs')],
      ['↗', 'Forecast lab', chart('forecast')],
      ['◇', 'Markov analysis', chart('markov')],
      ['≋', 'Monte Carlo simulations', chart('brownian')],
      ['▥', 'Strategy comparison', chart('compare')]
    ],
    More: [
      ['▦', 'Your dashboard', '/dashboard', 'Charts, saved work and interview prep'],
      ['♙', 'Your account', '/account'],
      ['→', 'Sign in', '/login'],
      ['◇', 'Project architecture', '/workspace/architecture.html'],
      ['▤', 'Interview preparation', '/workspace/interview-prep.html']
    ]
  };
  function close(focus = false) {
    menu.hidden = true; toggle.setAttribute('aria-expanded', 'false');
    if (focus) toggle.focus();
  }
  function render(group) {
    menu.replaceChildren();
    if (group) {
      const head = document.createElement('div'); head.className = 'menu-title';
      const back = document.createElement('button'); back.textContent = '‹'; back.setAttribute('aria-label', 'Back to destinations'); back.onclick = () => { render(); menu.querySelector('button').focus(); };
      const title = document.createElement('strong'); title.textContent = group;
      const exit = document.createElement('button'); exit.textContent = '×'; exit.setAttribute('aria-label', 'Close navigation'); exit.onclick = () => close(true);
      head.append(back, title, exit); menu.append(head);
      const label = document.createElement('div'); label.className = 'menu-label'; label.textContent = group === 'More' ? 'YOUR QUANTSTACK' : 'EXPLORE ' + group.toUpperCase(); menu.append(label);
      for (const [icon, title, href, description] of groups[group]) {
        const link = document.createElement('a'); link.className = 'menu-row'; link.href = href;
        const glyph = document.createElement('span'); glyph.className = 'menu-icon'; glyph.textContent = icon; glyph.setAttribute('aria-hidden', 'true');
        const text = document.createElement('span'); text.textContent = title;
        if (description) { const small = document.createElement('small'); small.textContent = description; text.append(small); }
        link.append(glyph, text); menu.append(link);
      }
    } else {
      for (const [group, icon] of [['Products', '◇'], ['Markets', '◉'], ['Research', '⌁'], ['More', '•••']]) {
        const button = document.createElement('button'); button.className = 'menu-row';
        button.innerHTML = `<span class="menu-icon" aria-hidden="true">${icon}</span><span>${group}</span><span class="arrow" aria-hidden="true">›</span>`;
        button.onclick = () => { render(group); menu.querySelector('button').focus(); }; menu.append(button);
      }
    }
  }
  toggle.onclick = () => {
    if (!menu.hidden) return close();
    render(); menu.hidden = false; toggle.setAttribute('aria-expanded', 'true');
  };
  toggle.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown') { event.preventDefault(); if (menu.hidden) toggle.click(); menu.querySelector('button').focus(); }
  });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !menu.hidden) close(true); });
  document.addEventListener('click', event => { if (!event.composedPath().includes(toggle.parentElement)) close(); });
  document.addEventListener('focusin', event => { if (!event.target.closest('.destination')) close(); });
  const search = document.getElementById('symbolSearch');
  const suggestions = document.createElement('datalist'); suggestions.id = 'homeSymbolSuggestions';
  search.after(suggestions); search.setAttribute('list', suggestions.id);
  search.setAttribute('autocomplete', 'off'); search.setAttribute('spellcheck', 'false');
  search.placeholder = 'Search symbol or company'; search.setAttribute('aria-label', 'Search symbol or company');
  let catalog = [], catalogRequest = null;
  function suggest() {
    search.setCustomValidity('');
    const term = search.value.trim().toUpperCase();
    const matches = catalog.filter(asset => asset.symbol.includes(term) || String(asset.name || '').toUpperCase().includes(term));
    matches.sort((a,b) => Number(b.symbol.startsWith(term)) - Number(a.symbol.startsWith(term)) || a.symbol.localeCompare(b.symbol));
    suggestions.replaceChildren(...matches.slice(0,12).map(asset => {
      const option = document.createElement('option'); option.value = asset.symbol; option.label = asset.name || asset.symbol; return option;
    }));
  }
  // Fetch only when someone uses search; custom shortcuts remain immediately usable.
  search.addEventListener('focus', () => {
    if (catalogRequest) return;
    catalogRequest = fetch('/workspace/api/symbols').then(response => response.ok ? response.json() : []).then(items => {
      catalog = Array.isArray(items) ? items.filter(item => item && typeof item.symbol === 'string' && /^[A-Z0-9^][A-Z0-9.^=-]{0,29}$/.test(item.symbol)) : [];
      suggest();
    }).catch(() => { catalogRequest = null; });
  });
  search.addEventListener('input', suggest);
  search.form.addEventListener('submit', event => {
    const value = search.value.trim();
    const named = catalog.find(asset => String(asset.name || '').toLowerCase() === value.toLowerCase());
    search.value = named?.symbol || value.toUpperCase().replace(/^([A-Z0-9]+)\/USD$/, '$1-USD');
    if (!/^[A-Z0-9^][A-Z0-9.^=-]{0,29}$/.test(search.value)) {
      event.preventDefault(); search.setCustomValidity('Choose a suggested company or enter a ticker such as AAPL or BTC-USD.'); search.reportValidity();
    }
  });
})();

(() => {
  'use strict';
  const key = 'atlas.home.shortcuts.v1', limit = 48;
  const colors = new Set(['blue', 'violet', 'green', 'amber', 'rose', 'slate']);
  const defaults = [
    {id: 'finviz', name: 'Finviz', url: 'https://finviz.com/', color: 'green'},
    {id: 'yahoo', name: 'Yahoo Finance', url: 'https://finance.yahoo.com/', color: 'violet'},
    {id: 'tradingview', name: 'TradingView', url: 'https://www.tradingview.com/', color: 'blue'},
    {id: 'investing', name: 'Investing.com', url: 'https://www.investing.com/', color: 'amber'},
    {id: 'fred', name: 'FRED', url: 'https://fred.stlouisfed.org/', color: 'slate'},
    {id: 'coingecko', name: 'CoinGecko', url: 'https://www.coingecko.com/', color: 'rose'}
  ];
  const el = id => document.getElementById(id);
  const grid = el('shortcutGrid'), dialog = el('shortcutDialog');
  let items = [], customizing = false, editing = null, removed = null, drag = null;
  let readable = true;
  const status = message => { el('shortcutStatus').textContent = message; };

  function address(value) {
    const input = value.trim();
    if (!input || input.length > 2048) throw Error('Enter a website address.');
    const relative = input.startsWith('/') && !input.startsWith('//');
    const url = new URL(relative ? input : /^[a-z][a-z\d+.-]*:/i.test(input) ? input : 'https://' + input, location.origin);
    if (!['https:', 'http:'].includes(url.protocol) || !url.hostname || url.username || url.password) {
      throw Error('Use an http or https website address without embedded sign-in details.');
    }
    return relative ? url.pathname + url.search + url.hash : url.href;
  }

  function load() {
    try {
      const raw = localStorage.getItem(key);
      const saved = raw === null ? {version: 1, items: defaults} : JSON.parse(raw);
      if (saved?.version !== 1 || !Array.isArray(saved.items) || saved.items.length > limit) throw Error();
      const ids = new Set();
      items = saved.items.map(item => {
        if (!item || typeof item.id !== 'string' || !/^[\w-]{1,80}$/.test(item.id) || ids.has(item.id) ||
            typeof item.name !== 'string' || !item.name.trim() || item.name.length > 50 ||
            typeof item.url !== 'string' || !colors.has(item.color)) throw Error();
        ids.add(item.id);
        return {id: item.id, name: item.name.trim(), url: address(item.url), color: item.color};
      });
      readable = true;
    } catch {
      items = defaults.map(item => ({...item})); readable = false;
      status('Saved shortcuts could not be read. Changes will last for this visit only; saved data has not been overwritten.');
    }
  }

  function save(message) {
    if (!readable) { status(message + ' For this visit only; saved data could not be read.'); return; }
    try {
      localStorage.setItem(key, JSON.stringify({version: 1, items}));
      status(message + ' Saved in this browser.');
    } catch { status(message + ' Could not save. Changes will last for this visit only.'); }
  }

  function button(text, label, action) {
    const node = document.createElement('button'); node.type = 'button'; node.textContent = text;
    node.setAttribute('aria-label', label); node.title = label; node.onclick = action;
    return node;
  }

  function focusHandle(id) {
    [...grid.children].find(card => card.dataset.id === id)?.querySelector('.shortcut-handle').focus();
  }

  function move(id, to) {
    const from = items.findIndex(item => item.id === id);
    if (from < 0 || to < 0 || to >= items.length || from === to) return;
    const [item] = items.splice(from, 1); items.splice(to, 0, item);
    save(`${item.name} moved to position ${to + 1}.`); render(); focusHandle(id);
  }

  function render() {
    grid.replaceChildren(); grid.classList.toggle('customizing', customizing);
    el('shortcutHelp').hidden = !customizing;
    el('shortcutCustomize').setAttribute('aria-pressed', String(customizing));
    el('shortcutCustomize').textContent = customizing ? 'Done' : 'Customize';
    el('shortcutEmpty').hidden = items.length > 0;
    el('shortcutAdd').disabled = items.length >= limit;
    items.forEach((item, index) => {
      const card = document.createElement('li'); card.className = 'shortcut-card'; card.dataset.id = item.id;
      const link = document.createElement('a'); link.className = 'shortcut-link'; link.href = item.url; link.draggable = false;
      const url = new URL(item.url, location.origin), external = url.origin !== location.origin;
      if (external) { link.target = '_blank'; link.rel = 'noopener noreferrer'; }
      link.setAttribute('aria-label', item.name + (external ? ' (opens in new tab)' : ''));
      const icon = document.createElement('span'); icon.className = 'shortcut-icon'; icon.dataset.color = item.color;
      icon.textContent = item.name.split(/\s+/).map(word => [...word][0]).slice(0, 2).join('').toUpperCase(); icon.setAttribute('aria-hidden', 'true');
      const name = document.createElement('strong'); name.textContent = item.name;
      const host = document.createElement('span'); host.className = 'shortcut-host'; host.textContent = url.hostname.replace(/^www\./, '');
      link.append(icon, name, host); card.append(link);
      if (customizing) {
        const handle = button('⠿', 'Reorder ' + item.name, () => status('Drag the handle, or use arrow keys to move this shortcut.'));
        handle.className = 'shortcut-handle'; handle.setAttribute('aria-describedby', 'shortcutHelp');
        handle.onkeydown = event => {
          const offsets = {ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1};
          if (Object.hasOwn(offsets, event.key)) { event.preventDefault(); move(item.id, index + offsets[event.key]); }
        };
        handle.onpointerdown = event => startDrag(event, item.id, handle, card);
        handle.onpointermove = trackDrag; handle.onpointerup = finishDrag;
        handle.onpointercancel = cancelDrag;
        const controls = document.createElement('div'); controls.className = 'shortcut-controls';
        const earlier = button('←', 'Move ' + item.name + ' earlier', () => move(item.id, index - 1)); earlier.disabled = index === 0;
        const later = button('→', 'Move ' + item.name + ' later', () => move(item.id, index + 1)); later.disabled = index === items.length - 1;
        const edit = button('Edit', 'Edit ' + item.name, () => openEditor(item));
        const remove = button('×', 'Remove ' + item.name, () => {
          removed = {item, index}; items.splice(index, 1); save(item.name + ' removed.');
          el('shortcutUndo').hidden = false; render();
          if (items.length) focusHandle(items[Math.min(index, items.length - 1)].id); else el('shortcutAdd').focus();
        });
        controls.append(earlier, later, edit, remove); card.append(handle, controls);
      }
      grid.append(card);
    });
  }

  function openEditor(item = null) {
    editing = item?.id ?? null;
    el('shortcutDialogTitle').textContent = item ? 'Edit shortcut' : 'Add a shortcut';
    el('shortcutName').value = item?.name ?? ''; el('shortcutUrl').value = item?.url ?? '';
    el('shortcutColor').value = item?.color ?? 'blue'; el('shortcutError').textContent = '';
    dialog.showModal(); el('shortcutName').focus();
  }
  el('shortcutCustomize').onclick = () => { customizing = !customizing; render(); };
  el('shortcutAdd').onclick = () => openEditor();
  el('shortcutCancel').onclick = el('shortcutClose').onclick = () => dialog.close();
  el('shortcutForm').onsubmit = event => {
    event.preventDefault();
    try {
      const name = el('shortcutName').value.trim();
      if (!name) throw Error('Enter a name for this shortcut.');
      const url = address(el('shortcutUrl').value), color = el('shortcutColor').value;
      const index = items.findIndex(item => item.id === editing);
      if (editing && index < 0) throw Error('This shortcut was removed in another tab. Close this editor and add a new shortcut.');
      if (!editing && items.length >= limit) throw Error('You can save up to 48 shortcuts. Remove one before adding another.');
      if (items.some(item => item.id !== editing && item.url === url)) throw Error('That website is already in your shortcuts.');
      const item = {id: editing ?? crypto.randomUUID(), name, url, color};
      if (editing) items[index] = item; else items.push(item);
      save(name + (editing ? ' updated.' : ' added.')); render(); dialog.close();
    } catch (error) { el('shortcutError').textContent = error instanceof TypeError ? 'Enter a valid website address, such as https://finviz.com.' : error.message; }
  };
  el('shortcutUndo').onclick = () => {
    if (!removed) return;
    if (items.length >= limit) { status('Remove a shortcut before restoring this one.'); return; }
    if (items.some(item => item.url === removed.item.url)) { status('That website is already in your shortcuts.'); return; }
    items.splice(Math.min(removed.index, items.length), 0, removed.item);
    save(removed.item.name + ' restored.'); removed = null; el('shortcutUndo').hidden = true; render();
  };

  function startDrag(event, id, handle, card) {
    if (event.button !== 0 || !event.isPrimary) return;
    const rect = card.getBoundingClientRect();
    drag = {id, handle, card, pointer: event.pointerId, x: event.clientX, y: event.clientY,
      offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top, width: rect.width, target: null, ghost: null};
    handle.setPointerCapture(event.pointerId);
  }
  function trackDrag(event) {
    if (!drag || drag.pointer !== event.pointerId) return;
    if (!drag.ghost && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 8) return;
    if (!drag.ghost) {
      drag.ghost = drag.card.cloneNode(true); drag.ghost.className = 'shortcut-card shortcut-ghost';
      drag.ghost.setAttribute('aria-hidden', 'true'); drag.ghost.inert = true; drag.ghost.style.width = drag.width + 'px';
      document.body.append(drag.ghost); drag.card.classList.add('shortcut-moving');
    }
    drag.ghost.style.left = event.clientX - drag.offsetX + 'px'; drag.ghost.style.top = event.clientY - drag.offsetY + 'px';
    drag.target?.classList.remove('shortcut-drop-target');
    drag.target = document.elementFromPoint(event.clientX, event.clientY)?.closest('#shortcutGrid .shortcut-card') ?? null;
    if (drag.target === drag.card) drag.target = null;
    drag.target?.classList.add('shortcut-drop-target');
    if (event.clientY > innerHeight - 60) window.scrollBy(0, 18);
    else if (event.clientY < 100) window.scrollBy(0, -18);
  }
  function cancelDrag() {
    if (!drag) return;
    drag.ghost?.remove(); drag.card.classList.remove('shortcut-moving'); drag.target?.classList.remove('shortcut-drop-target');
    if (drag.handle.hasPointerCapture(drag.pointer)) drag.handle.releasePointerCapture(drag.pointer);
    drag = null;
  }
  function finishDrag(event) {
    if (!drag || drag.pointer !== event.pointerId) return;
    const id = drag.id, to = drag.ghost && drag.target ? items.findIndex(item => item.id === drag.target.dataset.id) : -1;
    cancelDrag(); if (to >= 0) move(id, to);
  }
  document.addEventListener('keydown', event => { if (event.key === 'Escape') cancelDrag(); });
  window.addEventListener('blur', cancelDrag);
  window.addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    cancelDrag(); removed = null; el('shortcutUndo').hidden = true;
    load(); render(); if (readable) status('Shortcuts updated from another tab.');
  });
  load(); render();
})();
