// Search the full stored symbol catalog without changing the active watchlist.
(() => {
  'use strict';

  function initialize() {
    if (!document.body.classList.contains('workspace') || document.body.dataset.toolPage) return;
    const trigger = document.getElementById('workspaceSymbol');
    if (!trigger || document.getElementById('workspaceSymbolDialog')) return;

    const storageKey = 'atlas.workspace.recentSymbols.v1';
    const recentLimit = 8, pageSize = 60;
    const catalog = () => typeof assets !== 'undefined' && Array.isArray(assets) ? assets : [];
    const currentSymbol = () => document.getElementById('symbol')?.textContent.trim() || '';
    let recent = [], category = 'All', results = [], activeIndex = -1, resultLimit = pageSize;
    let returnFocus = null;
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || '[]');
      if (Array.isArray(stored)) recent = [...new Set(stored.filter(value => typeof value === 'string' && value.length <= 80))].slice(0, recentLimit);
    } catch {}

    const dialog = document.createElement('dialog');
    dialog.id = 'workspaceSymbolDialog';
    dialog.setAttribute('aria-labelledby', 'workspaceSymbolSearchTitle');
    dialog.innerHTML = `
      <div class="workspace-symbol-search-heading">
        <h2 id="workspaceSymbolSearchTitle">Symbol search</h2>
        <button type="button" id="workspaceSymbolSearchClose" aria-label="Close symbol search" title="Close (Escape)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6"/></svg></button>
      </div>
      <div class="workspace-symbol-search-field">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></svg>
        <input id="workspaceSymbolSearchInput" type="search" role="combobox" aria-label="Search symbol or company" aria-autocomplete="list" aria-expanded="false" aria-controls="workspaceSymbolSearchResults" aria-describedby="workspaceSymbolSearchStatus" placeholder="Search symbol or company" maxlength="80" autocomplete="off" spellcheck="false" autofocus>
        <button type="button" id="workspaceSymbolSearchClear" aria-label="Clear search" title="Clear search" hidden>×</button>
      </div>
      <div id="workspaceSymbolSearchFilters" class="workspace-symbol-search-filters" role="group" aria-label="Asset type">
        <button type="button" data-symbol-category="All" aria-pressed="true">All</button>
        <button type="button" data-symbol-category="Stocks" aria-pressed="false">Stocks</button>
        <button type="button" data-symbol-category="Crypto" aria-pressed="false">Crypto</button>
      </div>
      <div class="workspace-symbol-search-summary">
        <span id="workspaceSymbolSearchStatus" role="status" aria-live="polite" aria-atomic="true"></span>
        <button type="button" id="workspaceSymbolSearchClearRecent" hidden>Clear recent</button>
      </div>
      <div id="workspaceSymbolSearchResults" role="listbox" aria-label="Available symbols"></div>
      <div id="workspaceSymbolSearchEmpty" class="workspace-symbol-search-empty" hidden><strong>No symbols found</strong><p>Try another ticker or company name, or select All.</p></div>
      <button type="button" id="workspaceSymbolSearchMore" hidden>Show more symbols</button>
      <div class="workspace-symbol-search-footer"><span>Available in your dataset</span><span class="workspace-symbol-search-keys"><kbd>↑</kbd><kbd>↓</kbd> Navigate <kbd>Enter</kbd> Open <kbd>Esc</kbd> Close</span></div>`;
    document.body.append(dialog);
    const find = id => document.getElementById(id);
    const input = find('workspaceSymbolSearchInput');
    const list = find('workspaceSymbolSearchResults');
    const status = find('workspaceSymbolSearchStatus');
    const empty = find('workspaceSymbolSearchEmpty');
    const clear = find('workspaceSymbolSearchClear');
    const clearRecent = find('workspaceSymbolSearchClearRecent');
    const more = find('workspaceSymbolSearchMore');

    const shortcut = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘ K' : 'Ctrl K';
    const hint = document.createElement('kbd');
    hint.className = 'workspace-symbol-shortcut';
    hint.textContent = shortcut;
    hint.setAttribute('aria-hidden', 'true');
    trigger.append(hint);
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.setAttribute('aria-controls', dialog.id);
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-keyshortcuts', '/ Control+k Meta+k');
    trigger.setAttribute('aria-label', `Search all symbols (${shortcut})`);
    trigger.title = `Search all symbols (${shortcut} or /)`;

    function saveRecent(symbol) {
      if (!symbol || !catalog().some(asset => asset.symbol === symbol) || recent[0] === symbol) return;
      recent = [symbol, ...recent.filter(value => value !== symbol)].slice(0, recentLimit);
      try { localStorage.setItem(storageKey, JSON.stringify(recent)); } catch {}
    }

    function markActive(index, scroll = false) {
      const previous = list.children[activeIndex];
      if (previous) previous.setAttribute('aria-selected', 'false');
      activeIndex = results.length ? Math.max(0, Math.min(index, results.length - 1)) : -1;
      const option = list.children[activeIndex];
      if (!option) {
        input.removeAttribute('aria-activedescendant');
        return;
      }
      option.setAttribute('aria-selected', 'true');
      input.setAttribute('aria-activedescendant', option.id);
      if (scroll) option.scrollIntoView({block: 'nearest'});
    }

    function textElement(tag, className, text) {
      const element = document.createElement(tag);
      element.className = className;
      element.textContent = text;
      return element;
    }

    function render(reset = true) {
      const query = input.value.trim().toLocaleLowerCase();
      const terms = query.split(/\s+/).filter(Boolean);
      const available = catalog().filter(asset => asset && typeof asset.symbol === 'string');
      const eligible = available.filter(asset => category === 'All' || asset.kind === category);
      const matches = eligible.filter(asset => terms.every(term => `${asset.symbol} ${asset.name || ''}`.toLocaleLowerCase().includes(term)));
      const rank = asset => {
        const symbol = asset.symbol.toLocaleLowerCase();
        if (symbol === query) return 0;
        if (symbol.startsWith(query)) return 1;
        if (symbol.includes(query)) return 2;
        return 3;
      };
      if (query) matches.sort((a, b) => rank(a) - rank(b) || a.symbol.localeCompare(b.symbol));
      const recentSet = new Set(recent);
      const recentAssets = query ? [] : recent.map(symbol => eligible.find(asset => asset.symbol === symbol)).filter(Boolean);
      const ordered = query ? matches : [...recentAssets, ...matches.filter(asset => !recentSet.has(asset.symbol))];
      const oldSymbol = reset ? null : results[activeIndex]?.symbol;
      results = ordered.slice(0, resultLimit);
      const fragment = document.createDocumentFragment();
      results.forEach((asset, index) => {
        const option = document.createElement('div');
        option.id = `workspaceSymbolOption-${index}`;
        option.className = 'workspace-symbol-search-option';
        option.setAttribute('role', 'option');
        option.setAttribute('aria-selected', 'false');
        option.dataset.symbolIndex = String(index);
        const avatar = textElement('span', 'workspace-symbol-search-avatar', asset.symbol.slice(0, 2));
        avatar.setAttribute('aria-hidden', 'true');
        avatar.classList.toggle('is-crypto', asset.kind === 'Crypto');
        const identity = textElement('span', 'workspace-symbol-search-identity', '');
        const ticker = textElement('strong', 'workspace-symbol-search-ticker', asset.symbol);
        if (!query && index < recentAssets.length) ticker.append(textElement('small', 'workspace-symbol-search-recent', 'Recent'));
        identity.append(ticker, textElement('span', 'workspace-symbol-search-company', asset.name || asset.symbol));
        const venue = textElement('span', 'workspace-symbol-search-venue', '');
        venue.append(textElement('span', '', asset.exchange || asset.kind || ''), textElement('small', '', asset.kind || ''));
        option.append(avatar, identity, venue);
        fragment.append(option);
      });
      list.replaceChildren(fragment);
      clear.hidden = !input.value;
      clearRecent.hidden = Boolean(query) || !recentAssets.length;
      more.hidden = ordered.length <= results.length;
      more.textContent = `Show ${Math.min(pageSize, ordered.length - results.length)} more symbols`;
      empty.hidden = Boolean(results.length);
      list.hidden = !results.length;
      const loading = !available.length && find('refresh')?.disabled;
      empty.querySelector('strong').textContent = loading ? 'Loading symbols…' : available.length ? 'No symbols found' : 'No symbols available yet';
      empty.querySelector('p').textContent = loading ? 'Your market data catalog will appear here.' : available.length ? 'Try another ticker or company name, or select All.' : 'Add symbols to your local collection, then refresh the dataset.';
      status.textContent = query ? `${matches.length.toLocaleString()} ${matches.length === 1 ? 'symbol' : 'symbols'} found` : `${eligible.length.toLocaleString()} ${eligible.length === 1 ? 'symbol' : 'symbols'}${recentAssets.length ? ' · Recent first' : ' available'}`;
      activeIndex = -1;
      markActive(oldSymbol ? Math.max(0, results.findIndex(asset => asset.symbol === oldSymbol)) : 0);
      if (reset) list.scrollTop = 0;
    }

    function open() {
      if (dialog.open) { input.focus(); return; }
      if (document.querySelector('dialog[open]')) return;
      returnFocus = document.activeElement;
      input.value = '';
      category = 'All';
      resultLimit = pageSize;
      find('workspaceSymbolSearchFilters').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.symbolCategory === category)));
      saveRecent(currentSymbol());
      render();
      dialog.showModal();
      trigger.setAttribute('aria-expanded', 'true');
      input.setAttribute('aria-expanded', 'true');
      input.focus({preventScroll: true});
    }

    function choose(index) {
      const asset = results[index];
      if (!asset || typeof select !== 'function') return;
      saveRecent(asset.symbol);
      dialog.close();
      Promise.resolve(select(asset.symbol)).catch(error => {
        if (typeof showError === 'function') showError(error?.message || 'Unable to open this symbol.');
      });
    }

    trigger.addEventListener('click', event => {
      event.preventDefault();
      event.stopImmediatePropagation();
      open();
    }, true);
    find('workspaceSymbolSearchClose').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => {
      trigger.setAttribute('aria-expanded', 'false');
      input.setAttribute('aria-expanded', 'false');
      input.removeAttribute('aria-activedescendant');
      (returnFocus?.isConnected && typeof returnFocus.focus === 'function' ? returnFocus : trigger).focus({preventScroll: true});
    });
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    });
    input.addEventListener('input', () => { resultLimit = pageSize; render(); });
    clear.addEventListener('click', () => { input.value = ''; resultLimit = pageSize; render(); input.focus(); });
    clearRecent.addEventListener('click', () => {
      recent = [];
      try { localStorage.removeItem(storageKey); } catch {}
      render();
      input.focus();
    });
    find('workspaceSymbolSearchFilters').addEventListener('click', event => {
      const button = event.target.closest('[data-symbol-category]');
      if (!button) return;
      category = button.dataset.symbolCategory;
      resultLimit = pageSize;
      for (const item of button.parentElement.children) item.setAttribute('aria-pressed', String(item === button));
      render();
      input.focus({preventScroll: true});
    });
    more.addEventListener('click', () => { resultLimit += pageSize; render(false); input.focus({preventScroll: true}); });
    list.addEventListener('click', event => {
      const option = event.target.closest('[data-symbol-index]');
      if (option) choose(Number(option.dataset.symbolIndex));
    });
    list.addEventListener('pointermove', event => {
      if (event.pointerType !== 'mouse') return;
      const option = event.target.closest('[data-symbol-index]');
      if (option) markActive(Number(option.dataset.symbolIndex));
    });
    list.addEventListener('mousedown', event => {
      if (event.target.closest('[data-symbol-index]')) event.preventDefault();
    });

    // Capture at window so the old document shortcut cannot focus the watchlist.
    window.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.isComposing) return;
      if (dialog.open) {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopImmediatePropagation();
          dialog.close();
        } else if (event.target === input && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && ['ArrowDown', 'ArrowUp', 'Enter'].includes(event.key)) {
          event.preventDefault();
          event.stopImmediatePropagation();
          if (event.key === 'Enter') choose(activeIndex);
          else if (results.length) markActive((activeIndex + (event.key === 'ArrowDown' ? 1 : -1) + results.length) % results.length, true);
        }
        return;
      }
      const plainSlash = event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey;
      const commandK = event.key.toLowerCase() === 'k' && (event.ctrlKey !== event.metaKey) && !event.altKey && !event.shiftKey;
      if ((!plainSlash && !commandK) || event.repeat || document.querySelector('dialog[open]')) return;
      const target = event.target instanceof Element ? event.target : document.activeElement;
      if (target?.closest('input, textarea, select, [role="textbox"], [role="combobox"]') || target?.isContentEditable) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      open();
    }, true);

    // Update only when catalog loading or a selected symbol changes; no timers.
    const observer = new MutationObserver(() => {
      saveRecent(currentSymbol());
      if (dialog.open) render(false);
    });
    for (const id of ['symbol', 'libraryCount']) {
      const element = find(id);
      if (element) observer.observe(element, {childList: true, characterData: true, subtree: true});
    }
    saveRecent(currentSymbol());
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, {once: true});
  else initialize();
})();
