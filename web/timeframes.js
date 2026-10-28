// Favorite intervals are local preferences; changing them never starts a data pull.
(() => {
  const options = [
    ['1m', '1 minute', 'Minutes'], ['3m', '3 minutes', 'Minutes'],
    ['5m', '5 minutes', 'Minutes'], ['15m', '15 minutes', 'Minutes'], ['30m', '30 minutes', 'Minutes'],
    ['1h', '1 hour', 'Hours'], ['2h', '2 hours', 'Hours'], ['4h', '4 hours', 'Hours'],
    ['1d', '1 day', 'Longer'], ['1w', '1 week', 'Longer'], ['1mo', '1 month', 'Longer']
  ];
  const key = 'atlas.timeframeFavorites.v1', defaults = ['1m', '5m', '15m', '30m', '1h', '1d'];
  const valid = new Set(options.map(([id]) => id));
  const allowed = id => !window.ATLAS_STATIC || ['1d', '1w', '1mo'].includes(id);
  const label = id => ({'1d': '1D', '1w': '1W', '1mo': '1M'})[id] || id;
  let favorites = defaults;
  function read() {
    try {
      const value = JSON.parse(localStorage.getItem(key));
      favorites = Array.isArray(value) ? [...new Set(value.filter(id => valid.has(id)))] : defaults;
    } catch { favorites = defaults; }
  }
  read();
  const dialog = document.createElement('dialog');
  dialog.id = 'timeframeDialog'; dialog.setAttribute('aria-labelledby', 'timeframeTitle');
  dialog.innerHTML = '<header><strong id="timeframeTitle">Timeframes</strong><button type="button" id="timeframeClose" aria-label="Close timeframes">×</button></header><p>Star intervals to keep them in your toolbar.</p><div id="timeframeOptions"></div>';
  document.body.append(dialog);
  function render() {
    $('intervals').innerHTML = options.filter(([id]) => allowed(id) && (favorites.includes(id) || id === interval))
      .map(([id, name]) => `<button type="button" data-interval="${id}" title="${name}" aria-label="${name}" aria-pressed="${id === interval}" class="${id === interval ? 'active' : ''}">${label(id)}</button>`).join('')
      + '<button type="button" id="timeframePicker" title="Choose timeframe and favorites" aria-label="Choose timeframe and favorites" aria-haspopup="dialog">▾</button>';
    let group = '';
    $('timeframeOptions').innerHTML = options.map(([id, name, category]) => {
      const heading = group !== category ? `<h3>${category}</h3>` : ''; group = category;
      return `${heading}<div class="timeframe-row"><button type="button" data-timeframe="${id}" aria-pressed="${id === interval}" ${allowed(id) ? '' : 'disabled title="Intraday charts require the local site"'}><span>${name}</span><small>${label(id)}</small></button><button type="button" class="timeframe-star" data-timeframe-star="${id}" aria-label="Favorite ${name}" aria-pressed="${favorites.includes(id)}">${favorites.includes(id) ? '★' : '☆'}</button></div>`;
    }).join('');
  }
  function open() {
    render();
    const box = $('timeframePicker').getBoundingClientRect();
    dialog.style.left = Math.max(8, Math.min(box.left, innerWidth - 296)) + 'px';
    dialog.style.top = Math.max(8, Math.min(box.bottom + 6, innerHeight - 580)) + 'px';
    dialog.showModal();
    dialog.querySelector(`[data-timeframe="${interval}"]`)?.focus();
  }
  function choose(id) {
    if (!selected || !valid.has(id) || !allowed(id)) return;
    interval = id;
    if (!['MAX', 'CUSTOM'].includes(period)) period = ({'1d':'1Y', '1w':'5Y', '1mo':'MAX'})[id] || '5D';
    document.querySelectorAll('[data-period]').forEach(b => b.classList.toggle('active', b.dataset.period === period));
    if (period !== 'CUSTOM') setDates();
    dialog.close(); render(); loadDrawings(); loadHistory();
  }
  $('intervals').onclick = event => {
    const button = event.target.closest('button');
    if (button?.id === 'timeframePicker') open();
    else if (button?.dataset.interval) choose(button.dataset.interval);
  };
  $('timeframeOptions').onclick = event => {
    const button = event.target.closest('button');
    const id = button?.dataset.timeframeStar;
    if (id) {
      favorites = favorites.includes(id) ? favorites.filter(item => item !== id) : [...favorites, id];
      try { localStorage.setItem(key, JSON.stringify(favorites)); } catch {}
      render(); dialog.querySelector(`[data-timeframe-star="${id}"]`)?.focus();
    } else if (button?.dataset.timeframe) choose(button.dataset.timeframe);
  };
  $('timeframeClose').onclick = () => dialog.close();
  dialog.addEventListener('click', event => { if (event.target === dialog) {
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  }});
  dialog.addEventListener('close', () => $('timeframePicker')?.focus());
  window.addEventListener('storage', event => { if (event.key === key) { read(); render(); } });
  const originalLoad = loadHistory;
  loadHistory = async function() { render(); return originalLoad(); };
  render();
})();
