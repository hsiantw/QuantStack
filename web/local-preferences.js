// Runs before theme/chart modules so a fresh browser session starts with saved settings.
(() => {
  const seed = window.quantLocalPreferenceSeed;
  delete window.quantLocalPreferenceSeed;
  if (!seed) return;
  const keys = new Set(seed.keys), profile = seed.profile;
  const marker = 'quantstack.localPreferenceProfile';
  const pendingKey = 'quantstack.localPreferences.pending.' + profile;
  const nativeSet = Storage.prototype.setItem, nativeRemove = Storage.prototype.removeItem, nativeClear = Storage.prototype.clear;
  let pending = {}, timer, running, paused = false, retry = 1000, status = 'Saved on this PC';
  const put = (key, value) => value === null ? nativeRemove.call(localStorage,key) : nativeSet.call(localStorage,key,value);
  function persistPending() { try { nativeSet.call(localStorage,pendingKey,JSON.stringify(pending)); } catch {} }
  function show(text) {
    status = text;
    const el = document.getElementById('localPreferenceStatus');
    if (el) { el.textContent = text; el.title = 'Watchlists, favorites and chart preferences are backed up locally on this PC.'; }
  }
  try {
    const previous = localStorage.getItem(marker);
    const stored = JSON.parse(localStorage.getItem(pendingKey) || '{}');
    if (stored && typeof stored === 'object' && !Array.isArray(stored)) {
      for (const [key,value] of Object.entries(stored)) if (keys.has(key) && (value === null || typeof value === 'string')) pending[key] = value;
    }
    if (previous && previous !== profile) for (const key of keys) put(key,null);
    for (const key of keys) {
      if (Object.hasOwn(pending,key)) put(key,pending[key]);
      else if (!previous && localStorage.getItem(key) !== null && seed.values[key] !== null) pending[key] = localStorage.getItem(key);
      else if (Object.hasOwn(seed.values,key)) put(key,seed.values[key]);
      else if ((!previous || previous === profile) && localStorage.getItem(key) !== null) pending[key] = localStorage.getItem(key);
    }
    nativeSet.call(localStorage,marker,profile);
    persistPending();
  } catch { show('Browser storage unavailable'); return; }
  function schedule() {
    if (paused) return;
    clearTimeout(timer); timer = setTimeout(() => flush(),150);
    show('Saving preferences…');
  }
  function changed(key, value) {
    if (paused || !keys.has(key)) return;
    pending[key] = value; persistPending(); schedule();
  }
  Storage.prototype.setItem = function(key,value) {
    nativeSet.call(this,key,value);
    if (this === localStorage) changed(String(key),String(value));
  };
  Storage.prototype.removeItem = function(key) {
    nativeRemove.call(this,key);
    if (this === localStorage) changed(String(key),null);
  };
  Storage.prototype.clear = function() {
    const removed = this === localStorage ? [...keys].filter(key => localStorage.getItem(key) !== null) : [];
    nativeClear.call(this);
    for (const key of removed) changed(key,null);
  };
  async function flush() {
    clearTimeout(timer);
    if (running) { const saved = await running; if (saved && Object.keys(pending).length) return flush(); return saved; }
    if (paused || !Object.keys(pending).length) return;
    const sent = {...pending}, body = JSON.stringify({profile,changes:sent});
    running = (async () => {
      try {
        const response = await fetch('/workspace/api/account/local-preferences', {method:'POST',
          credentials:'same-origin',headers:{'Content-Type':'application/json'},body,
          keepalive:new TextEncoder().encode(body).length < 60000});
        if (response.status === 409) { paused = true; throw Error('Account changed; reload to save preferences'); }
        if (!response.ok) throw Error('Saved in browser; retrying local backup');
        for (const [key,value] of Object.entries(sent)) if (pending[key] === value) delete pending[key];
        persistPending(); retry = 1000; show('Saved on this PC'); return true;
      } catch (error) {
        show(error.message);
        if (!paused) { timer = setTimeout(() => flush(),retry); retry = Math.min(30000,retry*2); }
        return false;
      }
    })();
    const saved = await running; running = null;
    if (saved && Object.keys(pending).length) return flush();
    return saved;
  }
  window.QuantLocalPreferences = {flush, pause() { paused=true; clearTimeout(timer); }};
  window.addEventListener('pagehide', () => flush());
  document.addEventListener('visibilitychange', () => { if (document.hidden) flush(); });
  window.addEventListener('online', () => flush());
  window.addEventListener('storage', event => {
    if (event.key === 'quantstack.accountChanged') { paused=true; clearTimeout(timer); }
  });
  document.addEventListener('DOMContentLoaded', () => {
    const el = document.createElement('small'); el.id='localPreferenceStatus'; el.setAttribute('role','status');
    el.style.cssText='display:block;padding:4px 12px;font-size:10px;color:var(--atlas-muted,#66758b)';
    (document.getElementById('workspaceSideWatchlist') || document.querySelector('footer'))?.append(el);
    show(status);
  });
  if (Object.keys(pending).length) schedule();
})();
