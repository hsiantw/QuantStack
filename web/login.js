(() => {
  const el = id => document.getElementById(id);
  let mode = 'login', busy = false, generation = 0;
  const clock = () => { el('clock').textContent = new Date().toISOString().slice(11, 19) + ' UTC'; };
  clock(); setInterval(clock, 1000);
  async function api(action, body) {
    const response = await fetch('/workspace/api/account/' + action, {
      method: body ? 'POST' : 'GET', credentials: 'same-origin', cache: 'no-store',
      ...(body ? {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {})
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !Object.hasOwn(data, 'user')) throw Error(data.error || 'Account service is unavailable. Please try again.');
    return data;
  }
  function setMode(next) {
    if (busy) return;
    generation++;
    mode = next;
    el('signinTab').setAttribute('aria-pressed', String(mode === 'login'));
    el('registerTab').setAttribute('aria-pressed', String(mode === 'register'));
    el('loginForm').hidden = false; el('sessionPanel').hidden = true;
    el('password').autocomplete = mode === 'register' ? 'new-password' : 'current-password';
    el('passwordHint').textContent = mode === 'register' ? 'Use 8–128 characters. Keep your password safe; recovery is not available yet.' : 'Your session stays signed in for up to 7 days.';
    el('submit').textContent = mode === 'register' ? 'Create account & open Supercharts ↗' : 'Sign in & open Supercharts ↗';
    el('status').textContent = '';
  }
  el('signinTab').onclick = () => setMode('login');
  el('registerTab').onclick = () => setMode('register');
  el('switchAccount').onclick = () => { setMode('login'); el('username').focus(); };
  el('showPassword').onclick = () => {
    const show = el('password').type === 'password';
    el('password').type = show ? 'text' : 'password';
    el('showPassword').textContent = show ? 'Hide' : 'Show';
    el('showPassword').setAttribute('aria-label', show ? 'Hide password' : 'Show password');
    el('showPassword').setAttribute('aria-pressed', String(show));
  };
  el('loginForm').onsubmit = async event => {
    event.preventDefault(); if (busy) return;
    generation++; busy = true;
    for (const id of ['submit', 'signinTab', 'registerTab']) el(id).disabled = true;
    el('status').textContent = mode === 'register' ? 'Creating your account…' : 'Signing in…';
    try {
      await window.QuantLocalPreferences?.flush();
      await api(mode, {username: el('username').value.trim(), password: el('password').value});
      window.QuantLocalPreferences?.pause();
      el('password').value = '';
      try { localStorage.setItem('quantstack.accountChanged', crypto.randomUUID()); } catch {}
      location.assign('/workspace/');
    } catch (error) {
      el('status').textContent = error.message;
    } finally {
      busy = false;
      for (const id of ['submit', 'signinTab', 'registerTab']) el(id).disabled = false;
    }
  };
  const initialGeneration = generation;
  api('session').then(({user}) => {
    if (!user || generation !== initialGeneration) return;
    el('loginForm').hidden = true; el('sessionPanel').hidden = false;
    el('identity').textContent = 'Signed in as ' + user.username + '.';
  }).catch(error => { if (generation === initialGeneration) el('status').textContent = error.message; });
})();
