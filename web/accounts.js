// Accounts store explicit workspace snapshots; browser-only work stays local until saved.
(() => {
  const button = document.createElement('button');
  button.id = 'accountButton'; button.type = 'button'; button.textContent = 'Account';
  document.querySelector('.terminal-toolbar').append(button);
  const dialog = document.createElement('dialog');
  dialog.id = 'accountDialog'; dialog.setAttribute('aria-labelledby', 'accountTitle');
  dialog.innerHTML = `<div class="dialog-head"><h2 id="accountTitle">Your QuantStack account</h2><button type="button" id="accountClose">Close</button></div>
    <p>Save notes, drawings, watchlists and settings to your account. Sign in on another device using this same QuantStack server to load them.</p>
    <form id="accountForm"><label>Username<input id="accountUsername" autocomplete="username" minlength="3" maxlength="32" pattern="[A-Za-z0-9_.-]{3,32}" required></label>
    <label>Password<input id="accountPassword" type="password" autocomplete="current-password" minlength="8" maxlength="128" required></label>
    <p class="muted">Use 8–128 characters. Password recovery is not available yet; keep your password in a password manager.</p>
    <div class="account-actions"><button id="accountLogin" type="submit">Sign in</button><button id="accountRegister" type="button">Create account</button></div></form>
    <section id="accountWorkspace" hidden><p id="accountIdentity"></p>
    <p>Saving uploads the current browser workspace. Loading replaces this browser's notes, drawings and settings with the account copy. Save notebook entries before uploading. Market prices are shared and are not copied into accounts.</p>
    <div class="account-actions"><button type="button" id="accountSave">Save to account</button><button type="button" id="accountLoad">Load saved workspace</button><button type="button" id="accountLogout">Sign out &amp; clear browser workspace</button></div></section>
    <p id="accountStatus" role="status" aria-live="polite"></p>`;
  document.body.append(dialog);
  const el = id => document.getElementById(id);
  let user = null, revision = null, busy = false;
  const key = () => 'quantstack.accountRevision.' + user.id;
  const workspace = () => Object.fromEntries(Object.keys(localStorage).filter(k => k.startsWith('atlas.')).map(k => [k, localStorage.getItem(k)]));
  const signalAccountChange = () => { try { localStorage.setItem('quantstack.accountChanged', crypto.randomUUID()); } catch {} };
  async function api(action, body) {
    const response = await fetch(new URL('./api/account/' + action, location.href), {
      method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin', cache: 'no-store',
      ...(body === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    });
    const data = await response.json().catch(() => ({error: 'Accounts require the QuantStack app server. Open the app with Run-Local-Website.bat; static sites do not support accounts.'}));
    if (!response.ok || !Object.hasOwn(data, 'user')) throw Error(data.error || 'Account service is unavailable.');
    return data;
  }
  function render() {
    button.textContent = user ? 'Account: ' + user.username : 'Account';
    el('accountForm').hidden = !!user; el('accountWorkspace').hidden = !user;
    if (user) {
      el('accountIdentity').textContent = `Signed in as ${user.username}. ` + (user.updated_at ? `Last account save: ${new Date(user.updated_at * 1000).toLocaleString()}.` : 'No workspace saved yet.');
      el('accountSave').disabled = busy || revision === null;
      el('accountLoad').disabled = busy || !user.revision;
    }
    for (const id of ['accountLogin', 'accountRegister', 'accountLogout', 'accountClose']) el(id).disabled = busy;
  }
  function setUser(next) {
    user = next; revision = null;
    if (user) {
      try {
        const saved = sessionStorage.getItem(key());
        revision = saved === null ? (user.revision === 0 ? 0 : null) : Number(saved);
        if (!Number.isSafeInteger(revision) || revision < 0) revision = null;
      } catch { revision = user.revision === 0 ? 0 : null; }
    }
    render();
  }
  async function perform(action) {
    if (busy) return;
    busy = true; render(); el('accountStatus').textContent = 'Working…';
    try { await action(); }
    catch (error) { el('accountStatus').textContent = error.message || 'Account request failed. Your browser workspace is unchanged.'; }
    finally { busy = false; render(); }
  }
  button.onclick = () => {
    dialog.showModal();
    perform(async () => {
      setUser((await api('session')).user);
      el('accountStatus').textContent = user ? (revision === null ? 'Load your saved workspace before saving from this tab.' : 'Browser changes are local until you choose Save to account.') : 'Sign in or create an account. Existing browser work will not be uploaded automatically.';
    });
  };
  el('accountClose').onclick = () => dialog.close();
  dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  async function authenticate(action) {
    if (!el('accountForm').reportValidity()) return;
    await perform(async () => {
      const data = await api(action, {username: el('accountUsername').value.trim(), password: el('accountPassword').value});
      el('accountPassword').value = ''; setUser(data.user);
      signalAccountChange();
      el('accountStatus').textContent = revision === null ? 'Signed in. Load your saved workspace to continue on this device.' : 'Signed in. Choose Save to account to upload this browser workspace.';
    });
  }
  el('accountForm').onsubmit = event => { event.preventDefault(); authenticate('login'); };
  el('accountRegister').onclick = () => authenticate('register');
  el('accountSave').onclick = () => {
    if (!user || revision === null || !confirm(`Save this browser's notes, drawings and settings to ${user.username}'s account? This replaces the previous account snapshot.`)) return;
    perform(async () => {
      const data = await api('workspace', {account_id: user.id, revision, entries: workspace()});
      user = data.user; revision = user.revision;
      try { sessionStorage.setItem(key(), String(revision)); } catch {}
      el('accountStatus').textContent = 'Saved to your account. You can load this workspace on another device.';
    });
  };
  function replaceWorkspace(entries) {
    if (!entries || typeof entries !== 'object' || Array.isArray(entries) || Object.entries(entries).some(([k,v]) => !k.startsWith('atlas.') || typeof v !== 'string')) throw Error('Invalid account workspace. Nothing was changed.');
    const before = workspace();
    try {
      Object.keys(before).forEach(k => localStorage.removeItem(k));
      Object.entries(entries).forEach(([k,v]) => localStorage.setItem(k,v));
    } catch {
      try {
        Object.keys(workspace()).forEach(k => localStorage.removeItem(k));
        Object.entries(before).forEach(([k,v]) => localStorage.setItem(k,v));
      } catch { throw Error('Browser storage failed. Your account copy is safe; use a browser with storage enabled.'); }
      throw Error('Browser storage is full or unavailable. Your previous browser workspace was restored.');
    }
  }
  el('accountLoad').onclick = () => {
    if (!user || !confirm('Replace this browser workspace with your account copy and reload? Local-only changes and unsaved notebook edits will be lost. Download a notes/drawings backup first if needed.')) return;
    perform(async () => {
      const data = await api('workspace');
      if (data.user.id !== user.id) throw Error('The signed-in account changed. Reload before loading a workspace.');
      replaceWorkspace(data.entries); user = data.user; revision = user.revision;
      try { sessionStorage.setItem(key(), String(revision)); } catch {}
      signalAccountChange();
      location.reload();
    });
  };
  el('accountLogout').onclick = () => {
    if (!confirm('Sign out and clear notes, drawings and settings from this browser? Save to your account first to keep local changes.')) return;
    perform(async () => {
      await api('logout', {account_id: user.id});
      try { sessionStorage.removeItem(key()); } catch {}
      setUser(null);
      // Clear all same-origin tabs' shared browser workspace before reloading.
      replaceWorkspace({});
      signalAccountChange();
      location.reload();
    });
  };
  // Other tabs must not upload a workspace under a newly switched account.
  window.addEventListener('storage', event => {
    if (event.key === 'quantstack.accountChanged') { location.reload(); return; }
    if (event.key?.startsWith('atlas.') && dialog.open) el('accountStatus').textContent = 'The browser workspace changed in another tab. Reload before saving to your account.';
  });
  render();
})();
