(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const PREFS = 'atlas.account.preferences.v1', PROFILES = 'atlas.appearance.presets.v1';
  const themes = ['system','light','dark','midnight','forest','paper'];
  const styles = ['candles','line','area','bars'];
  let user = null, entries = {}, busy = false, epoch = 0;
  const json = (key, fallback) => entries[key] === undefined ? fallback : JSON.parse(entries[key]);
  const date = value => value ? new Date(value).toLocaleString() : 'Not saved yet';
  function message(text) { $('status').textContent = text; }
  function signedOut() { epoch++; user = null; entries = {}; $('accountContent').hidden = true; $('signedOut').hidden = false; $('identity').textContent = 'Signed out'; }
  async function api(action, body) {
    const response = await fetch('/workspace/api/account/' + action, {method:body === undefined ? 'GET' : 'POST', credentials:'same-origin', cache:'no-store', ...(body === undefined ? {} : {headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !Object.hasOwn(data,'user')) throw Error(data.error || 'Account service is unavailable.');
    return data;
  }
  async function verify() {
    const result = await api('session');
    if (!user || result.user?.id !== user.id) { signedOut(); throw Error('Your session changed or expired. Sign in again.'); }
  }
  async function perform(action) {
    if (busy) return;
    busy = true; controls(true); message('Working…');
    try { await action(); }
    catch (error) { message(error.message || 'Unable to complete this action.'); }
    finally { busy = false; controls(false); }
  }
  function controls(disabled) {
    document.querySelectorAll('main button, main input, main select').forEach(node => { node.disabled = disabled || !user; });
    $('reload').disabled = disabled;
  }
  function profiles() {
    const value = json(PROFILES, []);
    if (!Array.isArray(value) || value.some(p => !p || typeof p.name !== 'string' || p.format !== 'quantstack-appearance' || p.version !== 1 || !themes.includes(p.theme) || !styles.includes(p.style) || !p.appearance || typeof p.appearance !== 'object' || Array.isArray(p.appearance))) throw Error('Saved chart profiles could not be read. Export a workspace backup before repairing them.');
    return value;
  }
  function notes() {
    const value = json('atlas.ideaNotes.v1', []);
    if (!Array.isArray(value) || value.some(note => !note || typeof note.title !== 'string' || typeof note.body !== 'string')) throw Error('Saved notebook data could not be read. Download the workspace backup to preserve it.');
    return value;
  }
  function renderSummary() {
    const prefs = json(PREFS, {});
    $('identity').textContent = (prefs.displayName ? prefs.displayName + ' · ' : '') + '@' + user.username;
    $('saveDate').textContent = date(user.updated_at && user.updated_at * 1000);
    $('revision').textContent = 'Workspace revision ' + user.revision;
    // The API enforces ASCII-encoded JSON size, including separators.
    const encoded = JSON.stringify(entries).replace(/[\u007f-\uffff]/g, char => '\\u' + char.charCodeAt(0).toString(16).padStart(4,'0'));
    const bytes = new TextEncoder().encode(encoded).length;
    $('storage').textContent = (bytes / 1000).toFixed(1) + ' KB / 2,000 KB'; $('storageMeter').value = bytes;
    $('noteCount').textContent = notes().length.toLocaleString();
  }
  function row(title, description) {
    const item = document.createElement('article'); item.className = 'saved-item';
    const content = document.createElement('div'), heading = document.createElement('h3'), detail = document.createElement('p');
    heading.textContent = title; detail.textContent = description; content.append(heading, detail); item.append(content); return item;
  }
  function renderProfiles() {
    const list = $('profileList'); list.replaceChildren();
    for (const profile of profiles()) {
      const item = row(profile.name, profile.theme + ' theme · ' + profile.style);
      const actions = document.createElement('div'); actions.className = 'actions';
      for (const [action,label] of [['apply','Set as account default'],['export','Export'],['delete','Delete']]) {
        const button = document.createElement('button'); button.type = 'button'; button.textContent = label;
        button.onclick = () => perform(async () => {
          await verify();
          if (action === 'export') { download(profile, 'chart-profile.json'); message('Chart profile exported.'); return; }
          if (action === 'delete') {
            if (!confirm(`Delete the saved profile “${profile.name}”? Your current chart settings are kept.`)) { message('Deletion canceled.'); return; }
            await save({...entries, [PROFILES]:JSON.stringify(profiles().filter(p => p.name !== profile.name))});
          } else {
            await save({...entries, 'atlas.theme':profile.theme, 'atlas.terminal':JSON.stringify({...json('atlas.terminal',{}), ...profile.appearance, style:profile.style})});
            $('theme').value = profile.theme;
            window.atlasTheme.apply(profile.theme, true);
          }
          renderSummary(); renderProfiles(); message(action === 'delete' ? 'Profile deleted.' : 'Account default updated. Load your saved workspace in Supercharts to apply it.');
        });
        actions.append(button);
      }
      item.append(actions); list.append(item);
    }
    if (!list.children.length) list.textContent = 'No chart profiles saved yet.';
  }
  function renderNotes() {
    const query = $('noteSearch').value.toLowerCase(), filter = $('noteFilter').value;
    const selected = notes().filter(note => (filter === 'all' || (filter === 'trashed' ? note.trashed : filter === 'archived' ? note.archived && !note.trashed : !note.archived && !note.trashed)) && `${note.title} ${note.body} ${note.symbol || ''}`.toLowerCase().includes(query));
    $('notebookList').replaceChildren(...selected.slice(0,50).map(note => row(note.title || 'Untitled note', [note.category, note.symbol, note.trashed ? 'Deleted' : note.archived ? 'Archived' : 'Active', note.body.slice(0,180)].filter(Boolean).join(' · '))));
    if (!selected.length) $('notebookList').textContent = 'No matching notes in your saved account copy.';
    $('notebookSummary').textContent = `Showing ${Math.min(50,selected.length)} of ${selected.length} matching notes. Exports include all notes and categories.`;
  }
  function renderSecurity(data) { $('sessionInfo').textContent = `${data.sessions} active session${data.sessions === 1 ? '' : 's'}. This session expires ${date(data.expires * 1000)}.`; }
  async function load() {
    const token = ++epoch;
    const session = await api('session');
    if (token !== epoch) return;
    if (!session.user) { signedOut(); message('Sign in to continue.'); return; }
    const [workspace, security] = await Promise.all([api('workspace'), api('security')]);
    if (token !== epoch) return;
    if (workspace.user.id !== session.user.id || security.user.id !== session.user.id) { signedOut(); throw Error('Account changed while loading. Sign in again.'); }
    user = workspace.user; entries = workspace.entries;
    $('accountContent').hidden = false; $('signedOut').hidden = true;
    $('displayName').value = json(PREFS,{}).displayName || '';
    $('theme').value = themes.includes(entries['atlas.theme']) ? entries['atlas.theme'] : 'system';
    const style = json('atlas.terminal',{}).style; $('profileStyle').value = styles.includes(style) ? style : 'candles';
    renderSummary(); renderProfiles(); renderNotes(); renderSecurity(security); message('Saved account data loaded.');
  }
  async function save(next) {
    const token = epoch, account = user.id;
    const data = await api('workspace', {account_id:account, revision:user.revision, entries:next});
    if (token !== epoch || !user || user.id !== account || data.user.id !== account) throw Error('Account changed. Reload before continuing.');
    user = data.user; entries = data.entries;
    // This page updated the server copy, not the chart browser's snapshot.
    try { sessionStorage.removeItem('quantstack.accountRevision.' + user.id); localStorage.setItem('quantstack.accountWorkspaceChanged', JSON.stringify({id:user.id,revision:user.revision})); } catch {}
  }
  function download(value, filename, type='application/json') {
    const blob = new Blob([typeof value === 'string' ? value : JSON.stringify(value,null,2)], {type});
    const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'quantstack-' + filename; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  $('preferencesForm').onsubmit = event => { event.preventDefault(); perform(async () => {
    await verify();
    await save({...entries, [PREFS]:JSON.stringify({displayName:$('displayName').value.trim()}), 'atlas.theme':$('theme').value});
    window.atlasTheme.apply($('theme').value, true);
    renderSummary(); message('Preferences saved. Your theme now applies across this browser. Load your saved workspace on another device to use your account preferences there.');
  }); };
  $('profileForm').onsubmit = event => { event.preventDefault(); perform(async () => {
    await verify(); const name = $('profileName').value.trim(); if (!name) throw Error('Enter a profile name.');
    const current = profiles(), index = current.findIndex(p => p.name.toLowerCase() === name.toLowerCase());
    if (index < 0 && current.length >= 20) throw Error('You can save up to 20 chart profiles.');
    if (index >= 0 && !confirm(`Replace the saved profile “${current[index].name}”?`)) { message('Replacement canceled.'); return; }
    const {style,comparisons,...appearance} = json('atlas.terminal',{});
    const profile = {format:'quantstack-appearance', version:1, name, theme:$('theme').value, style:$('profileStyle').value, appearance};
    const next = current.slice(); if (index >= 0) next[index] = profile; else next.push(profile);
    await save({...entries, [PROFILES]:JSON.stringify(next)}); renderSummary(); renderProfiles(); message('Chart profile saved to your account.');
  }); };
  $('exportNotes').onclick = () => perform(async () => {
    await verify();
    const supported = key => ['atlas.ideaNotes.v1','atlas.noteCategories.v1'].includes(key) || /^atlas\.(notes|source)\./.test(key);
    download({format:'quantstack-personal-work',version:1,exportedAt:new Date().toISOString(),entries:Object.fromEntries(Object.entries(entries).filter(([key]) => supported(key)))},'notebooks.json');
    message('Saved notebooks and symbol notes exported as JSON.');
  });
  $('exportMarkdown').onclick = () => perform(async () => {
    await verify();
    const text = ['# QuantStack notebooks', ...notes().map(note => `## ${note.title || 'Untitled'}\n\n${[note.category,note.symbol,note.trashed ? 'Deleted' : note.archived ? 'Archived' : 'Active'].filter(Boolean).join(' · ')}\n\n${note.body}\n\n${(Array.isArray(note.items) ? note.items : []).map(item => `- [${item.done ? 'x' : ' '}] ${item.text}`).join('\n')}`), ...Object.entries(entries).filter(([key]) => key.startsWith('atlas.notes.')).map(([key,value]) => `## ${key.slice(12)} — symbol notes\n\n${value}`)].join('\n\n');
    download(text,'notebooks.md','text/markdown;charset=utf-8'); message('Saved notebooks exported as Markdown.');
  });
  $('exportWorkspace').onclick = () => perform(async () => { await verify(); download({format:'quantstack-account-workspace',version:1,exportedAt:new Date().toISOString(),user:{username:user.username,revision:user.revision},entries},'workspace.json'); message('Full saved workspace downloaded. Keep this personal backup in a safe place.'); });
  $('passwordForm').onsubmit = event => { event.preventDefault(); perform(async () => {
    if ($('newPassword').value !== $('confirmPassword').value) throw Error('New passwords do not match.');
    await verify();
    const data = await api('security', {account_id:user.id,action:'password',current_password:$('currentPassword').value,new_password:$('newPassword').value});
    $('passwordForm').reset(); renderSecurity(data); message('Password changed. Other sessions have been signed out.');
  }); };
  $('revokeOthers').onclick = () => perform(async () => {
    if (!confirm('Sign out all other sessions? This session will stay signed in.')) { message('Session changes canceled.'); return; }
    await verify(); renderSecurity(await api('security', {account_id:user.id,action:'revoke_others'})); message('Other sessions signed out.');
  });
  $('signOut').onclick = () => perform(async () => {
    if (!confirm('Sign out and clear this browser workspace? Save recent chart changes to your account first.')) { message('Sign-out canceled.'); return; }
    await verify(); const id = user.id; await api('logout',{account_id:id});
    signedOut();
    try { Object.keys(localStorage).filter(key => key.startsWith('atlas.')).forEach(key => localStorage.removeItem(key)); sessionStorage.removeItem('quantstack.accountRevision.' + id); localStorage.setItem('quantstack.accountChanged',crypto.randomUUID()); }
    catch { message('Signed out. Browser storage could not be cleared; clear site data before using a shared device.'); return; }
    location.assign('/');
  });
  $('noteSearch').oninput = renderNotes; $('noteFilter').onchange = renderNotes;
  $('reload').onclick = () => perform(load);
  window.addEventListener('storage', event => { if (event.key === 'quantstack.accountChanged') { signedOut(); location.reload(); } if (event.key === 'quantstack.accountWorkspaceChanged' && user) message('The account workspace changed in another tab. Reload saved data before editing.'); });
  perform(load);
})();
