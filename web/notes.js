// Keep-compatible notebook fields extend the existing backup/account snapshot format.
(() => {
  'use strict';
  const KEY = 'atlas.ideaNotes.v1';
  const categories = ['Idea', 'Research', 'To do', 'Watchlist'];
  const colors = ['default', 'yellow', 'green', 'blue', 'pink'];
  const text = (value, limit) => typeof value === 'string' && value.length <= limit;
  function valid(note) {
    return note && text(note.id, 200) && !!note.id && text(note.title, 120) && text(note.body, 20000) &&
      categories.includes(note.category) && (note.symbol === undefined || text(note.symbol, 30)) &&
      text(note.updatedAt, 80) && Number.isFinite(Date.parse(note.updatedAt)) &&
      ['pinned', 'archived', 'trashed'].every(k => note[k] === undefined || typeof note[k] === 'boolean') &&
      (note.color === undefined || colors.includes(note.color)) &&
      (note.labels === undefined || (Array.isArray(note.labels) && note.labels.length <= 20 && note.labels.every(v => text(v, 40)))) &&
      (note.items === undefined || (Array.isArray(note.items) && note.items.length <= 100 && note.items.every(v => v && text(v.text, 300) && typeof v.done === 'boolean')));
  }
  function read() {
    const notes = JSON.parse(localStorage.getItem(KEY) || '[]');
    if (!Array.isArray(notes) || notes.length > 10000 || !notes.every(valid) || new Set(notes.map(n => n.id)).size !== notes.length)
      throw Error('Saved notes could not be read. Download a backup before repairing browser data.');
    return notes;
  }
  let flush = () => true, cleanup = () => {}, selectedId = null;
  const discard = () => { cleanup(); flush = () => true; selectedId = null; };
  window.QuantNotes = {valid, flush: () => flush(), mount, discard, unmount() { if (!flush()) return false; cleanup(); flush = () => true; return true; }};
  window.addEventListener('pagehide', () => flush());
  window.addEventListener('beforeunload', event => { if (!flush()) { event.preventDefault(); event.returnValue = ''; } });
  document.addEventListener('visibilitychange', () => { if (document.hidden) flush(); });
  window.addEventListener('workspace:dock', () => flush());

  function mount(pane, symbol) {
    cleanup();
    pane.innerHTML = `<section class="research-tool idea-notebook">
      <div class="research-heading"><div><span class="eyebrow">NOTES & RESEARCH</span><h2>Idea notebook</h2><p>Capture an idea. Make a checklist. Pick up where you left off.</p></div><div class="idea-note-actions"><button id="ideaExpand" type="button">Expand notebook</button><button id="ideaNew" type="button">New note</button></div></div>
      <div class="idea-toolbar"><label>Search<input id="ideaSearch" type="search" placeholder="Title, text, symbol or label"></label>
      <label>View<select id="ideaView"><option value="notes">Notes</option><option value="archive">Archive</option><option value="trash">Trash</option><option value="all">All notes</option></select></label>
      <label>Category<select id="ideaCategoryFilter"><option value="">All categories</option>${categories.map(c => `<option>${c}</option>`).join('')}</select></label>
      <button id="ideaAccount" type="button">Save to account</button><button id="ideaBackup" type="button">Download backup</button></div>
      <p id="ideaStatus" role="status" aria-live="polite">Notes autosave in this browser. Use Save to account for a server copy.</p>
      <div class="idea-notebook-layout"><section class="idea-notes-list"><p id="ideaCount" class="research-help"></p><div id="ideaNotesList" aria-label="Saved notes"></div><button id="ideaMore" type="button" hidden>Show more notes</button></section>
      <form id="ideaEditor" class="idea-note-editor"><label for="ideaTitle">Title</label><input id="ideaTitle" maxlength="120" placeholder="Untitled note">
      <div class="idea-fields"><label>Category<select id="ideaCategory">${categories.map(c => `<option>${c}</option>`).join('')}</select></label><label>Symbol<input id="ideaSymbol" maxlength="30" placeholder="e.g. NVDA"></label><label>Colour<select id="ideaColor">${colors.map(c => `<option value="${c}">${c[0].toUpperCase()+c.slice(1)}</option>`).join('')}</select></label></div>
      <label for="ideaLabels">Labels (comma separated)</label><input id="ideaLabels" maxlength="819" placeholder="Earnings, long term">
      <label for="ideaBody">Notes</label><textarea id="ideaBody" maxlength="20000" rows="7" placeholder="Write your thesis, links or next steps..."></textarea>
      <fieldset class="idea-checklist"><legend>Checklist</legend><div id="ideaItems"></div><button id="ideaAddItem" type="button">Add checklist item</button></fieldset>
      <p id="ideaNoteDate" class="idea-note-date"></p><div class="idea-note-actions"><button id="ideaSave" type="submit">Save note</button><button id="ideaPin" type="button" aria-pressed="false">Pin</button><button id="ideaArchive" type="button">Archive</button><button id="ideaDuplicate" type="button">Duplicate</button><button id="ideaDelete" type="button">Move to trash</button><button id="ideaRestore" type="button" hidden>Restore note</button><button id="ideaPurge" type="button" hidden>Delete forever</button></div></form></div></section>`;
    const el = id => pane.querySelector('#' + id), editor = el('ideaEditor');
    let notes = [], draft = null, base = null, dirty = false, timer, searchTimer, limit = 60, blocked = false;
    const status = (message, error = false) => { el('ideaStatus').textContent = message; el('ideaStatus').dataset.error = String(error); };
    const clone = value => JSON.parse(JSON.stringify(value));
    const normal = n => ({color:'default', labels:[], items:[], pinned:false, archived:false, trashed:false, ...n});
    function list() {
      const query = el('ideaSearch').value.trim().toLowerCase(), category = el('ideaCategoryFilter').value, view = el('ideaView').value;
      const matching = notes.filter(n => (view === 'trash' ? n.trashed : !n.trashed && (view === 'all' || Boolean(n.archived) === (view === 'archive'))) &&
        (!category || category === n.category) && `${n.title} ${n.body} ${n.symbol || ''} ${(n.labels || []).join(' ')} ${(n.items || []).map(i => i.text).join(' ')}`.toLowerCase().includes(query))
        .sort((a,b) => Number(!!b.pinned)-Number(!!a.pinned) || b.updatedAt.localeCompare(a.updatedAt));
      const fragment = document.createDocumentFragment();
      for (const n of matching.slice(0, limit)) {
        const card = document.createElement('button'); card.type = 'button'; card.className = 'idea-note-item'; card.dataset.ideaOpen = n.id; card.dataset.color = n.color || 'default';
        card.classList.toggle('active', n.id === draft?.id); card.setAttribute('aria-pressed', String(n.id === draft?.id));
        const title = document.createElement('strong'); title.textContent = (n.pinned ? 'Pinned · ' : '') + (n.title || 'Untitled note');
        const body = document.createElement('span'); body.textContent = n.body.slice(0, 160) || (n.items || []).map(i => (i.done ? '[x] ' : '[ ] ') + i.text).join('\n').slice(0, 160);
        const meta = document.createElement('small'); meta.textContent = [n.symbol, n.category, ...(n.labels || []), n.items?.length ? `${n.items.filter(i => i.done).length}/${n.items.length} done` : ''].filter(Boolean).join(' · ');
        card.append(title, body, meta); fragment.append(card);
      }
      if (!matching.length) { const empty = document.createElement('p'); empty.className = 'research-help'; empty.textContent = 'No notes here yet. Create a note or change your filters.'; fragment.append(empty); }
      el('ideaNotesList').replaceChildren(fragment); el('ideaCount').textContent = `${matching.length} notes${matching.length > limit ? ` · showing ${limit}` : ''}`;
      el('ideaMore').hidden = matching.length <= limit;
    }
    function controls() {
      el('ideaPin').textContent = draft.pinned ? 'Unpin' : 'Pin'; el('ideaPin').setAttribute('aria-pressed', String(draft.pinned));
      el('ideaArchive').textContent = draft.archived ? 'Unarchive' : 'Archive';
      el('ideaRestore').hidden = !draft.trashed;
      el('ideaPurge').hidden = !draft.trashed;
      for (const id of ['ideaSave','ideaPin','ideaArchive','ideaDelete','ideaDuplicate','ideaAddItem']) el(id).disabled = draft.trashed || blocked;
      for (const input of editor.querySelectorAll('input,textarea,select')) input.disabled = draft.trashed || blocked;
      el('ideaDelete').disabled ||= !base;
      editor.dataset.color = draft.color;
    }
    function items() {
      const fragment = document.createDocumentFragment();
      draft.items.forEach((item, index) => {
        const row = document.createElement('div'); row.className = 'idea-check-row'; row.dataset.index = index;
        const check = document.createElement('input'); check.type = 'checkbox'; check.checked = item.done; check.setAttribute('aria-label', `Complete item ${index+1}`);
        const input = document.createElement('input'); input.type = 'text'; input.maxLength = 300; input.value = item.text; input.setAttribute('aria-label', `Checklist item ${index+1}`);
        const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove'; remove.dataset.remove = index; remove.disabled = draft.trashed; remove.setAttribute('aria-label', `Remove item ${index+1}`);
        row.append(check, input, remove); fragment.append(row);
      });
      el('ideaItems').replaceChildren(fragment); controls();
    }
    function show(note) {
      base = note ? clone(note) : null;
      draft = normal(note ? clone(note) : {id:crypto.randomUUID(), title:'', body:'', symbol:symbol || '', category:'Idea'});
      selectedId = draft.id; dirty = false;
      for (const [id,k] of [['ideaTitle','title'],['ideaBody','body'],['ideaSymbol','symbol'],['ideaCategory','category'],['ideaColor','color']]) el(id).value = draft[k] || '';
      el('ideaLabels').value = draft.labels.join(', ');
      el('ideaNoteDate').textContent = draft.updatedAt ? 'Updated ' + new Date(draft.updatedAt).toLocaleString() : 'New note · autosaves as you type';
      items(); list();
    }
    function capture() {
      for (const [id,k] of [['ideaTitle','title'],['ideaBody','body'],['ideaSymbol','symbol'],['ideaCategory','category'],['ideaColor','color']]) draft[k] = el(id).value;
      draft.symbol = draft.symbol.trim().toUpperCase(); draft.labels = [...new Set(el('ideaLabels').value.split(',').map(v => v.trim()).filter(Boolean))];
      draft.items = [...el('ideaItems').children].map(row => ({text:row.querySelector('[type=text]').value, done:row.querySelector('[type=checkbox]').checked}));
    }
    function save() {
      clearTimeout(timer);
      if (!dirty) return true;
      capture();
      if (!base && !draft.title.trim() && !draft.body.trim() && !draft.items.some(i => i.text.trim())) { dirty = false; return true; }
      try {
        const latest = read(), current = latest.find(n => n.id === draft.id);
        const now = new Date().toISOString();
        let next = {...draft, createdAt:draft.createdAt || now, updatedAt:now};
        const conflict = base && JSON.stringify(current) !== JSON.stringify(base);
        if (conflict) next = {...next, id:crypto.randomUUID(), title:(next.title || 'Untitled note').slice(0, 104) + ' (conflict copy)'};
        if (!valid(next)) throw Error('Use at most 20 labels of 40 characters each and 100 checklist items. Your edits are still in the editor.');
        const updated = [next, ...latest.filter(n => n.id !== next.id)];
        if (updated.length > 10000) throw Error('Notebook is full. Export a backup before removing stored notes.');
        localStorage.setItem(KEY, JSON.stringify(updated));
        notes = updated; draft = clone(next); base = clone(next); selectedId = next.id; dirty = false;
        if (conflict) el('ideaTitle').value = next.title;
        el('ideaNoteDate').textContent = 'Updated ' + new Date(now).toLocaleString();
        controls(); list(); status(conflict ? 'Another tab changed this note. Saved your edits as a conflict copy.' : 'Saved in this browser. Use Save to account for a server copy.');
        return true;
      } catch (error) { status(error.message || 'Could not save. Browser storage is full or unavailable; keep this tab open and copy your edits.', true); return false; }
    }
    flush = save;
    function changed() { if (blocked || draft.trashed) return; dirty = true; status('Saving...'); clearTimeout(timer); timer = setTimeout(save, 350); }
    editor.oninput = changed;
    editor.onchange = () => { changed(); controls(); };
    editor.onsubmit = event => { event.preventDefault(); save(); };
    editor.onkeydown = event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); } };
    el('ideaNew').onclick = () => { if (save()) { show(null); el('ideaTitle').focus(); } };
    el('ideaExpand').onclick = () => { document.getElementById('workspaceMaximizeDock').click(); el('ideaExpand').textContent = document.body.classList.contains('workspace-dock-maximized') ? 'Restore notebook size' : 'Expand notebook'; };
    el('ideaNotesList').onclick = event => { const card = event.target.closest('[data-idea-open]'); if (card && save()) { notes = read(); show(notes.find(n => n.id === card.dataset.ideaOpen)); } };
    el('ideaSearch').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { limit = 60; list(); }, 120); };
    for (const id of ['ideaCategoryFilter', 'ideaView']) el(id).onchange = () => { limit = 60; list(); };
    el('ideaMore').onclick = () => { limit += 60; list(); };
    el('ideaAddItem').onclick = () => { capture(); if (draft.items.length >= 100) { status('A note can contain up to 100 checklist items.', true); return; } draft.items.push({text:'',done:false}); items(); changed(); el('ideaItems').lastElementChild.querySelector('[type=text]').focus(); };
    el('ideaItems').onclick = event => { const button = event.target.closest('[data-remove]'); if (!button) return; capture(); draft.items.splice(Number(button.dataset.remove), 1); items(); changed(); };
    for (const [id, field] of [['ideaPin','pinned'],['ideaArchive','archived'],['ideaDelete','trashed']]) el(id).onclick = () => { if (!save()) return; draft[field] = !draft[field]; dirty = true; save(); };
    el('ideaRestore').onclick = () => { draft.trashed = false; draft.archived = false; dirty = true; if (save()) { el('ideaView').value = 'notes'; show(base); } };
    el('ideaPurge').onclick = () => {
      if (!draft.trashed || !confirm('Permanently delete this note? This cannot be undone.')) return;
      try {
        const latest = read(), current = latest.find(n => n.id === draft.id);
        if (JSON.stringify(current) !== JSON.stringify(base)) throw Error('This note changed in another tab. Reopen it before deleting.');
        const updated = latest.filter(n => n.id !== draft.id);
        localStorage.setItem(KEY, JSON.stringify(updated)); notes = updated; show(null); status('Note permanently deleted from this browser.');
      } catch (error) { status(error.message, true); }
    };
    el('ideaDuplicate').onclick = () => { if (!save()) return; draft = {...clone(draft), id:crypto.randomUUID(), title:(draft.title || 'Untitled note').slice(0, 113)+' (copy)', createdAt:undefined}; base = null; el('ideaTitle').value = draft.title; dirty = true; save(); };
    el('ideaAccount').onclick = () => { if (save()) document.getElementById('accountButton')?.click(); };
    el('ideaBackup').onclick = () => { if (save()) document.getElementById('personalWorkExport')?.click(); };
    function onStorage(event) {
      if (event.key !== KEY && event.key !== null) return;
      try { notes = read(); if (!dirty) show(notes.find(n => n.id === selectedId) || null); else { list(); status('Notes changed in another tab. Your pending edits will be preserved.'); } }
      catch (error) { status(error.message, true); }
    }
    window.addEventListener('storage', onStorage);
    cleanup = () => { clearTimeout(timer); clearTimeout(searchTimer); window.removeEventListener('storage', onStorage); };
    try { notes = read(); show(notes.find(n => n.id === selectedId) || notes.filter(n => !n.trashed && !n.archived).sort((a,b) => b.updatedAt.localeCompare(a.updatedAt))[0] || null); }
    catch (error) { blocked = true; show(null); editor.inert = true; el('ideaNew').disabled = true; status(error.message, true); }
  }
})();
