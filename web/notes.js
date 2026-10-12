// Keep-compatible notebook fields extend the existing backup/account snapshot format.
(() => {
  'use strict';
  const KEY = 'atlas.ideaNotes.v1';
  const CATEGORY_KEY = 'atlas.noteCategories.v1';
  const validCategory = value => text(value, 40) && !!value.trim() && value === value.trim();
  const categories = ['Idea', 'Research', 'To do', 'Watchlist'];
  const colors = ['default', 'yellow', 'green', 'blue', 'pink'];
  const text = (value, limit) => typeof value === 'string' && value.length <= limit;
  function valid(note) {
    return note && text(note.id, 200) && !!note.id && text(note.title, 120) && text(note.body, 20000) && (note.bodyHtml === undefined || text(note.bodyHtml, 200000)) &&
      validCategory(note.category) && (note.symbol === undefined || text(note.symbol, 30)) &&
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
  window.QuantNotes = {valid, validCategory, flush: () => flush(), mount, discard, unmount() { if (!flush()) return false; cleanup(); flush = () => true; return true; }};
  window.addEventListener('pagehide', () => flush());
  window.addEventListener('beforeunload', event => { if (!flush()) { event.preventDefault(); event.returnValue = ''; } });
  document.addEventListener('visibilitychange', () => { if (document.hidden) flush(); });
  window.addEventListener('workspace:dock', () => flush());

  function mount(pane, symbol) {
    cleanup();
    pane.innerHTML = `<section class="research-tool idea-notebook">
      <div class="research-heading"><div><span class="eyebrow">NOTES & RESEARCH</span><h2>My notebook</h2><p>Your ideas, research and next steps in one place.</p></div><div class="idea-note-actions"><button id="ideaExpand" type="button">Expand notebook</button><button id="ideaNew" type="button">+ New note</button><button id="ideaNewChecklist" type="button">+ Checklist</button></div></div>
      <div class="idea-toolbar"><label>Search<input id="ideaSearch" type="search" placeholder="Title, text, symbol or label"></label>
      <label>View<select id="ideaView"><option value="notes">All active notes</option><option value="pinned">Pinned notes</option><option value="archive">Archive</option><option value="trash">Trash</option><option value="all">All notes</option></select></label>
      <select id="ideaCategoryFilter" aria-label="Filter by category" hidden><option value="">All categories</option></select>
      <button id="ideaAccount" type="button">Save to account</button><button id="ideaBackup" type="button">Download backup</button></div>
      <p id="ideaStatus" role="status" aria-live="polite">Notes autosave in this browser. Use Save to account for a server copy.</p>
      <div class="idea-shell"><div class="idea-category-sidebar"><nav id="ideaCategories" aria-label="Note categories"></nav>
      <details class="idea-category-manager"><summary>Manage categories</summary><label for="ideaCategoryName">Category name</label><input id="ideaCategoryName" maxlength="40" placeholder="e.g. Earnings"><button id="ideaCategoryCreate" type="button">Add category</button><button id="ideaCategoryRename" type="button">Rename selected</button><button id="ideaCategoryRemove" type="button">Remove selected</button><p>Removing a category moves its notes to Idea.</p></details></div><div class="idea-notebook-layout"><section class="idea-notes-list"><button id="ideaQuickNote" type="button">Take a note...</button><p id="ideaCount" class="research-help"></p><div id="ideaNotesList" aria-label="Saved notes"></div><button id="ideaMore" type="button" hidden>Show more notes</button></section>
      <form id="ideaEditor" class="idea-note-editor"><label for="ideaTitle">Title</label><input id="ideaTitle" maxlength="120" placeholder="Untitled note">
      <div class="idea-fields"><label>Category<select id="ideaCategory">${categories.map(c => `<option>${c}</option>`).join('')}</select></label><label>Symbol<input id="ideaSymbol" maxlength="30" placeholder="e.g. NVDA"></label><label>Colour<select id="ideaColor">${colors.map(c => `<option value="${c}">${c[0].toUpperCase()+c.slice(1)}</option>`).join('')}</select></label></div>
      <label for="ideaLabels">Labels (comma separated)</label><input id="ideaLabels" maxlength="819" placeholder="Earnings, long term">
      <span class="idea-body-label">Notes</span><div id="ideaRichEditor"></div>
      <fieldset class="idea-checklist"><legend>Checklist</legend><div id="ideaItems"></div><button id="ideaAddItem" type="button">Add checklist item</button></fieldset>
      <p id="ideaNoteDate" class="idea-note-date"></p><div class="idea-note-actions"><button id="ideaSave" type="submit">Save note</button><button id="ideaPin" type="button" aria-pressed="false">Pin</button><button id="ideaArchive" type="button">Archive</button><button id="ideaDuplicate" type="button">Duplicate</button><button id="ideaDelete" type="button">Move to trash</button><button id="ideaRestore" type="button" hidden>Restore note</button><button id="ideaPurge" type="button" hidden>Delete forever</button></div></form></div></div></section>`;
    const el = id => pane.querySelector('#' + id), editor = el('ideaEditor');
    let customCategories = [], notes = [], draft = null, base = null, dirty = false, timer, searchTimer, limit = 60, blocked = false;
    const status = (message, error = false) => { el('ideaStatus').textContent = message; el('ideaStatus').dataset.error = String(error); };
    const rich = QuantNoteEditor.create(el('ideaRichEditor'), {onChange:changed, onBeforeExport:save, getTitle:()=>el('ideaTitle').value, onMessage:status});
    const clone = value => JSON.parse(JSON.stringify(value));
    const normal = n => ({color:'default', labels:[], items:[], pinned:false, archived:false, trashed:false, ...n});
    function readCategories() {
      const saved = JSON.parse(localStorage.getItem(CATEGORY_KEY) || '[]');
      if (!Array.isArray(saved) || saved.length > 100 || !saved.every(validCategory)) throw Error('Saved categories could not be read. Download a backup before editing.');
      return saved;
    }
    function allCategories() { return [...new Set([...categories, ...customCategories, ...notes.map(n => n.category)])]; }
    function categoryNavigation() {
      const current = el('ideaCategoryFilter').value, names = allCategories();
      const counts = new Map(); let total = 0;
      for (const note of notes) if (!note.trashed && !note.archived) { total++; counts.set(note.category,(counts.get(note.category) || 0)+1); }
      for (const [id, value, blank] of [['ideaCategoryFilter',current,true], ['ideaCategory',draft?.category || 'Idea',false]]) {
        const select = el(id); select.replaceChildren();
        if (blank) select.add(new Option('All categories',''));
        names.forEach(name => select.add(new Option(name,name)));
        select.value = value;
      }
      const fragment = document.createDocumentFragment();
      for (const name of ['', ...names]) {
        const button = document.createElement('button'); button.type = 'button'; button.dataset.category = name;
        button.className = 'idea-category-link'; button.setAttribute('aria-pressed', String(current === name));
        const title = document.createElement('span'); title.textContent = name || 'All categories';
        const count = document.createElement('small'); count.textContent = name ? counts.get(name) || 0 : total;
        button.append(title,count); fragment.append(button);
      }
      el('ideaCategories').replaceChildren(fragment);
      el('ideaCategoryRename').disabled = !current || categories.includes(current) || blocked;
      el('ideaCategoryRemove').disabled = !current || categories.includes(current) || blocked;
    }
    function chooseCategory(name) {
      if (!save()) return;
      el('ideaCategoryFilter').value = name; el('ideaCategoryName').value = name;
      limit = 60; list(); categoryNavigation();
      [...el('ideaCategories').children].find(button => button.dataset.category === name)?.focus();
      status(name ? `Category: ${name}. New notes will be added here.` : 'Showing all categories.');
    }
    function manageCategory(action) {
      if (!save() || blocked) return;
      try {
        const old = el('ideaCategoryFilter').value, name = el('ideaCategoryName').value.trim();
        notes = read(); customCategories = readCategories();
        if (action !== 'remove' && !validCategory(name)) throw Error('Enter a category name of 1 to 40 characters.');
        if (action !== 'remove' && allCategories().some(c => c.toLowerCase() === name.toLowerCase())) throw Error('That category already exists. Choose it from the sidebar.');
        if (action !== 'create' && (!old || categories.includes(old))) throw Error('Select a custom category to rename or remove.');
        if (action === 'remove' && !confirm(`Remove the category "${old}" and move its notes to Idea? Notes will be kept.`)) return;
        const nextCategories = action === 'create' ? [...customCategories,name] : customCategories.filter(c => c !== old).concat(action === 'rename' ? [name] : []);
        if (nextCategories.length > 100) throw Error('You can create up to 100 custom categories.');
        const nextNotes = action === 'create' ? notes : notes.map(n => n.category === old ? {...n,category:action === 'rename' ? name : 'Idea'} : n);
        const beforeNotes = localStorage.getItem(KEY);
        let wroteNotes = false;
        try {
          if (action !== 'create') { localStorage.setItem(KEY,JSON.stringify(nextNotes)); wroteNotes = true; }
          localStorage.setItem(CATEGORY_KEY,JSON.stringify(nextCategories));
        } catch (error) {
          if (wroteNotes) {
            try { beforeNotes === null ? localStorage.removeItem(KEY) : localStorage.setItem(KEY,beforeNotes); }
            catch { throw Error('Storage failed during the category change. Your note text is safe; reopen the notebook to check its categories.'); }
          }
          throw error;
        }
        notes = nextNotes; customCategories = nextCategories;
        categoryNavigation(); el('ideaCategoryFilter').value = action === 'remove' ? 'Idea' : name;
        if (base) show(notes.find(n => n.id === base.id));
        else show(null);
        categoryNavigation(); list();
        status(action === 'remove' ? 'Category removed. Its notes are now in Idea.' : 'Category saved in this browser.');
      } catch (error) { status(error.message,true); }
    }
    function list() {
      const query = el('ideaSearch').value.trim().toLowerCase(), category = el('ideaCategoryFilter').value, view = el('ideaView').value;
      const matching = notes.filter(n => (view === 'trash' ? n.trashed : !n.trashed && (view === 'all' || (view === 'pinned' ? n.pinned && !n.archived : Boolean(n.archived) === (view === 'archive')))) &&
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
      categoryNavigation();
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
      rich.setEditable(!draft.trashed && !blocked);
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
      draft = normal(note ? clone(note) : {id:crypto.randomUUID(), title:'', body:'', symbol:symbol || '', category:el('ideaCategoryFilter').value || 'Idea'});
      selectedId = draft.id; dirty = false;
      categoryNavigation();
      for (const [id,k] of [['ideaTitle','title'],['ideaSymbol','symbol'],['ideaCategory','category'],['ideaColor','color']]) el(id).value = draft[k] || '';
      rich.setContent(draft);
      el('ideaLabels').value = draft.labels.join(', ');
      el('ideaNoteDate').textContent = draft.updatedAt ? 'Updated ' + new Date(draft.updatedAt).toLocaleString() : 'New note · autosaves as you type';
      items(); list();
    }
    function capture() {
      for (const [id,k] of [['ideaTitle','title'],['ideaSymbol','symbol'],['ideaCategory','category'],['ideaColor','color']]) draft[k] = el(id).value;
      Object.assign(draft,rich.getContent());
      draft.symbol = draft.symbol.trim().toUpperCase(); draft.labels = [...new Set(el('ideaLabels').value.split(',').map(v => v.trim()).filter(Boolean))];
      draft.items = [...el('ideaItems').children].map(row => ({text:row.querySelector('[type=text]').value, done:row.querySelector('[type=checkbox]').checked}));
    }
    function save() {
      clearTimeout(timer);
      if (!dirty) return true;
      capture();
      if (!base && !draft.title.trim() && !draft.body.trim() && !rich.hasContent() && !draft.items.some(i => i.text.trim())) { dirty = false; return true; }
      try {
        const latest = read(), current = latest.find(n => n.id === draft.id);
        const now = new Date().toISOString();
        let next = {...draft, createdAt:draft.createdAt || now, updatedAt:now};
        const conflict = base && JSON.stringify(current) !== JSON.stringify(base);
        if (conflict) next = {...next, id:crypto.randomUUID(), title:(next.title || 'Untitled note').slice(0, 104) + ' (conflict copy)'};
        if (!valid(next)) throw Error('Note limit: 20,000 text characters, 200 KB of formatting, 20 labels (40 characters each) and 100 separate checklist items. Your edits are still in the editor.');
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
    function changed() { if (blocked || !draft || draft.trashed) return; dirty = true; status('Saving...'); clearTimeout(timer); timer = setTimeout(save, 350); }
    editor.oninput = changed;
    editor.onchange = () => { changed(); controls(); };
    editor.onsubmit = event => { event.preventDefault(); save(); };
    editor.onkeydown = event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); } };
    el('ideaNew').onclick = () => { if (save()) { el('ideaView').value = 'notes'; show(null); el('ideaTitle').focus(); editor.scrollIntoView({block:'nearest'}); } };
    el('ideaQuickNote').onclick = () => el('ideaNew').click();
    el('ideaNewChecklist').onclick = () => { if (save()) { el('ideaView').value = 'notes'; show(null); el('ideaAddItem').click(); } };
    el('ideaCategories').onclick = event => { const button = event.target.closest('[data-category]'); if (button) chooseCategory(button.dataset.category); };
    for (const [id,action] of [['ideaCategoryCreate','create'],['ideaCategoryRename','rename'],['ideaCategoryRemove','remove']]) el(id).onclick = () => manageCategory(action);
    el('ideaExpand').onclick = () => { document.getElementById('workspaceMaximizeDock').click(); el('ideaExpand').textContent = document.body.classList.contains('workspace-dock-maximized') ? 'Restore notebook size' : 'Expand notebook'; };
    el('ideaNotesList').onclick = event => { const card = event.target.closest('[data-idea-open]'); if (card && save()) { notes = read(); show(notes.find(n => n.id === card.dataset.ideaOpen)); editor.scrollIntoView({block:'nearest'}); el('ideaTitle').focus({preventScroll:true}); } };
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
      if (![KEY,CATEGORY_KEY,null].includes(event.key)) return;
      try { notes = read(); customCategories = readCategories(); if (!dirty) show(notes.find(n => n.id === selectedId) || null); else { list(); status('Notes changed in another tab. Your pending edits will be preserved.'); } }
      catch (error) { status(error.message, true); }
    }
    window.addEventListener('storage', onStorage);
    cleanup = () => { rich.destroy(); clearTimeout(timer); clearTimeout(searchTimer); window.removeEventListener('storage', onStorage); };
    try { notes = read(); customCategories = readCategories(); show(notes.find(n => n.id === selectedId) || notes.filter(n => !n.trashed && !n.archived).sort((a,b) => b.updatedAt.localeCompare(a.updatedAt))[0] || null); }
    catch (error) { blocked = true; show(null); editor.inert = true; el('ideaNew').disabled = true; status(error.message, true); }
  }
})();
