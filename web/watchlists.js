// Named watchlists and symbol flags share the existing chart and Saved favorites.
(() => {
  'use strict';
  const key = 'atlas.watchlists.v1';
  const colors = {red:'#ff4d57',blue:'#2979ff',green:'#72bd83',orange:'#ffb52e',purple:'#ad63d2',cyan:'#00bfd8',pink:'#f486b0'};
  const fields = {
    close:{name:'Last price',short:'Last',help:'Latest stored price in the instrument’s quote currency.'},
    change_1d_pct:{name:'Daily change %',short:'Day %',help:'Latest completed daily close versus the previous daily close.'},
    return_1w_pct:{name:'Weekly change %',short:'Week %',help:'5 completed trading sessions for stocks; 7 daily bars for crypto.'},
    return_1m_pct:{name:'Monthly change %',short:'Month %',help:'21 completed trading sessions for stocks; 30 daily bars for crypto.'},
    market_cap:{name:'Market cap',short:'Mkt cap',help:'Stored provider market cap, in its reported currency. No currency conversion.'},
    volume:{name:'Daily volume',short:'Volume',help:'Volume of the latest completed daily bar.'}
  };
  const defaultView = () => ({sort:'symbol',direction:'asc',columns:['close','change_1d_pct'],names:true});
  const title = value => value[0].toUpperCase() + value.slice(1);
  const defaults = () => ({lists:[], labels:{}, active:'all', color:'all',view:defaultView()});
  function normalize(value) {
    const result=defaults(), ids=new Set();
    if (!value || typeof value!=='object') return result;
    if(Array.isArray(value.lists)) for(const list of value.lists.slice(0,50)) {
      if(!list || typeof list.id!=='string' || !list.id.startsWith('list-') || ids.has(list.id) || typeof list.name!=='string' || !list.name.trim() || !Array.isArray(list.symbols)) continue;
      ids.add(list.id);result.lists.push({id:list.id,name:list.name.trim().slice(0,60),symbols:[...new Set(list.symbols.filter(s=>typeof s==='string'&&s.length<=30))].slice(0,10000),...(Object.hasOwn(colors,list.color)?{color:list.color}:{})});
    }
    if(value.labels && typeof value.labels==='object') for(const [symbol,color] of Object.entries(value.labels).slice(0,50000)) {
      if(Object.hasOwn(colors,color) && /^[A-Z0-9^][A-Z0-9.^=-]{0,29}$/.test(symbol)) result.labels[symbol]=color;
    }
    if(['all','saved',...ids].includes(value.active)) result.active=value.active;
    if(['all','none',...Object.keys(colors)].includes(value.color)) result.color=value.color;
    if(value.view && typeof value.view==='object'){
      if(['symbol','name',...Object.keys(fields)].includes(value.view.sort))result.view.sort=value.view.sort;
      if(['asc','desc'].includes(value.view.direction))result.view.direction=value.view.direction;
      if(Array.isArray(value.view.columns))result.view.columns=[...new Set(value.view.columns.filter(key=>Object.hasOwn(fields,key)))];
      if(typeof value.view.names==='boolean')result.view.names=value.view.names;
    }
    return result;
  }
  function read() {try{return normalize(JSON.parse(localStorage.getItem(key)));}catch{return defaults();}}
  let state=read(), target=null, opener=null;
  const resolveList = list => list?.color ? {...list,symbols:Object.keys(state.labels).filter(symbol=>state.labels[symbol]===list.color)} : list;
  const activeList = () => resolveList(state.lists.find(list=>list.id===state.active));
  const choices = () => [{id:'saved',name:'Saved favorites',symbols:[...saved]},...state.lists.map(resolveList)];
  const status=document.createElement('p');status.id='watchlistStatus';status.setAttribute('role','status');status.setAttribute('aria-live','polite');status.hidden=true;
  const pane=$('workspaceSideWatchlist'), heading=pane.querySelector('.sidebar-heading');
  for(const node of [...heading.childNodes]) if(node.nodeType===Node.TEXT_NODE)node.remove();
  const selector=document.createElement('select');selector.id='watchlistSelect';selector.setAttribute('aria-label','Watchlist');
  heading.prepend(selector);
  const manage=document.createElement('button');manage.id='watchlistManage';manage.textContent='⋯';manage.title='Manage watchlists';manage.setAttribute('aria-label','Manage watchlists');manage.setAttribute('aria-haspopup','menu');heading.append(manage);
  const tools=document.createElement('div');tools.className='watchlist-controls';
  tools.innerHTML='<div id="watchlistColor" role="group" aria-label="Filter by color label"></div><button id="watchlistAddSymbol" type="button">+ Add symbol</button>';
  const saveColor=document.createElement('button');saveColor.id='watchlistSaveColor';saveColor.textContent='Save color as watchlist';saveColor.title='Save a named list that automatically follows this color label';tools.append(saveColor);
  pane.querySelector('#filters').after(tools);tools.after(status);
  const displayTools=document.createElement('div');displayTools.className='watchlist-display-tools';
  displayTools.innerHTML='<label>Sort <select id="watchlistSort" aria-label="Sort watchlist"></select></label><button id="watchlistDirection" type="button"></button><button id="watchlistColumns" type="button">Columns ±</button>';
  tools.after(displayTools);
  $('watchlistSort').replaceChildren(new Option('Symbol','symbol'),new Option('Name','name'),...Object.entries(fields).map(([key,field])=>new Option(field.name,key)));
  pane.querySelector('.list-heading').lastElementChild.textContent='';
  $('watchlistColor').innerHTML='<button type="button" data-label-filter="all" aria-label="All labels" title="All labels">All</button>'+Object.entries(colors).map(([color,hex])=>`<button type="button" class="watchlist-color-dot" data-label-filter="${color}" aria-label="${title(color)} labels" title="${title(color)} labels" style="--swatch:${hex}"></button>`).join('')+'<button type="button" class="watchlist-color-none" data-label-filter="none" aria-label="Unlabeled" title="Unlabeled">×</button>';
  function announce(message) {status.textContent=message;status.hidden=!message;}
  function refresh() {
    selector.replaceChildren(new Option('All instruments','all'),...choices().map(list=>new Option(list.name,list.id)));
    selector.value=state.active;
    for(const button of $('watchlistColor').querySelectorAll('button')){
      button.setAttribute('aria-pressed',String(button.dataset.labelFilter===(activeList()?.color||state.color)));
      button.disabled=!!activeList()?.color;
    }
    saveColor.disabled=!Object.hasOwn(colors,activeList()?.color||state.color);
    updateViewControls();
    renderAssets();
  }
  function updateViewControls(){
    $('watchlistSort').value=state.view.sort;
    $('watchlistDirection').textContent=state.view.direction==='asc'?'↑':'↓';
    $('watchlistDirection').setAttribute('aria-label',state.view.direction==='asc'?'Ascending; switch to descending':'Descending; switch to ascending');
    $('watchlistDirection').title=state.view.direction==='asc'?'Ascending':'Descending';
    $('assets').style.setProperty('--watchlist-grid',`minmax(90px,1fr)${' 76px'.repeat(state.view.columns.length)}`);
    $('assets').style.setProperty('--watchlist-table-width',`${132+76*state.view.columns.length}px`);
  }
  function sortBy(key){mutate(next=>{next.view.direction=next.view.sort===key?(next.view.direction==='asc'?'desc':'asc'):(['symbol','name'].includes(key)?'asc':'desc');next.view.sort=key;},'');}
  $('watchlistSort').onchange=()=>sortBy($('watchlistSort').value);
  $('watchlistDirection').onclick=()=>mutate(next=>{next.view.direction=next.view.direction==='asc'?'desc':'asc';},'');
  function compactNumber(value){
    for(const [scale,suffix] of [[1e12,'T'],[1e9,'B'],[1e6,'M'],[1e3,'K']])if(Math.abs(value)>=scale)return fmt(value/scale,2)+suffix;
    return fmt(value,0);
  }
  function metric(asset,key){
    const value=asset[key],field=fields[key],isReturn=key.endsWith('_pct'),valid=typeof value==='number'&&Number.isFinite(value);
    const stamp=isReturn||key==='volume'?asset.performance_asof:key==='market_cap'?asset.market_cap_asof:asset.quote_timestamp;
    const currency=key==='market_cap'?asset.market_cap_currency||asset.currency:asset.currency;
    const tip=field.help+(isReturn?` ${asset.performance_basis||'Unknown'} price basis.`:'')+(stamp?` As of ${stamp}.`:'')+(['market_cap','close'].includes(key)&&currency?` Currency: ${currency}.`:'');
    const text=!valid?'—':isReturn?pct(value):['market_cap','volume'].includes(key)?compactNumber(value):fmt(value);
    return `<span class="watchlist-cell ${valid&&isReturn?(value>=0?'positive':'negative'):''}" data-watchlist-cell="${key}" title="${esc(valid?tip:'Unavailable: '+tip)}">${text}</span>`;
  }
  function commit(next,message) {
    try {localStorage.setItem(key,JSON.stringify(next));state=next;refresh();announce(message);return true;}
    catch {announce('Browser storage is unavailable. Your watchlist changes were not saved.');return false;}
  }
  function mutate(fn,message) {const next=structuredClone(state);fn(next);return commit(next,message);}
  function toggleMembership(id,symbol) {
    if(id==='saved') {
      const next=new Set(saved);next.has(symbol)?next.delete(symbol):next.add(symbol);
      try {localStorage.setItem('atlas.saved',JSON.stringify([...next]));saved=next;if(selected)updateSaved();refresh();window.dispatchEvent(new Event('watchlist-favorites-changed'));announce(`${symbol} ${next.has(symbol)?'added to':'removed from'} Saved favorites.`);return true;}
      catch {announce('Browser storage is unavailable. Favorites were not changed.');return false;}
    }
    const list=state.lists.find(item=>item.id===id);if(!list)return false;
    if(list.color) {
      const removing=state.labels[symbol]===list.color;
      return mutate(next=>{if(removing)delete next.labels[symbol];else next.labels[symbol]=list.color;},`${symbol}: ${removing?'label cleared':title(list.color)+' label saved'}. Color watchlists updated.`);
    }
    return mutate(next=>{const item=next.lists.find(l=>l.id===id);item.symbols=item.symbols.includes(symbol)?item.symbols.filter(s=>s!==symbol):[...item.symbols,symbol];},`${symbol} ${list.symbols.includes(symbol)?'removed from':'added to'} ${list.name}.`);
  }
  window.quantWatchlists={
    signature:()=>state.active+'|'+state.color+'|'+state.view.sort+'|'+state.view.direction,
    sort:items=>items.slice().sort((a,b)=>{
      const key=state.view.sort,x=a[key],y=b[key],numeric=!['symbol','name'].includes(key);
      const missing=value=>numeric?typeof value!=='number'||!Number.isFinite(value):typeof value!=='string'||!value;
      if(missing(x)!==missing(y))return missing(x)?1:-1;
      const order=missing(x)?0:numeric?x-y:x.localeCompare(y,undefined,{numeric:true});
      return (state.view.direction==='asc'?order:-order)||a.symbol.localeCompare(b.symbol);
    }),
    header:()=>`<div class="watchlist-column-head">${['symbol',...state.view.columns].map(key=>`<button type="button" data-watchlist-sort="${key}" title="${esc(key==='symbol'?'Sort by symbol':fields[key].help)}" aria-label="Sort by ${esc(key==='symbol'?'symbol':fields[key].name)}" aria-pressed="${state.view.sort===key}">${key==='symbol'?'Symbol':fields[key].short}${state.view.sort===key?(state.view.direction==='asc'?' ↑':' ↓'):''}</button>`).join('')}</div>`,
    content:asset=>`<button class="asset watchlist-table-asset ${asset.symbol===selected?.symbol?'active':''}" data-symbol="${esc(asset.symbol)}"><span class="asset-info"><strong>${esc(asset.symbol)}${saved.has(asset.symbol)?' ★':''}</strong>${state.view.names?`<small>${esc(asset.name)}</small>`:''}</span>${state.view.columns.map(key=>metric(asset,key)).join('')}</button>`,
    accepts:symbol=>{
      const list=state.lists.find(l=>l.id===state.active);
      return (state.active==='all'||(state.active==='saved'?saved.has(symbol):list?.color?state.labels[symbol]===list.color:list?.symbols.includes(symbol))) && (state.color==='all'||(state.color==='none'?!state.labels[symbol]:state.labels[symbol]===state.color));
    },
    row:(symbol,html)=>{
      const color=state.labels[symbol];
      return `<div class="watchlist-row${color?' has-label':''}"${color?` style="--symbol-label:${colors[color]}"`:''}>${color?`<span class="watchlist-flag" role="img" aria-label="${title(color)} label" title="${title(color)} label"></span>`:''}${html}<button class="watchlist-row-menu" data-watchlist-menu="${esc(symbol)}" aria-label="Options for ${esc(symbol)}" aria-haspopup="menu" title="Instrument options">⋯</button></div>`;
    }
  };
  $('assets').classList.add('watchlist-table');
  $('assets').addEventListener('click',event=>{const button=event.target.closest('[data-watchlist-sort]');if(button){sortBy(button.dataset.watchlistSort);$('assets').querySelector(`[data-watchlist-sort="${button.dataset.watchlistSort}"]`)?.focus();}});
  selector.onchange=()=>{closeMenu(false);mutate(next=>{next.active=selector.value;if(state.lists.find(l=>l.id===next.active)?.color||activeList()?.color)next.color='all';},'');};
  $('watchlistColor').onclick=event=>{const button=event.target.closest('[data-label-filter]');if(button&&!button.disabled)mutate(next=>{next.color=button.dataset.labelFilter;},'');};
  saveColor.onclick=()=>openName('create',null,activeList()?.color||state.color);

  // A single bounded popover serves mouse, touch and keyboard entry points.
  const menu=document.createElement('div');menu.id='watchlistMenu';menu.className='watchlist-menu';menu.setAttribute('role','menu');menu.hidden=true;document.body.append(menu);
  let anchor={x:0,y:0};
  function closeMenu(restore=true) {menu.hidden=true;manage.setAttribute('aria-expanded','false');if(restore){const focus=opener?.isConnected?opener:target?$('assets').querySelector(`[data-watchlist-menu="${CSS.escape(target)}"]`)||selector:selector;focus.focus({preventScroll:true});}}
  function position() {menu.style.left='0px';menu.style.top='0px';const box=menu.getBoundingClientRect();menu.style.left=Math.max(8,Math.min(anchor.x,innerWidth-box.width-8))+'px';menu.style.top=Math.max(8,Math.min(anchor.y,innerHeight-box.height-8))+'px';}
  function showMenu(html,label) {menu.innerHTML=html;menu.setAttribute('aria-label',label);menu.hidden=false;position();menu.querySelector('button:not(:disabled)')?.focus({preventScroll:true});}
  const action=(name,label,disabled=false)=>`<button type="button" role="menuitem" data-action="${name}"${disabled?' disabled':''}>${label}</button>`;
  function instrumentMenu() {
    const color=state.labels[target],list=state.active==='saved'?choices()[0]:activeList();
    showMenu(`<div class="watchlist-menu-title">Flag / unflag <strong>${esc(target)}</strong></div><div class="watchlist-swatches" role="group" aria-label="Color label">${Object.entries(colors).map(([name,hex])=>`<button type="button" role="menuitemradio" aria-checked="${name===color}" aria-label="${title(name)} label" title="${title(name)}" data-color="${name}" style="--swatch:${hex}"></button>`).join('')}</div>${action('clear','Clear color label',!color)}<hr>${action('lists',`Add / remove ${esc(target)} in watchlists <span aria-hidden="true">›</span>`)}${action('compare',`Add ${esc(target)} to compare`,target===selected?.symbol)}${action('note',`Add note for ${esc(target)}`)}${list?action('remove',`Remove from ${esc(list.name)}`,!list.symbols.includes(target)):''}<hr>${action('create','Create watchlist…')}${action('add','Add symbol…')}`,`Options for ${target}`);
  }
  function listMenu() {
    showMenu(`${action('back','‹ Instrument options')}<div class="watchlist-menu-title">Watchlists for <strong>${esc(target)}</strong></div>${choices().map(list=>`<button type="button" role="menuitemcheckbox" aria-checked="${list.symbols.includes(target)}" data-list="${esc(list.id)}"><span>${esc(list.name)}</span><span aria-hidden="true">${list.symbols.includes(target)?'✓':'+'}</span></button>`).join('')}<hr>${action('create','Create watchlist…')}`,`Watchlists for ${target}`);
  }
  function openInstrument(symbol,element,x,y) {target=symbol;opener=element;anchor={x,y};instrumentMenu();}
  $('assets').addEventListener('contextmenu',event=>{
    const row=event.target.closest('.watchlist-row'),button=row?.querySelector('.asset');if(!button)return;
    event.preventDefault();openInstrument(button.dataset.symbol,button,event.clientX,event.clientY);
  });
  $('assets').addEventListener('click',event=>{
    const button=event.target.closest('[data-watchlist-menu]');if(!button)return;
    event.stopImmediatePropagation();const box=button.getBoundingClientRect();openInstrument(button.dataset.watchlistMenu,button,box.right,box.bottom);
  },true);
  $('assets').addEventListener('keydown',event=>{
    if(event.key!=='ContextMenu'&&!(event.shiftKey&&event.key==='F10'))return;
    const button=event.target.closest('.asset');if(!button)return;
    event.preventDefault();event.stopPropagation();const box=button.getBoundingClientRect();openInstrument(button.dataset.symbol,button,box.left,box.bottom);
  });
  manage.onclick=()=>{
    if(!menu.hidden&&opener===manage){closeMenu();return;}
    target=null;opener=manage;const box=manage.getBoundingClientRect();anchor={x:box.right,y:box.bottom};manage.setAttribute('aria-expanded','true');
    showMenu(`<div class="watchlist-menu-title">Watchlists</div>${action('create','Create watchlist…')}${action('rename','Rename current watchlist…',!activeList())}${action('delete','Delete current watchlist…',!activeList())}<hr>${action('add','Add symbol…')}`,'Manage watchlists');
  };
  menu.addEventListener('keydown',event=>{
    if(event.key==='Escape'){event.preventDefault();event.stopPropagation();closeMenu();return;}
    if(event.key==='Tab'){closeMenu();return;}
    if(!['ArrowDown','ArrowUp','ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();event.stopPropagation();const items=[...menu.querySelectorAll('button:not(:disabled)')],index=items.indexOf(document.activeElement);
    const next=event.key==='Home'?0:event.key==='End'?items.length-1:(index+(['ArrowUp','ArrowLeft'].includes(event.key)?-1:1)+items.length)%items.length;
    items[next]?.focus();
  });
  document.addEventListener('pointerdown',event=>{if(!menu.hidden&&!menu.contains(event.target)&&!manage.contains(event.target))closeMenu(false);});
  window.addEventListener('resize',()=>closeMenu(false));$('assets').addEventListener('scroll',()=>closeMenu(false));
  menu.onclick=async event=>{
    const button=event.target.closest('button');if(!button||button.disabled)return;
    if(button.dataset.color){const color=button.dataset.color,removing=state.labels[target]===color;mutate(next=>{if(removing)delete next.labels[target];else next.labels[target]=color;},`${target}: ${removing?'label cleared':title(color)+' label saved'}. Saved in this browser; color watchlists updated.`);closeMenu();return;}
    if(button.dataset.list){toggleMembership(button.dataset.list,target);listMenu();return;}
    switch(button.dataset.action) {
      case 'clear':mutate(next=>{delete next.labels[target];},`${target} label cleared. Saved in this browser; color watchlists updated.`);closeMenu();break;
      case 'lists':listMenu();break;
      case 'back':instrumentMenu();break;
      case 'remove':toggleMembership(state.active,target);closeMenu();break;
      case 'create':openName('create',target);break;
      case 'rename':openName('rename');break;
      case 'delete':openName('delete');break;
      case 'add':openAdd();break;
      case 'compare': {const error=window.addChartComparison(target);closeMenu();announce(error||`${target} added to chart comparison.`);break;}
      case 'note': {
        const symbol=target;closeMenu(false);await select(symbol);
        if($('workspaceSidenotes').hidden||$('workspaceSidebar').inert)document.querySelector('[data-side-panel="notes"]').click();
        $('workspaceNotes').focus();break;
      }
    }
  };

  const nameDialog=document.createElement('dialog');nameDialog.id='watchlistNameDialog';nameDialog.className='watchlist-dialog';nameDialog.setAttribute('aria-labelledby','watchlistNameTitle');
  nameDialog.innerHTML='<form id="watchlistNameForm"><div class="dialog-head"><h2 id="watchlistNameTitle"></h2><button type="button" id="watchlistNameClose" aria-label="Close watchlist dialog">×</button></div><p id="watchlistNameHelp"></p><label id="watchlistNameLabel">Name<input id="watchlistName" maxlength="60" required autocomplete="off"></label><p id="watchlistNameError" role="alert"></p><div class="dialog-actions"><button type="button" id="watchlistNameCancel">Cancel</button><button id="watchlistNameSubmit" type="submit">Create</button></div></form>';
  document.body.append(nameDialog);let nameMode, nameSymbol, nameId, nameColor;
  function openName(mode,symbol=null,color=null) {
    closeMenu(false);nameMode=mode;nameSymbol=symbol;nameId=state.active;nameColor=Object.hasOwn(colors,color)?color:null;
    const deleting=mode==='delete';$('watchlistNameTitle').textContent=nameColor?'Save color watchlist':{create:'Create watchlist',rename:'Rename watchlist',delete:'Delete watchlist'}[mode];
    $('watchlistNameHelp').textContent=deleting?`Delete “${activeList()?.name}”? Instruments, color labels and notes are kept.`:nameColor?`Includes all instruments with a ${title(nameColor)} label, across every watchlist. Updates automatically when labels change. Saved in this browser.`:symbol?`${symbol} will be added to your new watchlist.`:'Watchlists are saved in this browser.';
    $('watchlistNameLabel').hidden=deleting;$('watchlistName').disabled=deleting;$('watchlistName').value=mode==='rename'?activeList().name:nameColor?title(nameColor)+' flags':'';
    $('watchlistNameError').textContent='';$('watchlistNameSubmit').textContent={create:'Create',rename:'Rename',delete:'Delete'}[mode];nameDialog.showModal();
    if(!deleting)$('watchlistName').focus();
  }
  for(const id of ['watchlistNameClose','watchlistNameCancel'])$(id).onclick=()=>nameDialog.close();
  nameDialog.addEventListener('close',()=>selector.focus());
  $('watchlistNameForm').onsubmit=event=>{
    event.preventDefault();const name=$('watchlistName').value.trim();
    if(nameMode!=='delete'&&(!name||choices().some(list=>(nameMode!=='rename'||list.id!==nameId)&&list.name.toLowerCase()===name.toLowerCase())||name.toLowerCase()==='all instruments')){$('watchlistNameError').textContent='Enter a unique watchlist name.';return;}
    if(nameMode==='create'&&state.lists.length>=50){$('watchlistNameError').textContent='You can save up to 50 watchlists.';return;}
    const ok=mutate(next=>{
      if(nameMode==='create'){const id='list-'+crypto.randomUUID();next.lists.push({id,name,symbols:nameSymbol?[nameSymbol]:[],...(nameColor?{color:nameColor}:{})});next.active=id;next.color='all';}
      if(nameMode==='rename'){const list=next.lists.find(l=>l.id===nameId);if(list)list.name=name;}
      if(nameMode==='delete'){next.lists=next.lists.filter(l=>l.id!==nameId);next.active='all';}
    },nameMode==='delete'?'Watchlist deleted.':`Watchlist “${name}” saved.`);
    if(ok){if(nameMode==='create'){filter='All';$('search').value='';[...$('filters').children].forEach(b=>b.classList.toggle('active',b.dataset.filter===filter));renderAssets();}nameDialog.close();}
    else $('watchlistNameError').textContent='Could not save. Browser storage is unavailable.';
  };

  const addDialog=document.createElement('dialog');addDialog.id='watchlistAddDialog';addDialog.className='watchlist-dialog';addDialog.setAttribute('aria-labelledby','watchlistAddTitle');
  addDialog.innerHTML='<div class="dialog-head"><h2 id="watchlistAddTitle">Add symbol</h2><button id="watchlistAddClose" aria-label="Close add symbol">×</button></div><label>Watchlist<select id="watchlistAddTarget"></select></label><label>Find an instrument<input id="watchlistAddSearch" type="search" placeholder="Company or symbol"></label><div id="watchlistAddResults"></div><p id="watchlistAddStatus" role="status"></p><div id="watchlistCollect"><p>Missing an instrument? Add it to local collection first.</p></div>';
  document.body.append(addDialog);
  function renderAdd() {
    const list=choices().find(l=>l.id===$('watchlistAddTarget').value),q=$('watchlistAddSearch').value.trim().toLowerCase();
    let help=$('watchlistColorHelp');if(!help){help=document.createElement('p');help.id='watchlistColorHelp';$('watchlistAddResults').before(help);}
    help.hidden=!list?.color;help.textContent=list?.color?`Adding assigns the ${title(list.color)} label, replacing any previous label. Removing clears it. All color watchlists update automatically.`:'';
    $('watchlistAddResults').innerHTML=assets.filter(a=>`${a.symbol} ${a.name}`.toLowerCase().includes(q)).slice(0,60).map(a=>`<button type="button" data-add-symbol="${esc(a.symbol)}" aria-pressed="${!!list?.symbols.includes(a.symbol)}"><span><strong>${esc(a.symbol)}</strong><small>${esc(a.name)}</small></span><span>${list?.symbols.includes(a.symbol)?'Added ✓':'+ Add'}</span></button>`).join('')||'<p>No matching stored instruments.</p>';
  }
  function openAdd() {closeMenu(false);$('watchlistAddTarget').replaceChildren(...choices().map(l=>new Option(l.name,l.id)));$('watchlistAddTarget').value=state.active==='all'?'saved':state.active;$('watchlistAddSearch').value='';$('watchlistAddStatus').textContent='';renderAdd();addDialog.showModal();$('watchlistAddSearch').focus();}
  $('watchlistAddSymbol').onclick=openAdd;$('watchlistAddClose').onclick=()=>addDialog.close();
  $('watchlistAddSearch').oninput=$('watchlistAddTarget').onchange=renderAdd;
  $('watchlistAddResults').onclick=event=>{const button=event.target.closest('[data-add-symbol]');if(!button)return;toggleMembership($('watchlistAddTarget').value,button.dataset.addSymbol);renderAdd();$('watchlistAddStatus').textContent=status.textContent;$('watchlistAddResults').querySelector(`[data-add-symbol="${CSS.escape(button.dataset.addSymbol)}"]`)?.focus();};
  const columnDialog=document.createElement('dialog');columnDialog.id='watchlistColumnsDialog';columnDialog.className='watchlist-dialog';columnDialog.setAttribute('aria-labelledby','watchlistColumnsTitle');
  columnDialog.innerHTML='<div class="dialog-head"><h2 id="watchlistColumnsTitle">Watchlist columns</h2><button id="watchlistColumnsClose" aria-label="Close watchlist columns">×</button></div><p>Check to add a column; uncheck to remove it. Symbol always stays visible. Scroll the watchlist sideways when you add more columns.</p><div id="watchlistColumnChoices"></div><p>Returns use completed daily bars: stocks 5/21 trading sessions for week/month; crypto 7/30 daily bars. Market caps and prices retain their reported currencies; sorting does not convert currencies. Missing data stays last in either sort direction.</p><p id="watchlistColumnsStatus" role="status"></p><div class="dialog-actions"><button id="watchlistColumnsReset">Reset columns</button><button id="watchlistColumnsDone">Done</button></div>';
  document.body.append(columnDialog);
  function renderColumnChoices(){
    $('watchlistColumnChoices').innerHTML=`<label><input type="checkbox" data-watchlist-column="names" ${state.view.names?'checked':''}><span>Company / instrument name</span></label>`+Object.entries(fields).map(([key,field])=>`<label><input type="checkbox" data-watchlist-column="${key}" ${state.view.columns.includes(key)?'checked':''}><span>${esc(field.name)}<small>${esc(field.help)}</small></span></label>`).join('');
  }
  $('watchlistColumns').onclick=()=>{renderColumnChoices();$('watchlistColumnsStatus').textContent='Preferences save in this browser.';columnDialog.showModal();};
  $('watchlistColumnChoices').onchange=event=>{
    const box=event.target.closest('[data-watchlist-column]');if(!box)return;
    const key=box.dataset.watchlistColumn;
    const ok=mutate(next=>{if(key==='names')next.view.names=box.checked;else next.view.columns=Object.keys(fields).filter(field=>field===key?box.checked:next.view.columns.includes(field));},'');
    if(!ok)box.checked=key==='names'?state.view.names:state.view.columns.includes(key);
    $('watchlistColumnsStatus').textContent=ok?'Columns saved.':'Could not save: browser storage is unavailable.';
  };
  $('watchlistColumnsReset').onclick=()=>{if(mutate(next=>{next.view.columns=defaultView().columns;next.view.names=true;},'')){renderColumnChoices();$('watchlistColumnsStatus').textContent='Default columns restored.';}};
  for(const id of ['watchlistColumnsClose','watchlistColumnsDone'])$(id).onclick=()=>columnDialog.close();
  columnDialog.addEventListener('close',()=>$('watchlistColumns').focus());
  document.addEventListener('DOMContentLoaded', () => {
    const button = $('schedulerButton');
    if (!button) return;
    const collection = document.createElement('section');
    collection.className = 'watchlist-collection';
    collection.setAttribute('aria-labelledby', 'watchlistCollectionTitle');
    collection.innerHTML = '<span id="watchlistCollectionTitle" class="eyebrow">LOCAL DATA COLLECTION</span>';
    button.setAttribute('aria-haspopup', 'dialog');
    button.setAttribute('aria-controls', 'schedulerDialog');
    collection.append(button);
    heading.after(collection);
    // Keep the missing-instrument shortcut in the existing watchlist picker.
    const shortcut = document.createElement('button');
    shortcut.id = 'watchlistCollectButton';
    shortcut.type = 'button';
    shortcut.textContent = '+ Add symbols to local collection';
    shortcut.setAttribute('aria-haspopup', 'dialog');
    shortcut.setAttribute('aria-controls', 'schedulerDialog');
    shortcut.onclick = () => { addDialog.close(); button.click(); };
    $('watchlistCollect').append(shortcut);
    $('schedulerDialog').addEventListener('close', () => button.focus());
  });
  window.addEventListener('storage',event=>{
    if(event.key===key||event.key===null){state=read();closeMenu(false);refresh();}
    if(event.key==='atlas.saved'){
      try {const value=JSON.parse(event.newValue||'[]');saved=new Set(Array.isArray(value)?value.filter(s=>typeof s==='string'):[]);if(selected)updateSaved();refresh();if(!menu.hidden&&target)listMenu();}catch {}
    }
  });
  refresh();
})();
