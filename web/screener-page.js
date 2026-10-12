// Compact quick filters and an atomic, multi-condition manual filter editor.
window.initializeScreenerPage = () => {
  'use strict';
  const api = window.ScreenerControls, el = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const categories = ['country', 'exchange', 'sector', 'industry', 'currency'];
  const numeric = Object.keys(api.fields).filter(key => !['text', 'date'].includes(api.fields[key].type));
  const groups = {'Descriptive': categories, 'Fundamental': ['market_cap','pe','forward_pe','pb','beta','dividend_yield_pct','revenue_growth_pct','profit_margin_pct'], 'Performance': numeric.filter(key => /^(close|change_|return_|low_|high_|distance_|range_)/.test(key)), 'Technical': numeric.filter(key => !/^(close|change_|return_|low_|high_|distance_|range_)/.test(key) && !['market_cap','pe','forward_pe','pb','beta','dividend_yield_pct','revenue_growth_pct','profit_margin_pct'].includes(key))};
  const quick = new Set(['country','exchange','close','change_1d_pct','market_cap','pe','dividend_yield_pct','sector','return_1m_pct','revenue_growth_pct','relative_volume','rsi14','beta']);
  const rule = (field, operator, value, maximum = '') => ({field, operator, value:String(value), maximum:String(maximum), currency: 'USD'});
  function presets(key) {
    if (key === 'close') return [['Above 100','gt',100],['10 to 100','between',10,100],['10 and below','lte',10],['5 and below','lte',5],['Above 20','gt',20]];
    if (key === 'market_cap') return [['Mega · above 200B','gte',200],['Large · 10B to 200B','between',10,200],['Mid · 2B to 10B','between',2,10],['Small · 300M to 2B','between',.3,2]];
    if (key === 'rsi14') return [['Oversold · below 30','lt',30],['Neutral · 30 to 70','between',30,70],['Overbought · above 70','gt',70]];
    if (['pe','forward_pe'].includes(key)) return [['Positive · 0 to 15','between',0,15],['Positive · 0 to 25','between',0,25],['Above 25','gt',25]];
    if (api.fields[key].scale === 1e6) return [['Above 500K','gt',.5],['Above 1M','gt',1],['Above 5M','gt',5],['Above 10M','gt',10]];
    if (key === 'relative_volume') return [['Above 1×','gt',1],['Above 2×','gt',2],['Above 3×','gt',3],['Below 1×','lt',1]];
    if (key === 'beta') return [['Below 1','lt',1],['1 to 2','between',1,2],['Above 2','gt',2]];
    if (key === 'dividend_yield_pct') return [['Above 0%','gt',0],['Above 2%','gt',2],['Above 5%','gt',5]];
    return [['Above 0','gt',0],['Above 5','gt',5],['Above 10','gt',10],['Below 0','lt',0],['Below 20','lt',20]];
  }
  document.body.classList.add('screener-page');
  const head = document.createElement('section'); head.id = 'screenControlBar';
  head.innerHTML = `<div class="screen-title-row"><div><span class="screen-kicker">Stock Screener</span><h2 id="screenNameTitle">All stocks</h2></div><button id="screenSavedToggle">Saved screens ⌄</button><button id="screenManual">Manual setup <span id="screenConditionCount"></span></button><button id="screenReset">Reset all</button><span class="screen-data-label">Stored daily data</span></div><div id="screenQuickFilters" class="screen-quick-filters"></div><div id="screenSavedPanel" hidden></div>`;
  el('screenerContent').prepend(head);
  el('screenSavedPanel').append(document.querySelector('.screener-saved'));
  el('screenSavedToggle').setAttribute('aria-expanded', 'false');
  el('screenSavedToggle').onclick = () => { const panel = el('screenSavedPanel'); panel.hidden = !panel.hidden; el('screenSavedToggle').setAttribute('aria-expanded', String(!panel.hidden)); };
  const toolbar = document.createElement('div'); toolbar.className = 'screen-table-toolbar';
  el('screenerColumnSets').before(toolbar);
  toolbar.append(el('screenerColumnSets'), el('refreshScreener'));
  const columnButton = document.createElement('button'); columnButton.textContent = '+'; columnButton.setAttribute('aria-label', 'Customize table columns');
  columnButton.onclick = () => { const details = el('screenerColumnsDetails'); details.open = !details.open; if (details.open) el('screenerColumnSearch').focus(); };
  toolbar.append(columnButton);
  const density = document.createElement('button'); density.id='screenDensity';
  density.title='Switch between comfortable and compact table rows';
  let compact=false;try{compact=localStorage.getItem('atlas.screener.compact')==='true';}catch{}
  function applyDensity(){document.body.classList.toggle('screen-compact',compact);density.textContent=compact?'Compact rows':'Comfortable rows';density.setAttribute('aria-pressed',String(compact));}
  density.onclick=()=>{compact=!compact;applyDensity();try{localStorage.setItem('atlas.screener.compact',String(compact));}catch{}};applyDensity();
  const focus = document.createElement('button');focus.id='screenFocus';focus.textContent='Focus table';focus.setAttribute('aria-pressed','false');
  function setFocus(value){document.body.classList.toggle('screen-focus',value);focus.textContent=value?'Show filters':'Focus table';focus.setAttribute('aria-pressed',String(value));}
  focus.onclick=()=>setFocus(!document.body.classList.contains('screen-focus'));
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'&&document.body.classList.contains('screen-focus')&&!document.querySelector('dialog[open]')){
      event.preventDefault();event.stopImmediatePropagation();setFocus(false);focus.focus();
    }
  },true);
  toolbar.append(density,focus);
  el('screenerColumnSets').before(document.querySelector('.screener-search'));
  el('screenerSearch').placeholder = 'Search symbols';
  const more = document.createElement('details'); more.className = 'screen-extra-options'; more.innerHTML = '<summary>Table options & data coverage</summary>';
  el('screenerContent').append(more);
  for (const node of [document.querySelector('.tool-advanced'), el('screenerBreadth'), el('workspaceScreenerFilters'), document.querySelector('.screener-methodology')]) if (node) more.append(node);
  const pop = document.createElement('div'); pop.id = 'screenFilterPopover'; pop.hidden = true; pop.setAttribute('role','dialog'); pop.setAttribute('aria-label','Filter options'); document.body.append(pop);
  let opener = null;
  function closePop(focus = false) { pop.hidden = true; if (opener?.isConnected) { opener.setAttribute('aria-expanded','false'); if (focus) opener.focus(); } }
  function showPop(button) {
    closePop(); opener = button; button.setAttribute('aria-expanded','true'); pop.hidden = false;
    const rect = button.getBoundingClientRect();
    pop.style.left = Math.max(12, Math.min(rect.left, innerWidth - 336)) + 'px';
    pop.style.top = Math.min(rect.bottom + 6, innerHeight - 220) + 'px';
    pop.style.maxHeight = Math.max(190, innerHeight - parseFloat(pop.style.top) - 16) + 'px';
    pop.querySelector('input,button')?.focus();
  }
  function apply(draft) { try { api.apply(draft); closePop(); } catch (error) { el('screenerNotice').textContent = error.message; } }
  function choose(key, spec) {
    const draft = api.read(); draft.rules = draft.rules.filter(item => item.field !== key);
    if (spec) draft.rules.push(rule(key, ...spec.slice(1)));
    apply(draft);
  }
  function filterMenu(key, button) {
    const field = api.fields[key];
    pop.innerHTML = `<div class="screen-pop-title"><strong>${esc(field.label)}</strong><button id="screenRemoveFilter" aria-label="Remove ${esc(field.label)} filter">×</button></div><input id="screenPresetSearch" type="search" placeholder="Search" aria-label="Search filter choices"><div id="screenPresetChoices"></div><button id="screenManualFromPop" class="screen-pop-footer">☷ Manual setup</button>`;
    const options = categories.includes(key) ? [['Any', ''], ...api.choices(key).map(value => [value,value])] : [['Any', null], ...presets(key).map(spec => [spec[0],spec])];
    for (const [label, value] of options) {
      const b = document.createElement('button'); b.textContent = label; b.className = 'screen-preset-option';
      b.onclick = () => { if (categories.includes(key)) { const d = api.read(); d[key] = value; if (key === 'sector') d.industry = ''; if (key === 'currency') d.capTier = ''; apply(d); } else choose(key,value); }; el('screenPresetChoices').append(b);
    }
    if (key === 'close') for (const [label, operator] of [['Above EMA 50','gt'],['Below EMA 50','lt']]) {
      const b = document.createElement('button'); b.textContent = label; b.className = 'screen-preset-option'; b.onclick = () => choose('ema50_distance_pct', [label,operator,0]); el('screenPresetChoices').append(b);
    }
    el('screenRemoveFilter').onclick = () => { const d=api.read(); d.rules=d.rules.filter(r=>r.field!==key); if(categories.includes(key)) d[key]=''; if(key==='sector') d.industry=''; apply(d); };
    el('screenPresetSearch').oninput = event => { for(const b of el('screenPresetChoices').children) b.hidden=!b.textContent.toLowerCase().includes(event.target.value.toLowerCase()); };
    el('screenManualFromPop').onclick = () => { closePop(); openManual(key); };
    showPop(button);
  }
  function picker(button) {
    pop.innerHTML = '<div class="screen-pop-title"><strong>Add filter</strong></div><input id="screenFilterSearch" type="search" placeholder="Search all filters" aria-label="Search all filters"><div id="screenFilterCatalog"></div>';
    for(const [group, keys] of Object.entries(groups)) {
      const section=document.createElement('section'); section.innerHTML=`<h3>${esc(group)}</h3>`;
      for(const key of keys) { const b=document.createElement('button'); b.textContent=api.fields[key].label; b.onclick=()=>{quick.add(key); renderQuick(); filterMenu(key,el('screenQuickFilters').querySelector(`[data-quick="${key}"]`));}; section.append(b); }
      el('screenFilterCatalog').append(section);
    }
    el('screenFilterSearch').oninput=event=>{for(const section of el('screenFilterCatalog').children){let count=0;for(const b of section.querySelectorAll('button')){b.hidden=!b.textContent.toLowerCase().includes(event.target.value.toLowerCase());if(!b.hidden)count++;}section.hidden=!count;}};
    showPop(button);
  }
  function renderQuick() {
    const state=api.read(), container=el('screenQuickFilters');
    for(const r of state.rules) quick.add(r.field);
    container.replaceChildren();
    for(const key of quick) {
      const matches=state.rules.filter(r=>r.field===key), active=categories.includes(key)?state[key]:matches.length;
      const b=document.createElement('button'); b.dataset.quick=key; b.className='screen-filter-chip'; b.classList.toggle('active',!!active); b.setAttribute('aria-expanded','false');
      const symbols={gt:'>',gte:'≥',lt:'<',lte:'≤',eq:'=',between:'',missing:'Missing',present:'Available'};
      const chosen=matches[0];
      const detail=chosen?`${symbols[chosen.operator]} ${['missing','present'].includes(chosen.operator)?'':chosen.value}${chosen.operator==='between'?'–'+chosen.maximum:''}${api.fields[key].monetary?' '+chosen.currency:''}${api.fields[key].unit?' '+api.fields[key].unit:''}`:'';
      b.textContent=api.fields[key].label+(active?' · '+(categories.includes(key)?state[key]:matches.length===1?detail:matches.length+' rules'):'')+' ⌄';
      b.onclick=()=>filterMenu(key,b); container.append(b);
    }
    const add=document.createElement('button'); add.textContent='+'; add.className='screen-filter-chip'; add.setAttribute('aria-label','Add filter'); add.onclick=()=>picker(add); container.append(add);
    const count=state.rules.length+categories.filter(k=>state[k]).length+Number(state.onlyFavorites)+Number(state.onlyHistory)+Number(!!state.capTier)+Number(!!state.priceSince);
    el('screenConditionCount').textContent=count?`(${count})`:'';
    el('screenNameTitle').textContent=count||state.search?'Custom screen':'All stocks';
  }
  const dialog=document.createElement('dialog'); dialog.id='screenManualDialog'; dialog.setAttribute('aria-labelledby','screenManualTitle');
  dialog.innerHTML=`<div class="screen-manual-heading"><div><h2 id="screenManualTitle">Manual setup</h2><p>Combine conditions across categories. Changes apply together.</p></div><button id="screenManualClose" aria-label="Close manual setup">×</button></div><div class="screen-manual-toolbar"><nav id="screenManualTabs" aria-label="Filter categories"></nav><input id="screenManualSearch" type="search" placeholder="Find a condition…" aria-label="Find a condition"><label>Numeric conditions<select id="screenManualMode"><option value="all">Match all (AND)</option><option value="any">Match any (OR)</option></select></label></div><div class="screen-manual-note">Descriptive filters always apply together. Monetary values use the currency selected in each condition; market cap is in billions, volume in millions.</div><div id="screenManualGrid"></div><footer class="screen-manual-footer"><button id="screenManualClear">Clear conditions</button><p id="screenManualPreview" role="status" aria-live="polite"></p><button id="screenManualCancel">Cancel</button><button id="screenManualApply" class="screen-primary">Apply filters</button></footer>`;
  document.body.append(dialog);
  let draft, tab='All', manualOpener;
  function preview() { const result=api.preview(draft); el('screenManualPreview').textContent=result.errors.length?result.errors.join(' '):`${result.count.toLocaleString()} matching stocks · ${draft.rules.length} numeric conditions`; el('screenManualApply').disabled=!!result.errors.length; }
  function control(tag, attrs={}) { const node=document.createElement(tag); for(const [k,v] of Object.entries(attrs)) node.setAttribute(k,v); return node; }
  function numericCell(key, existing) {
    const field=api.fields[key], cell=control('div',{'class':'screen-manual-cell','data-manual-field':key});
    const title=document.createElement('label'); title.textContent=field.label+(field.unit?` (${field.unit})`:field.type==='percent'?' (%)':'');
    const select=control('select',{'aria-label':field.label+' condition'});
    const options=presets(key); select.innerHTML='<option value="">Any</option>'+options.map((p,i)=>`<option value="${i}">${esc(p[0])}</option>`).join('')+'<option value="custom">Custom…</option>';
    let current=existing;
    const found=current?options.findIndex(p=>p[1]===current.operator&&String(p[2])===String(current.value)&&(p[1]!=='between'||String(p[3])===String(current.maximum))):-1;
    select.value=current?found>=0?String(found):'custom':'';
    title.append(select); cell.append(title);
    const custom=document.createElement('div'); custom.className='screen-custom-condition';
    const op=control('select',{'aria-label':field.label+' comparison'}); op.innerHTML=Object.entries(api.operators).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join(''); op.value=current?.operator||'gte';
    const value=control('input',{'type':'number','step':'any','placeholder':'Value','aria-label':field.label+' value'});value.value=current?.value||'';
    const maximum=control('input',{'type':'number','step':'any','placeholder':'Maximum','aria-label':field.label+' maximum'});maximum.value=current?.maximum||'';
    const currency=control('select',{'aria-label':field.label+' currency'}); currency.innerHTML=[...new Set(['USD',...api.choices('currency'),current?.currency].filter(Boolean))].map(c=>`<option>${esc(c)}</option>`).join(''); currency.value=current?.currency||'USD';
    custom.append(op,value,maximum);cell.append(custom);if(field.monetary)cell.append(currency);
    function visibility(){custom.hidden=select.value!=='custom';value.hidden=['missing','present'].includes(op.value);maximum.hidden=op.value!=='between';currency.hidden=!field.monetary||!select.value||['missing','present'].includes(op.value);}
    function update(){if(current)draft.rules=draft.rules.filter(r=>r!==current);current=null;if(select.value){current=select.value==='custom'?rule(key,op.value,value.value,maximum.value):rule(key,...options[Number(select.value)].slice(1));current.currency=currency.value;draft.rules.push(current);}visibility();preview();}
    select.onchange=()=>{if(select.value==='custom'&&current){op.value=current.operator;value.value=current.value;maximum.value=current.maximum||'';}update();};
    op.onchange=value.oninput=maximum.oninput=currency.onchange=update;visibility();
    const extra=control('button',{'type':'button','class':'screen-add-condition'});extra.textContent='+ Another condition';extra.onclick=()=>{draft.rules.push(rule(key,'gte',''));renderManual();};cell.append(extra);
    return cell;
  }
  function renderManual() {
    const grid=el('screenManualGrid');grid.replaceChildren();const query=el('screenManualSearch').value.toLowerCase().trim();
    for(const [group,keys] of Object.entries(groups)) {
      if(tab!=='All'&&tab!==group)continue;
      const visible=keys.filter(k=>api.fields[k].label.toLowerCase().includes(query));if(!visible.length)continue;
      const section=document.createElement('section');section.innerHTML=`<h3>${esc(group)}</h3><div class="screen-condition-matrix"></div>`;
      for(const key of visible) {
        if(categories.includes(key)) {
          const label=control('label',{'class':'screen-manual-cell'});label.textContent=api.fields[key].label;
          const select=control('select',{'aria-label':api.fields[key].label+' filter'});
          select.innerHTML='<option value="">Any</option>'+[...new Set([...api.choices(key),draft[key]].filter(Boolean))].map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');select.value=draft[key];select.onchange=()=>{draft[key]=select.value;if(key==='sector'){draft.industry='';renderManual();}if(key==='currency')draft.capTier='';preview();};label.append(select);section.lastElementChild.append(label);
        }else{const rules=draft.rules.filter(r=>r.field===key);for(const r of rules.length?rules:[null])section.lastElementChild.append(numericCell(key,r));}
      }
      if(group==='Descriptive'&&!query){
        for(const [key,label] of [['onlyFavorites','Saved stocks only'],['onlyHistory','Charts available only']]){
          const cell=control('label',{'class':'screen-manual-cell'});cell.textContent=label;
          const input=control('input',{'type':'checkbox','aria-label':label});input.checked=draft[key];input.onchange=()=>{draft[key]=input.checked;preview();};cell.append(input);section.lastElementChild.append(cell);
        }
        const dateLabel=control('label',{'class':'screen-manual-cell'});dateLabel.textContent='Price data since';const date=control('input',{'type':'date','aria-label':'Price data since'});date.value=draft.priceSince;date.onchange=()=>{draft.priceSince=date.value;preview();};dateLabel.append(date);section.lastElementChild.append(dateLabel);
        const capLabel=control('label',{'class':'screen-manual-cell'});capLabel.textContent='Market cap tier (USD)';const cap=control('select',{'aria-label':'Market cap tier'});cap.innerHTML='<option value="">Any</option>'+['mega','large','mid','small','micro'].map(v=>`<option value="${v}">${v}</option>`).join('');cap.value=draft.capTier;cap.onchange=()=>{draft.capTier=cap.value;preview();};capLabel.append(cap);section.lastElementChild.append(capLabel);
      }
      grid.append(section);
    }
    if(!grid.childElementCount)grid.textContent='No matching conditions.';
    for(const button of el('screenManualTabs').children)button.setAttribute('aria-pressed',String(button.textContent===tab));preview();
  }
  for(const name of ['All',...Object.keys(groups)]){const button=document.createElement('button');button.textContent=name;button.onclick=()=>{tab=name;renderManual();};el('screenManualTabs').append(button);}
  function openManual(key) { manualOpener=document.activeElement;draft=api.read();tab='All';el('screenManualSearch').value='';el('screenManualMode').value=draft.matchMode;renderManual();dialog.showModal();if(key){const node=dialog.querySelector(`[data-manual-field="${key}"]`);if(node){node.scrollIntoView({block:'center'});node.querySelector('select').focus();}} }
  function closeManual(){dialog.close();if(manualOpener?.isConnected)manualOpener.focus();}
  el('screenManualSearch').oninput=renderManual;
  el('screenManualMode').onchange=e=>{draft.matchMode=e.target.value;preview();};
  el('screenManualClear').onclick=()=>{draft.rules=[];for(const key of categories)draft[key]='';draft.capTier='';draft.onlyFavorites=false;draft.onlyHistory=false;draft.priceSince='';draft.matchMode='all';el('screenManualMode').value='all';renderManual();};
  el('screenManualApply').onclick=()=>{try{api.apply(draft);closeManual();}catch(error){el('screenManualPreview').textContent=error.message;}};
  el('screenManualClose').onclick=el('screenManualCancel').onclick=closeManual;
  dialog.addEventListener('cancel',e=>{e.preventDefault();closeManual();});
  el('screenManual').onclick=()=>openManual();el('screenReset').onclick=()=>el('resetScreener').click();
  document.addEventListener('click',e=>{if(!pop.hidden&&!e.composedPath().includes(pop)&&!e.composedPath().includes(opener))closePop();});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!pop.hidden){e.preventDefault();closePop(true);}});
  document.addEventListener('focusin',e=>{if(!pop.hidden&&!pop.contains(e.target)&&e.target!==opener)closePop();});
  window.addEventListener('resize',()=>closePop());window.addEventListener('scroll',()=>closePop(),{passive:true});
  window.addEventListener('screener:change',renderQuick);renderQuick();
};
