// Consolidated research tools share chart context, theme, data and dock controls.
(() => {
  'use strict';
  const el=id=>document.getElementById(id),pane=el('workspaceToolsPanel');
  const E=esc,number=(n,d=2)=>Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}):'Unavailable';
  const percent=n=>Number.isFinite(n)?number(n*100)+'%':'Unavailable';
  const tools={
    portfolio:{title:'Portfolio & allocation',icon:'M4 7h16v13H4zM9 7V4h6v3M4 12h16',description:'Holdings, allocation, optimization, rebalancing and portfolio stress tests.'},
    pairs:{title:'Pairs & statistical analysis',icon:'M3 6l5 5 5-7 8 8M3 18l5-7 5 8 8-9',description:'Aligned returns, correlation, log-price hedge ratio and spread z-score.'},
    options:{title:'Options lab',icon:'M3 19h18M4 17l8-8 8 8M12 9V3',description:'European option pricing, Greeks and expiry payoff scenarios.'},
    fundamentals:{title:'Company fundamentals',icon:'M4 21V5h10v16M14 10h6v11M7 8h4M7 12h4M7 16h4',description:'Stored company profile, valuation, profitability and metadata freshness.'},
    liquidity:{title:'Volume & liquidity',icon:'M4 20V13h3v7M11 20V4h3v16M18 20V9h3v11',description:'Relative volume, turnover, OBV and an approximate volume profile.'},
    forecast:{title:'Forecast lab',icon:'M3 18l5-6 4 2 4-8M16 6l5-3M16 6l5 5',description:'Walk-forward AR(1) evaluation against a no-change baseline.'},
    compare:{title:'Strategy comparison',icon:'M4 20V4M4 20h17M7 16l4-7 4 3 6-8',description:'Compare the native strategy presets on the same bars and costs.'},
    markets:{title:'Market overview',icon:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0M3 12h18M12 3v18',description:'Stocks, crypto and other stored instruments; daily change and data freshness.'},
    rotation:{title:'Sector rotation',icon:'M4 19V5m0 14h17M7 15l4-5 4 3 6-8',description:'Rank Finviz sector performance across multiple timeframes and track relative momentum shifts.'},
    ideas:{title:'Idea notebook',icon:'M5 4h14v16H5zM8 8h8M8 12h8M8 16h5',description:'Save, organize, search and revisit market ideas in this browser.'},
    journal:{title:'Research journal',icon:'M5 3h14v18H5zM8 7h8M8 11h8M8 15h5',description:'Symbol notes and source links saved in this browser.'}
  };
  const existing=[
    ['screener','Stock screener','Fundamental filters, momentum rankings and saved screens.'],
    ['strategy','Strategy tester','Custom rules, breakout strategies, signals, trades and equity curves.'],
    ['risk','Returns & risk','Historical VaR, expected shortfall, drawdown and return statistics.'],
    ['brownian','Monte Carlo / Brownian motion','Seeded price simulations, percentiles and scenario exports.'],
    ['markov','Markov analysis','Regimes, transition probabilities and state-conditioned outcomes.'],
    ['data','Stored market data','Daily and hourly bars, freshness, API activity and CSV export.']
  ];
  const library=el('workspaceSidetools');
  library.insertAdjacentHTML('beforeend','<div class="research-library"><input id="researchSearch" type="search" placeholder="Find an analysis tool" aria-label="Search analysis tools"><div id="researchLibraryList"></div></div>');
  const entries=[...Object.entries(tools).map(([key,t])=>[key,t.title,t.description]),...existing];
  const groups = [
    ['Markets & data', ['markets','rotation','screener','fundamentals','liquidity','data']],
    ['Strategies & relationships', ['strategy','compare','pairs']],
    ['Portfolio & risk', ['portfolio','risk','options']],
    ['Models & forecasts', ['forecast','markov','brownian']],
    ['Notes & research', ['ideas','journal']]
  ];
  function showLibrary() {
    const q=el('researchSearch').value.toLowerCase();
    el('researchLibraryList').innerHTML=groups.map(([title,keys])=>{
      const matches=keys.map(key=>entries.find(e=>e[0]===key)).filter(e=>(title+' '+e.join(' ')).toLowerCase().includes(q));
      return matches.length?`<section><h3>${E(title)}</h3>${matches.map(([key,name,description])=>`<button data-research-open="${key}"><strong>${E(name)}</strong><small>${E(description)}</small></button>`).join('')}</section>`:'';
    }).join('')||'<p>No matching tools.</p>';
  }
  el('researchSearch').oninput=showLibrary;showLibrary();
  const rail=document.querySelector('.workspace-right-rail'),slot=el('workspaceSettingsDock');
  const divider=document.createElement('div');divider.className='research-rail-separator';rail.insertBefore(divider,slot);
  for(const [key,t] of Object.entries(tools)) {
    const b=document.createElement('button');b.dataset.researchOpen=key;b.title=t.title;b.setAttribute('aria-label',t.title);b.setAttribute('aria-controls',pane.id);b.setAttribute('aria-pressed','false');
    b.innerHTML=`<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${t.icon}"/></svg>`;
    rail.insertBefore(b,slot);
  }
  let active='portfolio',worker=null,rejectWork=null,revision=0,result=null;
  const drafts=new Map();
  let holdings={};try{const saved=JSON.parse(localStorage.getItem('atlas.portfolios.v1')||'{}');if(saved&&typeof saved==='object'&&!Array.isArray(saved))holdings=saved;}catch{}
  let screener=null;
  const ideaNotesKey='atlas.ideaNotes.v1';
  let ideaNotes=[];
  let selectedIdeaId=null;
  const input=(id,label,value,min,max,step='any')=>`<label>${label}<input id="rt-${id}" type="number" value="${value}" min="${min}" max="${max}" step="${step}" required></label>`;
  const basis=()=>'<label>Price basis<select id="rt-adjusted"><option value="true">Adjusted close</option><option value="false">Raw close</option></select></label>';
  const get=id=>el('rt-'+id),num=id=>Number(get(id).value);
  function stop(){if(worker){worker.terminate();worker=null;const reject=rejectWork;rejectWork=null;reject?.(Error('Analysis canceled.'));}}
  function invalidate(message='Chart or settings changed. Run again for the current inputs.') {revision++;stop();result=null;if(el('researchOutput'))el('researchOutput').innerHTML='';if(el('researchExport'))el('researchExport').disabled=true;if(el('researchStatus')){el('researchStatus').textContent=message;el('researchStatus').dataset.error='false';}if(el('researchRun'))el('researchRun').disabled=false;}
  function calculate(kind,args){return new Promise((resolve,reject)=>{worker=new Worker('./research-worker.js');rejectWork=reject;worker.onmessage=({data})=>{worker.terminate();worker=null;rejectWork=null;data.error?reject(Error(data.error)):resolve(data.result);};worker.onerror=()=>{stop();};worker.postMessage({kind,args});});}
  function metrics(items){return '<div class="research-metrics">'+items.map(([k,v])=>`<div><span>${E(k)}</span><strong>${E(v)}</strong></div>`).join('')+'</div>';}
  function table(headers,records){return '<div class="research-table"><table><thead><tr>'+headers.map(h=>`<th>${E(h)}</th>`).join('')+'</tr></thead><tbody>'+records.map(r=>'<tr>'+r.map(v=>`<td>${E(v)}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';}
  function plot(curve,label) {
    const values=curve.map(r=>r.value),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;
    const sampled=curve.filter((r,i)=>i%Math.max(1,Math.ceil(curve.length/800))===0||i===curve.length-1);
    const points=sampled.map(r=>`${60+curve.indexOf(r)/(curve.length-1||1)*710},${175-(r.value-lo)/span*145}`).join(' ');
    return `<svg class="research-plot" viewBox="0 0 800 210" role="img" aria-label="${E(label)}"><title>${E(label)}</title><line class="axis" x1="60" x2="770" y1="175" y2="175"/><polyline points="${points}" fill="none" stroke="currentColor" stroke-width="2"/><text x="8" y="30">${number(hi)}</text><text x="8" y="175">${number(lo)}</text><text x="60" y="200">${E(curve[0].date)}</text><text x="770" y="200" text-anchor="end">${E(curve.at(-1).date)}</text></svg>`;
  }
  const help={
    portfolio:'One line per holding: symbol, units, average cost per unit, target weight %. Use 2–12 assets in the same currency. Analysis uses shared timestamps and a fixed initial allocation with no periodic rebalancing, fees or cash yield. Suggested trades use last stored raw closes. Optimization is fitted to this history, not out-of-sample.',
    pairs:'OLS fits log(A) = intercept + beta × log(B) over shared timestamps. Correlation uses simple returns; the final spread z-score uses the chosen trailing window. This is an in-sample descriptive fit, not a cointegration test or trading signal.',
    options:'Black–Scholes with continuous dividend yield and ACT/365 expiry. Price and Greeks are per underlying unit: vega per 1 volatility percentage point, theta per day. European exercise only. These are model values, not an option chain or executable quotes.',
    fundamentals:'Stored company metadata is independent of the chart date range. Missing values remain unavailable; no live filing or valuation feed is assumed.',
    liquidity:'Computed from stored bars. Turnover is close × volume. Volume profile assigns each bar’s entire volume to its closing-price bucket; it is not trade-level volume. OHLCV cannot identify dark pools, actual order flow or bid/ask spreads.',
    forecast:'AR(1) models log returns. The final 20% is evaluated one bar at a time using only earlier observations, then the full sample is fitted for the forward path. MAE is measured in log returns. The baseline predicts no price change. This is an experimental model, not a promise of price performance.',
    compare:'Uses the same next-open execution engine as Strategy tester. Each preset retains its own warmup. Results are in-sample and may have different first trade dates; ranking does not establish future performance. Open a preset in Strategy tester to inspect its rules and trades.',
    markets:'Latest stored daily changes and timestamps. Prices in different currencies are not aggregated. No live quotes, on-chain metrics, futures curves or macro calendar are inferred.',
    rotation:'Finviz Group Screener sector returns are snapshots, not a historical sector-index series. Rotation compares each sector’s Finviz weekly performance rank with its quarterly rank: a rise of at least two places is Improving, a fall of at least two is Cooling, otherwise Steady. Performance is delayed as reported by Finviz and is not a trading signal.',
    journal:'Keep research alongside the selected chart. Notes use the same browser storage as the Notes panel. Source links are your references; no automatic news feed or sentiment score is generated.'
  };
  function context(){if(el('researchContext'))el('researchContext').textContent=`${selected?.symbol||'Select a symbol'} · ${interval} · ${rows.length.toLocaleString()} loaded bars${rows.length?' · '+rows[0].date+' to '+rows.at(-1).date:''}`;}
  function form(key) {
    const symbol=selected?.symbol||'AAPL',peer=assets.find(a=>a.symbol!==symbol&&a.currency===selected?.currency&&a.has_daily!==false)?.symbol||'MSFT';
    if(key==='portfolio')return `<label>Holdings<textarea id="rt-holdings" rows="3" required>${E(symbol)}, 10, 0, 50\n${E(peer)}, 10, 0, 50</textarea></label><label>Allocation method<select id="rt-method"><option value="target">Entered targets</option><option value="equal">Equal weight</option><option value="inverse">Inverse volatility</option><option value="minimum">Minimum variance · long only</option></select></label>${basis()}${input('shock','Uniform price shock %',-20,-100,100)}<label>Saved portfolio<select id="rt-saved"><option value="">Choose…</option>${Object.keys(holdings).map(n=>`<option>${E(n)}</option>`).join('')}</select></label><label class="research-wide">Portfolio name<input id="rt-name" maxlength="60" placeholder="My portfolio"></label><button type="button" id="rt-save">Save holdings</button><button type="button" id="rt-delete">Delete saved</button>`;
    if(key==='pairs')return `<label class="research-wide">Compare symbol<input id="rt-peer" value="${E(peer)}" maxlength="30" required></label>${input('window','Spread window · bars',60,10,500,1)}${basis()}`;
    if(key==='options')return `<label>Option type<select id="rt-type"><option value="call">Call</option><option value="put">Put</option></select></label>${input('spot','Underlying spot',rows.at(-1)?.close||100,.000001,1e9)}${input('strike','Strike',rows.at(-1)?.close||100,.000001,1e9)}${input('days','Days to expiry',30,1,3650,1)}${input('vol','Volatility % p.a.',25,.01,500)}${input('rate','Risk-free rate % p.a.',0,-100,100)}${input('dividend','Dividend yield % p.a.',0,0,100)}`;
    if(key==='forecast')return `${input('horizon','Forward horizon · bars',20,1,120,1)}${basis()}`;
    if(key==='compare')return `${input('capital','Initial capital',10000,100,1e9)}${input('commission','Commission %',.1,0,10)}${input('slippage','Slippage %',.05,0,10)}${basis()}`;
    if(key==='markets')return '<label>Asset group<select id="rt-group"><option value="All">All assets</option><option value="Stocks">Stocks</option><option value="Crypto">Crypto</option><option value="Saved">Saved watchlist</option></select></label><label class="research-wide">Search<input id="rt-search" placeholder="Company or ticker"></label>';
    if(key==='rotation')return '<label>Sort by<select id="rt-period"><option value="change_1d_pct">Day</option><option value="week_pct">Week</option><option value="month_pct">Month</option><option value="quarter_pct">Quarter</option><option value="half_year_pct">Half-year</option><option value="ytd_pct">Year to date</option><option value="year_pct">Year</option></select></label>';
    if(key==='journal')return '<label class="research-wide">Symbol notes<textarea id="rt-notes" rows="5" maxlength="20000"></textarea></label><label class="research-wide">Source URL (optional)<input id="rt-source" type="url" placeholder="https://…"></label>';
    return '';
  }
  function renderTool(key){
    if(el('researchForm')&&active!=='journal') drafts.set(active,Object.fromEntries([...el('researchForm').querySelectorAll('input,textarea,select')].map(field=>[field.id,field.value])));
    invalidate();active=key;
    if(key==='ideas') { renderIdeaNotebook(); return; }
    pane.innerHTML=`<section class="research-tool"><div class="research-heading"><div><span class="eyebrow">CHART ANALYSIS</span><h2>${E(tools[key].title)}</h2><p id="researchContext"></p></div><button id="researchExport" disabled>Export result JSON</button></div><form id="researchForm" class="research-controls">${form(key)}<button type="submit" id="researchRun">${key==='journal'?'Save research':key==='markets'?'Refresh overview':key==='rotation'?'Refresh sector data':'Run analysis'}</button><button type="button" id="researchCancel">Cancel</button></form><p class="research-help">${E(help[key])}</p><p id="researchStatus" role="status" aria-live="polite">Ready. Calculations run only when requested.</p><div id="researchOutput"></div></section>`;
    if(drafts.has(key))for(const [id,value] of Object.entries(drafts.get(key)))if(el(id))el(id).value=value;
    context();el('researchForm').oninput=()=>invalidate();el('researchForm').onsubmit=run;el('researchCancel').onclick=()=>invalidate('Canceled.');
    el('researchExport').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`quantstack-${active}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    if(key==='rotation'&&window.ATLAS_STATIC){el('researchRun').disabled=true;el('researchStatus').textContent='Sector rotation snapshots are available on the local dashboard only.';el('researchOutput').innerHTML='<p class="research-help">Open the <a href="https://finviz.com/groups?g=sector&v=140" target="_blank" rel="noopener noreferrer">Finviz sector performance screener</a>.</p>';}
    if(key==='portfolio') {
      get('save').onclick=()=>{try{parseHoldings();const name=get('name').value.trim();if(!name)throw Error('Enter a portfolio name.');if(Object.keys(holdings).length>=30&&!Object.hasOwn(holdings,name))throw Error('Save up to 30 portfolios.');holdings={...holdings,[name]:get('holdings').value};localStorage.setItem('atlas.portfolios.v1',JSON.stringify(holdings));get('saved').innerHTML='<option value="">Choose…</option>'+Object.keys(holdings).map(n=>`<option>${E(n)}</option>`).join('');get('saved').value=name;el('researchStatus').textContent='Holdings saved in this browser.';}catch(e){statusError(e);}};
      get('saved').onchange=()=>{const n=get('saved').value;if(Object.hasOwn(holdings,n)){get('name').value=n;get('holdings').value=holdings[n];invalidate();}};
      get('delete').onclick=()=>{const n=get('saved').value;if(!n)return;const next={...holdings};delete next[n];try{localStorage.setItem('atlas.portfolios.v1',JSON.stringify(next));holdings=next;get('saved').selectedOptions[0].remove();el('researchStatus').textContent='Saved portfolio removed; the editable holdings remain.';}catch(e){statusError(e);}};
    }
    if(key==='journal') {
      try{get('notes').value=localStorage.getItem('atlas.notes.'+selected?.symbol)||'';get('source').value=localStorage.getItem('atlas.source.'+selected?.symbol)||'';}catch{}
    }
  }
  function renderIdeaNotebook() {
    pane.innerHTML=`<section class="research-tool idea-notebook"><div class="research-heading"><div><span class="eyebrow">NOTES & RESEARCH</span><h2>${E(tools.ideas.title)}</h2><p>Keep ideas organized and available for later in this browser.</p></div><button id="ideaNew" type="button">New note</button></div><div class="idea-notebook-layout"><section class="idea-notes-list"><label for="ideaSearch">Find a note</label><input id="ideaSearch" type="search" placeholder="Search title, ticker or text"><label for="ideaCategoryFilter">Category</label><select id="ideaCategoryFilter"><option value="">All categories</option><option>Idea</option><option>Research</option><option>To do</option><option>Watchlist</option></select><div id="ideaNotesList" aria-label="Saved notes"></div></section><form id="ideaEditor" class="idea-note-editor"><label for="ideaTitle">Title</label><input id="ideaTitle" maxlength="120" required placeholder="Give this note a useful title"><label for="ideaCategory">Category</label><select id="ideaCategory"><option>Idea</option><option>Research</option><option>To do</option><option>Watchlist</option></select><label for="ideaSymbol">Related symbol (optional)</label><input id="ideaSymbol" maxlength="30" placeholder="e.g. NVDA"><label for="ideaBody">Notes</label><textarea id="ideaBody" maxlength="20000" rows="12" placeholder="Capture the thesis, questions, links or next steps…" required></textarea><p id="ideaNoteDate" class="idea-note-date"></p><div class="idea-note-actions"><button id="ideaSave" type="submit">Save note</button><button id="ideaDelete" type="button" class="secondary" disabled>Delete</button></div></form></div><p id="ideaStatus" role="status" aria-live="polite">Notes are saved in this browser.</p></section>`;
    const list=el('ideaNotesList'), editor=el('ideaEditor'), status=el('ideaStatus');
    const field=id=>el(id);
    const formatDate=value=>value?new Date(value).toLocaleString():'';
    function showStatus(message,isError=false) {
      status.textContent=message;
      status.dataset.error=String(isError);
    }
    function loadIdeas() {
      try {
        const stored=localStorage.getItem(ideaNotesKey);
        const parsed=stored?JSON.parse(stored):[];
        if(!Array.isArray(parsed)||parsed.some(note=>!note||typeof note.id!=='string'||typeof note.title!=='string'||typeof note.category!=='string'||typeof note.body!=='string'||typeof note.updatedAt!=='string'))
          throw Error('Saved notes have an invalid format. They have not been changed.');
        ideaNotes=parsed;
        showStatus(`${ideaNotes.length} saved ${ideaNotes.length===1?'note':'notes'} · stored in this browser.`);
      } catch(error) {
        ideaNotes=[];
        showStatus(error instanceof SyntaxError?'Saved notes could not be read because the browser data is invalid.':error.message||'Browser storage is unavailable. Notes have not been changed.',true);
        return false;
      }
      return true;
    }
    function clearEditor() {
      selectedIdeaId=null;
      editor.reset();
      field('ideaCategory').value='Idea';
      field('ideaSymbol').value=selected?.symbol||'';
      field('ideaNoteDate').textContent='New note';
      field('ideaDelete').disabled=true;
      list.querySelectorAll('[data-idea-open]').forEach(button=>button.classList.remove('active'));
    }
    function openIdea(id) {
      const note=ideaNotes.find(item=>item.id===id);
      if(!note)return;
      selectedIdeaId=id;
      field('ideaTitle').value=note.title;
      field('ideaCategory').value=note.category;
      field('ideaSymbol').value=note.symbol||'';
      field('ideaBody').value=note.body;
      field('ideaNoteDate').textContent=`Updated ${formatDate(note.updatedAt)}`;
      field('ideaDelete').disabled=false;
      list.querySelectorAll('[data-idea-open]').forEach(button=>button.classList.toggle('active',button.dataset.ideaOpen===id));
      showStatus(`Editing “${note.title}”.`);
    }
    function renderList() {
      const search=field('ideaSearch').value.trim().toLowerCase(),category=field('ideaCategoryFilter').value;
      const visible=ideaNotes.filter(note=>(!category||note.category===category)&&`${note.title} ${note.symbol||''} ${note.category} ${note.body}`.toLowerCase().includes(search))
        .sort((a,b)=>b.updatedAt.localeCompare(a.updatedAt));
      list.innerHTML=visible.map(note=>`<button type="button" data-idea-open="${E(note.id)}" class="idea-note-item ${selectedIdeaId===note.id?'active':''}"><strong>${E(note.title)}</strong><small>${E(note.category)}${note.symbol?' · '+E(note.symbol):''} · ${E(formatDate(note.updatedAt))}</small><span>${E(note.body.slice(0,110))}</span></button>`).join('')||'<p class="research-help">No saved notes match. Create a note to keep an idea for later.</p>';
    }
    const notesAvailable=loadIdeas();
    if(notesAvailable) {
      renderList();
      if(ideaNotes.length)openIdea(ideaNotes.find(note=>note.id===selectedIdeaId)?.id||ideaNotes.slice().sort((a,b)=>b.updatedAt.localeCompare(a.updatedAt))[0].id);
      else clearEditor();
    } else {
      renderList();
      clearEditor();
      editor.inert=true;
      field('ideaNew').disabled=true;
    }
    field('ideaSearch').oninput=renderList;
    field('ideaCategoryFilter').onchange=renderList;
    field('ideaNew').onclick=()=>{clearEditor();showStatus('New note. Save it to keep it in this browser.');field('ideaTitle').focus();};
    list.onclick=event=>{const button=event.target.closest('[data-idea-open]');if(button)openIdea(button.dataset.ideaOpen);};
    editor.onsubmit=event=>{
      event.preventDefault();
      if(!editor.reportValidity())return;
      const now=new Date().toISOString(),existing=ideaNotes.find(note=>note.id===selectedIdeaId);
      const note={id:existing?.id||crypto.randomUUID(),title:field('ideaTitle').value.trim(),category:field('ideaCategory').value,symbol:field('ideaSymbol').value.trim().toUpperCase(),body:field('ideaBody').value.trim(),createdAt:existing?.createdAt||now,updatedAt:now};
      if(!note.title||!note.body){showStatus('Add a title and note before saving.',true);return;}
      const updated=[note,...ideaNotes.filter(item=>item.id!==note.id)];
      try { localStorage.setItem(ideaNotesKey,JSON.stringify(updated)); }
      catch { showStatus('Could not save this note. Browser storage is unavailable or full.',true);return; }
      ideaNotes=updated;selectedIdeaId=note.id;field('ideaNoteDate').textContent=`Updated ${formatDate(now)}`;field('ideaDelete').disabled=false;
      renderList();showStatus(`Saved “${note.title}” in this browser.`);
    };
    field('ideaDelete').onclick=()=>{
      const note=ideaNotes.find(item=>item.id===selectedIdeaId);
      if(!note||!window.confirm(`Delete “${note.title}”?`))return;
      const updated=ideaNotes.filter(item=>item.id!==note.id);
      try { localStorage.setItem(ideaNotesKey,JSON.stringify(updated)); }
      catch { showStatus('Could not delete this note from browser storage.',true);return; }
      ideaNotes=updated;selectedIdeaId=null;renderList();clearEditor();showStatus(`Deleted “${note.title}”.`);
    };
  }
  function statusError(error){el('researchStatus').textContent=error.message;el('researchStatus').dataset.error='true';}
  function open(key){if(tools[key]){if(key!==active||!el('researchForm'))renderTool(key);ChartWorkspace.openDock('tools');}else ChartWorkspace.openDock(key);syncRail(tools[key]?'tools':key);}
  window.openChartResearch=open;
  function syncRail(dock){rail.querySelectorAll('[data-research-open]').forEach(b=>{const on=dock==='tools'&&b.dataset.researchOpen===active;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});}
  window.addEventListener('workspace:dock',e=>{syncRail(e.detail);if(e.detail!=='tools'&&worker)invalidate('Analysis canceled when the panel closed.');});
  document.addEventListener('click',e=>{const b=e.target.closest('[data-research-open]');if(b){if(b.parentElement===rail&&b.dataset.researchOpen===active&&!pane.hidden&&!el('workspaceDock').hidden)el('workspaceCloseDock').click();else open(b.dataset.researchOpen);}const chart=e.target.closest('[data-research-symbol]');if(chart)select(chart.dataset.researchSymbol);const preset=e.target.closest('[data-research-preset]');if(preset){ChartWorkspace.openDock('strategy');el('st-kind').value=preset.dataset.researchPreset;el('st-kind').dispatchEvent(new Event('change',{bubbles:true}));}});
  function parseHoldings(){
    const lines=get('holdings').value.trim().split('\n').filter(s=>s.trim());if(lines.length<2||lines.length>12)throw Error('Enter 2–12 holdings.');
    const parsed=lines.map(line=>{const fields=line.split(',').map(s=>s.trim());if(fields.length!==4||fields.some(s=>!s))throw Error('Each line needs symbol, units, cost, target %.');const [symbol,...values]=fields;const [units,cost,target]=values.map(Number);if(![units,cost,target].every(Number.isFinite)||units<=0||cost<0||target<0||target>100)throw Error('Units must be positive; cost and target cannot be negative.');return {symbol:symbol.toUpperCase(),units,cost,target};});
    if(new Set(parsed.map(h=>h.symbol)).size!==parsed.length)throw Error('Use each symbol once.');return parsed;
  }
  async function historyFor(symbol,token){if(token!==revision)throw Error('Canceled.');if(!assets.some(a=>a.symbol===symbol))throw Error(`${symbol} is not in the stored catalog. Add it to the local scheduler first.`);const q=query();q.set('symbol',symbol);return symbol===selected.symbol?rows.slice():api('/api/history?'+q);}
  async function run(event){
    event.preventDefault();invalidate('Running analysis…');const token=revision,key=active,ctx={symbol:selected?.symbol,interval,start:rows[0]?.date,end:rows.at(-1)?.date};el('researchRun').disabled=true;
    const adjusted=get('adjusted')?.value!=='false';let data,html='';
    try {
      if(!selected&&key!=='rotation')throw Error('Select a chart symbol first.');
      if(key==='portfolio') {
        const h=parseHoldings(),catalog=h.map(v=>assets.find(a=>a.symbol===v.symbol));
        if(catalog.some(a=>!a?.currency)||new Set(catalog.map(a=>a.currency)).size!==1)throw Error('All holdings must have the same known currency; currency conversion is not available.');
        const histories=[];for(const holding of h){histories.push(await historyFor(holding.symbol,token));}
        if(token!==revision)return;
        let weights=h.map(v=>v.target/100);if(get('method').value!=='target')weights=await calculate('allocation',[histories,get('method').value,adjusted]);
        if(token!==revision)return;
        const annual=interval==='1h'?(catalog.every(a=>a.kind==='Crypto')?8760:1638):(catalog.every(a=>a.kind==='Crypto')?365:252);
        data=await calculate('portfolio',[histories,weights,annual,adjusted]);
        const last=histories.map(r=>r.at(-1));if(last.some(r=>!Number.isFinite(r?.close)||r.close<=0))throw Error('Each holding needs a valid last close.');
        const total=h.reduce((sum,v,i)=>sum+v.units*last[i].close,0),cost=h.reduce((sum,v)=>sum+v.units*v.cost,0);
        data.holdings=h.map((v,i)=>({...v,price:last[i].close,date:last[i].date,value:v.units*last[i].close,target:weights[i]*100,tradeUnits:(total*weights[i]-v.units*last[i].close)/last[i].close}));
        data.currency=catalog[0].currency;data.shockPct=num('shock');data.annualization=annual;
        html=metrics([['Stored holdings value',number(total)+' '+data.currency],['Cost-basis P/L',h.every(v=>v.cost>0)?number(total-cost):'Enter all costs'],['Historical allocation return',percent(data.return)],['Annualized volatility',percent(data.volatility)],['Maximum drawdown',percent(data.maxDrawdown)],['Shock value',number(total*(1+num('shock')/100))]])+plot(data.curve,'Historical allocation growth, starting at 1')+table(['Symbol','Price date','Value','Target %','Rebalance units'],data.holdings.map(h=>[h.symbol,h.date,number(h.value),number(h.target),number(h.tradeUnits,4)]))+`<p class="research-help">Positive units are hypothetical buys; negative units are sells. No orders are submitted. Quote dates may differ. Annualization assumes ${annual} bars/year.</p>`+table(['Return correlation',...h.map(v=>v.symbol)],data.correlations.map((r,i)=>[h[i].symbol,...r.map(v=>number(v,3))]));
      } else if(key==='pairs') {
        const peer=get('peer').value.trim().toUpperCase();if(peer===selected.symbol)throw Error('Choose a different comparison symbol.');const other=await historyFor(peer,token);if(token!==revision)return;
        data=await calculate('pairs',[rows,other,num('window'),adjusted]);data.peer=peer;
        html=metrics([['Shared bars',data.count],['Return correlation',number(data.correlation,4)],['Log hedge ratio',number(data.beta,4)],['Spread z-score',number(data.z,3)]])+plot(data.curve,`${selected.symbol} / ${peer} fitted log spread`);
      } else if(key==='options') {
        const params={spot:num('spot'),strike:num('strike'),days:num('days'),volatility:num('vol')/100,rate:num('rate')/100,dividend:num('dividend')/100,type:get('type').value};
        data=await calculate('option',[params]);data.parameters=params;html=metrics([['Model premium',number(data.price,4)],['Delta',number(data.delta,4)],['Gamma',number(data.gamma,6)],['Vega / 1 vol point',number(data.vega,4)],['Theta / day',number(data.theta,4)]])+plot(data.curve,'Long option expiry profit per unit versus underlying price');
      } else if(key==='liquidity') {
        data=await calculate('liquidity',[rows]);html=metrics([['Relative volume',number(data.relativeVolume)+'×'],['Mean turnover · 20 bars',number(data.turnover)+' '+(selected.currency||'')],['Amihud × 1 million',number(data.amihud,6)],['OBV',number(data.obv,0)],['Accumulation / distribution',number(data.ad,0)]])+plot(data.curve,'On-balance volume')+table(['Price bucket midpoint','Attributed volume'],data.profile.map(r=>[number(r.price),number(r.volume,0)]));
      } else if(key==='forecast') {
        data=await calculate('forecast',[rows,num('horizon'),adjusted]);html=metrics([['Walk-forward model MAE',number(data.modelMAE,6)],['No-change baseline MAE',number(data.baselineMAE,6)],['AR coefficient',number(data.slope,4)],['Validation observations',data.validation.length]])+plot(data.curve,'Experimental AR(1) forward price path')+table(['Date','Predicted log return','Observed log return'],data.validation.slice(-20).map(r=>[r.date,number(r.predicted,6),number(r.actual,6)]));
      } else if(key==='compare') {
        data=await calculate('compare',[rows,{capital:num('capital'),commission:num('commission'),slippage:num('slippage'),adjusted}]);
        data.sort((a,b)=>(b.returnPct??-Infinity)-(a.returnPct??-Infinity));html=table(['Strategy','Return %','Drawdown %','Trades','Fees / status'],data.map(r=>[r.strategy,number(r.returnPct),number(r.maxDrawdown),r.trades??'—',r.error||number(r.fees)]))+'<div class="research-links">'+data.filter(r=>!r.error).map(r=>`<button data-research-preset="${E(r.kind)}">Inspect ${E(r.strategy)}</button>`).join('')+'</div>';
      } else if(key==='fundamentals') {
        if(!screener)screener=await api('/api/screener');data=screener.rows.find(r=>r.symbol===selected.symbol);if(!data)throw Error('No company fundamentals stored for this instrument. Price-based tools remain available.');
        const fields=['name','sector','industry','country','exchange','currency','market_cap','pe','forward_pe','pb','dividend_yield','revenue_growth','profit_margin','beta','metadata_date','metadata_source'];
        html=table(['Company field','Stored value'],fields.map(k=>[k.replaceAll('_',' '),data[k]==null?'Unavailable':typeof data[k]==='number'?number(data[k],4):data[k]]));
      } else if(key==='rotation') {
        data=await api('/api/sector-rotation');
        const period=get('period').value,labels={change_1d_pct:'Day',week_pct:'Week',month_pct:'Month',quarter_pct:'Quarter',half_year_pct:'Half-year',ytd_pct:'YTD',year_pct:'Year'};
        const sorted=data.sectors.slice().sort((a,b)=>(b[period]??-Infinity)-(a[period]??-Infinity));
        const pctCell=value=>Number.isFinite(value)?`<span class="${value>=0?'positive':'negative'}">${E(number(value))}%</span>`:'Unavailable';
        html=metrics([['Sectors',data.sectors.length],['Top sector',sorted[0]?.name||'Unavailable'],['Top '+labels[period],Number.isFinite(sorted[0]?.[period])?number(sorted[0][period])+'%':'Unavailable'],['Finviz snapshot',new Date(data.generated_at).toLocaleString()]])+`<div class="research-table"><table><thead><tr>${['Rank','Sector','Day','Week','Month','Quarter','Half-year','YTD','Year','Rotation'].map(h=>`<th>${E(h)}</th>`).join('')}</tr></thead><tbody>${sorted.map((sector,index)=>`<tr><td>${index+1}</td><td>${E(sector.name)}</td><td>${pctCell(sector.change_1d_pct)}</td><td>${pctCell(sector.week_pct)}</td><td>${pctCell(sector.month_pct)}</td><td>${pctCell(sector.quarter_pct)}</td><td>${pctCell(sector.half_year_pct)}</td><td>${pctCell(sector.ytd_pct)}</td><td>${pctCell(sector.year_pct)}</td><td>${E(sector.rotation)}</td></tr>`).join('')}</tbody></table></div><p class="research-help">Source: <a href="${E(data.source)}" target="_blank" rel="noopener noreferrer">Finviz Group Screener · Sector Performance</a></p>`;
      } else if(key==='markets') {
        const group=get('group').value,q=get('search').value.toLowerCase();data=assets.filter(a=>(group==='All'||a.kind===group||group==='Saved'&&saved.has(a.symbol))&&`${a.symbol} ${a.name}`.toLowerCase().includes(q)).slice().sort((a,b)=>(b.change??-Infinity)-(a.change??-Infinity));
        html=metrics([['Matching assets',data.length],['Display limit',100],['Positive daily change',data.filter(a=>a.change>0).length],['Negative daily change',data.filter(a=>a.change<0).length]])+table(['Symbol','Name','Close','Currency','Daily change %','As of'],data.slice(0,100).map(a=>[a.symbol,a.name,number(a.close),a.currency||'Unknown',number(a.change),a.quote_timestamp||a.date]))+'<div class="research-links">'+data.slice(0,20).map(a=>`<button data-research-symbol="${E(a.symbol)}">${E(a.symbol)} chart</button>`).join('')+'</div>';
      } else if(key==='journal') {
        const source=get('source').value.trim();if(source&&!['http:','https:'].includes(new URL(source).protocol))throw Error('Use an HTTP or HTTPS source URL.');data={symbol:selected.symbol,notes:get('notes').value,source,updated:new Date().toISOString()};
        localStorage.setItem('atlas.notes.'+selected.symbol,data.notes);localStorage.setItem('atlas.source.'+selected.symbol,source);el('workspaceNotes').value=data.notes;
        html='<p>Research saved in this browser.</p>'+(source?`<a href="${E(source)}" target="_blank" rel="noopener noreferrer">Open saved source</a>`:'');
      }
      if(token!==revision)return;
      result={tool:key,context:ctx,priceBasis:adjusted?'adjusted':'raw',result:data};el('researchOutput').innerHTML=html;el('researchExport').disabled=false;el('researchStatus').textContent=ctx.symbol?'Complete · '+ctx.symbol+' · '+ctx.interval:'Complete · Sector rotation';
    }catch(error){if(token===revision)statusError(error);}finally{if(token===revision)el('researchRun').disabled=false;}
  }
  const oldLoad=loadHistory;loadHistory=async function(){invalidate();await oldLoad();context();if(active==='journal'){try{get('notes').value=localStorage.getItem('atlas.notes.'+selected?.symbol)||'';get('source').value=localStorage.getItem('atlas.source.'+selected?.symbol)||'';}catch{}}if(active==='options'&&rows.length){get('spot').value=rows.at(-1).close;get('strike').value=rows.at(-1).close;}};
  // Clear metadata cache only when the user explicitly reloads the dataset.
  el('refresh').addEventListener('click',()=>{screener=null;});
  renderTool(active);
})();
