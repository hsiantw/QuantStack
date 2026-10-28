// Read-only inspection of the market SQLite database. No arbitrary SQL is exposed.
(() => {
  const by=id=>document.getElementById(id);
  const dialog=document.createElement('dialog');dialog.id='databaseDialog';dialog.setAttribute('aria-labelledby','databaseTitle');
  dialog.innerHTML=`<div class="db-heading"><div><span class="eyebrow">LOCAL STORAGE · READ ONLY</span><h2 id="databaseTitle">Market database</h2><p id="databaseInfo">Inspect collected prices and collector records.</p></div><button id="databaseClose" type="button">Close</button></div>
    <form id="databaseForm"><div class="db-filters"><label>Data table<select id="databaseTable" required></select></label><label data-db-field="symbol">Symbol<input id="databaseSymbol" placeholder="All symbols" maxlength="100"></label><label data-db-field="interval">Interval<select id="databaseInterval"><option value="">All intervals</option><option>1m</option><option>3m</option><option>5m</option><option>15m</option><option>30m</option><option>60m</option><option>1d</option></select></label><label data-db-field="source">Source<input id="databaseSource" list="databaseSources" placeholder="All sources" maxlength="100"><datalist id="databaseSources"><option>Yahoo Finance</option><option>Bitstamp</option><option>Kraken</option><option>Coinbase</option></datalist></label><label data-db-field="status">Status<input id="databaseStatus" placeholder="All statuses" maxlength="100"></label><label data-db-field="date">From<input id="databaseStart" type="date"></label><label data-db-field="date">Through<input id="databaseEnd" type="date"></label><label>Rows per page<select id="databaseLimit"><option>25</option><option selected>50</option><option>100</option><option>200</option></select></label></div><p id="databaseTableHelp"></p>
    <div class="db-actions"><button id="databaseApply" type="submit">Apply filters</button><button id="databaseClear" type="button">Clear filters</button><button id="databaseCurrent" type="button">Current chart symbol</button><button id="databaseRefresh" type="button">Refresh</button><button id="databaseExport" type="button" disabled>Export this page CSV</button></div></form>
    <p id="databasePending" role="status" hidden>Filters changed. Apply them to update the results.</p><p id="databaseStatusLine" role="status" aria-live="polite">Choose a table.</p><p id="databaseError" role="alert" hidden></p>
    <div id="databaseRows" class="db-scroll" tabindex="0" aria-label="Database rows"></div>
    <div class="db-pagination"><button id="databasePrev" type="button" disabled>Previous</button><span id="databasePage"></span><button id="databaseNext" type="button" disabled>Next</button></div>
    <details><summary>Table schema and storage details</summary><p id="databaseStorage"></p><div id="databaseSchema" class="db-scroll"></div></details>
    <p class="db-note">Rows are shown newest first where a date column exists. Counts match the active filters. Collector writes can change counts between pages. Dates filter stored timestamps; intraday timestamps are UTC. CSV exports only the displayed page. This view cannot edit or delete data.</p>`;
  document.body.append(dialog);
  let metadata=null, offset=0, result=null, version=0, loading=false, dirty=false, applied=null;
  const names={prices:'Daily prices',intraday_prices:'Intraday prices',crypto_exchange_prices:'Crypto exchange candles',state:'Daily collector status',intraday_state:'Intraday collector status',attempts:'Collection attempts',api_requests:'Provider requests',membership:'Index membership',stock_metadata:'Company details',btc_backfill_windows:'BTC archive checkpoints',btc_backfill_meta:'BTC archive details',liquidation_events:'Observed liquidations',liquidation_collector_state:'Liquidation collector status'};
  const priceTables=new Set(['prices','intraday_prices','crypto_exchange_prices']);
  const fieldIds={symbol:'databaseSymbol',interval:'databaseInterval',source:'databaseSource',status:'databaseStatus'};
  async function request(params='') {
    const response=await fetch('./api/database'+(params?'?'+params:''));
    const text=await response.text();let data;try{data=JSON.parse(text);}catch{throw Error('Database inspection is available on the local QuantStack app.');}
    if(!response.ok)throw Error(data.error||'Unable to read the database.');return data;
  }
  function error(message){by('databaseError').textContent=message;by('databaseError').hidden=!message;}
  function table(headers,rows){return `<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>v===null?'<td class="db-null">NULL</td>':`<td title="${esc(v)}">${esc(typeof v==='number' && Number.isFinite(v)?v.toLocaleString(undefined,{maximumFractionDigits:8}):v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;}
  function schema(){
    const rows=metadata?.schemas[by('databaseTable').value]||[],columns=rows.map(r=>r.name);
    by('databaseTableHelp').textContent=names[by('databaseTable').value]?`${names[by('databaseTable').value]} · ${by('databaseTable').value}`:by('databaseTable').value;
    const dateColumn=['timestamp','date','requested_at','started','observed_date'].find(c=>columns.includes(c));
    dialog.querySelectorAll('[data-db-field]').forEach(label=>{const enabled=label.dataset.dbField==='date'?!!dateColumn:columns.includes(label.dataset.dbField);label.hidden=!enabled;label.querySelectorAll('input,select').forEach(input=>input.disabled=!enabled);});
    by('databaseSchema').innerHTML=table(['Column','Type','Required','Primary key order','Default'],rows.map(r=>[r.name,r.type,r.notnull?'Yes':'No',r.pk||'—',r.dflt_value]));
  }
  function query(){const q=new URLSearchParams({table:by('databaseTable').value,offset,limit:by('databaseLimit').value});for(const [name,id] of Object.entries(fieldIds)){if(!by(id).disabled && by(id).value.trim())q.set(name,by(id).value.trim());}for(const [key,id] of [['start','databaseStart'],['end','databaseEnd']])if(!by(id).disabled && by(id).value)q.set(key,by(id).value);return q;}
  function controls(){by('databaseApply').disabled=loading;by('databasePrev').disabled=loading||dirty||!result||offset===0;by('databaseNext').disabled=loading||dirty||!result?.has_more||offset+(result?.limit||50)>100000;by('databaseExport').disabled=loading||!result?.rows.length;by('databaseRefresh').disabled=loading||dirty;by('databaseCurrent').disabled=loading||by('databaseSymbol').disabled;by('databasePending').hidden=!dirty;}
  async function load(apply=true){
    if(apply){offset=0;applied=query();dirty=false;}
    if(!applied)return;
    const params=new URLSearchParams(applied);params.set('offset',offset);
    const token=++version;loading=true;controls();error('');by('databaseStatusLine').textContent='Reading filtered rows…';by('databaseRows').setAttribute('aria-busy','true');
    try{const data=await request(params);if(token!==version)return;result=data;offset=data.offset;
      by('databaseRows').innerHTML=data.rows.length?table(data.columns,data.rows.map(r=>data.columns.map(c=>r[c]))):'<p class="db-empty">No rows match these filters. Clear filters or choose another table.</p>';
      by('databaseStatusLine').textContent=`${data.table} · ${data.count.toLocaleString()} matching rows · ${data.columns.length} columns · Read only`;
      by('databasePage').textContent=data.rows.length?`${offset+1}–${offset+data.rows.length} of ${data.count.toLocaleString()}`:'0 rows';
    }catch(e){if(token===version){result=null;by('databaseRows').textContent='';by('databasePage').textContent='';by('databaseStatusLine').textContent='Could not load rows.';error(e.message);}}
    finally{if(token===version){loading=false;controls();by('databaseRows').setAttribute('aria-busy','false');}}
  }
  async function open(){
    if(!dialog.open)dialog.showModal();error('');
    try{metadata=await request();by('databaseInfo').textContent=`${metadata.tables.length} tables · Read only · Data stays on this computer`;
      by('databaseStorage').textContent=`${metadata.database} · ${(metadata.bytes/1024/1024).toLocaleString(undefined,{maximumFractionDigits:1})} MB · Write-ahead log ${(metadata.wal_bytes/1024/1024).toFixed(1)} MB`;
      const previous=by('databaseTable').value;
      by('databaseTable').replaceChildren(...[...metadata.tables].sort((a,b)=>Number(priceTables.has(b))-Number(priceTables.has(a))||a.localeCompare(b)).map(name=>{const option=document.createElement('option');option.value=name;option.textContent=names[name]||name.replaceAll('_',' ');return option;}));
      if(previous && metadata.tables.includes(previous))by('databaseTable').value=previous;
      else{by('databaseTable').value=selected?.symbol==='BTC-USD'?'crypto_exchange_prices':'prices';if(!by('databaseTable').value)by('databaseTable').selectedIndex=0;by('databaseSymbol').value=selected?.symbol||'';by('databaseEnd').value=by('end')?.value||new Date().toISOString().slice(0,10);by('databaseStart').value=new Date(Date.parse(by('databaseEnd').value+'T00:00:00Z')-6*86400000).toISOString().slice(0,10);}
      schema();await load();
    }catch(e){error(e.message);}
  }
  by('databaseForm').onsubmit=e=>{e.preventDefault();load();};
  by('databaseTable').onchange=()=>{Object.values(fieldIds).forEach(id=>by(id).value='');by('databaseStart').value='';by('databaseEnd').value='';if(priceTables.has(by('databaseTable').value))by('databaseSymbol').value=selected?.symbol||'';schema();load();};
  by('databaseForm').addEventListener('input',e=>{if(e.target.id!=='databaseTable'){dirty=true;controls();}});
  by('databaseClear').onclick=()=>{Object.values(fieldIds).forEach(id=>by(id).value='');by('databaseStart').value='';by('databaseEnd').value='';load();};
  by('databaseCurrent').onclick=()=>{by('databaseSymbol').value=selected?.symbol||'';load();};
  by('databaseRefresh').onclick=()=>load(false);
  by('databasePrev').onclick=()=>{offset=Math.max(0,offset-result.limit);load(false);};
  by('databaseNext').onclick=()=>{offset+=result.limit;load(false);};
  by('databaseClose').onclick=()=>dialog.close();
  by('databaseExport').onclick=()=>{if(!result)return;const cell=v=>{let text=String(v??'');if(typeof v==='string' && /^[=+@\-]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};const csv=[result.columns.map(cell).join(','),...result.rows.map(r=>result.columns.map(c=>cell(r[c])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`${result.table}-rows-${result.offset+1}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  const button=document.createElement('button');button.id='databaseOpen';button.type='button';button.textContent='Database';button.onclick=open;
  document.querySelector('.terminal-actions')?.append(button);
  const context=document.createElement('button');context.id='databasePanelOpen';context.type='button';context.className='secondary analysis-context-button';context.textContent='Browse database tables';context.onclick=open;by('workspaceDataPanel')?.prepend(context);
  window.openMarketDatabase=open;
})();
