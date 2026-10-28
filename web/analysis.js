// Python research jobs share the chart context, with immutable result snapshots.
(() => {
  const by = id => document.getElementById(id);
  const labels = {strategy: 'Strategy validation', portfolio: 'Portfolio report', factor: 'Factor research', exchange: 'Exchange data'};
  const dialog = document.createElement('dialog');
  dialog.id = 'analysisDialog'; dialog.className = 'analysis-dialog';
  dialog.setAttribute('aria-labelledby', 'analysisTitle');
  dialog.innerHTML = `<header class="analysis-head"><div><span class="eyebrow">QUANTSTACK RESEARCH</span><h2 id="analysisTitle">Research tools</h2><p>Test ideas, understand risk, and inspect the data behind your chart.</p></div><button id="analysisClose" type="button" aria-label="Close research tools">Close</button></header>
    <nav class="analysis-tabs" aria-label="Research tools">${Object.entries(labels).map(([id,label]) => `<button type="button" data-analysis-kind="${id}" aria-pressed="false">${label}</button>`).join('')}</nav>
    <div class="analysis-layout"><form id="analysisForm"><h3 id="analysisModeTitle"></h3><p id="analysisDescription"></p><p id="analysisContextNote" class="analysis-context-note" hidden></p>
    <label>Symbols <input id="analysisSymbols" required placeholder="AAPL, MSFT, NVDA" maxlength="650"></label>
    <label data-analysis-field="portfolio">Initial weights (%) <input id="analysisWeights" placeholder="50, 30, 20"><small>One positive weight per symbol, totaling 100. Uses a buy-and-hold allocation.</small></label>
    <div class="analysis-fields"><label>Source<select id="analysisSource"><option>Yahoo Finance</option><option>Bitstamp</option><option>Kraken</option><option>Coinbase</option></select></label>
    <label>Bar interval<select id="analysisInterval"><option value="1m">1 minute</option><option value="5m">5 minutes</option><option value="15m">15 minutes</option><option value="60m">1 hour</option><option value="1d">1 day</option></select></label>
    <label>From<input id="analysisStart" type="date" required></label><label>Through<input id="analysisEnd" type="date" required></label></div>
    <div class="analysis-fields" data-analysis-field="strategy"><label>Fast moving average<input id="analysisFast" type="number" min="2" max="499" value="10" required></label><label>Slow moving average<input id="analysisSlow" type="number" min="3" max="500" value="30" required></label><label>Initial capital<input id="analysisCapital" type="number" min="100" max="100000000" value="10000" required></label><label>Fees per side (%)<input id="analysisFees" type="number" min="0" max="5" step="any" value="0.10" required></label><label>Slippage per side (%)<input id="analysisSlippage" type="number" min="0" max="5" step="any" value="0.05" required></label></div>
    <div class="analysis-fields" data-analysis-field="factor"><label>Momentum lookback (bars)<input id="analysisLookback" type="number" min="2" max="252" value="20" required></label><label>Forward horizon (bars)<input id="analysisHorizon" type="number" min="1" max="60" value="5" required></label></div>
    <p id="analysisWriteNote" data-analysis-field="exchange">Saves public USD spot candles on this computer. Choose up to 31 days; available history varies by exchange.</p>
    <button type="button" class="secondary" id="analysisUseChart">Use chart settings</button>
    <div class="analysis-actions"><button id="analysisRun" type="submit">Run research</button><button id="analysisCancel" type="button" class="secondary" hidden>Cancel job</button></div>
    <p class="analysis-footnote">Closing this window keeps your inputs and leaves the job running.</p><details class="analysis-help"><summary>How this works</summary><p id="analysisMethod"></p><p>Uses completed candles stored on this computer. Limit: 50,000 source bars per symbol; choose a shorter range if needed.</p></details></form>
    <section class="analysis-output" aria-labelledby="analysisResultTitle"><h3 id="analysisResultTitle">Results</h3><p id="analysisStatus" role="status" aria-live="polite">Choose a tool and run an analysis.</p><div id="analysisError" role="alert" hidden></div>
    <div id="analysisResult" hidden></div><div class="analysis-actions"><button id="analysisDownload" class="secondary" hidden>Download report JSON</button><button id="analysisCurveDownload" class="secondary" hidden>Download curve CSV</button><button id="analysisViewChart" hidden>View pulled candles on chart</button></div></section></div>`;
  document.body.append(dialog);
  const descriptions = {
    strategy: 'Compare a moving-average strategy with buy-and-hold, including trading costs. Tests the last 30% of your date range.',
    portfolio: 'See returns, drawdowns and correlations for your initial allocation. Use stocks or crypto in one currency.',
    factor: 'Compare momentum rankings across 5–20 symbols to see how they relate to later returns.',
    exchange: 'Download candles from an exchange, then view them on your chart. Use a USD pair such as BTC-USD or ETH-USD.'
  };
  const methods = {
    strategy: 'VectorBT uses simple moving averages and long positions only. Signals fill at the next available open; fees and slippage apply on each buy or sell. QuantStats summarizes daily risk. These settings are separate from the browser strategy tester.',
    portfolio: 'QuantStats uses common daily observations and buy-and-hold weights that drift with prices. Mixed stock/crypto calendars and currency conversion are not supported.',
    factor: 'Alphalens reports quintile returns, rank correlation and turnover. This describes historical relationships; it is not a trading simulation.',
    exchange: 'CCXT requests public spot data, excludes incomplete candles, and retains the exchange label. A pull makes at most 20 requests; exchange retention may limit coverage.'
  };
  let kind = 'strategy', job = null, result = null, timer = null, busy = false, available = false;
  const drafts = new Map();
  const draftIds = ['analysisSymbols','analysisWeights','analysisSource','analysisInterval','analysisStart','analysisEnd','analysisFast','analysisSlow','analysisCapital','analysisFees','analysisSlippage','analysisLookback','analysisHorizon'];
  let initialized = false, pollRevision = 0;
  function saveDraft() { if(initialized) drafts.set(kind,Object.fromEntries(draftIds.map(id=>[id,by(id).value]))); }
  const remember = id => { try { localStorage.setItem('quantstack.analysis.lastJob', id); } catch {} };
  try { job = localStorage.getItem('quantstack.analysis.lastJob'); } catch {}
  async function request(path = '', body) {
    const response = await fetch('./api/analysis' + path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const text = await response.text();
    let value; try { value = JSON.parse(text); } catch { throw Error('Research is available on the local QuantStack app.'); }
    if (!response.ok) throw Error(value.error || 'Unable to run research.');
    return value;
  }
  function failure(message) { by('analysisError').textContent = message; by('analysisError').hidden = !message; }
  function state(running) {
    busy = running; by('analysisRun').disabled = running || !available;
    by('analysisCancel').hidden = !running;
    by('analysisResult').setAttribute('aria-busy', String(running));
  }
  function mode(next) {
    kind = next;
    by('analysisModeTitle').textContent = labels[kind]; by('analysisDescription').textContent = descriptions[kind];
    by('analysisMethod').textContent = methods[kind];
    by('analysisSymbols').placeholder = kind === 'factor' ? 'At least 5 symbols: AAPL, MSFT, NVDA, AMZN, GOOG' : kind === 'exchange' ? 'BTC-USD' : 'AAPL, MSFT, NVDA';
    dialog.querySelectorAll('[data-analysis-kind]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.analysisKind === kind)));
    dialog.querySelectorAll('[data-analysis-field]').forEach(field => {
      field.hidden = field.dataset.analysisField !== kind;
      field.querySelectorAll('input,select').forEach(input => input.disabled = field.hidden);
    });
    by('analysisInterval').disabled = ['portfolio','factor'].includes(kind);
    if (by('analysisInterval').disabled) by('analysisInterval').value = '1d';
    by('analysisSource').options[0].disabled = kind === 'exchange';
    if (kind === 'exchange' && by('analysisSource').value === 'Yahoo Finance') by('analysisSource').value = 'Bitstamp';
    by('analysisRun').textContent = kind === 'exchange' ? 'Pull and save candles' : 'Run research';
  }
  function changeMode(next) {
    saveDraft();
    mode(next);
    if(drafts.has(next)) {
      for(const [id,value] of Object.entries(drafts.get(next))) by(id).value=value;
      by('analysisContextNote').hidden=true;
      mode(next);
    } else useChart();
    initialized=true;
  }
  function useChart() {
    const symbol = selected?.symbol || 'BTC-USD';
    by('analysisSymbols').value = symbol;
    by('analysisWeights').value = '100';
    const supported = {'1m':'1m','5m':'5m','15m':'15m','1h':'60m','1d':'1d'};
    by('analysisInterval').value = supported[interval] || '1d';
    by('analysisSource').value = by('analysisChartSource')?.value || (symbol === 'BTC-USD' && by('analysisInterval').value !== '1d' ? 'Bitstamp' : 'Yahoo Finance');
    by('analysisStart').value = by('start')?.value || new Date(Date.now()-30*86400000).toISOString().slice(0,10);
    by('analysisEnd').value = by('end')?.value || new Date().toISOString().slice(0,10);
    const notices=[];
    if (kind === 'exchange' && !symbol.endsWith('-USD')) {
      by('analysisSymbols').value='';
      notices.push('Enter the crypto USD pair you want to download.');
    }
    if (kind === 'factor') notices.push('Add at least five symbols for a meaningful ranking comparison.');
    const maxDays=kind==='exchange'?31:['portfolio','factor'].includes(kind)?null:by('analysisInterval').value==='1m'?7:by('analysisInterval').value!=='1d'?30:null;
    if(maxDays && by('analysisEnd').value) {
      const earliest=new Date(Date.parse(by('analysisEnd').value+'T00:00:00Z')-(maxDays-1)*86400000).toISOString().slice(0,10);
      if(by('analysisStart').value<earliest){by('analysisStart').value=earliest;notices.push(`Starts with the last ${maxDays} days of the chart range. You can adjust the dates.`);}
    }
    by('analysisContextNote').textContent=notices.join(' ');by('analysisContextNote').hidden=!notices.length;
    if (by('st-kind')?.value === 'sma') {
      for (const [target,source] of [['analysisFast','st-fast'],['analysisSlow','st-slow'],['analysisCapital','st-capital']]) if(by(source)) by(target).value=by(source).value;
      if(by('st-commission')) by('analysisFees').value=Number(by('st-commission').value);
      if(by('st-slippage')) by('analysisSlippage').value=Number(by('st-slippage').value);
    }
    mode(kind);
  }
  const number = x => typeof x === 'number' && Number.isFinite(x) ? x.toLocaleString(undefined,{maximumFractionDigits:4}) : 'Unavailable';
  const percentageKeys = new Set(['total_return','max_drawdown','annualized_volatility','top_minus_bottom','mean_top_quintile_turnover']);
  const title = key => key.replaceAll('_',' ').replace(/^./, s => s.toUpperCase());
  function table(headers, records) {
    return `<div class="analysis-table"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${records.map(row=>`<tr>${row.map(v=>`<td>${esc(v ?? 'Unavailable')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function plot(points, isFactor) {
    const values = points.flatMap(p=>[p.value,p.benchmark]).filter(x=>typeof x==='number' && Number.isFinite(x));
    if (!values.length) return '';
    const min=Math.min(...values),max=Math.max(...values),span=max-min||1;
    const path = key => points.map((p,i)=> typeof p[key]==='number' ? `${i?'L':'M'}${(25+i/(points.length-1||1)*750).toFixed(2)},${(215-(p[key]-min)/span*190).toFixed(2)}` : '').join(' ');
    return `<figure class="analysis-curve"><figcaption>${isFactor?'Daily rank information coefficient':'Value over time'} · ${esc(points[0].time.slice(0,10))} – ${esc(points.at(-1).time.slice(0,10))}</figcaption><svg viewBox="0 0 800 250" role="img" aria-label="${isFactor?'Factor rank correlation':'Research result value'} over time"><text x="25" y="16" fill="currentColor">${esc(number(max))}</text><text x="25" y="242" fill="currentColor">${esc(number(min))}</text><path d="M25 215H775" stroke="currentColor" opacity=".2"/><path d="${path('value')}" fill="none" stroke="#298fa3" stroke-width="2"/>${points.some(p=>p.benchmark!=null)?`<path d="${path('benchmark')}" fill="none" stroke="#c68b3c" stroke-width="2"/><text x="480" y="16" fill="currentColor">Teal: strategy · Gold: buy &amp; hold</text>`:''}</svg></figure>`;
  }
  function render(value) {
    result=value; by('analysisResult').hidden=false;
    by('analysisResultTitle').textContent=value.title;
    const r=value.request;
    let html=`<p class="analysis-snapshot"><strong>${esc(r.symbols.join(', '))}</strong> · ${esc(r.source)} · ${esc(r.interval)}<br>${esc(r.start)} through ${esc(r.end)} · ${esc(value.currency||'')}</p>`;
    const stats={...(value.metrics||{})};
    if(value.net_return!=null) stats.net_return=value.net_return;
    if(value.benchmark_return!=null) stats.benchmark_return=value.benchmark_return;
    if(value.trade_count!=null) stats.trades=value.trade_count;
    if(value.stored_rows!=null) stats.stored_candles=value.stored_rows;
    html+=`<div class="analysis-metrics">${Object.entries(stats).map(([key,v])=>`<div><span>${esc(title(key))}</span><strong>${esc(typeof v==='number'?(percentageKeys.has(key)||['net_return','benchmark_return'].includes(key)?number(v*100)+'%':number(v)):v??'Unavailable')}</strong></div>`).join('')}</div>`;
    if(value.curve?.length) html+=plot(value.curve,r.kind==='factor');
    if(value.evaluation_start) html+=`<p>Evaluation begins ${esc(value.evaluation_start)}. Positions still open at the end are marked to market.</p>`;
    if(value.quintiles) html+=table(['Quintile','Mean forward return'],value.quintiles.map(q=>[q.quantile,number(q.mean_forward_return*100)+'%']));
    if(value.weights) html+=table(['Holding','Initial weight'],Object.entries(value.weights).map(([s,w])=>[s,number(w)+'%']));
    if(value.correlation) {const symbols=Object.keys(value.correlation);html+=`<details><summary>Return correlations</summary>${table(['Symbol',...symbols],symbols.map(s=>[s,...symbols.map(t=>number(value.correlation[s][t]))]))}</details>`;}
    if(value.audit) html+=`<details open><summary>Data coverage and quality</summary>${table(['Symbol','Bars','First completed bar','Latest completed bar','Age (hours)','Grid gaps','Price basis'],value.audit.map(a=>[a.symbol,a.bars,a.first,a.last,a.age_hours,`${a.missing_grid_slots} — ${a.gap_interpretation}`,a.adjusted?'Adjusted OHLC':'Raw OHLC']))}</details>`;
    if(value.trades?.length) {const headers=Object.keys(value.trades[0]);html+=`<details><summary>Trades (first ${Math.min(value.trade_count,500)} of ${value.trade_count})</summary>${table(headers,value.trades.map(t=>headers.map(h=>typeof t[h]==='number'?number(t[h]):t[h])))}</details>`;}
    if(value.preview) html+=`<details><summary>Latest ${value.preview.length} pulled candles</summary>${table(['Time','Open','High','Low','Close','Volume'],value.preview.map(r=>[r.time,...['open','high','low','close','volume'].map(k=>number(r[k]))]))}</details>`;
    html+=`<details><summary>Method and limitations</summary><ul>${value.assumptions.map(a=>`<li>${esc(a)}</li>`).join('')}</ul><p>Calculated ${esc(value.generated_at)}<br>${Object.entries(value.versions).map(([k,v])=>esc(k+' '+v)).join(' · ')}</p></details>`;
    by('analysisResult').innerHTML=html;
    by('analysisDownload').hidden=false; by('analysisCurveDownload').hidden=!value.curve?.length;
    by('analysisViewChart').hidden=r.kind!=='exchange';
  }
  async function poll() {
    clearTimeout(timer);
    if (!job || !dialog.open || document.hidden) return;
    const revision=++pollRevision,currentJob=job;
    try {
      const data=await request('?id='+encodeURIComponent(job));
      if(revision!==pollRevision || currentJob!==job)return;
      by('analysisStatus').textContent=title(data.status)+': '+data.message;
      state(['queued','running'].includes(data.status));
      if(data.status==='complete') render(data.result);
      if(data.status==='failed') failure(data.message);
      if(busy && dialog.open && !document.hidden) timer=setTimeout(poll,1200);
    } catch(error) {if(revision===pollRevision){state(false);failure(error.message);}}
  }
  async function open(next='strategy') {
    changeMode(next); failure('');
    if(!dialog.open) dialog.showModal();
    try {available=!window.ATLAS_STATIC && (await request()).available;state(busy);if(!available) failure('Research environment is unavailable on this server. Use the local app with .venv-analysis installed.'); else if(job) await poll();}
    catch(error){available=false;state(false);failure(error.message);}
  }
  by('analysisForm').onsubmit=async event=>{
    event.preventDefault();failure('');clearTimeout(timer);
    const symbols=by('analysisSymbols').value.trim().toUpperCase().split(/[\s,;]+/).filter(Boolean);
    const payload={kind,symbols,source:by('analysisSource').value,interval:by('analysisInterval').value,start:by('analysisStart').value,end:by('analysisEnd').value};
    if(kind==='strategy') Object.assign(payload,{fast:Number(by('analysisFast').value),slow:Number(by('analysisSlow').value),capital:Number(by('analysisCapital').value),fee_bps:Number(by('analysisFees').value)*100,slippage_bps:Number(by('analysisSlippage').value)*100});
    if(kind==='portfolio') payload.weights=by('analysisWeights').value.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
    if(kind==='factor') Object.assign(payload,{lookback:Number(by('analysisLookback').value),horizon:Number(by('analysisHorizon').value)});
    if(!symbols.length || kind==='factor' && symbols.length<5){failure(kind==='factor'?'Add at least five symbols to compare their rankings.':'Enter at least one symbol.');by('analysisSymbols').focus();return;}
    if(by('analysisStart').value>by('analysisEnd').value){failure('The start date must be on or before the end date.');by('analysisStart').focus();return;}
    if(kind==='strategy' && payload.fast>=payload.slow){failure('The fast moving average must be shorter than the slow moving average.');by('analysisFast').focus();return;}
    if(kind==='portfolio' && (payload.weights.length!==symbols.length || payload.weights.some(w=>!Number.isFinite(w)||w<=0) || Math.abs(payload.weights.reduce((sum,w)=>sum+w,0)-100)>0.000001)){failure('Enter one positive weight per symbol, totaling 100%.');by('analysisWeights').focus();return;}
    saveDraft();++pollRevision;state(true);by('analysisCancel').hidden=true;by('analysisStatus').textContent='Submitting research job…';
    try {const data=await request('',payload);job=data.id;remember(job);result=null;by('analysisResult').hidden=true;['analysisDownload','analysisCurveDownload','analysisViewChart'].forEach(id=>by(id).hidden=true);await poll();}
    catch(error){state(false);failure(error.message);by('analysisStatus').textContent='Job was not submitted.';}
  };
  by('analysisCancel').onclick=async()=>{clearTimeout(timer);++pollRevision;try{await request('/cancel',{id:job});await poll();}catch(error){failure(error.message);}};
  by('analysisClose').onclick=()=>dialog.close(); dialog.addEventListener('close',()=>{saveDraft();clearTimeout(timer);++pollRevision;});
  document.addEventListener('visibilitychange',()=>{clearTimeout(timer);if(!document.hidden && dialog.open && job)poll();});
  by('analysisUseChart').onclick=useChart;
  dialog.querySelectorAll('[data-analysis-kind]').forEach(b=>b.onclick=()=>changeMode(b.dataset.analysisKind));
  function download(text,name,mime){const url=URL.createObjectURL(new Blob([text],{type:mime})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  by('analysisDownload').onclick=()=>download(JSON.stringify(result,null,2),'quantstack-research-'+job+'.json','application/json');
  by('analysisCurveDownload').onclick=()=>download('time,value,benchmark\n'+result.curve.map(p=>[p.time,p.value,p.benchmark??''].join(',')).join('\n'),'quantstack-curve-'+job+'.csv','text/csv');
  const sourceLabel=document.createElement('label'); sourceLabel.className='analysis-source';
  sourceLabel.innerHTML='Candle source <select id="analysisChartSource" aria-label="Chart candle source"><option value="">Automatic</option><option>Yahoo Finance</option><option>Bitstamp</option><option>Kraken</option><option>Coinbase</option></select>';
  document.querySelector('.terminal-actions')?.append(sourceLabel);
  by('analysisChartSource').onchange=()=>loadHistory();
  by('analysisViewChart').onclick=async()=>{
    const r=result.request;dialog.close();
    try {
      by('analysisChartSource').value=r.source;
      if(!assets.some(a=>a.symbol===r.symbols[0])) assets.push({symbol:r.symbols[0],name:r.symbols[0],kind:'Crypto',currency:'USD',has_data:true});
      await select(r.symbols[0]);interval=r.interval==='60m'?'1h':r.interval;
      by('start').value=r.start;by('end').value=r.end;await loadHistory();
      document.querySelectorAll('[data-interval]').forEach(b=>b.classList.toggle('active',b.dataset.interval===interval));
    }catch(error){showError(error.message);}
  };
  const launch=document.createElement('button');launch.id='analysisOpen';launch.type='button';launch.textContent='Research tools';launch.onclick=()=>open('strategy');
  document.querySelector('.terminal-actions')?.prepend(launch);
  for(const [target,which,label] of [['strategyForm','strategy','Validate moving-average strategy'],['workspaceRiskPanel','portfolio','Portfolio report'],['workspaceScreenerPanel','factor','Test factor rankings'],['workspaceDataPanel','exchange','Pull exchange candles']]){
    const host=by(target);if(!host)continue;const b=document.createElement('button');b.type='button';b.className='secondary analysis-context-button';b.textContent=label;b.onclick=()=>open(which);host.prepend(b);
  }
  window.openQuantResearch=open;
})();
