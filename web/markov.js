(() => {
  'use strict';
  const el=id=>document.getElementById(id), pane=el('workspaceMarkovPanel');
  const workerURL=new URL('./markov-worker.js',document.currentScript.src);
  const storageKey='atlas.markov.v1';
  const number=(x,d=2)=>x===null || x===undefined || !Number.isFinite(x)?'Unavailable':x.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
  const percent=x=>x===null || x===undefined?'Unavailable':`${number(x*100,1)}%`;
  const signed=x=>x===null || x===undefined?'Unavailable':`${x>0?'+':''}${number(x,2)}%`;
  const input=(name,label,min,max,step=1)=>`<label>${label}<input id="mk-${name}" type="number" min="${min}" max="${max}" step="${step}" required></label>`;
  pane.innerHTML=`<div class="markov-header"><div><span class="eyebrow">RESEARCH LAB</span><h2>Markov chain analysis</h2><p id="markovContext"></p></div><div class="markov-actions"><button id="markovExport" class="secondary" disabled>Export analysis JSON</button><button id="markovCSV" class="secondary" disabled>Transitions CSV</button></div></div>
    <form id="markovForm" class="markov-config">
      <label>State definition<select id="mk-mode"><option value="quantile">Training quantiles</option><option value="fixed">Fixed down / flat / up</option></select></label>
      <label id="mk-states-label">Number of states<select id="mk-states"><option value="3">3 states</option><option value="5">5 states</option></select></label>
      <span id="mk-threshold-label">${input('threshold','Flat band ± log return %',0,100,.01)}</span>
      ${input('window','Latest returns · 0 = all',0,50000)}
      ${input('train','Initial training %',50,90)}
      <label>Validation update<select id="mk-validation"><option value="expanding">Expanding · score then update</option><option value="frozen">Frozen training model</option></select></label>
      ${input('alpha','Smoothing α per cell',0,10,.1)}${input('horizon','Forecast horizon · bars',1,250)}
      ${input('paths','Simulation paths',100,10000,100)}${input('seed','Simulation seed',0,4294967295)}
      <label>Price basis<select id="mk-adjusted"><option value="true">Adjusted close</option><option value="false">Raw close</option></select></label>
      <div class="markov-actions"><button id="markovRun" type="submit">Run analysis</button><button id="markovCancel" class="secondary" type="button" hidden>Cancel</button></div>
    </form><p class="markov-help">States describe one-bar log returns. Quantile boundaries use the initial training segment and remain fixed. The current forecast refits transitions on all returns in the selected estimation window.</p>
    <p id="markovStatus" role="status" aria-live="polite">Load price history, choose your settings, then run an analysis.</p><p id="markovError" role="alert" hidden></p>
    <div id="markovResults" hidden><div id="markovMetrics" class="markov-metrics"></div><div id="markovWarnings" class="markov-warnings"></div>
      <nav class="markov-tabs" role="tablist" aria-label="Markov analysis results">${['Overview','Transitions','Forecasts','Validation'].map((name,i)=>`<button type="button" role="tab" id="markovTab${name}" aria-controls="markov${name}" aria-selected="${i===0}" tabindex="${i===0?0:-1}">${name}</button>`).join('')}</nav>
      ${['Overview','Transitions','Forecasts','Validation'].map((name,i)=>`<section id="markov${name}" class="markov-section" role="tabpanel" aria-labelledby="markovTab${name}" ${i?'hidden':''}></section>`).join('')}
    </div>
    <details class="markov-method"><summary>Model definitions, formulas &amp; limitations</summary>
      <h3>States and estimation</h3><p>Each observation is r[t] = ln(close[t]) − ln(close[t−1]). The estimation window includes its preceding price bar. Fixed states are down (r &lt; −band), flat (−band ≤ r ≤ band), and up (r &gt; band). Quantile states are ordered from lowest to highest return; exact boundaries belong to the lower state. Quantiles use linear interpolation on the initial training returns. Ties can leave empty states. These are observed return states, not hidden regimes.</p>
      <p>C[i,j] counts adjacent transitions from state i to j. P[i,j] = (C[i,j] + α) / (Σj C[i,j] + Kα). Positive α is a symmetric Dirichlet prior; α = 0 gives empirical frequencies, with a disclosed uniform fallback for unobserved origin rows. Every row sums to one. Missing, nonpositive or nonchronological prices reject the analysis instead of silently joining gaps.</p>
      <h3>Forecasts and long-run behavior</h3><p>The next-state distribution is the current state's row of P. At h bars it is e[current] P^h. “Visit by horizon” includes the current observation (h = 0). Expected run length 1 / (1 − P[i,i]) includes the first observation and assumes constant transition probabilities; an absorbing state has an infinite run length. A unique stationary distribution solves πP = π and Σπ = 1. Multiple closed classes have no unique stationary distribution; periodic chains can have stationary occupancy without horizon-by-horizon convergence.</p>
      <p>Entropy is measured in bits. The displayed conditional entropy weights row entropies by observed outgoing counts. Half-sample drift is the total variation distance ½Σj|Pearly[i,j] − Plate[i,j]|, using fixed boundaries and smoothing, excluding the transition across the split. It is descriptive, not a significance test. The 95% Wilson intervals refer to unsmoothed transition counts, not posterior credible intervals; approximate binomial assumptions can understate uncertainty with dependent or changing data.</p>
      <h3>Chronological validation</h3><p>The initial training segment determines boundaries, transition counts and baseline frequencies. Each validation forecast uses the previous observed state. Expanding validation scores the next observation before updating counts; frozen validation never updates training counts. The baseline predicts historical state frequencies with the same smoothing and update schedule. Brier score is Σj(p[j] − 1[j = actual])² (0 to 2); log loss is −ln(p[actual]), floored at 10⁻¹⁵ for zero probabilities. Lower is better. Skill is 1 − Brier(model) / Brier(baseline); negative values favor the baseline. Ties in predicted class use the first state. Calibration bins compare top-class confidence with observed accuracy. These scores validate one-step state probabilities, not trading profitability or multi-step price ranges. Repeated parameter tuning on this validation segment makes it unsuitable as an untouched test set.</p>
      <h3>Return simulations</h3><p>Each seeded path starts in the latest state, draws its next state from P, then samples a historical log return from that destination state's pool. Log returns accumulate and convert with 100 × (exp(sum) − 1). Shaded ranges are pointwise 5th–95th and 25th–75th percentiles of simulated cumulative returns, not parameter confidence intervals or simultaneous path bounds. Drawdown uses simulated bar closes and includes the initial value. If any reachable state has no observed return, simulation is unavailable.</p>
      <p>The first-order, time-homogeneous assumption ignores longer memory, changing volatility within a state and external information. Simulation holds fitted parameters fixed, assumes returns depend only on the destination state, and cannot generate shocks outside observed state pools. Price basis follows stored provider adjustments; raw returns can contain corporate-action jumps. Intervals count observed bars, not elapsed calendar time; missing sessions and overnight gaps are not reconstructed. Forecasts are historical statistical estimates, without costs, execution rules or investment recommendations.</p>
      <p>References: <a href="https://www.stat.berkeley.edu/~aldous/150/Lectures/lecture_9_post.pdf" target="_blank" rel="noopener">Berkeley: stationary distributions</a> · <a href="https://otexts.com/fpp3/tscv.html" target="_blank" rel="noopener">Forecasting: time series cross-validation</a>.</p>
    </details>`;
  let result=null, worker=null, contextSnapshot=null, request=0;
  let preferences={...AtlasMarkov.defaults};
  try {preferences=AtlasMarkov.options({...preferences,...JSON.parse(localStorage.getItem(storageKey)||'{}')});} catch {}
  for(const name of Object.keys(AtlasMarkov.defaults)) el(`mk-${name}`).value=String(preferences[name]);
  function configure() {
    const fixed=el('mk-mode').value==='fixed';
    el('mk-states-label').hidden=fixed;el('mk-states').disabled=fixed;
    el('mk-threshold-label').hidden=!fixed;el('mk-threshold').disabled=!fixed;
  }
  function parameters() {
    return Object.fromEntries(Object.keys(AtlasMarkov.defaults).map(name=>[name,
      ['mode','validation'].includes(name)?el(`mk-${name}`).value:name==='adjusted'?el('mk-adjusted').value==='true':el(`mk-${name}`).value===''?NaN:Number(el(`mk-${name}`).value)]));
  }
  function context() {el('markovContext').textContent=`${selected?.symbol||'No symbol'} · ${interval} · ${rows.length.toLocaleString()} loaded bars${rows.length?` · ${rows[0].date} to ${rows.at(-1).date}`:''}`;}
  function busy(value) {el('markovRun').disabled=value;el('markovCancel').hidden=!value;pane.setAttribute('aria-busy',String(value));}
  function invalidate(message) {
    request++;if(worker) worker.terminate();worker=null;busy(false);result=null;contextSnapshot=null;
    el('markovResults').hidden=true;el('markovExport').disabled=el('markovCSV').disabled=true;
    el('markovError').hidden=true;el('markovStatus').textContent=message;
  }
  el('markovForm').addEventListener('input',()=>{configure();invalidate('Settings changed. Run analysis to update results.');});
  el('markovCancel').onclick=()=>invalidate('Analysis canceled.');
  const load=loadHistory;
  loadHistory=async function() {invalidate('Chart data changed. Run analysis for the new symbol and range.');await load();context();};
  function fail(message) {busy(false);el('markovError').textContent=message;el('markovError').hidden=false;el('markovStatus').textContent='Analysis could not be completed.';if(worker)worker.terminate();worker=null;}
  el('markovForm').onsubmit=event=>{
    event.preventDefault();invalidate('Fitting transitions, validating forecasts and simulating paths…');
    try {
      const settings=AtlasMarkov.options(parameters());
      try {localStorage.setItem(storageKey,JSON.stringify(settings));} catch {}
      const token=request;contextSnapshot={symbol:selected?.symbol||'',currency:selected?.currency||'',interval};
      worker=new Worker(workerURL);busy(true);
      worker.onmessage=event=>{
        if(token!==request)return;
        if(event.data.error) {fail(event.data.error);return;}
        result=event.data.result;worker.terminate();worker=null;busy(false);render();
        el('markovStatus').textContent=`Analysis complete · ${result.data.returns.toLocaleString()} returns · ${result.validation.count.toLocaleString()} chronological validation forecasts · ${result.simulation.available?`${result.parameters.paths.toLocaleString()} simulated paths`:'return simulation unavailable'}.`;
      };
      worker.onerror=()=>{if(token===request)fail('The analysis worker could not run. Reload the page and try again.');};
      worker.postMessage({bars:rows.map(b=>({date:b.date,close:b.close,adjusted_close:b.adjusted_close})),parameters:settings});
    } catch(error) {fail(error.message);}
  };
  const names=()=>result.parameters.mode==='fixed'?['Down','Flat','Up']:result.parameters.states===3?['Low return','Middle return','High return']:['Lowest return','Lower return','Middle return','Higher return','Highest return'];
  const badge=(i)=>`<span class="markov-state mk-state-${i}">S${i+1} · ${names()[i]}</span>`;
  function range(i) {
    const e=result.edges.map(x=>`${number(x*100,4)}%`), last=result.parameters.states-1;
    if(result.parameters.mode==='fixed') return i===0?`r < ${e[0]}`:i===2?`r > ${e[1]}`:`${e[0]} ≤ r ≤ ${e[1]}`;
    return i===0?`r ≤ ${e[0]}`:i===last?`r > ${e.at(-1)}`:`${e[i-1]} < r ≤ ${e[i]}`;
  }
  const table=(headers,body,caption='')=>`<div class="markov-table-wrap"><table>${caption?`<caption>${caption}</caption>`:''}<thead><tr>${headers.map(x=>`<th scope="col">${x}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`;
  const cell=x=>`<td>${x}</td>`;
  function render() {
    const r=result, n=r.parameters.states, next=r.forecasts[0].probabilities, likely=next.indexOf(Math.max(...next));
    el('markovMetrics').innerHTML=[['Current state',badge(r.current)],['Observed current run',`${r.streak} bars`],['Most likely next state',`${badge(likely)} <small>${percent(next[likely])}</small>`],['Validation Brier skill',percent(r.validation.skill)],['Conditional entropy',`${number(r.conditionalEntropy,3)} bits`]].map(([label,value])=>`<div><span>${label}</span><strong>${value}</strong></div>`).join('');
    el('markovWarnings').innerHTML=r.warnings.length?`<details open><summary>${r.warnings.length} model diagnostics</summary><ul>${r.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`:'';
    el('markovOverview').innerHTML=`<h3>State definitions &amp; persistence</h3><p>Estimation: ${esc(r.data.start)} to ${esc(r.data.end)}. Boundaries fitted through ${esc(r.data.trainingEnd)}. ${r.counts.flat().reduce((a,b)=>a+b,0).toLocaleString()} adjacent transitions.</p>`+
      table(['State','Log return bounds','Observations','Sample share','Mean simple return','Stay probability','Expected run · bars','Stationary share'],r.summary.map(s=>`<tr><th scope="row">${badge(s.state)}</th>${[range(s.state),s.count.toLocaleString(),percent(s.frequency),signed(s.meanSimpleReturn===null?null:s.meanSimpleReturn*100),percent(s.stay),s.dwell===null?'∞':number(s.dwell),percent(s.stationary)].map(cell).join('')}</tr>`).join(''))+
      `<p>${r.chain.irreducible?'All states communicate.':`${r.chain.closed.length} closed communicating class(es).`} ${r.chain.stationary?`Stationary equation residual: ${r.chain.residual.toExponential(1)}.`:'Stationary occupancy is unavailable; see model diagnostics.'} Mean returns describe state membership; they are not forward returns conditional on today's state.</p><h3>Recent observed states</h3><div class="markov-state-strip" aria-label="Latest 120 return states">${r.history.slice(-120).map(x=>`<span class="mk-state-${x.state}" title="${esc(x.date)} · S${x.state+1} · ${number(x.logReturn*100,3)}% log return"></span>`).join('')}</div><p>Oldest to newest, last ${Math.min(120,r.history.length)} observations. Full timestamped state history is included in the JSON export.</p>`;
    el('markovTransitions').innerHTML=`<div class="markov-section-head"><div><h3>Transition matrix</h3><p>Rows: current state → columns: next state. Select a row to inspect counts and uncertainty.</p></div><label>Inspect origin<select id="markovOrigin">${names().map((name,i)=>`<option value="${i}">S${i+1} · ${name}</option>`).join('')}</select></label></div>`+
      table(['From / to',...names().map((_,i)=>badge(i)),'Outgoing N'],r.matrix.map((row,i)=>`<tr><th scope="row"><button type="button" data-markov-origin="${i}" aria-label="Inspect transitions from state ${i+1}">${badge(i)}</button></th>${row.map((v,j)=>`<td class="markov-heat" style="--heat:${Math.round(v*75)}%"><strong>${percent(v)}</strong><small>${r.counts[i][j]} observations</small></td>`).join('')}<td>${r.summary[i].outgoing}</td></tr>`).join(''))+
      '<div id="markovRowDetails"></div>';
    el('markovOrigin').value=String(r.current);el('markovOrigin').onchange=renderRow;
    el('markovTransitions').querySelectorAll('[data-markov-origin]').forEach(button=>button.onclick=()=>{el('markovOrigin').value=button.dataset.markovOrigin;renderRow();});renderRow();
    el('markovForecasts').innerHTML=`<h3>State probabilities from ${badge(r.current)}</h3><p>Horizons count observed bars in ${esc(contextSnapshot.interval)}. Visit probabilities include the current state at bar 0.</p>`+
      table(['Horizon',...names().map((_,i)=>badge(i))],r.forecasts.filter(f=>[1,2,5,10,20,60,120,250,r.parameters.horizon].includes(f.horizon)).map(f=>`<tr><th scope="row">${f.horizon} bars</th>${f.probabilities.map(x=>cell(percent(x))).join('')}</tr>`).join(''))+
      `<h3>Visit at least once by ${r.parameters.horizon} bars</h3><div class="markov-probabilities">${r.forecasts.at(-1).hit.map((v,i)=>`<div>${badge(i)}<strong>${percent(v)}</strong><progress max="1" value="${v}" aria-label="Visit probability for state ${i+1}"></progress></div>`).join('')}</div>`;
    const sim=r.simulation;
    el('markovForecasts').insertAdjacentHTML('beforeend',sim.available?`<h3>Simulated cumulative return distribution</h3><p>${r.parameters.paths.toLocaleString()} seeded paths · outer band 5–95% · inner band 25–75% · median line. Move the horizon slider to inspect exact values. These ranges hold estimated parameters fixed.</p><div id="markovFan"></div><label class="markov-horizon-label">Inspect horizon <input id="markovInspectHorizon" type="range" min="1" max="${r.parameters.horizon}" value="${r.parameters.horizon}"></label><div id="markovFanReadout" aria-live="polite"></div><p>Simulated maximum drawdown: median ${signed(sim.drawdownMedian)}; 5th percentile ${signed(sim.drawdownP05)} (more negative is worse).</p>`:`<p>${esc(sim.reason)}</p>`);
    if(sim.available) {drawFan();el('markovInspectHorizon').oninput=fanReadout;fanReadout();}
    renderValidation();el('markovResults').hidden=false;el('markovExport').disabled=el('markovCSV').disabled=false;
  }
  function renderRow() {
    const i=Number(el('markovOrigin').value), s=result.summary[i];
    el('markovRowDetails').innerHTML=`<h3>${badge(i)} · observed outgoing transitions</h3>`+table(['Next state','Count','Empirical probability','Smoothed probability','Approx. 95% Wilson interval'],result.matrix[i].map((v,j)=>`<tr><th scope="row">${badge(j)}</th>${[result.counts[i][j],s.outgoing?percent(result.counts[i][j]/s.outgoing):'Unavailable',percent(v),s.intervals[j]?s.intervals[j].map(percent).join(' to '):'Unavailable'].map(cell).join('')}</tr>`).join(''))+`<p>Row entropy ${number(s.entropy,3)} bits · early / late half total variation ${percent(s.drift)}. Intervals use raw counts and an approximate binomial model; they do not account for changing transition probabilities.</p>`;
  }
  function drawFan() {
    const fan=[{horizon:0,p05:0,p25:0,median:0,p75:0,p95:0},...result.simulation.fan], w=800,h=240,pad=52;
    let low=Math.min(0,...fan.map(x=>x.p05)),high=Math.max(0,...fan.map(x=>x.p95));
    if(high-low<.01) {low-=.01;high+=.01;}
    const x=i=>pad+i/result.parameters.horizon*(w-pad-14),y=v=>h-30-(v-low)/(high-low)*(h-48);
    const path=key=>fan.map((v,i)=>`${i?'L':'M'}${x(v.horizon).toFixed(2)},${y(v[key]).toFixed(2)}`).join(' ');
    const band=(upper,lower)=>path(upper)+' '+fan.slice().reverse().map(v=>`L${x(v.horizon).toFixed(2)},${y(v[lower]).toFixed(2)}`).join(' ')+' Z';
    el('markovFan').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Simulated cumulative return percentiles across ${result.parameters.horizon} bars"><title>Simulated cumulative return fan; exact values available using the horizon slider below</title>${[low,(low+high)/2,high].map(v=>`<line class="markov-grid" x1="${pad}" x2="${w-14}" y1="${y(v)}" y2="${y(v)}"/><text x="${pad-6}" y="${y(v)+4}" text-anchor="end">${number(v,1)}%</text>`).join('')}<path class="markov-fan-outer" d="${band('p95','p05')}"/><path class="markov-fan-inner" d="${band('p75','p25')}"/><path class="markov-fan-line" d="${path('median')}"/><text x="${pad}" y="${h-6}">Now</text><text x="${w-14}" y="${h-6}" text-anchor="end">${result.parameters.horizon} bars</text></svg>`;
  }
  function fanReadout() {
    const f=result.simulation.fan[Number(el('markovInspectHorizon').value)-1];
    el('markovFanReadout').innerHTML=table(['Horizon','5th percentile','25th percentile','Median','75th percentile','95th percentile','P(return < 0)'],`<tr>${[`${f.horizon} bars`,signed(f.p05),signed(f.p25),signed(f.median),signed(f.p75),signed(f.p95),percent(f.lossProbability)].map(cell).join('')}</tr>`);
  }
  function renderValidation() {
    const v=result.validation;
    el('markovValidation').innerHTML=`<h3>Chronological one-step validation</h3><p>${result.data.training.toLocaleString()} initial training returns, ending ${esc(result.data.trainingEnd)}. ${v.count.toLocaleString()} validation forecasts from ${esc(result.data.validationStart)}. ${result.parameters.validation==='expanding'?'Counts expand only after each outcome is scored.':'Transition and baseline estimates remain frozen at the training cutoff.'} Final forecast estimates use all selected returns.</p>`+
      table(['Model','Brier ↓','Log loss ↓','Top-class accuracy ↑'],[['Markov chain',v.model],['Historical frequencies',v.baseline]].map(([name,s])=>`<tr><th scope="row">${name}</th>${[number(s.brier,4),number(s.logLoss,4),percent(s.accuracy)].map(cell).join('')}</tr>`).join(''))+
      `<p>Brier skill against baseline: <strong>${percent(v.skill)}</strong>. Positive favors Markov; negative favors the baseline. Accuracy alone can hide poor probability estimates.</p><h3>Confusion matrix</h3>`+
      table(['Actual / predicted',...names().map((_,i)=>badge(i))],v.confusion.map((row,i)=>`<tr><th scope="row">${badge(i)}</th>${row.map(cell).join('')}</tr>`).join(''))+
      `<h3>Top-class probability calibration</h3>`+table(['Confidence bin','Forecast count','Mean confidence','Observed accuracy'],v.bins.map((b,i)=>`<tr><th scope="row">${i*20}–${(i+1)*20}%</th>${[b.n,percent(b.confidence),percent(b.accuracy)].map(cell).join('')}</tr>`).join(''))+
      '<p>Calibration bins are [lower, upper), with 100% included in the final bin. Small bin counts are inconclusive. Full per-observation probabilities and baseline forecasts are exported in JSON.</p>';
  }
  const tabs=[...pane.querySelectorAll('.markov-tabs [role="tab"]')];
  function selectTab(tab) {for(const button of tabs) {const active=button===tab;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;el(button.getAttribute('aria-controls')).hidden=!active;}}
  tabs.forEach(tab=>tab.onclick=()=>selectTab(tab));
  pane.querySelector('.markov-tabs').onkeydown=event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const i=tabs.indexOf(document.activeElement),next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:tabs.length-1))%tabs.length;
    selectTab(tabs[next]);tabs[next].focus();
  };
  function download(content,type,suffix) {
    const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');
    a.href=url;a.download=`${contextSnapshot.symbol.replace(/[^a-zA-Z0-9._-]/g,'_')}-markov-${suffix}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  el('markovExport').onclick=()=>{if(result)download(JSON.stringify({context:contextSnapshot,...result},null,2),'application/json','analysis.json');};
  el('markovCSV').onclick=()=>{
    if(!result)return;
    const fields=['symbol','interval','price_basis','start','end','alpha','from_state','to_state','count','outgoing_count','empirical_probability','model_probability','wilson95_low','wilson95_high'];
    const records=result.matrix.flatMap((row,i)=>row.map((v,j)=>[contextSnapshot.symbol,contextSnapshot.interval,result.parameters.adjusted?'adjusted':'raw',result.data.start,result.data.end,result.parameters.alpha,i+1,j+1,result.counts[i][j],result.summary[i].outgoing,result.summary[i].outgoing?result.counts[i][j]/result.summary[i].outgoing:'',v,...(result.summary[i].intervals[j]||['',''])]));
    const quote=value=>`"${String(typeof value==='string' && /^[=+\-@\t\r]/.test(value)?"'"+value:value).replace(/"/g,'""')}"`;
    download('\uFEFF'+[fields,...records].map(row=>row.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8','transitions.csv');
  };
  configure();context();
})();
