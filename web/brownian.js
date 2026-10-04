(() => {
  'use strict';
  const el=id=>document.getElementById(id),pane=el('workspaceBrownianPanel');
  const workerURL=window.ATLAS_BROWNIAN_WORKER_URL || new URL('./brownian-worker.js',document.currentScript.src);
  const number=(v,d=2)=>Number.isFinite(v)?v.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}):'Unavailable';
  const pct=v=>number(v*100)+'%';
  const input=(id,label,min,max,step=1)=>`<label>${label}<input id="bm-${id}" type="number" min="${min}" max="${max}" step="${step}" required></label>`;
  pane.innerHTML=`<div class="markov-header"><div><span class="eyebrow">STOCHASTIC MODELS</span><h2>Brownian motion</h2><p id="brownianContext"></p></div><div class="markov-actions"><button id="brownianExport" class="secondary" disabled>Export JSON</button><button id="brownianCSV" class="secondary" disabled>Percentiles CSV</button></div></div>
    <form id="brownianForm" class="markov-config">
      <label>Drift &amp; volatility<select id="bm-mode"><option value="historical">Fit both from history</option><option value="zero">Zero price drift · fitted volatility</option><option value="manual">Custom per-bar assumptions</option></select></label>
      ${input('window','Estimation window · returns',20,5000)}${input('horizon','Simulation horizon · bars',1,252)}${input('paths','Simulation paths',100,10000)}${input('seed','Random seed',0,4294967295)}
      <label>Price basis<select id="bm-adjusted"><option value="true">Adjusted close</option><option value="false">Raw close</option></select></label>
      ${input('drift','Price drift μ · % per bar',-20,20,.01)}${input('volatility','Volatility σ · % per √bar',0,100,.01)}
      <div class="markov-actions"><button id="brownianRun" type="submit">Simulate paths</button><button id="brownianCancel" type="button" class="secondary" hidden>Cancel</button></div>
    </form><p class="markov-help">Geometric Brownian motion keeps simulated prices positive. One step is one observed chart bar, not one calendar day. Assumptions are constant through the simulation; these are model scenarios, not a validated price forecast.</p>
    <p id="brownianStatus" role="status" aria-live="polite">Load a chart, choose assumptions, then simulate.</p><p id="brownianError" role="alert" hidden></p>
    <div id="brownianResults" hidden><div id="brownianMetrics" class="markov-metrics"></div><div class="markov-section">
      <h3>Simulated price distribution</h3><p id="brownianDescription"></p><label class="brownian-path-toggle"><input id="brownianShowPaths" type="checkbox"> Show 12 sample paths</label>
      <div id="brownianFan"></div><label class="markov-horizon-label">Inspect horizon<input id="brownianHorizon" type="range" min="1" max="60" value="60"></label><div id="brownianReadout" aria-live="polite"></div>
      <p id="brownianTheory"></p></div></div>
    <details class="markov-method"><summary>Model, calibration &amp; limitations</summary>
      <p>The model is dS = μS dt + σS dW. We use its exact transition S[t+1] = S[t] exp(μ − σ²/2 + σZ), with independent standard-normal Z and dt = 1 observed bar. A fixed seed reproduces the same paths and settings.</p>
      <p>For historical fitting, r[t] = ln(P[t]/P[t−1]); σ² is the sample variance of log returns and μ = mean(r) + σ²/2. The last selected adjusted or raw close starts every path. An estimation window of N returns needs N+1 bars; when fewer are loaded, all available returns are used, with a minimum of 20. We reject missing, nonpositive or nonchronological observations rather than bridging them silently. Missing market sessions are not reconstructed.</p>
      <p>Parameters use bar time. Daily crypto, daily stocks, and hourly bars have different trading calendars; no annualization or conversion to future calendar dates is assumed. In zero-price-drift mode μ = 0, so the expected price is flat but the median decreases when volatility is positive. Custom μ is the price-process drift, not the average log return.</p>
      <p>The bands are pointwise simulated 5–95% and 25–75% price percentiles. The loss probability counts terminal prices below the starting price at the inspected bar; it is not the chance of losing at any time along a path. The sample mean and percentiles vary with the seed and path count. Theoretical horizon mean is S₀ exp(μh), median is S₀ exp((μ−σ²/2)h), and log-price standard deviation is σ√h.</p>
      <p>This model assumes independent Gaussian log returns and constant parameters. It omits jumps, volatility clustering, liquidity, transaction costs, and parameter uncertainty. Historical drift is noisy, and raw prices can contain corporate-action jumps. Extreme shocks can fall outside these scenario ranges. This tool does not estimate investment suitability or validate a trading strategy.</p>
      <p><a href="https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf" target="_blank" rel="noopener">Reference: Columbia University — Geometric Brownian motion</a></p>
    </details><p class="brownian-signature">ian.h</p>`;
  let worker=null,result=null,snapshot=null,revision=0;
  let prefs={...AtlasBrownian.defaults};
  try{prefs=AtlasBrownian.options(JSON.parse(localStorage.getItem('atlas.brownian.v1')||'{}'));}catch{}
  for(const key of Object.keys(prefs))el('bm-'+key).value=String(prefs[key]);
  function configure(){const manual=el('bm-mode').value==='manual';el('bm-drift').disabled=el('bm-volatility').disabled=!manual;}
  function context(){el('brownianContext').textContent=`${selected?.symbol || 'No symbol'} · ${interval} · ${rows.length.toLocaleString()} loaded bars`;}
  function busy(value){el('brownianRun').disabled=value;el('brownianCancel').hidden=!value;pane.setAttribute('aria-busy',String(value));}
  function invalidate(message){revision++;worker?.terminate();worker=null;result=null;snapshot=null;busy(false);el('brownianResults').hidden=true;el('brownianExport').disabled=el('brownianCSV').disabled=true;el('brownianError').hidden=true;el('brownianStatus').textContent=message;}
  el('brownianForm').oninput=()=>{configure();invalidate('Settings changed. Run a new simulation.');};
  el('brownianCancel').onclick=()=>invalidate('Simulation canceled.');
  const load=loadHistory;loadHistory=async function(){invalidate('Chart data changed. Run a simulation for the selected history.');await load();context();};
  function fail(message){worker?.terminate();worker=null;busy(false);el('brownianError').textContent=message;el('brownianError').hidden=false;el('brownianStatus').textContent='Simulation could not be completed.';}
  el('brownianForm').onsubmit=event=>{
    event.preventDefault();invalidate('Simulating independent Brownian price paths…');
    try{
      const parameters=AtlasBrownian.options(Object.fromEntries(Object.keys(AtlasBrownian.defaults).map(key=>[key,key==='mode'?el('bm-'+key).value:key==='adjusted'?el('bm-adjusted').value==='true':el('bm-'+key).value===''?NaN:Number(el('bm-'+key).value)])));
      try{localStorage.setItem('atlas.brownian.v1',JSON.stringify(parameters));}catch{}
      snapshot={symbol:selected?.symbol || '',currency:selected?.currency || '',interval};
      const token=revision;worker=new Worker(workerURL);busy(true);
      worker.onmessage=event=>{if(token!==revision)return;if(event.data.error){fail(event.data.error);return;}result=event.data.result;worker.terminate();worker=null;busy(false);render();el('brownianStatus').textContent=`Completed ${parameters.paths.toLocaleString()} paths over ${parameters.horizon} bars.`;};
      worker.onerror=()=>{if(token===revision)fail('The simulation worker could not run. Reload the workspace and retry.');};
      worker.postMessage({bars:rows.slice(-parameters.window-1),parameters});
    }catch(error){fail(error.message);}
  };
  function render(){
    const r=result,last=r.fan.at(-1);
    el('brownianMetrics').innerHTML=[['Starting price',number(r.data.price)],['Price drift / bar',pct(r.mu)],['Volatility / √bar',pct(r.sigma)],['Horizon median',number(last.median)],['Horizon loss probability',pct(last.lossProbability)]].map(([label,value])=>`<div><span>${label}</span><strong>${value}</strong></div>`).join('');
    el('brownianDescription').textContent=`${snapshot.symbol} · ${r.parameters.adjusted?'adjusted':'raw'} prices in ${snapshot.currency || 'quote currency'} · ${r.data.count} estimation returns, ${r.data.start} to ${r.data.end}. Outer band 5–95%, inner band 25–75%, solid median line. Horizon counts ${snapshot.interval} bars.`;
    el('brownianTheory').textContent=`Model-implied horizon: mean ${number(r.theory.mean)}, median ${number(r.theory.median)}, 5–95% range ${number(r.theory.p05)}–${number(r.theory.p95)}, loss probability ${pct(r.theory.lossProbability)}. These analytic values hold the fitted assumptions fixed.`;
    el('brownianHorizon').max=el('brownianHorizon').value=r.parameters.horizon;
    el('brownianResults').hidden=false;el('brownianExport').disabled=el('brownianCSV').disabled=false;drawFan();readout();
  }
  function drawFan(){
    if(!result)return;
    const r=result,fan=r.fan,paths=el('brownianShowPaths').checked?r.samplePaths:[],w=840,h=260,left=85,right=20,bottom=30;
    const bounds=fan.flatMap(f=>[f.p05,f.p95]).concat(paths.flat());let low=Math.min(...bounds),high=Math.max(...bounds);
    const pad=(high-low || Math.abs(high)*.02 || 1)*.05;low-=pad;high+=pad;
    const x=i=>left+i/r.parameters.horizon*(w-left-right),y=v=>h-bottom-(v-low)/(high-low)*(h-bottom-15);
    const path=key=>fan.map((f,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(f[key]).toFixed(2)}`).join(' ');
    const band=(upper,lower)=>path(upper)+' '+fan.slice().reverse().map(f=>`L${x(f.bar).toFixed(2)},${y(f[lower]).toFixed(2)}`).join(' ')+' Z';
    el('brownianFan').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Geometric Brownian motion price scenarios"><title>Price percentiles by future bar; use the slider for exact values</title>${[low,(low+high)/2,high].map(v=>`<line class="markov-grid" x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-7}" y="${y(v)+4}" text-anchor="end">${number(v)}</text>`).join('')}<path class="markov-fan-outer" d="${band('p95','p05')}"/><path class="markov-fan-inner" d="${band('p75','p25')}"/>${paths.map(values=>`<path class="brownian-sample" d="${values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ')}"/>`).join('')}<path class="markov-fan-line" d="${path('median')}"/><text x="${left}" y="${h-6}">Now</text><text x="${w-right}" y="${h-6}" text-anchor="end">${r.parameters.horizon} bars</text></svg>`;
  }
  function readout(){if(!result)return;const f=result.fan[Number(el('brownianHorizon').value)];el('brownianReadout').innerHTML=`<div class="markov-table-wrap"><table><thead><tr>${['Future bar','5%','25%','Median','75%','95%','Mean','P(price < start)'].map(v=>`<th>${v}</th>`).join('')}</tr></thead><tbody><tr>${[f.bar,number(f.p05),number(f.p25),number(f.median),number(f.p75),number(f.p95),number(f.mean),pct(f.lossProbability)].map(v=>`<td>${v}</td>`).join('')}</tr></tbody></table></div>`;}
  el('brownianShowPaths').onchange=drawFan;el('brownianHorizon').oninput=readout;
  function download(content,type,extension){const url=URL.createObjectURL(new Blob([content],{type})),link=document.createElement('a');link.href=url;link.download=`${snapshot.symbol.replace(/[^a-zA-Z0-9._-]/g,'_')}-brownian.${extension}`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  el('brownianExport').onclick=()=>{if(result)download(JSON.stringify({context:snapshot,...result},null,2),'application/json','json');};
  el('brownianCSV').onclick=()=>{if(!result)return;const fields=['bar','p05','p25','median','p75','p95','mean','lossProbability'];download([fields.join(','),...result.fan.map(row=>fields.map(key=>row[key]).join(','))].join('\r\n'),'text/csv;charset=utf-8','csv');};
  configure();context();
})();
