(() => {
  'use strict';
  const el=id=>document.getElementById(id),pane=el('workspaceRiskPanel');
  const number=(value,digits=2)=>value==null||!Number.isFinite(value)?'Unavailable':value.toLocaleString(undefined,{maximumFractionDigits:digits,minimumFractionDigits:digits});
  const percent=value=>value==null||!Number.isFinite(value)?'Unavailable':`${number(value*100)}%`;
  const input=(id,label,min,max,step=1)=>`<label>${label}<input id="risk-${id}" type="number" min="${min}" max="${max}" step="${step}" required></label>`;
  pane.innerHTML=`<div class="markov-header"><div><span class="eyebrow">CHART STATISTICS</span><h2>Returns &amp; risk</h2><p id="riskContext"></p></div><div class="markov-actions"><button id="riskCSV" class="secondary" disabled>Export returns CSV</button></div></div>
    <form id="riskForm" class="markov-config">
      ${input('window','Estimation window · returns',0,50000)}
      <label>Price basis<select id="risk-adjusted"><option value="true">Adjusted close</option><option value="false">Raw close</option></select></label>
      <label>Tail confidence<select id="risk-confidence"><option value="0.9">90%</option><option value="0.95">95%</option><option value="0.99">99%</option></select></label>
      ${input('risk-free','Risk-free rate · % p.a.',0,100,.01)}
      <div class="markov-actions"><button id="riskRun" type="submit">Analyze returns</button></div>
    </form><p class="markov-help">Historical bar returns only. Annualization assumptions are shown for the selected chart; no missing sessions are reconstructed. Tail loss is empirical, not a forecast.</p>
    <p id="riskStatus" role="status" aria-live="polite">Load a chart, set the estimation window, then analyze.</p><p id="riskError" role="alert" hidden></p>
    <div id="riskResults" hidden><div id="riskMetrics" class="markov-metrics"></div><div class="markov-section"><h3>Observed return distribution</h3><p id="riskDescription"></p><div id="riskHistogram"></div><div id="riskReadout" class="risk-readout"></div></div></div>
    <details class="markov-method"><summary>Definitions &amp; limitations</summary><p>Returns are close-to-close simple returns; annualized return compounds the mean log return and annualized volatility scales sample log-return deviation by the square root of bars per year. Sharpe and Sortino use the configured annual risk-free rate. Maximum drawdown is the largest peak-to-subsequent-close decline in the selected price series.</p><p>Historical VaR is the selected lower-tail return quantile, reported as a nonnegative loss; expected shortfall is the mean of the worst empirical tail observations. These describe the chosen history, do not estimate future losses, and can understate tail risk. Annualization uses 252 bars for equity daily, 365 for crypto daily, 1,638 for equity hourly and 8,760 for crypto hourly. Statistical moments are descriptive and sensitive to outliers.</p></details>`;
  const storageKey='atlas.risk.v1';let result=null,contextSnapshot=null;
  let preferences={...AtlasRisk.defaults};
  try{preferences=AtlasRisk.options({...preferences,...JSON.parse(localStorage.getItem(storageKey)||'{}')});}catch{}
  for(const [key,id] of [['window','risk-window'],['adjusted','risk-adjusted'],['confidence','risk-confidence'],['riskFreeRate','risk-risk-free']])el(id).value=String(preferences[key]);
  function barsPerYear(){const crypto=/crypto/i.test(selected?.kind||'');return interval==='1h'?(crypto?8760:1638):(crypto?365:252);}
  function context(){
    const annual=barsPerYear();
    el('riskContext').textContent=`${selected?.symbol||'No symbol'} · ${interval} · ${rows.length.toLocaleString()} loaded bars${rows.length?` · ${rows[0].date} to ${rows.at(-1).date}`:''}`;
    pane.querySelector('.markov-help').textContent=`Historical bar returns only. Annualized figures use an assumed ${annual.toLocaleString()} bars per year; no missing sessions are reconstructed. Tail loss is empirical, not a forecast.`;
  }
  function invalidate(message){result=null;contextSnapshot=null;el('riskResults').hidden=true;el('riskCSV').disabled=true;el('riskError').hidden=true;el('riskStatus').textContent=message;}
  el('riskForm').oninput=()=>invalidate('Settings changed. Run the analysis again.');
  const load=loadHistory;
  loadHistory=async function(){invalidate('Chart data changed. Run analysis for the selected history.');await load();context();};
  function fail(message){el('riskError').textContent=message;el('riskError').hidden=false;el('riskStatus').textContent='Analysis could not be completed.';}
  function drawHistogram(){
    const bins=result.histogram,w=820,h=240,left=48,right=18,top=16,bottom=34,maxCount=Math.max(1,...bins.map(bin=>bin.count)),plotW=w-left-right,plotH=h-top-bottom;
    const min=bins[0].from,max=bins.at(-1).to,zero=Math.max(left,Math.min(w-right,left+(0-min)/(max-min)*plotW)),barW=plotW/bins.length;
    el('riskHistogram').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Histogram of observed returns"><title>Observed simple returns by frequency</title><line class="risk-zero" x1="${zero}" x2="${zero}" y1="${top}" y2="${h-bottom}"/>${bins.map((bin,index)=>{const height=bin.count/maxCount*plotH;return `<rect class="risk-bar" x="${(left+index*barW+1).toFixed(2)}" y="${(h-bottom-height).toFixed(2)}" width="${Math.max(1,barW-2).toFixed(2)}" height="${height.toFixed(2)}"><title>${percent(bin.from)} to ${percent(bin.to)} · ${bin.count} bars</title></rect>`;}).join('')}<text x="${left}" y="${h-8}">${percent(min)}</text><text x="${w-right}" y="${h-8}" text-anchor="end">${percent(max)}</text><text x="${zero+5}" y="${top+11}">0%</text></svg>`;
  }
  function render(){
    const r=result;
    el('riskMetrics').innerHTML=[['Observations',r.data.count.toLocaleString()],['Period return',percent(r.cumulativeReturn)],['Annualized return',percent(r.annualizedReturn)],['Annualized volatility',percent(r.annualizedVolatility)],['Sharpe ratio',number(r.sharpe)],['Sortino ratio',number(r.sortino)],['Maximum drawdown',percent(r.maxDrawdown)],['Historical VaR · '+number(r.confidence*100,0)+'%',percent(r.valueAtRisk)],['Expected shortfall',percent(r.expectedShortfall)]].map(([label,value])=>`<div><span>${label}</span><strong>${value}</strong></div>`).join('');
    el('riskDescription').textContent=`${contextSnapshot.symbol} · ${r.parameters.adjusted?'adjusted':'raw'} closes · ${r.data.start} to ${r.data.end} · tail confidence ${number(r.confidence*100,0)}% · risk-free rate ${number(r.parameters.riskFreeRate)}% p.a.`;
    el('riskReadout').innerHTML=`Log-return skewness <strong>${number(r.skewness)}</strong> · excess kurtosis <strong>${number(r.excessKurtosis)}</strong> · mean log return / bar <strong>${percent(r.meanLogReturn)}</strong>.`;
    drawHistogram();el('riskResults').hidden=false;el('riskCSV').disabled=false;
  }
  el('riskForm').onsubmit=event=>{
    event.preventDefault();invalidate('Calculating historical return and risk statistics…');
    try{
      const parameters=AtlasRisk.options({window:Number(el('risk-window').value),adjusted:el('risk-adjusted').value==='true',confidence:Number(el('risk-confidence').value),riskFreeRate:Number(el('risk-risk-free').value),barsPerYear:barsPerYear()});
      try{localStorage.setItem(storageKey,JSON.stringify({...parameters,barsPerYear:AtlasRisk.defaults.barsPerYear}));}catch{}
      contextSnapshot={symbol:selected?.symbol||'',interval};result=AtlasRisk.run(rows,parameters);render();el('riskStatus').textContent=`Analyzed ${result.data.count.toLocaleString()} returns from ${result.data.start} through ${result.data.end}.`;
    }catch(error){fail(error.message);}
  };
  el('riskCSV').onclick=()=>{
    if(!result)return;
    const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"',fields=['date','logReturn','simpleReturn','drawdown'];
    const csv=[['symbol',...fields].join(','),...result.observations.map(row=>[contextSnapshot.symbol,...fields.map(field=>row[field])].map(quote).join(','))].join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),link=document.createElement('a');link.href=url;link.download=`${contextSnapshot.symbol.replace(/[^a-zA-Z0-9._-]/g,'_')}-returns-risk.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  context();
})();