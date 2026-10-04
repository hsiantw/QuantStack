// Native chart research calculations. No network calls or timers.
(() => {
  'use strict';
  const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
  const variance = a => { const m=mean(a); return a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1); };
  const covariance = (a,b) => { const x=mean(a),y=mean(b); return a.reduce((s,v,i)=>s+(v-x)*(b[i]-y),0)/(a.length-1); };
  const correlation = (a,b) => { const d=Math.sqrt(variance(a)*variance(b)); return d>0?covariance(a,b)/d:null; };
  function prices(rows, adjusted=true) {
    if(!Array.isArray(rows)||rows.length<3||rows.length>50000) throw Error('Load between 3 and 50,000 price bars.');
    return rows.map((r,i)=>{
      const p=adjusted?r.adjusted_close:r.close;
      if(!Number.isFinite(p)||p<=0||!r.date||(i&&r.date<=rows[i-1].date)) throw Error('Prices must be positive, complete and ordered. Choose raw prices if adjusted closes are unavailable.');
      return {date:r.date,price:p};
    });
  }
  const returns = p => p.slice(1).map((v,i)=>v/p[i]-1);
  function align(histories, adjusted=true) {
    const series=histories.map(h=>prices(h,adjusted));
    const maps=series.map(s=>new Map(s.map(r=>[r.date,r.price])));
    const dates=series[0].map(r=>r.date).filter(d=>maps.every(m=>m.has(d)));
    if(dates.length<30) throw Error('At least 30 shared price timestamps are required. Expand the chart range.');
    return {dates,values:maps.map(m=>dates.map(d=>m.get(d)))};
  }
  function portfolio(histories, weights, annual=252, adjusted=true) {
    if(histories.length<2||histories.length>12||weights.length!==histories.length||weights.some(w=>!Number.isFinite(w)||w<0)||Math.abs(weights.reduce((s,w)=>s+w,0)-1)>1e-6) throw Error('Use 2–12 assets with nonnegative weights totaling 100%.');
    const {dates,values}=align(histories,adjusted),ret=values.map(returns);
    const curve=dates.map((date,i)=>({date,value:values.reduce((s,p,j)=>s+weights[j]*p[i]/p[0],0)}));
    let peak=1,maxDrawdown=0;
    curve.forEach(r=>{peak=Math.max(peak,r.value);r.drawdown=r.value/peak-1;maxDrawdown=Math.min(maxDrawdown,r.drawdown);});
    const daily=returns(curve.map(r=>r.value));
    const vol=Math.sqrt(variance(daily)*annual),average=mean(daily);
    return {curve,return:curve.at(-1).value-1,maxDrawdown,volatility:vol,sharpe:vol>0?average*annual/vol:null,
      covariance:ret.map(a=>ret.map(b=>covariance(a,b))),correlations:ret.map(a=>ret.map(b=>correlation(a,b))),count:dates.length};
  }
  function allocation(histories, method='equal', adjusted=true) {
    const {values}=align(histories,adjusted),n=values.length,r=values.map(returns);
    if(method==='equal') return Array(n).fill(1/n);
    const variances=r.map(variance);
    if(variances.some(v=>v<=1e-16)) throw Error('Allocation needs nonconstant return histories.');
    let w=variances.map(v=>1/Math.sqrt(v)); const sum=w.reduce((a,b)=>a+b,0);w=w.map(v=>v/sum);
    if(method==='inverse') return w;
    if(method!=='minimum') throw Error('Unknown allocation method.');
    const c=r.map(a=>r.map(b=>covariance(a,b)));
    const step=1/(2*Math.max(...c.map(row=>row.reduce((s,v)=>s+Math.abs(v),0))));
    for(let k=0;k<500;k++) {
      const next=w.map((v,i)=>v-step*2*c[i].reduce((s,x,j)=>s+x*w[j],0));
      const sorted=next.slice().sort((a,b)=>b-a); let total=0,theta=0;
      sorted.forEach((v,i)=>{total+=v;const t=(total-1)/(i+1);if(v>t)theta=t;});
      w=next.map(v=>Math.max(0,v-theta));
    }
    return w;
  }
  function regression(x,y) {
    const v=variance(x);if(v<1e-16) throw Error('Regression needs a changing explanatory series.');
    const slope=covariance(x,y)/v;return {slope,intercept:mean(y)-slope*mean(x)};
  }
  function pairs(left,right,window=60,adjusted=true) {
    if(!Number.isInteger(window)||window<10||window>500) throw Error('Use a spread window from 10 to 500 bars.');
    const {dates,values}=align([left,right],adjusted),[a,b]=values.map(s=>s.map(Math.log));
    if(dates.length<window) throw Error('Load more shared bars than the spread window.');
    const fit=regression(b,a),spread=a.map((v,i)=>v-fit.intercept-fit.slope*b[i]);
    const tail=spread.slice(-window),sd=Math.sqrt(variance(tail));
    return {beta:fit.slope,intercept:fit.intercept,correlation:correlation(returns(values[0]),returns(values[1])),
      z:sd>1e-12?(tail.at(-1)-mean(tail))/sd:null,count:dates.length,curve:dates.map((date,i)=>({date,value:spread[i]}))};
  }
  function liquidity(rows) {
    if(rows.length<21) throw Error('At least 21 bars are needed for volume analysis.');
    if(rows.some(r=>![r.close,r.high,r.low,r.volume].every(Number.isFinite)||r.close<=0||r.high<r.low||r.volume<0)) throw Error('Complete OHLC and nonnegative volume are required.');
    let obv=0,ad=0;const illiquid=[],curve=[];
    rows.forEach((r,i)=>{
      if(i) { obv+=Math.sign(r.close-rows[i-1].close)*r.volume;if(r.volume>0)illiquid.push(Math.abs(r.close/rows[i-1].close-1)/(r.close*r.volume)); }
      ad+=(r.high===r.low?0:(2*r.close-r.high-r.low)/(r.high-r.low))*r.volume;
      curve.push({date:r.date,value:obv});
    });
    const prior=mean(rows.slice(-21,-1).map(r=>r.volume)),last=rows.at(-1);
    const low=Math.min(...rows.map(r=>r.low)),high=Math.max(...rows.map(r=>r.high)),width=(high-low||1)/20;
    const profile=Array.from({length:20},(_,i)=>({price:low+(i+.5)*width,volume:0}));
    rows.forEach(r=>profile[Math.max(0,Math.min(19,Math.floor((r.close-low)/width)))].volume+=r.volume);
    return {relativeVolume:prior>0?last.volume/prior:null,turnover:mean(rows.slice(-20).map(r=>r.volume*r.close)),
      amihud:illiquid.length?mean(illiquid)*1e6:null,obv,ad,profile,curve};
  }
  function cdf(x) {const t=1/(1+.2316419*Math.abs(x)),d=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI),p=1-d*t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return x>=0?p:1-p;}
  function option({spot,strike,days,volatility,rate=0,dividend=0,type='call'}) {
    if(![spot,strike,days,volatility,rate,dividend].every(Number.isFinite)||spot<=0||strike<=0||days<=0||days>3650||volatility<=0||volatility>5||Math.abs(rate)>1||Math.abs(dividend)>1||!['call','put'].includes(type)) throw Error('Enter valid positive spot, strike, expiry and volatility.');
    const t=days/365,s=volatility,root=Math.sqrt(t),dr=Math.exp(-rate*t),dq=Math.exp(-dividend*t);
    const d1=(Math.log(spot/strike)+(rate-dividend+s*s/2)*t)/(s*root),d2=d1-s*root,pdf=Math.exp(-d1*d1/2)/Math.sqrt(2*Math.PI),call=type==='call';
    const price=call?spot*dq*cdf(d1)-strike*dr*cdf(d2):strike*dr*cdf(-d2)-spot*dq*cdf(-d1);
    const delta=call?dq*cdf(d1):dq*(cdf(d1)-1),gamma=dq*pdf/(spot*s*root),vega=spot*dq*pdf*root/100;
    const theta=(-spot*dq*pdf*s/(2*root)+(call?-rate*strike*dr*cdf(d2)+dividend*spot*dq*cdf(d1):rate*strike*dr*cdf(-d2)-dividend*spot*dq*cdf(-d1)))/365;
    return {price,delta,gamma,vega,theta,curve:Array.from({length:61},(_,i)=>{const x=spot*(.4+i*.02);return {date:x.toFixed(2),value:Math.max(0,call?x-strike:strike-x)-price};})};
  }
  function forecast(rows,horizon=20,adjusted=true) {
    const p=prices(rows,adjusted);if(p.length<100||p.length>5000||!Number.isInteger(horizon)||horizon<1||horizon>120) throw Error('Use 100–5,000 bars and a 1–120 bar forecast horizon.');
    const r=p.slice(1).map((v,i)=>Math.log(v.price/p[i].price)),start=Math.floor(r.length*.8);
    let modelError=0,baselineError=0;const validation=[];
    for(let i=start;i<r.length;i++) {
      const fit=regression(r.slice(0,i-1),r.slice(1,i)),predicted=fit.intercept+fit.slope*r[i-1];
      modelError+=Math.abs(predicted-r[i]);baselineError+=Math.abs(r[i]);validation.push({date:p[i+1].date,predicted,actual:r[i]});
    }
    const fit=regression(r.slice(0,-1),r.slice(1));let last=r.at(-1),price=p.at(-1).price;
    const curve=[{date:'Now',value:price}];
    for(let i=1;i<=horizon;i++){last=fit.intercept+fit.slope*last;price*=Math.exp(last);if(!Number.isFinite(price)||price<=0)throw Error('The fitted model is unstable for this horizon.');curve.push({date:`+${i} bars`,value:price});}
    return {...fit,modelMAE:modelError/validation.length,baselineMAE:baselineError/validation.length,validation,curve};
  }
  window.AtlasResearch={prices,align,portfolio,allocation,pairs,liquidity,option,forecast};
})();
