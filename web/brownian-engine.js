// Exact geometric Brownian motion transitions in observed-bar time (dt = 1).
(() => {
  'use strict';
  const defaults = {mode:'historical',window:252,horizon:60,paths:2000,seed:42,adjusted:true,drift:0,volatility:2};
  function options(input={}) {
    const p={...defaults,...input};
    for(const [key,min,max] of [['window',20,5000],['horizon',1,252],['paths',100,10000],['seed',0,4294967295]])
      if(!Number.isInteger(p[key]) || p[key]<min || p[key]>max)throw Error(`Invalid ${key}: use an integer from ${min} to ${max}.`);
    if(!['historical','zero','manual'].includes(p.mode) || typeof p.adjusted!=='boolean')throw Error('Invalid calibration mode or price basis.');
    if(!Number.isFinite(p.drift) || p.drift < -20 || p.drift > 20)throw Error('Drift must be between -20% and 20% per bar.');
    if(!Number.isFinite(p.volatility) || p.volatility<0 || p.volatility>100)throw Error('Volatility must be between 0% and 100% per bar.');
    return Object.fromEntries(Object.keys(defaults).map(key=>[key,p[key]]));
  }
  function calibrate(bars,p) {
    if(!Array.isArray(bars))throw Error('Load chart history first.');
    const data=bars.slice(-p.window-1),field=p.adjusted?'adjusted_close':'close';
    if(data.length<21)throw Error('At least 21 price bars (20 returns) are required. Expand the chart date range.');
    const prices=[];let previous=-Infinity;
    for(const row of data){
      const time=Date.parse(row.date),price=row[field];
      if(!Number.isFinite(time) || time<=previous)throw Error('Price dates must be unique and strictly chronological.');
      if(!Number.isFinite(price) || price<=0)throw Error(`Missing or nonpositive ${field}. Choose a valid price basis or range.`);
      previous=time;prices.push(price);
    }
    const returns=prices.slice(1).map((v,i)=>Math.log(v)-Math.log(prices[i]));
    const mean=returns.reduce((sum,r)=>sum+r,0)/returns.length;
    const variance=returns.reduce((sum,r)=>sum+(r-mean)**2,0)/(returns.length-1);
    return {start:data[0].date,end:data.at(-1).date,count:returns.length,price:prices.at(-1),logMean:mean,variance,mu:mean+variance/2,sigma:Math.sqrt(variance)};
  }
  function normalGenerator(seed) {
    let state=seed>>>0,spare=null;
    function uniform(){state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
    return ()=>{if(spare!==null){const result=spare;spare=null;return result;}const radius=Math.sqrt(-2*Math.log(1-uniform())),angle=2*Math.PI*uniform();spare=radius*Math.sin(angle);return radius*Math.cos(angle);};
  }
  function quantile(sorted,q){const i=(sorted.length-1)*q,k=Math.floor(i);return sorted[k]+(sorted[Math.min(k+1,sorted.length-1)]-sorted[k])*(i-k);}
  function cdf(x){const z=Math.abs(x),t=1/(1+.2316419*z),tail=Math.exp(-z*z/2)/Math.sqrt(2*Math.PI)*t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return x>=0?1-tail:tail;}
  function analytic(price,mu,sigma,horizon){
    const center=(mu-sigma*sigma/2)*horizon,spread=sigma*Math.sqrt(horizon);
    const value=z=>price*Math.exp(center+spread*z);
    return {mean:price*Math.exp(mu*horizon),median:value(0),p05:value(-1.6448536269514722),p95:value(1.6448536269514722),lossProbability:spread ? cdf(-center/spread) : center<0?1:0};
  }
  function run(bars,input={}){
    const parameters=options(input),fit=calibrate(bars,parameters);
    const mu=parameters.mode==='manual'?parameters.drift/100:parameters.mode==='zero'?0:fit.mu;
    const sigma=parameters.mode==='manual'?parameters.volatility/100:fit.sigma;
    const normal=normalGenerator(parameters.seed),logs=new Float64Array(parameters.paths),stepDrift=mu-sigma*sigma/2;
    const samplePaths=Array.from({length:Math.min(12,parameters.paths)},()=>[fit.price]);
    const fan=[{bar:0,p05:fit.price,p25:fit.price,median:fit.price,p75:fit.price,p95:fit.price,mean:fit.price,lossProbability:0}];
    for(let step=1;step<=parameters.horizon;step++){
      const values=new Float64Array(parameters.paths);let sum=0,losses=0;
      for(let i=0;i<parameters.paths;i++){
        logs[i]+=stepDrift+sigma*normal();const value=fit.price*Math.exp(logs[i]);
        if(!Number.isFinite(value) || value<=0)throw Error('Simulation exceeded numeric limits. Reduce volatility, drift or horizon.');
        values[i]=value;sum+=value;if(logs[i]<0)losses++;if(i<samplePaths.length)samplePaths[i].push(value);
      }
      if(!Number.isFinite(sum))throw Error('Simulation exceeded numeric limits. Reduce the input assumptions.');
      values.sort();fan.push({bar:step,p05:quantile(values,.05),p25:quantile(values,.25),median:quantile(values,.5),p75:quantile(values,.75),p95:quantile(values,.95),mean:sum/parameters.paths,lossProbability:losses/parameters.paths});
    }
    const theory=analytic(fit.price,mu,sigma,parameters.horizon);
    if(Object.values(theory).some(x=>!Number.isFinite(x)))throw Error('Model exceeded numeric limits. Reduce the input assumptions.');
    return {model:'Geometric Brownian motion',parameters,data:fit,mu,sigma,fan,samplePaths,theory};
  }
  globalThis.AtlasBrownian={defaults,options,calibrate,normalGenerator,quantile,analytic,run};
})();
