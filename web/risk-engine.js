// Historical return and downside-risk statistics for observed chart bars.
(() => {
  'use strict';
  const defaults={window:252,adjusted:true,confidence:.95,riskFreeRate:0,barsPerYear:252};
  const quantile=(sorted,q)=>{const index=(sorted.length-1)*q,lower=Math.floor(index);return sorted[lower]+(sorted[Math.min(lower+1,sorted.length-1)]-sorted[lower])*(index-lower);};
  function options(input={}){
    const p={...defaults,...input};
    if(!Number.isInteger(p.window)||p.window<0||p.window>50000||(p.window>0&&p.window<20))throw Error('Window must be 0 (all bars) or an integer from 20 to 50,000 returns.');
    if(typeof p.adjusted!=='boolean')throw Error('Choose adjusted or raw close prices.');
    if(![.9,.95,.99].includes(p.confidence))throw Error('Confidence must be 90%, 95% or 99%.');
    if(!Number.isFinite(p.riskFreeRate)||p.riskFreeRate<0||p.riskFreeRate>100)throw Error('Risk-free rate must be between 0% and 100% per year.');
    if(!Number.isFinite(p.barsPerYear)||p.barsPerYear<1||p.barsPerYear>8760)throw Error('Invalid annualization setting.');
    return Object.fromEntries(Object.keys(defaults).map(key=>[key,p[key]]));
  }
  function run(bars,input={}){
    const parameters=options(input);
    if(!Array.isArray(bars))throw Error('Load chart history first.');
    const data=parameters.window?bars.slice(-parameters.window-1):bars.slice(),field=parameters.adjusted?'adjusted_close':'close';
    if(data.length<21)throw Error('At least 21 valid price bars (20 returns) are required. Expand the chart date range.');
    let previous=-Infinity;
    const prices=data.map(row=>{
      const time=Date.parse(row.date),price=row[field];
      if(!Number.isFinite(time)||time<=previous)throw Error('Price dates must be unique and strictly chronological.');
      if(typeof price!=='number'||!Number.isFinite(price)||price<=0)throw Error(`Missing or nonpositive ${field}. Choose a valid price basis or range.`);
      previous=time;return price;
    });
    const logReturns=prices.slice(1).map((price,index)=>Math.log(price)-Math.log(prices[index]));
    const simpleReturns=prices.slice(1).map((price,index)=>price/prices[index]-1);
    const count=logReturns.length,meanLog=logReturns.reduce((sum,value)=>sum+value,0)/count;
    const meanSimple=simpleReturns.reduce((sum,value)=>sum+value,0)/count;
    const logVariance=logReturns.reduce((sum,value)=>sum+(value-meanLog)**2,0)/(count-1);
    const simpleVariance=simpleReturns.reduce((sum,value)=>sum+(value-meanSimple)**2,0)/(count-1);
    const logDeviation=Math.sqrt(logVariance),simpleDeviation=Math.sqrt(simpleVariance),annual=Math.sqrt(parameters.barsPerYear);
    const riskFreePerBar=(1+parameters.riskFreeRate/100)**(1/parameters.barsPerYear)-1;
    const excessMean=meanSimple-riskFreePerBar;
    const downside=Math.sqrt(simpleReturns.reduce((sum,value)=>sum+Math.min(0,value-riskFreePerBar)**2,0)/count);
    const sorted=simpleReturns.slice().sort((a,b)=>a-b),tailCount=Math.max(1,Math.ceil(count*(1-parameters.confidence)));
    const tail=sorted.slice(0,tailCount),varQuantile=quantile(sorted,1-parameters.confidence);
    let peak=prices[0],maxDrawdown=0;
    const observations=simpleReturns.map((value,index)=>{
      peak=Math.max(peak,prices[index+1]);
      const drawdown=prices[index+1]/peak-1;
      maxDrawdown=Math.min(maxDrawdown,drawdown);
      return {date:data[index+1].date,logReturn:logReturns[index],simpleReturn:value,drawdown};
    });
    const centered=logReturns.map(value=>value-meanLog),moment2=centered.reduce((sum,value)=>sum+value**2,0)/count;
    const skewness=moment2?centered.reduce((sum,value)=>sum+value**3,0)/count/moment2**1.5:null;
    const excessKurtosis=moment2?centered.reduce((sum,value)=>sum+value**4,0)/count/moment2**2-3:null;
    const histogramCount=20,min=Math.min(...simpleReturns),max=Math.max(...simpleReturns),spread=max-min||Math.max(Math.abs(max)*.02,.0001);
    const histogram=Array.from({length:histogramCount},(_,index)=>({from:min-spread*.02+index*(spread*1.04/histogramCount),to:min-spread*.02+(index+1)*(spread*1.04/histogramCount),count:0}));
    simpleReturns.forEach(value=>{const index=Math.min(histogramCount-1,Math.floor((value-histogram[0].from)/(histogram.at(-1).to-histogram[0].from)*histogramCount));histogram[index].count++;});
    const result={parameters,data:{start:data[0].date,end:data.at(-1).date,count,startPrice:prices[0],endPrice:prices.at(-1)},
      cumulativeReturn:prices.at(-1)/prices[0]-1,annualizedReturn:Math.expm1(meanLog*parameters.barsPerYear),
      annualizedVolatility:logDeviation*annual,sharpe:simpleDeviation?(excessMean/simpleDeviation)*annual:null,
      sortino:downside?(excessMean/downside)*annual:null,maxDrawdown,
      valueAtRisk:Math.max(0,-varQuantile),expectedShortfall:Math.max(0,-tail.reduce((sum,value)=>sum+value,0)/tail.length),
      skewness,excessKurtosis,meanLogReturn:meanLog,confidence:parameters.confidence,
      observations,histogram};
    if(Object.values(result).some(value=>typeof value==='number'&&!Number.isFinite(value)))throw Error('Statistics exceeded numeric limits. Reduce the estimation window.');
    return result;
  }
  globalThis.AtlasRisk={defaults,options,quantile,run};
})();