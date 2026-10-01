/* Finite-state return chains. Pure functions shared by the UI worker and tests. */
(() => {
  'use strict';
  const defaults = {mode: 'quantile', states: 3, threshold: 0.5, window: 1000,
    train: 70, alpha: 0.5, horizon: 20, paths: 2000, seed: 42, adjusted: true, validation: 'expanding'};
  const sum = a => a.reduce((x, y) => x + y, 0);
  const zeros = n => Array(n).fill(0);
  const matrix = n => Array.from({length: n}, () => zeros(n));
  const dot = (a, b) => sum(a.map((v, i) => v * b[i]));
  const step = (v, p) => v.map((_, j) => sum(v.map((x, i) => x * p[i][j])));
  const entropy = p => -sum(p.map(x => x > 0 ? x * Math.log2(x) : 0));
  const quantile = (sorted, q) => {
    const x = (sorted.length - 1) * q, i = Math.floor(x);
    return sorted[i] + (sorted[Math.min(i + 1, sorted.length - 1)] - sorted[i]) * (x - i);
  };
  function options(raw = {}) {
    const p = {...defaults, ...raw};
    for (const [name, low, high, integer] of [['states',3,5,true],['threshold',0,100,false],
      ['window',0,50000,true],['train',50,90,true],['alpha',0,10,false],['horizon',1,250,true],
      ['paths',100,10000,true],['seed',0,4294967295,true]]) {
      if (!Number.isFinite(p[name]) || p[name] < low || p[name] > high || (integer && !Number.isInteger(p[name]))) throw Error(`Invalid ${name}: expected ${integer ? 'an integer' : 'a number'} from ${low} to ${high}.`);
    }
    if (![3,5].includes(p.states) || !['quantile','fixed'].includes(p.mode) ||
      !['expanding','frozen'].includes(p.validation) || typeof p.adjusted !== 'boolean') throw Error('Invalid model settings.');
    if (p.mode === 'fixed') p.states = 3;
    if (p.window && p.window < 60) throw Error('Use at least 60 returns in the estimation window, or 0 for all loaded returns.');
    return p;
  }
  function prepare(bars, p) {
    if (!Array.isArray(bars)) throw Error('Price history must be an array.');
    const start = p.window ? Math.max(0, bars.length - p.window - 1) : 0;
    const data = bars.slice(start);
    if (data.length < 61) throw Error('Load at least 61 price bars (60 returns) to fit and validate a chain.');
    if (data.length > 50001) throw Error('Limit the estimation window to 50,000 returns.');
    let previous = -Infinity;
    const prices = data.map(b => {
      const time = Date.parse(b.date), value = p.adjusted ? b.adjusted_close : b.close;
      if (!Number.isFinite(time) || time <= previous) throw Error('Price bars must have unique, increasing timestamps.');
      previous = time;
      if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) throw Error(`Missing or nonpositive ${p.adjusted ? 'adjusted' : 'raw'} close at ${b.date}. Choose another price basis or range.`);
      return value;
    });
    const returns = prices.slice(1).map((price,i) => Math.log(price) - Math.log(prices[i]));
    return {returns, dates: data.slice(1).map(b => b.date), start: data[0].date, end: data.at(-1).date, lastPrice: prices.at(-1)};
  }
  function edgesFor(returns, p) {
    if (p.mode === 'fixed') return [-p.threshold / 100, p.threshold / 100];
    const sorted = returns.slice().sort((a,b) => a-b);
    return Array.from({length:p.states-1}, (_,i) => quantile(sorted,(i+1)/p.states));
  }
  // Boundaries belong to the middle state in fixed mode; quantile ties go lower.
  function classify(value, edges, mode) {
    if (mode === 'fixed') return value < edges[0] ? 0 : value > edges[1] ? 2 : 1;
    const i = edges.findIndex(e => value <= e);
    return i < 0 ? edges.length : i;
  }
  function countsFor(states, n) {
    const counts = matrix(n);
    for (let t=1;t<states.length;t++) counts[states[t-1]][states[t]]++;
    return counts;
  }
  function probabilities(counts, alpha) {
    const n = counts.length;
    // Explicit fallback for an unidentified row when smoothing is disabled.
    return counts.map(row => { const total = sum(row) + n * alpha; return row.map(x => total ? (x+alpha)/total : 1/n); });
  }
  function solve(a, b) {
    const n = b.length, m = a.map((row,i) => [...row,b[i]]);
    for (let col=0;col<n;col++) {
      let pivot=col;
      for (let i=col+1;i<n;i++) if (Math.abs(m[i][col]) > Math.abs(m[pivot][col])) pivot=i;
      if (Math.abs(m[pivot][col]) < 1e-13) return null;
      [m[col],m[pivot]]=[m[pivot],m[col]];
      const scale=m[col][col]; for(let j=col;j<=n;j++) m[col][j]/=scale;
      for(let i=0;i<n;i++) if(i!==col) { const factor=m[i][col]; for(let j=col;j<=n;j++) m[i][j]-=factor*m[col][j]; }
    }
    return m.map(row => row[n]);
  }
  function structure(p) {
    const n=p.length, reach=p.map((row,i)=>row.map((v,j)=>v>0 || i===j));
    for(let k=0;k<n;k++) for(let i=0;i<n;i++) for(let j=0;j<n;j++) reach[i][j] ||= reach[i][k] && reach[k][j];
    const seen=new Set(), classes=[];
    for(let i=0;i<n;i++) if(!seen.has(i)) {
      const group=Array.from({length:n},(_,j)=>j).filter(j=>reach[i][j] && reach[j][i]);
      group.forEach(j=>seen.add(j)); classes.push(group);
    }
    const closed=classes.filter(group=>group.every(i=>p[i].every((v,j)=>v===0 || group.includes(j))));
    const gcd=(a,b)=>b ? gcd(b,a%b) : a;
    const periods=closed.map(group=>{
      const levels={[group[0]]:0}, queue=[group[0]]; let d=0;
      for(const i of queue) for(const j of group) if(p[i][j]>0 && levels[j]===undefined) {levels[j]=levels[i]+1;queue.push(j);}
      for(const i of group) for(const j of group) if(p[i][j]>0) d=gcd(d,Math.abs(levels[i]+1-levels[j]));
      return d;
    });
    // A unique stationary distribution exists with one closed class, even with transient states.
    let stationary=null;
    if(closed.length===1) {
      const a=matrix(n), b=zeros(n); b[n-1]=1;
      for(let i=0;i<n-1;i++) for(let j=0;j<n;j++) a[i][j]=p[j][i]-(i===j?1:0);
      a[n-1]=Array(n).fill(1);
      const solution=solve(a,b);
      if(solution) {stationary=solution.map(x=>Math.max(0,x)); const total=sum(stationary);stationary=stationary.map(x=>x/total);}
    }
    return {classes,closed,periods,stationary,irreducible:classes.length===1,
      residual:stationary ? Math.max(...step(stationary,p).map((x,i)=>Math.abs(x-stationary[i]))) : null};
  }
  function wilson(success, total) {
    if(!total) return null;
    const z=1.959963984540054, z2=z*z, p=success/total, scale=1+z2/total;
    const center=(p+z2/(2*total))/scale, half=z*Math.sqrt(p*(1-p)/total+z2/(4*total*total))/scale;
    return [Math.max(0,center-half),Math.min(1,center+half)];
  }
  function validate(states, split, n, p) {
    const counts=countsFor(states.slice(0,split),n), frequencies=zeros(n);
    states.slice(0,split).forEach(s=>frequencies[s]++);
    const predictions=[], confusion=matrix(n), bins=Array.from({length:5},()=>({n:0,confidence:0,correct:0}));
    const scores={model:{brier:0,logLoss:0,correct:0},baseline:{brier:0,logLoss:0,correct:0}};
    for(let t=split;t<states.length;t++) {
      const forecast=probabilities(counts,p.alpha)[states[t-1]], total=sum(frequencies)+n*p.alpha;
      const baseline=frequencies.map(x=>(x+p.alpha)/total), actual=states[t], predicted=forecast.indexOf(Math.max(...forecast));
      for(const [key,dist] of [['model',forecast],['baseline',baseline]]) {
        const score=scores[key];score.brier+=sum(dist.map((v,j)=>(v-(j===actual?1:0))**2));
        score.logLoss+=-Math.log(Math.max(1e-15,dist[actual]));score.correct+=Number(dist.indexOf(Math.max(...dist))===actual);
      }
      confusion[actual][predicted]++;
      const confidence=Math.max(...forecast), bin=bins[Math.min(4,Math.floor(confidence*5))];
      bin.n++;bin.confidence+=confidence;bin.correct+=Number(predicted===actual);
      predictions.push({index:t,from:states[t-1],actual,probabilities:forecast,baseline});
      // Score first; the just-observed outcome only enters the NEXT forecast.
      if(p.validation==='expanding') {counts[states[t-1]][actual]++;frequencies[actual]++;}
    }
    const count=predictions.length;
    for(const score of Object.values(scores)) {score.brier/=count;score.logLoss/=count;score.accuracy=score.correct/count;}
    return {...scores,count,predictions,confusion,bins:bins.map(b=>({n:b.n,confidence:b.n?b.confidence/b.n:null,accuracy:b.n?b.correct/b.n:null})),
      skill:scores.baseline.brier>0?1-scores.model.brier/scores.baseline.brier:null};
  }
  function random(seed) {
    let state=seed>>>0;
    return () => {state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  }
  function simulate(p, pools, current, settings) {
    const n=p.length, reachable=new Set([current]);
    for(let k=0;k<n;k++) for(const i of [...reachable]) p[i].forEach((v,j)=>{if(v>0)reachable.add(j);});
    if([...reachable].some(i=>!pools[i].length)) return {available:false,reason:'A reachable state has no observed returns. Return simulations are unavailable; adjust states or load more history.'};
    const rng=random(settings.seed), state=Array(settings.paths).fill(current), cumulative=zeros(settings.paths), peak=zeros(settings.paths), drawdown=zeros(settings.paths), fan=[];
    for(let h=1;h<=settings.horizon;h++) {
      for(let path=0;path<settings.paths;path++) {
        const u=rng();let c=0,next=n-1;
        for(let j=0;j<n;j++) {c+=p[state[path]][j];if(u<c) {next=j;break;}}
        state[path]=next;cumulative[path]+=pools[next][Math.floor(rng()*pools[next].length)];
        peak[path]=Math.max(peak[path],cumulative[path]);drawdown[path]=Math.min(drawdown[path],cumulative[path]-peak[path]);
      }
      const sorted=cumulative.slice().sort((a,b)=>a-b);
      const convert=x=>100*Math.expm1(x);
      const values=[.05,.25,.5,.75,.95].map(q=>convert(quantile(sorted,q)));
      if(values.some(x=>!Number.isFinite(x))) return {available:false,reason:'Simulated returns exceed the numerical range. Shorten the horizon.'};
      fan.push({horizon:h,p05:values[0],p25:values[1],median:values[2],p75:values[3],p95:values[4],lossProbability:cumulative.filter(x=>x<0).length/settings.paths});
    }
    const dd=drawdown.map(x=>100*Math.expm1(x)).sort((a,b)=>a-b);
    return {available:true,fan,drawdownMedian:quantile(dd,.5),drawdownP05:quantile(dd,.05)};
  }
  function run(bars, raw={}) {
    const parameters=options(raw), data=prepare(bars,parameters), {returns}=data, n=parameters.states;
    const split=Math.floor(returns.length*parameters.train/100), edges=edgesFor(returns.slice(0,split),parameters);
    const states=returns.map(x=>classify(x,edges,parameters.mode)), counts=countsFor(states,n), p=probabilities(counts,parameters.alpha);
    const frequencies=zeros(n), pools=Array.from({length:n},()=>[]);states.forEach((s,i)=>{frequencies[s]++;pools[s].push(returns[i]);});
    const current=states.at(-1), chain=structure(p), warnings=[];
    if(new Set(edges).size!==edges.length) warnings.push('Repeated quantile boundaries: tied returns leave empty states. Consider fewer states or fixed thresholds.');
    counts.forEach((row,i)=>{if(sum(row)<30) warnings.push(`State ${i+1} has only ${sum(row)} outgoing transitions; estimates are weakly supported.`);});
    if(parameters.alpha===0 && counts.some(row=>sum(row)===0)) warnings.push('Unobserved transition rows use an explicit uniform fallback because smoothing is zero.');
    if(!chain.stationary) warnings.push(chain.closed.length>1
      ? 'Multiple closed classes: there is no unique stationary distribution. Long-run occupancy is not reported.'
      : 'The stationary system is numerically ill-conditioned. Long-run occupancy is unavailable; consider stronger smoothing.');
    if(chain.periods.some(x=>x>1)) warnings.push('The fitted chain is periodic: a stationary distribution does not imply convergence of individual horizon forecasts.');
    const validation=validate(states,split,n,parameters);
    if(validation.count<100) warnings.push('Fewer than 100 chronological validation forecasts; score comparisons may be unstable.');
    if(validation.skill!==null && validation.skill<=0) warnings.push('The Markov model does not beat the historical-frequency baseline on validation Brier score.');
    let distribution=zeros(n);distribution[current]=1;
    const forecasts=[], notHit=Array.from({length:n},(_,target)=>{const v=zeros(n);v[current]=current===target?0:1;return v;});
    for(let h=1;h<=parameters.horizon;h++) {
      distribution=step(distribution,p);
      notHit.forEach((v,target)=>{notHit[target]=step(v,p);notHit[target][target]=0;});
      forecasts.push({horizon:h,probabilities:distribution.slice(),hit: notHit.map(v=>Math.max(0,Math.min(1,1-sum(v)))),
        expectedLogReturn:pools.every(pool=>pool.length)?dot(distribution,pools.map(pool=>sum(pool)/pool.length)):null});
    }
    const outgoing=counts.map(sum), total=sum(outgoing), empirical=frequencies.map(x=>x/returns.length);
    const conditionalEntropy=dot(outgoing.map(x=>x/total),p.map(entropy));
    const midpoint=Math.floor(states.length/2), firstCounts=countsFor(states.slice(0,midpoint),n), secondCounts=countsFor(states.slice(midpoint),n);
    const firstP=probabilities(firstCounts,parameters.alpha), secondP=probabilities(secondCounts,parameters.alpha);
    const drift=firstP.map((row,i)=>sum(firstCounts[i]) && sum(secondCounts[i]) ? sum(row.map((x,j)=>Math.abs(x-secondP[i][j])))/2 : null);
    if(drift.some(x=>x!==null && x>.25)) warnings.push('Transition probabilities differ substantially between sample halves. A constant transition model may be unstable.');
    const summary=Array.from({length:n},(_,i)=>({state:i,count:frequencies[i],outgoing:outgoing[i],frequency:empirical[i],
      meanLogReturn:pools[i].length?sum(pools[i])/pools[i].length:null,meanSimpleReturn:pools[i].length?sum(pools[i].map(Math.expm1))/pools[i].length:null,
      stay:p[i][i],dwell:p[i][i]<1?1/(1-p[i][i]):null,stationary:chain.stationary?.[i]??null,entropy:entropy(p[i]),
      intervals:counts[i].map(x=>wilson(x,outgoing[i])),drift:drift[i]}));
    const simulation=simulate(p,pools,current,parameters);
    if(!simulation.available) warnings.push(simulation.reason);
    let streak=1;for(let i=states.length-2;i>=0 && states[i]===current;i--)streak++;
    return {version:1,parameters,data:{start:data.start,end:data.end,lastPrice:data.lastPrice,returns:returns.length,training:split,
      trainingEnd:data.dates[split-1],validationStart:data.dates[split]},edges,current,streak,counts,matrix:p,summary,chain,
      forecasts,simulation,validation,conditionalEntropy,warnings,
      history:states.map((state,i)=>({date:data.dates[i],state,logReturn:returns[i],partition:i<split?'training':'validation'}))};
  }
  globalThis.AtlasMarkov=Object.freeze({defaults,run,options,classify,countsFor,probabilities,step,structure,wilson,validate});
})();
