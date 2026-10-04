self.window = self;
importScripts('./research-engine.js', './strategy-engine.js');
self.onmessage = ({data}) => {
  try {
    let result;
    if(data.kind==='compare') {
      result=Object.entries(AtlasBacktest.catalog).map(([kind,item])=>{
        try {const r=AtlasBacktest.run(data.args[0],{...AtlasBacktest.defaults,...data.args[1],kind});return {strategy:item.label,kind,returnPct:r.returnPct,maxDrawdown:r.maxDrawdown,trades:r.trades.length,fees:r.fees};}
        catch(error){return {strategy:item.label,kind,error:error.message};}
      });
    } else {
      if(!['portfolio','allocation','pairs','option','liquidity','forecast'].includes(data.kind)) throw Error('Unknown analysis.');
      result=AtlasResearch[data.kind](...data.args);
    }
    self.postMessage({result});
  } catch(error) {self.postMessage({error:error.message});}
};
