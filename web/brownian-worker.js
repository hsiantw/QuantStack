importScripts('./brownian-engine.js');
self.onmessage=event=>{
  try{self.postMessage({result:AtlasBrownian.run(event.data.bars,event.data.parameters)});}
  catch(error){self.postMessage({error:error.message || 'Simulation failed.'});}
};
