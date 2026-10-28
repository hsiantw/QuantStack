// Use the same deterministic engine while leaving the interface responsive.
self.window = self;
importScripts('./strategy-engine.js');
self.onmessage = ({data}) => {
  try { self.postMessage({result: self.AtlasBacktest.run(data.rows, data.settings)}); }
  catch (error) { self.postMessage({error: error.message || 'Unable to run this strategy.'}); }
};
