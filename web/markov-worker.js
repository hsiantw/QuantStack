importScripts('./markov-engine.js');
self.onmessage = event => {
  try { self.postMessage({result: AtlasMarkov.run(event.data.bars, event.data.parameters)}); }
  catch (error) { self.postMessage({error: error.message || 'Analysis failed.'}); }
};
