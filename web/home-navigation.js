// Home links open the native analysis panel after chart scripts have initialized.
(() => {
  const allowed = new Set(['screener', 'strategy', 'portfolio', 'options', 'risk', 'markets', 'rotation', 'fundamentals', 'liquidity', 'data', 'ideas', 'journal', 'pairs', 'forecast', 'markov', 'brownian', 'compare']);
  function openDestination() {
    const query = new URLSearchParams(location.search);
    if (query.get('database') === '1' && window.openMarketDatabase) {
      window.openMarketDatabase(); return;
    }
    const research = query.get('research');
    if (['strategy','portfolio','factor','exchange'].includes(research) && window.openQuantResearch) {
      window.openQuantResearch(research); return;
    }
    const key = new URLSearchParams(location.hash.slice(1)).get('tool');
    if (allowed.has(key) && window.openChartResearch) window.openChartResearch(key);
  }
  window.addEventListener('hashchange', openDestination);
  window.openHomeDestination = openDestination;
})();
