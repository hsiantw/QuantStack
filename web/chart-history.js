// Slide a bounded candle window through the local archive while preserving the viewport.
(() => {
  let cursor = null, state = null;
  const originalQuery = query;
  query = function() {
    const params = originalQuery();
    if (cursor) params.set(cursor.kind, cursor.date);
    return params;
  };
  const originalLoad = loadHistory;
  loadHistory = async function() {
    cursor = null; state = null;
    await originalLoad();
  };
  window.queueChartHistory = function(direction = 0) {
    if (!isArchiveChart() || !rows.length) return;
    const pageSize = chartArchivePageSize();
    if (!state || state.generation !== generation) {
      state = {generation, busy: false, older: rows.length === pageSize, latest: rows.at(-1).date};
    }
    const current = state;
    if (current.busy) return;
    const origin = viewStart - chartScale.offset;
    const margin = Math.max(pageSize * .2, Math.min(pageSize * .25, viewCount));
    const older = direction <= 0 && current.older && origin < margin;
    const newer = direction >= 0 && rows.at(-1).date < current.latest && origin + viewCount > rows.length - margin;
    if (!older && !newer) return;
    // Include the entire viewport and the prefetch margin in the next window.
    const overlap = Math.min(rows.length - 1, Math.ceil(Math.max(pageSize * .4, viewCount + margin + pageSize * .025)));
    const nextCursor = older ? {kind: 'before', date: rows[overlap].date}
      : {kind: 'after', date: rows[rows.length - overlap - 1].date};
    const params = originalQuery(); params.set(nextCursor.kind, nextCursor.date);
    current.busy = true;
    $('chart').setAttribute('aria-busy', 'true');
    $('chartHistoryStatus').textContent = 'Loading earlier candles…';
    if (!older) $('chartHistoryStatus').textContent = 'Loading newer candles…';
    api('/api/history?' + params).then(async result => {
      if (state !== current || generation !== current.generation) return;
      const oldRows = rows;
      if (!result.length || (older && result[0].date >= oldRows[0].date)) {
        if (older) current.older = false;
        return;
      }
      // A user may have reversed direction during the request. Never jump away
      // from the visible candles to display a now irrelevant response.
      const anchor = Math.max(0, Math.min(rows.length - 1, Math.round(viewStart - chartScale.offset)));
      const nextAnchor = result.findIndex(row => row.date === rows[anchor].date);
      if (nextAnchor < 0) return;
      const shift = nextAnchor - anchor, origin = viewStart - chartScale.offset + shift;
      if (origin < 0 || origin + viewCount > result.length) return;
      rows = result; cursor = nextCursor;
      current.older = rows.length === pageSize;
      viewStart = clamp(Math.round(origin), 0, Math.max(0, rows.length - viewCount));
      chartScale.offset = viewStart - origin;
      if (chartNavigationGesture) chartNavigationGesture.start += shift;
      hover = -1; hoverY = null; hoverAnchor = null;
      showError();
      render();
      window.dispatchEvent(new Event('chart-history-page'));
      await Promise.all([reloadAdvanced(), reloadLiquidations()]);
      if (state === current && generation === current.generation) render();
    }).catch(error => {
      if (state === current && generation === current.generation) showError('Could not load previous bars: ' + error.message + '. Pan again to retry.');
    }).finally(() => {
      if (state !== current || generation !== current.generation) return;
      current.busy = false;
      $('chart').setAttribute('aria-busy', 'false');
      $('chartHistoryStatus').textContent = '';
    });
  };
  const status = document.createElement('span');
  status.id = 'chartHistoryStatus'; status.setAttribute('role', 'status');
  $('rangeSummary').after(status);
})();
