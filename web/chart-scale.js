// Price transforms are shared by rendering, crosshairs and drawing anchors.
const chartScale = {mode: 'linear', auto: true, inverted: false, priceOnly: true, bounds: null, offset: 0, context: ''};
const chartAppearance = {grid: true, lastPrice: true, crosshair: true, up: '#089981', down: '#f23645'};
try {
  const preferences = JSON.parse(localStorage.getItem('atlas.chartScale') || '{}');
  if (['linear', 'log', 'percent', 'indexed'].includes(preferences.mode)) chartScale.mode = preferences.mode;
  chartScale.inverted = preferences.inverted === true;
  chartScale.priceOnly = preferences.priceOnly !== false;
} catch {}
try {
  if (!localStorage.getItem('atlas.studies.v1')) { indicators.add('volume'); renderStudyChips(); }
} catch {}

function saveChartScale() {
  try { localStorage.setItem('atlas.chartScale', JSON.stringify({mode: chartScale.mode, inverted: chartScale.inverted, priceOnly: chartScale.priceOnly})); } catch {}
}

geometry = function() {
  const canvas = $('chart'), box = canvas.getBoundingClientRect(), data = visible();
  const w = box.width, h = box.height, left = 8, right = 82, top = 22, bottom = 30;
  const pw = Math.max(1, w - left - right), ph = Math.max(1, h - top - bottom);
  const context = `${selected?.symbol}:${interval}:${generation}`;
  if (chartScale.context !== context) {
    chartScale.context = context; chartScale.auto = true; chartScale.bounds = null; chartScale.offset = 0;
  }
  const transform = value => chartScale.mode === 'log' ? value > 0 ? Math.log10(value) : NaN : value;
  const inverse = value => chartScale.mode === 'log' ? 10 ** value : value;
  const count = Math.max(1, viewCount), x = i => left + (i + .5 + chartScale.offset) / count * pw;
  const shown = data.map((row, index) => ({row, index})).filter(({index}) => x(index) >= left - pw / count && x(index) <= left + pw + pw / count);
  let values = shown.flatMap(({row}) => [row.low, row.high]).filter(value => Number.isFinite(value) && (chartScale.mode !== 'log' || value > 0));
  if (!chartScale.priceOnly) {
    const series = [];
    for (const id of ['sma', 'ema', 'bb']) if (indicators.has(id)) series.push(...Object.values(quickStudySeries(id, studyConfig(id))));
    for (const item of advanced.values()) if (item.overlay && item.chartContext === studyContext()) series.push(...Object.values(item.outputs));
    for (const output of series) for (const {index} of shown) {
      const value = output[viewStart + index];
      if (Number.isFinite(value) && (chartScale.mode !== 'log' || value > 0)) values.push(value);
    }
  }
  if (!values.length) values = chartScale.mode === 'log' ? [1, 10] : [0, 1];
  const limits = values.reduce((range, value) => [Math.min(range[0], value), Math.max(range[1], value)], [Infinity, -Infinity]);
  let low = transform(limits[0]), high = transform(limits[1]);
  const padding = (high - low || Math.abs(high) * .02 || .02) * .09;
  low -= padding; high += padding;
  if (!chartScale.auto && chartScale.bounds) [low, high] = chartScale.bounds;
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) { low = 0; high = 1; }
  const spanT = high - low, min = inverse(low), max = inverse(high), span = max - min;
  const y = value => top + (chartScale.inverted ? (transform(value) - low) : (high - transform(value))) / spanT * ph;
  const price = pixel => inverse(chartScale.inverted ? low + (pixel - top) / ph * spanT : high - (pixel - top) / ph * spanT);
  const base = rows.find(row => Number.isFinite(row.close) && row.close > 0)?.close || 1;
  const label = value => chartScale.mode === 'percent' ? `${fmt((value / base - 1) * 100)}%` : chartScale.mode === 'indexed' ? fmt(value / base * 100) : fmt(value, Math.abs(value) >= 10000 ? 0 : Math.abs(value) < 1 ? 4 : 2);
  return {canvas, box, data, w, h, left, right, top, bottom, pw, ph, min, max, span, low, high, spanT, transform, inverse, x, y, price, label,
    index: pixel => Math.floor((pixel - left) / pw * count - chartScale.offset)};
};

// A shifted time viewport must use the same inverse transform for drawing input.
pointer = function(event, snap = tool !== 'cursor', g = geometry()) {
  const px = event.clientX - g.box.left, py = event.clientY - g.box.top;
  const i = Math.max(0, Math.min(g.data.length - 1, g.index(px))), bar = g.data[i];
  let price = g.price(Math.max(g.top, Math.min(g.top + g.ph, py))), label = null;
  if (magnet && snap && !event.shiftKey && bar) {
    const candidates = [['O', bar.open], ['H', bar.high], ['L', bar.low], ['C', bar.close]].filter(([, value]) => Number.isFinite(g.y(value)));
    if (candidates.length) [label, price] = candidates.reduce((best, item) => Math.abs(g.y(item[1]) - py) < Math.abs(g.y(best[1]) - py) ? item : best);
  }
  return {g, px, py, i, inside: px >= g.left && px <= g.left + g.pw && py >= g.top && py <= g.top + g.ph, anchor: {date: bar?.date, price, snap: label}};
};

draw = function() {
  const g = geometry(), {canvas, w, h, left, top, pw, ph, data, x, y} = g;
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(w * dpr)); canvas.height = Math.max(1, Math.round(h * dpr));
  const c = canvas.getContext('2d'); c.scale(dpr, dpr); c.fillStyle = window.atlasTheme.palette.surface; c.fillRect(0, 0, w, h);
  syncChartScaleControls();
  if (!data.length || w < 100 || h < 80) return;
  c.font = '11px Segoe UI, sans-serif'; c.lineWidth = 1;
  const ticks = Math.max(3, Math.min(12, Math.floor(ph / 65)));
  for (let i = 0; i <= ticks; i++) {
    const py = top + i * ph / ticks;
    if (chartAppearance.grid) { c.strokeStyle = window.atlasTheme.palette.grid; c.setLineDash([1, 4]); c.beginPath(); c.moveTo(left, py); c.lineTo(left + pw, py); c.stroke(); }
    c.fillStyle = window.atlasTheme.palette.text; c.fillText(g.label(g.price(py)), left + pw + 9, py + 4);
  }
  const dateTicks = Math.max(2, Math.floor(pw / 110));
  for (let i = 0; i <= dateTicks; i++) {
    const px = left + i * pw / dateTicks, index = g.index(px), bar = data[index];
    if (!bar) continue;
    if (chartAppearance.grid) { c.strokeStyle = window.atlasTheme.palette.grid; c.beginPath(); c.moveTo(x(index), top); c.lineTo(x(index), top + ph); c.stroke(); }
    c.fillStyle = window.atlasTheme.palette.muted; c.textAlign = i === 0 ? 'left' : i === dateTicks ? 'right' : 'center';
    const date = new Date(bar.date.length === 10 ? bar.date + 'T12:00:00Z' : bar.date);
    const label = interval === '1d' ? date.toLocaleDateString(undefined, {month: 'short', day: 'numeric', ...(viewCount > 750 ? {year: '2-digit'} : {})}) : date.toLocaleString(undefined, {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
    c.fillText(label, x(index), h - 10);
  }
  c.textAlign = 'left'; c.setLineDash([]);
  c.strokeStyle = '#e0e3eb'; c.beginPath(); c.moveTo(left + pw, 0); c.lineTo(left + pw, h); c.moveTo(left, top + ph); c.lineTo(w, top + ph); c.stroke();
  c.save(); c.beginPath(); c.rect(left, top, pw, ph); c.clip();
  if (indicators.has('volume')) drawVolume(c, g);
  if (chartType === 'candles' || chartType === 'bars') {
    const width = Math.max(1, Math.min(20, pw / Math.max(1, viewCount) * .68));
    data.forEach((row, index) => {
      const px = x(index); if (px < left - width || px > left + pw + width) return;
      const positions = [row.open, row.high, row.low, row.close].map(y); if (!positions.every(Number.isFinite)) return;
      const [oy, hy, ly, cy] = positions, color = row.close >= row.open ? chartAppearance.up : chartAppearance.down;
      c.strokeStyle = color; c.fillStyle = color; c.beginPath(); c.moveTo(px, hy); c.lineTo(px, ly); c.stroke();
      if (chartType === 'bars') { c.beginPath(); c.moveTo(px - width / 2, oy); c.lineTo(px, oy); c.moveTo(px, cy); c.lineTo(px + width / 2, cy); c.stroke(); }
      else c.fillRect(px - width / 2, Math.min(oy, cy), width, Math.max(1, Math.abs(cy - oy)));
    });
  } else {
    if (chartType === 'area') {
      const gradient = c.createLinearGradient(0, top, 0, top + ph);
      gradient.addColorStop(0, '#2962ff40'); gradient.addColorStop(1, '#2962ff02'); c.fillStyle = gradient;
      let segment = [];
      const fillSegment = () => {
        if (!segment.length) return;
        c.beginPath(); c.moveTo(segment[0][0], top + ph);
        segment.forEach(([px, py]) => c.lineTo(px, py));
        c.lineTo(segment.at(-1)[0], top + ph); c.closePath(); c.fill(); segment = [];
      };
      data.forEach((row, index) => { const py = y(row.close); if (Number.isFinite(row.close) && Number.isFinite(py)) segment.push([x(index), py]); else fillSegment(); }); fillSegment();
    }
    c.beginPath(); let started = false;
    data.forEach((row, index) => { const py = y(row.close); if (!Number.isFinite(py)) { started = false; return; } if (started) c.lineTo(x(index), py); else c.moveTo(x(index), py); started = true; });
    c.strokeStyle = '#2962ff'; c.lineWidth = 1.8; c.stroke(); c.lineWidth = 1;
  }
  drawIndicators(c, g); drawAnnotations(c, g);
  c.restore();
  const latest = rows.at(-1), latestY = y(latest.close), color = latest.close >= latest.open ? chartAppearance.up : chartAppearance.down;
  if (chartAppearance.lastPrice && Number.isFinite(latestY) && latestY >= top && latestY <= top + ph) {
    c.strokeStyle = color; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(left, latestY); c.lineTo(left + pw, latestY); c.stroke(); c.setLineDash([]);
    c.fillStyle = color; c.fillRect(left + pw, latestY - 10, g.right, 20); c.fillStyle = '#fff'; c.fillText(g.label(latest.close), left + pw + 7, latestY + 4);
  }
  if (chartAppearance.crosshair && hover >= 0 && hover < data.length && hoverY != null) {
    const px = x(hover), py = Math.max(top, Math.min(top + ph, hoverY));
    if (px >= left && px <= left + pw) {
      c.strokeStyle = '#9598a1'; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(px, top); c.lineTo(px, top + ph); c.moveTo(left, py); c.lineTo(left + pw, py); c.stroke(); c.setLineDash([]);
      c.fillStyle = '#363a45'; c.fillRect(left + pw, py - 10, g.right, 20); c.fillStyle = '#fff'; c.fillText(g.label(hoverAnchor?.price ?? g.price(py)), left + pw + 7, py + 4);
      const text = labelTime(data[hover].date), labelWidth = Math.min(pw, Math.max(92, c.measureText(text).width + 16));
      const labelX = Math.max(left, Math.min(left + pw - labelWidth, px - labelWidth / 2));
      c.fillStyle = '#363a45'; c.fillRect(labelX, top + ph, labelWidth, g.bottom); c.fillStyle = '#fff'; c.textAlign = 'center'; c.fillText(text, labelX + labelWidth / 2, h - 9); c.textAlign = 'left';
      if (tool !== 'cursor') { c.fillStyle = drawingStyle.color; c.beginPath(); c.arc(px, py, 4, 0, Math.PI * 2); c.fill(); }
    }
  }
  c.fillStyle = '#abb0ba'; c.font = '600 12px Segoe UI'; c.fillText('QUANTSTACK', left + 10, top + ph - 12);
  drawAdvancedPanels();
};

const scaleControls = document.createElement('div');
scaleControls.id = 'chartScaleControls'; scaleControls.className = 'chart-scale-controls';
scaleControls.innerHTML = '<span id="chartScaleStatus" class="chart-scale-status" role="status"></span><button id="chartResetView" title="Reset chart view (Alt+R)" aria-label="Reset chart view">↺</button><button id="chartLogScale" title="Logarithmic scale (Alt+L)" aria-pressed="false">log</button><button id="chartAutoScale" title="Auto: fit visible prices to the chart (Alt+A)" aria-pressed="true">auto</button><button id="chartScaleMenuButton" title="Price scale settings" aria-label="Price scale settings" aria-haspopup="menu" aria-expanded="false" aria-controls="chartScaleMenu">⚙</button>';
document.querySelector('.range-toolbar').append(scaleControls);
const scaleMenu = document.createElement('div');
scaleMenu.id = 'chartScaleMenu'; scaleMenu.className = 'chart-scale-menu'; scaleMenu.hidden = true; scaleMenu.setAttribute('role', 'menu'); scaleMenu.setAttribute('aria-label', 'Price scale settings');
scaleMenu.innerHTML = `<button id="chartMenuAuto" role="menuitemcheckbox"><span data-check></span>Auto (fits data to screen)<kbd>Alt A</kbd></button>
  <button id="chartPriceOnly" role="menuitemcheckbox"><span data-check></span>Scale price chart only</button>
  <button id="chartInvertScale" role="menuitemcheckbox"><span data-check></span>Invert scale<kbd>Alt I</kbd></button>
  <hr>${[['linear', 'Regular', ''], ['percent', 'Percent', 'Alt P'], ['indexed', 'Indexed to 100', ''], ['log', 'Logarithmic', 'Alt L']].map(([mode, label, shortcut]) => `<button role="menuitemradio" data-scale-mode="${mode}"><span data-check></span>${label}<kbd>${shortcut}</kbd></button>`).join('')}
  <hr><button id="chartMenuReset" role="menuitem"><span>↺</span>Reset chart view<kbd>Alt R</kbd></button>
  <p>Drag the chart to pan. Drag the price or time axis to scale. Double-click an axis to fit. Percent and indexed modes use the first loaded close.</p>`;
document.body.append(scaleMenu);

function syncChartScaleControls() {
  if (!$('chartAutoScale')) return;
  for (const [id, checked] of [['chartAutoScale', chartScale.auto], ['chartLogScale', chartScale.mode === 'log']]) {
    $(id).setAttribute('aria-pressed', String(checked)); $(id).classList.toggle('active', checked);
  }
  for (const [id, checked] of [['chartMenuAuto', chartScale.auto], ['chartPriceOnly', chartScale.priceOnly], ['chartInvertScale', chartScale.inverted]]) {
    $(id).setAttribute('aria-checked', String(checked)); $(id).querySelector('[data-check]').textContent = checked ? '✓' : '';
  }
  for (const button of scaleMenu.querySelectorAll('[data-scale-mode]')) {
    const checked = button.dataset.scaleMode === chartScale.mode;
    button.setAttribute('aria-checked', String(checked)); button.querySelector('[data-check]').textContent = checked ? '✓' : '';
  }
  $('chartScaleStatus').textContent = chartScale.mode === 'percent' ? '%' : chartScale.mode === 'indexed' ? '100' : chartScale.inverted ? 'Inverted' : '';
}

function setChartScaleMode(mode) {
  if (!['linear', 'log', 'percent', 'indexed'].includes(mode)) return;
  chartScale.mode = mode; chartScale.bounds = null; chartScale.auto = true;
  saveChartScale(); draw();
}
function toggleChartAuto() {
  const g = geometry(); chartScale.auto = !chartScale.auto;
  chartScale.bounds = chartScale.auto ? null : [g.low, g.high]; draw();
}
function resetChartView() {
  chartScale.auto = true; chartScale.bounds = null; chartScale.offset = 0;
  viewCount = Math.min(rows.length, interval === '1d' ? 260 : 300); viewStart = Math.max(0, rows.length - viewCount);
  hover = -1; hoverY = null; hoverAnchor = null; draw();
}
function closeScaleMenu(returnFocus = false) {
  scaleMenu.hidden = true; $('chartScaleMenuButton').setAttribute('aria-expanded', 'false');
  if (returnFocus) $('chartScaleMenuButton').focus();
}
function openScaleMenu(clientX, clientY) {
  scaleMenu.hidden = false; syncChartScaleControls();
  const control = $('chartScaleMenuButton').getBoundingClientRect(), box = scaleMenu.getBoundingClientRect();
  scaleMenu.style.left = `${Math.max(8, Math.min(innerWidth - box.width - 8, clientX == null ? control.right - box.width : clientX - box.width))}px`;
  scaleMenu.style.top = `${Math.max(8, Math.min(innerHeight - box.height - 8, (clientY ?? control.top) - box.height - 5))}px`;
  $('chartScaleMenuButton').setAttribute('aria-expanded', 'true'); scaleMenu.querySelector('button').focus();
}
$('chartAutoScale').onclick = $('chartMenuAuto').onclick = toggleChartAuto;
$('chartLogScale').onclick = () => setChartScaleMode(chartScale.mode === 'log' ? 'linear' : 'log');
$('chartResetView').onclick = $('chartMenuReset').onclick = resetChartView;
$('chartPriceOnly').onclick = () => { chartScale.priceOnly = !chartScale.priceOnly; chartScale.auto = true; saveChartScale(); draw(); };
$('chartInvertScale').onclick = () => { chartScale.inverted = !chartScale.inverted; saveChartScale(); draw(); };
$('chartScaleMenuButton').onclick = () => scaleMenu.hidden ? openScaleMenu() : closeScaleMenu();
scaleMenu.addEventListener('click', event => {
  const button = event.target.closest('button'); if (!button) return;
  if (button.dataset.scaleMode) setChartScaleMode(button.dataset.scaleMode);
  closeScaleMenu(true);
});
scaleMenu.addEventListener('keydown', event => {
  const buttons = [...scaleMenu.querySelectorAll('button')], index = buttons.indexOf(document.activeElement);
  if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeScaleMenu(true); }
  else if (['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
    event.preventDefault(); buttons[event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length].focus();
  } else if (event.key === 'Tab') closeScaleMenu();
});
document.addEventListener('pointerdown', event => { if (!scaleMenu.hidden && !scaleMenu.contains(event.target) && !$('chartScaleMenuButton').contains(event.target)) closeScaleMenu(); });
window.addEventListener('resize', () => closeScaleMenu());

let chartNavigationGesture = null;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
// Allow space around the complete loaded history, using the same limits for
// wheel zoom, time-axis scaling and panning.
const chartTimeCount = count => clamp(Math.round(count), Math.min(5, rows.length), rows.length * 5);
function moveChartTime(target, count = viewCount) {
  viewCount = chartTimeCount(count);
  const maximum = Math.max(0, rows.length - viewCount);
  const keepVisible = Math.min(rows.length, viewCount * .2);
  target = clamp(target, keepVisible - viewCount, rows.length - keepVisible);
  viewStart = clamp(Math.round(target), 0, maximum); chartScale.offset = viewStart - target;
}
function zoomChartTime(factor, ratio = .5) {
  const origin = viewStart - chartScale.offset, old = viewCount, next = chartTimeCount(old * factor);
  moveChartTime(origin + ratio * (old - next), next);
}
function zoomChartPrice(g, factor, anchor = (g.low + g.high) / 2) {
  const low = anchor + (g.low - anchor) * factor, high = anchor + (g.high - anchor) * factor;
  if (![low, high, g.inverse(low), g.inverse(high)].every(Number.isFinite) || high - low < 1e-10) return;
  chartScale.auto = false; chartScale.bounds = [low, high];
}
$('chart').addEventListener('pointerdown', event => {
  if (!rows.length || event.button !== 0) return;
  const g = geometry(), px = event.clientX - g.box.left, py = event.clientY - g.box.top;
  const kind = px > g.left + g.pw ? 'price' : py > g.top + g.ph ? 'time' : 'pan';
  if (kind === 'pan' && (tool !== 'cursor' || !pointer(event, false, g).inside || hitDrawing(g, px, py))) return;
  event.preventDefault(); event.stopImmediatePropagation(); closeScaleMenu(); finishStyleEdit();
  selectedDrawing = null; syncDrawingEditor(); $('chart').focus({preventScroll: true}); $('chart').setPointerCapture(event.pointerId);
  chartNavigationGesture = {kind, id: event.pointerId, x: event.clientX, y: event.clientY, g, start: viewStart - chartScale.offset, count: viewCount, wasAuto: chartScale.auto};
  $('chart').style.cursor = kind === 'price' ? 'ns-resize' : kind === 'time' ? 'ew-resize' : 'grabbing';
}, true);
window.addEventListener('pointermove', event => {
  const gesture = chartNavigationGesture;
  if (gesture && event.pointerId === gesture.id) {
    event.preventDefault(); event.stopImmediatePropagation();
    const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y, g = gesture.g;
    if (gesture.kind === 'price') zoomChartPrice(g, Math.exp(clamp(dy / 180, -5, 5)));
    else if (gesture.kind === 'time') {
      const next = chartTimeCount(gesture.count * Math.exp(clamp(-dx / 200, -5, 5)));
      moveChartTime(gesture.start + (gesture.count - next), next);
    } else {
      moveChartTime(gesture.start - dx / g.pw * gesture.count, gesture.count);
      // Auto keeps fitting the visible bars while panning through time, even
      // when the pointer drifts vertically. Only a manual scale pans in price.
      if (!gesture.wasAuto) {
        const shift = dy / g.ph * g.spanT * (chartScale.inverted ? -1 : 1);
        const bounds = [g.low + shift, g.high + shift];
        if (bounds.every(value => Number.isFinite(g.inverse(value)))) { chartScale.auto = false; chartScale.bounds = bounds; }
      }
    }
    hover = -1; hoverY = null; hoverAnchor = null; draw(); return;
  }
  if (event.target !== $('chart')) return;
  const g = geometry(), px = event.clientX - g.box.left, py = event.clientY - g.box.top;
  if (px > g.left + g.pw || py > g.top + g.ph) { $('chart').style.cursor = px > g.left + g.pw ? 'ns-resize' : 'ew-resize'; hover = -1; hoverY = null; draw(); }
}, true);
function finishChartNavigation(event) {
  if (!chartNavigationGesture || event?.pointerId != null && event.pointerId !== chartNavigationGesture.id) return;
  const id = chartNavigationGesture.id; chartNavigationGesture = null;
  if ($('chart').hasPointerCapture(id)) $('chart').releasePointerCapture(id);
  $('chart').style.cursor = 'grab';
}
window.addEventListener('pointerup', finishChartNavigation, true);
$('chart').addEventListener('pointercancel', finishChartNavigation, true);
$('chart').addEventListener('lostpointercapture', finishChartNavigation, true);
window.addEventListener('blur', () => finishChartNavigation());
$('chart').addEventListener('wheel', event => {
  if (!rows.length) return; event.preventDefault(); event.stopImmediatePropagation();
  if (chartNavigationGesture || drawingGesture) return;
  const g = geometry(), px = event.clientX - g.box.left, py = event.clientY - g.box.top;
  if (px >= g.left + g.pw) zoomChartPrice(g, Math.exp(clamp(event.deltaY / 600, -.5, .5)), g.transform(g.price(clamp(py, g.top, g.top + g.ph))));
  else if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) moveChartTime(viewStart - chartScale.offset + (event.deltaX || event.deltaY) / g.pw * viewCount);
  else zoomChartTime(Math.exp(clamp(event.deltaY / 600, -.5, .5)), clamp((px - g.left) / g.pw, 0, 1));
  hover = -1; hoverY = null; draw();
}, {capture: true, passive: false});
$('chart').addEventListener('dblclick', event => {
  if (tool !== 'cursor') return;
  const g = geometry(), px = event.clientX - g.box.left;
  if (px >= g.left + g.pw) { chartScale.auto = true; chartScale.bounds = null; draw(); }
  else resetChartView();
});
$('chart').addEventListener('contextmenu', event => {
  const g = geometry(); if (event.clientX - g.box.left < g.left + g.pw) return;
  event.preventDefault(); event.stopImmediatePropagation(); openScaleMenu(event.clientX, event.clientY);
}, true);
document.addEventListener('keydown', event => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName) || event.target.isContentEditable || document.querySelector('dialog[open]')) return;
  if (event.key === 'Escape') { finishChartNavigation(); closeScaleMenu(); }
  if (!event.altKey || event.ctrlKey || event.metaKey) return;
  const actions = {a: toggleChartAuto, l: () => setChartScaleMode(chartScale.mode === 'log' ? 'linear' : 'log'), p: () => setChartScaleMode(chartScale.mode === 'percent' ? 'linear' : 'percent'), i: () => $('chartInvertScale').click(), r: resetChartView};
  const action = actions[event.key.toLowerCase()]; if (action) { event.preventDefault(); action(); }
});
syncChartScaleControls();
