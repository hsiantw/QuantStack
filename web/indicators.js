// Study calculations, display settings, and library interactions.
const studyStorageKey = 'atlas.studies.v1';
const studySources = ['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4'];
const quickStudies = {
  sma: {name: 'Simple moving average', parameters: {timeperiod: 20}, outputs: ['sma'], colors: ['#f5c451'], sources: studySources},
  ema: {name: 'Exponential moving average', parameters: {timeperiod: 50}, outputs: ['ema'], colors: ['#4c8dff'], sources: studySources},
  bb: {name: 'Bollinger Bands', parameters: {timeperiod: 20, multiplier: 2}, outputs: ['upper', 'middle', 'lower'], colors: ['#a879ff', '#c3a3ff', '#a879ff'], sources: studySources},
  volume: {name: 'Volume', parameters: {height: 22}, outputs: ['up', 'down'], colors: ['#26a69a', '#ef5350'], sources: [], transparency: 76}
};
const studyConfigurations = new Map();
const wantedAdvanced = new Set();
const studyRequests = new Map();
const pendingStudies = new Map();
const studyErrors = new Map();
let catalogPromise = null;
let settingsStudy = 'sma';
let settingsSession = 0;
let quickSeriesCache = {rows: null, entries: new Map()};

function studyDescriptor(id) {
  return quickStudies[id] || advancedCatalog.find(item => item.id === id);
}

function studyParameterMeta(id, key) {
  if (quickStudies[id]) {
    if (key === 'height') return {label: 'Chart height (%)', min: 5, max: 50, step: 1};
    if (key === 'multiplier') return {label: 'Standard deviations', min: 0.1, max: 10, step: 0.1};
    return {label: 'Period', min: 1, max: 1000, step: 1};
  }
  return studyDescriptor(id)?.parameter_meta?.[key] || {min: 0, max: 100000, step: 'any'};
}

function studyConfig(id) {
  const descriptor = studyDescriptor(id) || {};
  const stored = studyConfigurations.get(id) || {};
  const params = {...(descriptor.parameters || {})};
  for (const key of Object.keys(params)) {
    const value = Number(stored.params?.[key]);
    const meta = studyParameterMeta(id, key);
    if (stored.params?.[key] != null && Number.isFinite(value) &&
        (meta.min == null || value >= meta.min) && (meta.max == null || value <= meta.max) &&
        (!meta.options || Object.hasOwn(meta.options, String(value))) &&
        (meta.step !== 1 || Number.isInteger(value))) params[key] = value;
  }
  const sources = descriptor.sources || [];
  const source = sources.includes(stored.source) ? stored.source : descriptor.source || (sources.includes('close') ? 'close' : sources[0]) || 'close';
  const colors = {};
  (descriptor.outputs || ['real']).forEach((name, index) => {
    const value = stored.colors?.[name];
    colors[name] = /^#[0-9a-f]{6}$/i.test(value || '') ? value : descriptor.colors?.[index] || indicatorColors[index % indicatorColors.length];
  });
  const width = Number(stored.width), transparency = Number(stored.transparency);
  return {params, source, colors,
    width: Number.isFinite(width) && width >= 0.5 && width <= 6 ? width : 1.7,
    transparency: Number.isFinite(transparency) && transparency >= 0 && transparency <= 100 ? transparency : descriptor.transparency || 0};
}

function restoreStudySettings() {
  try {
    const data = JSON.parse(localStorage.getItem(studyStorageKey) || 'null');
    if (!data || typeof data !== 'object') return;
    for (const [id, config] of Object.entries(data.configurations || {})) {
      if (config && typeof config === 'object') studyConfigurations.set(id, config);
    }
    for (const id of Array.isArray(data.activeQuick) ? data.activeQuick : []) if (quickStudies[id]) indicators.add(id);
    if (!window.ATLAS_STATIC) {
      for (const id of (Array.isArray(data.activeAdvanced) ? data.activeAdvanced : []).slice(0, 8)) {
        if (typeof id === 'string') wantedAdvanced.add(id);
      }
    }
  } catch { /* A private browser or an older saved format can use defaults. */ }
}

function saveStudySettings() {
  try {
    localStorage.setItem(studyStorageKey, JSON.stringify({configurations: Object.fromEntries(studyConfigurations), activeQuick: [...indicators], activeAdvanced: [...wantedAdvanced]}));
  } catch { /* Study edits remain usable when storage is unavailable. */ }
}

function indicatorLabel(id) {
  const config = studyConfig(id);
  if (id === 'volume') return 'Volume';
  if (quickStudies[id]) return `${id.toUpperCase()} ${config.params.timeperiod}`;
  const descriptor = studyDescriptor(id);
  const values = Object.keys(descriptor?.parameters || {}).map(key => config.params[key]);
  return values.length ? `${id} (${values.join(', ')})` : id;
}

function studyColor(config, output, index = 0) {
  const hex = config.colors[output] || indicatorColors[index % indicatorColors.length];
  const value = parseInt(hex.slice(1), 16);
  return `rgba(${value >> 16},${value >> 8 & 255},${value & 255},${(100 - config.transparency) / 100})`;
}

function studySource(row, source) {
  if (source === 'hl2') return (row.high + row.low) / 2;
  if (source === 'hlc3') return (row.high + row.low + row.close) / 3;
  if (source === 'ohlc4') return (row.open + row.high + row.low + row.close) / 4;
  return row[source] ?? row.close;
}

function sma(period, source = 'close') {
  const output = Array(rows.length).fill(null);
  let sum = 0;
  for (let i = 0; i < rows.length; i++) {
    sum += studySource(rows[i], source);
    if (i >= period) sum -= studySource(rows[i - period], source);
    if (i >= period - 1) output[i] = sum / period;
  }
  return output;
}

function ema(period, source = 'close') {
  const output = Array(rows.length).fill(null), weight = 2 / (period + 1);
  if (!rows.length) return output;
  output[0] = studySource(rows[0], source);
  for (let i = 1; i < rows.length; i++) output[i] = studySource(rows[i], source) * weight + output[i - 1] * (1 - weight);
  return output;
}

function bollinger(period = 20, multiple = 2, source = 'close') {
  const middle = sma(period, source), upper = Array(rows.length).fill(null), lower = Array(rows.length).fill(null);
  for (let i = period - 1; i < rows.length; i++) {
    let variance = 0;
    for (let j = i - period + 1; j <= i; j++) variance += (studySource(rows[j], source) - middle[i]) ** 2;
    const deviation = Math.sqrt(variance / period) * multiple;
    upper[i] = middle[i] + deviation;
    lower[i] = middle[i] - deviation;
  }
  return {upper, middle, lower};
}

function quickStudySeries(id, config) {
  if (quickSeriesCache.rows !== rows) quickSeriesCache = {rows, entries: new Map()};
  const key = `${id}:${JSON.stringify(config.params)}:${config.source}`;
  if (!quickSeriesCache.entries.has(key)) {
    const period = config.params.timeperiod;
    const result = id === 'bb' ? bollinger(period, config.params.multiplier, config.source) : {[id]: id === 'sma' ? sma(period, config.source) : ema(period, config.source)};
    quickSeriesCache.entries.set(key, result);
  }
  return quickSeriesCache.entries.get(key);
}

function drawVolume(c, g) {
  const config = studyConfig('volume');
  const max = Math.max(1, ...g.data.map(row => row.volume || 0));
  const base = g.h - g.bottom, height = g.ph * config.params.height / 100, width = Math.max(1, g.pw / g.data.length * 0.65);
  g.data.forEach((row, index) => {
    const bar = (row.volume || 0) / max * height;
    c.fillStyle = studyColor(config, row.close >= row.open ? 'up' : 'down');
    c.fillRect(g.x(index) - width / 2, base - bar, width, bar);
  });
}

function studyContext() {
  return selected ? `${generation}:${query().toString()}` : '';
}

function drawIndicators(c, g) {
  for (const id of ['bb', 'sma', 'ema']) {
    if (!indicators.has(id)) continue;
    const config = studyConfig(id);
    Object.entries(quickStudySeries(id, config)).forEach(([name, series], index) => plotSeries(c, g, series, studyColor(config, name, index), config.width));
  }
  const context = studyContext();
  for (const item of advanced.values()) {
    if (!item.overlay || item.chartContext !== context) continue;
    const config = studyConfig(item.id);
    Object.entries(item.outputs).forEach(([name, series], index) => plotSeries(c, g, series, studyColor(config, name, index), config.width));
  }
}

async function fetchAdvanced(name, config = studyConfig(name)) {
  if (!selected) throw Error('Choose an asset before adding a study.');
  const q = query(), context = studyContext();
  const request = (studyRequests.get(name) || 0) + 1;
  studyRequests.set(name, request);
  pendingStudies.set(name, request);
  q.set('name', name);
  q.set('params', JSON.stringify(config.params));
  q.set('source', config.source);
  const isCurrent = () => wantedAdvanced.has(name) && studyRequests.get(name) === request && studyContext() === context;
  try {
    const data = await api('/api/indicator?' + q);
    if (!isCurrent()) return null;
    data.chartContext = context;
    advanced.set(name, data);
    studyErrors.delete(name);
    return data;
  } catch (error) {
    if (!isCurrent()) return null;
    studyErrors.set(name, error.message);
    throw error;
  } finally {
    if (pendingStudies.get(name) === request) pendingStudies.delete(name);
  }
}

async function reloadAdvanced() {
  const context = studyContext();
  for (const [id, item] of advanced) if (item.chartContext !== context) advanced.delete(id);
  syncAdvancedPanels();
  if (!wantedAdvanced.size || window.ATLAS_STATIC) return;
  try {
    await ensureIndicatorCatalog();
    if (studyContext() !== context) return;
    for (const id of wantedAdvanced) if (!studyDescriptor(id)) wantedAdvanced.delete(id);
    const names = [...wantedAdvanced];
    const requests = names.map(id => fetchAdvanced(id));
    renderStudyChips();
    const results = await Promise.allSettled(requests);
    if (studyContext() !== context) return;
    const errors = results.flatMap((result, index) => result.status === 'rejected' ? [`${names[index]}: ${result.reason.message}`] : []);
    if (errors.length) showError(errors.join(' '));
  } catch (error) {
    if (studyContext() === context) showError(error.message);
  }
  syncAdvancedPanels();
  renderAdvancedList();
  renderStudyChips();
}

function syncAdvancedPanels() {
  $('indicatorPanels').innerHTML = [...advanced.values()].filter(item => !item.overlay && item.chartContext === studyContext()).map(item => `
    <div class="indicator-panel" data-panel="${esc(item.id)}">
      <button class="indicator-panel-label study-panel-settings" data-study-settings="${esc(item.id)}" title="Edit ${esc(item.name)} settings">${esc(indicatorLabel(item.id))} · ${esc(item.name)} <span aria-hidden="true">⚙</span></button>
      <button class="indicator-panel-close" data-remove-indicator="${esc(item.id)}" aria-label="Remove ${esc(item.id)}">×</button><canvas aria-label="${esc(item.name)} indicator"></canvas>
    </div>`).join('');
  drawAdvancedPanels();
}

function drawAdvancedPanels() {
  for (const item of advanced.values()) {
    if (item.overlay || item.chartContext !== studyContext()) continue;
    const panel = document.querySelector(`[data-panel="${CSS.escape(item.id)}"]`);
    if (!panel) continue;
    const config = studyConfig(item.id), canvas = panel.querySelector('canvas'), box = canvas.getBoundingClientRect();
    const mainGeometry = geometry();
    const dpr = devicePixelRatio || 1, w = box.width, h = box.height, left = mainGeometry.left, right = mainGeometry.right, top = 27, bottom = 22, pw = w - left - right, ph = h - top - bottom;
    canvas.width = Math.max(1, w * dpr);
    canvas.height = Math.max(1, h * dpr);
    const c = canvas.getContext('2d');
    c.scale(dpr, dpr);
    c.fillStyle = window.atlasTheme.palette.surface;
    c.fillRect(0, 0, w, h);
    const entries = Object.entries(item.outputs);
    const values = entries.flatMap(([, series]) => series.slice(viewStart, viewStart + viewCount).filter(value => value != null && Number.isFinite(value)));
    if (!values.length || pw <= 0 || ph <= 0) continue;
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * 0.08;
    min -= pad; max += pad;
    const y = value => top + (max - value) / (max - min) * ph, x = index => mainGeometry.x(index);
    c.font = '9px Segoe UI';
    for (let i = 0; i < 3; i++) {
      const py = top + i * ph / 2, value = max - i * (max - min) / 2;
      c.strokeStyle = window.atlasTheme.palette.grid; c.lineWidth = 1; c.beginPath(); c.moveTo(left, py); c.lineTo(w - right, py); c.stroke();
      c.fillStyle = window.atlasTheme.palette.muted; c.fillText(fmt(value, Math.abs(value) > 1000 ? 0 : 2), w - right + 7, py + 3);
    }
    if (min < 0 && max > 0) {
      c.strokeStyle = '#526075'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(left, y(0)); c.lineTo(w - right, y(0)); c.stroke(); c.setLineDash([]);
    }
    entries.forEach(([name, series], index) => {
      c.save(); c.beginPath(); c.rect(left, top, pw, ph); c.clip();
      c.beginPath();
      let started = false;
      for (let i = 0; i < viewCount; i++) {
        const value = series[viewStart + i];
        if (value == null || !Number.isFinite(value)) { started = false; continue; }
        if (started) c.lineTo(x(i), y(value)); else c.moveTo(x(i), y(value));
        started = true;
      }
      c.strokeStyle = studyColor(config, name, index); c.lineWidth = config.width; c.stroke();
      c.restore();
      const last = series.slice(viewStart, viewStart + viewCount).findLast(value => value != null && Number.isFinite(value));
      if (last != null) { c.fillStyle = studyColor(config, name, index); c.fillText(`${name} ${fmt(last, 2)}`, left + index * 120, h - 6); }
    });
  }
}

async function ensureIndicatorCatalog() {
  if (advancedCatalog.length) return;
  if (!catalogPromise) catalogPromise = api('/api/indicators').then(data => { advancedCatalog = data; }).finally(() => { catalogPromise = null; });
  await catalogPromise;
  $('indicatorCount').textContent = advancedCatalog.length;
  renderAdvancedList();
}

function renderAdvancedList() {
  if (!advancedCatalog.length) return;
  const search = $('indicatorSearch').value.trim().toLowerCase();
  const filtered = advancedCatalog.filter(item => `${item.id} ${item.name} ${item.group}`.toLowerCase().includes(search)), groups = new Map();
  for (const item of filtered) {
    if (!groups.has(item.group)) groups.set(item.group, []);
    groups.get(item.group).push(item);
  }
  $('activeAdvanced').innerHTML = [...wantedAdvanced].map(id => `<span class="study-library-active"><button data-study-settings="${esc(id)}" title="Edit study">${esc(indicatorLabel(id))} ⚙</button><button data-remove-indicator="${esc(id)}" aria-label="Remove ${esc(id)}">×</button></span>`).join('');
  $('indicatorList').innerHTML = [...groups].map(([group, items]) => `<section class="indicator-group"><h3>${esc(group)} · ${items.length}</h3><div class="indicator-grid">${items.map(item => `
    <div class="study-library-row ${wantedAdvanced.has(item.id) ? 'active' : ''}">
      <button class="indicator-item ${wantedAdvanced.has(item.id) ? 'active' : ''}" data-advanced="${esc(item.id)}" aria-pressed="${wantedAdvanced.has(item.id)}"><span><strong>${esc(item.id)} · ${esc(item.name)}</strong><small>${item.overlay ? 'Price overlay' : item.pattern ? 'Candlestick signal' : 'Separate panel'}${wantedAdvanced.has(item.id) ? ' · Added' : ''}</small></span></button>
      <button class="study-library-settings" data-study-settings="${esc(item.id)}" aria-label="Configure ${esc(item.id)}" title="Configure inputs and appearance">⚙</button>
    </div>`).join('')}</div></section>`).join('') || '<div class="no-results">No matching indicators.</div>';
}

function renderStudyChips() {
  for (const button of $('indicatorTools').querySelectorAll('[data-indicator]')) {
    const id = button.dataset.indicator;
    button.textContent = id === 'volume' ? 'VOL' : indicatorLabel(id);
    button.classList.toggle('active', indicators.has(id));
    button.setAttribute('aria-pressed', String(indicators.has(id)));
    button.title = `${quickStudies[id].name} · use Study settings to edit`;
  }
  const ids = [...indicators, ...wantedAdvanced];
  $('studyChips').innerHTML = ids.length ? ids.map(id => {
    const loading = pendingStudies.has(id), error = studyErrors.get(id), config = studyConfig(id);
    const color = Object.values(config.colors)[0];
    return `<span class="study-chip ${error ? 'study-chip-error' : ''}"><button data-study-settings="${esc(id)}" title="${esc(error || 'Edit inputs, colors and transparency')}"><i style="background:${esc(color)}"></i>${esc(indicatorLabel(id))}${loading ? ' …' : ' ⚙'}</button><button data-remove-study="${esc(id)}" aria-label="Remove ${esc(indicatorLabel(id))}">×</button></span>`;
  }).join('') : '<span class="study-empty">Add indicators or choose Study settings to customize a study.</span>';
}

function removeAdvanced(id) {
  wantedAdvanced.delete(id); advanced.delete(id); studyErrors.delete(id); pendingStudies.delete(id);
  studyRequests.set(id, (studyRequests.get(id) || 0) + 1);
  saveStudySettings(); syncAdvancedPanels(); renderAdvancedList(); renderStudyChips(); render();
}

async function toggleAdvanced(id) {
  if (wantedAdvanced.has(id)) { removeAdvanced(id); return; }
  if (wantedAdvanced.size >= 8) { showError('Up to eight library indicators can be displayed at once. Remove a study to add another.'); return; }
  wantedAdvanced.add(id); saveStudySettings();
  const request = fetchAdvanced(id);
  renderStudyChips(); renderAdvancedList();
  try { await request; } catch (error) { showError(`${id}: ${error.message}`); }
  syncAdvancedPanels(); renderAdvancedList(); renderStudyChips(); render();
}

function parameterLabel(key, meta) {
  return meta.label || key.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, value => value.toUpperCase());
}

function populateStudySettings(id) {
  settingsStudy = id;
  const descriptor = studyDescriptor(id), config = studyConfig(id);
  if (!descriptor) return;
  $('studySettingsError').hidden = true;
  $('studySettingsDescription').textContent = descriptor.description || descriptor.name;
  const quickOptions = Object.entries(quickStudies).map(([key, item]) => `<option value="${key}">${esc(item.name)} (quick)</option>`).join('');
  const libraryOptions = advancedCatalog.map(item => `<option value="${esc(item.id)}">${esc(item.id)} · ${esc(item.name)}</option>`).join('');
  $('studySettingsSelect').innerHTML = `<optgroup label="Quick studies">${quickOptions}</optgroup>${libraryOptions ? `<optgroup label="Indicator library">${libraryOptions}</optgroup>` : ''}`;
  $('studySettingsSelect').value = id;
  $('studyParameterFields').innerHTML = Object.entries(config.params).map(([key, value]) => {
    const meta = studyParameterMeta(id, key), fieldId = `studyParam-${key}`;
    // Numeric min values are validation bounds, not an offset for decimal steps.
    const step = meta.step === 1 ? 1 : 'any';
    const field = meta.options ? `<select id="${esc(fieldId)}" data-study-param="${esc(key)}">${Object.entries(meta.options).map(([option, label]) => `<option value="${esc(option)}" ${String(value) === option ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select>` : `<input id="${esc(fieldId)}" data-study-param="${esc(key)}" type="number" value="${value}" ${meta.min != null ? `min="${meta.min}"` : ''} ${meta.max != null ? `max="${meta.max}"` : ''} step="${step}" required>`;
    return `<label for="${esc(fieldId)}">${esc(parameterLabel(key, meta))}${field}</label>`;
  }).join('');
  const sources = descriptor.sources || [];
  $('studySourceField').hidden = !sources.length;
  $('studySource').innerHTML = sources.map(source => `<option value="${esc(source)}">${esc({hl2: 'HL2 · (High + Low) / 2', hlc3: 'HLC3 · (High + Low + Close) / 3', ohlc4: 'OHLC4 · (Open + High + Low + Close) / 4'}[source] || source.charAt(0).toUpperCase() + source.slice(1))}</option>`).join('');
  $('studySource').value = config.source;
  $('studyColorFields').innerHTML = Object.entries(config.colors).map(([name, color], index) => `<label for="studyColor-${index}"><span>${esc(name.replace(/_/g, ' '))}</span><input type="color" id="studyColor-${index}" data-study-output="${esc(name)}" value="${color}"></label>`).join('');
  $('studyLineWidthField').hidden = id === 'volume';
  $('studyLineWidth').value = config.width;
  $('studyTransparency').value = config.transparency;
  $('studyTransparencyValue').textContent = `${config.transparency}%`;
  $('studyApply').textContent = (quickStudies[id] ? indicators.has(id) : wantedAdvanced.has(id)) ? 'Apply settings' : 'Add to chart';
}

async function openStudySettings(id) {
  const session = ++settingsSession;
  try {
    if (!quickStudies[id]) await ensureIndicatorCatalog();
    if (session !== settingsSession) return;
    populateStudySettings(id);
    $('studySettingsForm').inert = false;
    $('studyApply').disabled = false;
    if (!$('studySettingsDialog').open) $('studySettingsDialog').showModal();
  } catch (error) { showError(error.message); }
}

async function applyStudySettings(event) {
  event.preventDefault();
  const form = $('studySettingsForm');
  if (!form.reportValidity()) return;
  const id = settingsStudy, config = studyConfig(id), session = settingsSession;
  for (const field of form.querySelectorAll('[data-study-param]')) config.params[field.dataset.studyParam] = Number(field.value);
  if (!$('studySourceField').hidden) config.source = $('studySource').value;
  for (const field of form.querySelectorAll('[data-study-output]')) config.colors[field.dataset.studyOutput] = field.value;
  config.transparency = Number($('studyTransparency').value);
  config.width = Number($('studyLineWidth').value);
  const wasActive = wantedAdvanced.has(id);
  if (!quickStudies[id] && !wasActive && wantedAdvanced.size >= 8) {
    showStudySettingsError('Up to eight library indicators can be displayed at once. Remove a study to add another.'); return;
  }
  $('studyApply').disabled = true;
  form.inert = true;
  try {
    if (quickStudies[id]) indicators.add(id);
    else {
      wantedAdvanced.add(id);
      const request = fetchAdvanced(id, config);
      renderStudyChips();
      const data = await request;
      if (!data) {
        if (session === settingsSession) showStudySettingsError('The chart or study changed while loading. Apply again to use these settings on the current chart.');
        return;
      }
    }
    studyConfigurations.set(id, config);
    saveStudySettings(); syncAdvancedPanels(); renderAdvancedList(); renderStudyChips(); render();
    if (session === settingsSession) $('studySettingsDialog').close();
  } catch (error) {
    if (!wasActive) wantedAdvanced.delete(id);
    if (session === settingsSession) showStudySettingsError(error.message);
  } finally {
    if (session === settingsSession) { $('studyApply').disabled = false; form.inert = false; }
    renderStudyChips(); renderAdvancedList();
  }
}

function showStudySettingsError(message) {
  $('studySettingsError').textContent = message;
  $('studySettingsError').hidden = false;
}

function installStudyControls() {
  $('indicatorTools').insertAdjacentHTML('beforeend', '<button id="studySettingsButton" title="Customize indicator inputs and appearance">⚙ Study settings</button>');
  document.querySelector('.range-toolbar').insertAdjacentHTML('beforebegin', '<div id="studyChips" class="study-chips" aria-label="Active chart studies"></div>');
  document.body.insertAdjacentHTML('beforeend', `
    <dialog id="studySettingsDialog" class="study-settings-dialog" aria-labelledby="studySettingsTitle">
      <div class="dialog-head"><div><span class="eyebrow">TECHNICAL ANALYSIS</span><h2 id="studySettingsTitle">Study settings</h2><p id="studySettingsDescription" class="muted"></p></div><button id="closeStudySettings" type="button" class="icon-button" aria-label="Close study settings">×</button></div>
      <form id="studySettingsForm"><div class="study-settings-body">
        <label for="studySettingsSelect">Study<select id="studySettingsSelect"></select></label>
        <fieldset><legend>Inputs</legend><div id="studyParameterFields" class="study-parameter-fields"></div><label id="studySourceField" for="studySource">Price source<select id="studySource"></select></label></fieldset>
        <fieldset><legend>Appearance</legend><div id="studyColorFields" class="study-color-fields"></div><div class="study-appearance-fields"><label id="studyLineWidthField" for="studyLineWidth">Line width (px)<input id="studyLineWidth" type="number" min="0.5" max="6" step="0.1" required></label><label for="studyTransparency">Transparency <output id="studyTransparencyValue" for="studyTransparency">0%</output><input id="studyTransparency" type="range" min="0" max="100" step="1"><small>0% opaque · 100% invisible</small></label></div></fieldset>
        <p id="studySettingsError" class="error" role="alert" hidden></p>
      </div><div class="dialog-actions"><button id="cancelStudySettings" type="button" class="secondary">Cancel</button><button id="studyApply" type="submit" class="study-apply">Apply settings</button></div></form>
    </dialog>`);

  $('studySettingsButton').onclick = () => openStudySettings([...indicators, ...wantedAdvanced][0] || 'sma');
  $('closeStudySettings').onclick = $('cancelStudySettings').onclick = () => $('studySettingsDialog').close();
  $('studySettingsDialog').addEventListener('close', () => { settingsSession++; });
  $('studySettingsSelect').onchange = event => populateStudySettings(event.target.value);
  $('studyTransparency').oninput = event => { $('studyTransparencyValue').textContent = `${event.target.value}%`; };
  $('studySettingsForm').onsubmit = applyStudySettings;
  $('studyChips').onclick = event => {
    const edit = event.target.closest('[data-study-settings]'), remove = event.target.closest('[data-remove-study]');
    if (edit) openStudySettings(edit.dataset.studySettings);
    if (remove) {
      const id = remove.dataset.removeStudy;
      if (quickStudies[id]) { indicators.delete(id); saveStudySettings(); renderStudyChips(); render(); }
      else removeAdvanced(id);
    }
  };
  $('indicatorTools').onclick = event => {
    const button = event.target.closest('[data-indicator]');
    if (!button) return;
    const id = button.dataset.indicator;
    indicators.has(id) ? indicators.delete(id) : indicators.add(id);
    saveStudySettings(); renderStudyChips(); render();
  };
  $('indicatorLibrary').onclick = async () => {
    $('indicatorDialog').showModal();
    try { await ensureIndicatorCatalog(); }
    catch (error) { $('indicatorList').innerHTML = `<div class="error">${esc(error.message)}</div>`; }
  };
  $('closeIndicators').onclick = () => $('indicatorDialog').close();
  $('indicatorSearch').oninput = renderAdvancedList;
  const libraryClick = event => {
    const add = event.target.closest('[data-advanced]'), remove = event.target.closest('[data-remove-indicator]'), edit = event.target.closest('[data-study-settings]');
    if (edit) openStudySettings(edit.dataset.studySettings);
    else if (remove) removeAdvanced(remove.dataset.removeIndicator);
    else if (add) toggleAdvanced(add.dataset.advanced);
  };
  $('indicatorList').onclick = $('activeAdvanced').onclick = $('indicatorPanels').onclick = libraryClick;
  $('clearAdvanced').onclick = () => {
    for (const id of wantedAdvanced) studyRequests.set(id, (studyRequests.get(id) || 0) + 1);
    wantedAdvanced.clear(); advanced.clear(); pendingStudies.clear(); studyErrors.clear();
    saveStudySettings(); syncAdvancedPanels(); renderAdvancedList(); renderStudyChips(); render();
  };
}

restoreStudySettings();
installStudyControls();
renderStudyChips();
