(() => {
  'use strict';
  const el = id => document.getElementById(id), key = 'atlas.strategy.v1';
  const pane = el('workspaceStrategyPanel');
  const catalog = AtlasBacktest.catalog;
  const names = {...Object.fromEntries(Object.entries(catalog).map(([id, item]) => [id, item.label])), custom: 'Custom entry / exit'};
  const descriptions = {...Object.fromEntries(Object.entries(catalog).map(([id, item]) => [id, item.description])), custom: 'Build separate entry and exit conditions. Indicators use the selected raw or adjusted OHLC. Periods count bars in the selected chart interval.'};
  const strategyOptions = ['Trend', 'Reversion', 'Momentum', 'Breakout'].map(family => `<optgroup label="${family}">${Object.entries(catalog).filter(([, item]) => item.family === family).map(([id, item]) => `<option value="${id}">${item.label}</option>`).join('')}</optgroup>`).join('') + '<option value="custom">Custom entry / exit</option>';
  const input = (id, label, value, min, max, step = 'any', group = '') => `<label ${group ? `data-strategy-group="${group}"` : ''}>${label}<input id="st-${id}" type="number" value="${value}" min="${min}" max="${max}" step="${step}" required></label>`;
  pane.innerHTML = `<div class="strategy-header"><div><span class="eyebrow">RESEARCH LAB</span><h2>Strategy tester <span class="strategy-badge">Long only</span></h2><p id="strategyContext">Select a symbol and load price history.</p></div><button id="strategyExport" class="secondary" disabled>Export trades CSV</button></div>
    <form id="strategyForm"><div class="strategy-config"><label>Strategy · ${Object.keys(catalog).length} presets<select id="st-kind">${strategyOptions}</select></label>
    ${input('fast', 'Fast period', 10, 2, 500, 1, 'ma')}${input('slow', 'Slow period', 30, 2, 500, 1, 'ma')}${input('lookback', 'Lookback', 14, 2, 500, 1, 'lookback')}${input('lower', 'RSI lower threshold', 30, 0, 99, 1, 'rsi')}${input('upper', 'RSI upper threshold', 70, 1, 100, 1, 'rsi')}
    ${input('capital', 'Initial capital', 10000, 1, 1e12)}${input('allocation', 'Position size %', 100, 0.01, 100)}${input('commission', 'Commission % / side', 0.1, 0, 10, 0.01)}${input('slippage', 'Slippage % / side', 0.05, 0, 10, 0.01)}${input('stop', 'Stop loss % · 0 = off', 0, 0, 99.99)}${input('target', 'Take profit % · 0 = off', 0, 0, 1000)}
    <label>Price basis<select id="st-adjusted"><option value="true">Adjusted OHLC</option><option value="false">Raw OHLC</option></select></label><button class="strategy-run" id="strategyRun">Run backtest</button></div><p id="strategyDescription"></p></form>
    <p id="strategyStatus" role="status" aria-live="polite">Set your parameters, then run a test on the loaded chart range.</p><p id="strategyError" role="alert" hidden></p>
    <div id="strategyResults" hidden><div id="strategyMetrics" class="strategy-metrics"></div><div class="strategy-result-tabs" role="tablist" aria-label="Backtest results"><button role="tab" id="strategyOverviewTab" aria-controls="strategyOverview" aria-selected="true">Overview</button><button role="tab" id="strategyTradesTab" aria-controls="strategyTrades" aria-selected="false" tabindex="-1">List of trades</button></div>
    <div id="strategyOverview" role="tabpanel" aria-labelledby="strategyOverviewTab"><div class="strategy-legend"><span>● Strategy equity</span><span>● Buy &amp; hold</span><span>Drawdown below · hover to inspect</span></div><canvas id="strategyCurve" aria-label="Strategy equity, buy and hold equity, and drawdown over the test period" role="img"></canvas><p id="strategyCurveValue" aria-live="off"></p></div>
    <div id="strategyTrades" role="tabpanel" aria-labelledby="strategyTradesTab" hidden><div class="strategy-table-scroll"><table><thead><tr><th>#</th><th>Entry</th><th>Exit</th><th>Entry price</th><th>Exit price</th><th>Quantity</th><th>Net P&amp;L</th><th>Return</th><th>Fees</th><th>Bars</th><th>Exit reason</th></tr></thead><tbody id="strategyTradeRows"></tbody></table></div><div class="strategy-pagination"><span id="strategyTradePage"></span><button id="strategyPrev" class="secondary">Previous</button><button id="strategyNext" class="secondary">Next</button></div></div></div>
    <details class="strategy-method"><summary>Execution model &amp; assumptions</summary><p>Signals use completed closes and execute at the next bar’s open. One long position at a time, fractional units, no leverage or shorting. Position size is a percentage of available cash including entry commission. After a stop or target, a still-active entry condition can enter again on the next bar.</p><p>Stops and targets use the executed entry price. Opening gaps fill at the open; when both levels touch within one bar, the stop fills first. Slippage is adverse on every fill, and commission applies on both sides. Any remaining position closes on the final bar. Drawdown uses bar-close equity, not intrabar lows.</p><p>Warmup bars cannot trade. Buy &amp; hold invests all initial capital at the first eligible open after warmup and pays the same costs. Adjusted OHLC scales each bar by adjusted close / close (provider split/dividend adjustments); raw OHLC does not account for corporate actions. No separate dividends, cash interest, taxes or market impact. Missing OHLC or adjusted values reject the test. Stored history may have missing sessions. Results describe this symbol and range, not out-of-sample performance.</p></details>`;
  pane.querySelector('.strategy-result-tabs').insertAdjacentHTML('beforeend', '<button role="tab" id="strategyPerformanceTab" aria-controls="strategyPerformance" aria-selected="false" tabindex="-1">Performance</button>');
  el('strategyResults').insertAdjacentHTML('beforeend', '<div id="strategyPerformance" class="strategy-performance" role="tabpanel" aria-labelledby="strategyPerformanceTab" hidden></div>');
  el('strategyDescription').insertAdjacentHTML('beforebegin', `<div class="strategy-builder-actions"><button id="strategyCustomize" type="button" class="secondary">Customize this preset</button></div><div id="strategyRuleBuilder" class="strategy-rule-builder" hidden><p class="strategy-rule-help">Signals are evaluated at the close and filled at the next open. Crossing rules require the previous bar to be on the other side or equal. All indicators must finish warming up before trading begins. Stops and targets remain active alongside your exit rules.</p>${['entry', 'exit'].map(side => `<section class="strategy-condition-group" aria-labelledby="st-${side}-title"><div class="strategy-condition-heading"><h3 id="st-${side}-title">${side === 'entry' ? 'Entry · buy when' : 'Exit · sell when'}</h3><label>Match<select id="st-${side}-mode"><option value="all">All conditions (AND)</option><option value="any">Any condition (OR)</option></select></label><button type="button" class="secondary" data-add-condition="${side}">+ Add condition</button></div><div id="st-${side}-rules"></div></section>`).join('')}</div><div class="strategy-saved"><label>Setup name<input id="strategySetupName" maxlength="80" placeholder="My trend strategy"></label><button id="strategySaveSetup" type="button" class="secondary">Save setup</button><label>Saved setups<select id="strategySavedSetup" aria-label="Saved strategy setups"></select></label><button id="strategyDeleteSetup" type="button" class="secondary" disabled>Delete setup</button></div><p id="strategySetupNotice" role="status"></p>`);
  let customConditions = AtlasBacktest.rulesFor(AtlasBacktest.defaults), setups = Object.create(null);
  el('strategyDescription').insertAdjacentHTML('afterend', '<div id="strategyRulePreview" class="strategy-rule-preview" aria-label="Strategy entry and exit summary"></div>');
  const setupsKey = 'atlas.strategy.setups.v1';
  try {
    const stored = JSON.parse(localStorage.getItem(setupsKey) || '{}');
    for (const [name, settings] of Object.entries(stored)) if (name.length <= 80 && settings && typeof settings === 'object') setups[name] = settings;
  } catch {}
  let result = null, resultSymbol = '', resultCurrency = '', resultInterval = '', tradePage = 0;
  try {
    const prefs = JSON.parse(localStorage.getItem(key) || '{}');
    applySettings(prefs);
  } catch {}
  function applySettings(settings) {
    for (const [name, fallback] of Object.entries(AtlasBacktest.defaults)) el(`st-${name}`).value = String(settings[name] ?? fallback);
    if (!Object.hasOwn(names, el('st-kind').value)) el('st-kind').value = 'sma';
    const candidate = settings.conditions;
    if (candidate && ['entry', 'exit'].every(side => ['all', 'any'].includes(candidate[side]?.mode) && Array.isArray(candidate[side]?.rules) && candidate[side].rules.length <= 12 && candidate[side].rules.every(r => r && Object.hasOwn(AtlasBacktest.operators, r.operator) && [r.left, r.right].every(o => o && Object.hasOwn(AtlasBacktest.operands, o.type))))) customConditions = structuredClone(candidate);
  }
  function params() {
    const settings = Object.fromEntries(Object.keys(AtlasBacktest.defaults).map(name => [name, name === 'kind' ? el('st-kind').value : name === 'adjusted' ? el('st-adjusted').value === 'true' : el(`st-${name}`).value === '' ? NaN : Number(el(`st-${name}`).value)]));
    if (settings.kind === 'custom') settings.conditions = structuredClone(customConditions);
    return settings;
  }
  function configure() {
    const kind = el('st-kind').value;
    pane.querySelectorAll('[data-strategy-group]').forEach(label => {
      const group = label.dataset.strategyGroup;
      label.hidden = !(catalog[kind]?.controls || []).includes(group);
      label.querySelector('input').disabled = label.hidden;
    });
    el('strategyDescription').textContent = descriptions[kind] || '';
    el('strategyRuleBuilder').hidden = kind !== 'custom';
    el('strategyCustomize').hidden = kind === 'custom';
    el('strategyRuleBuilder').querySelectorAll('input,select,button').forEach(control => { control.disabled = kind !== 'custom' || control.dataset.atLimit === 'true'; });
    const describe = o => o.type === 'constant' ? String(o.value ?? '…') : `${AtlasBacktest.operands[o.type]}${AtlasBacktest.periodTypes.includes(o.type) ? ` (${o.period ?? '…'})` : ''}`;
    const rules = AtlasBacktest.rulesFor(params());
    el('strategyRulePreview').innerHTML = ['entry', 'exit'].map(side => `<div><strong>${side === 'entry' ? 'Enter' : 'Exit'} · ${rules[side].mode === 'all' ? 'ALL' : 'ANY'}</strong><span>${rules[side].rules.map(r => esc(`${describe(r.left)} ${AtlasBacktest.operators[r.operator].toLowerCase()} ${describe(r.right)}`)).join(`<b> ${rules[side].mode === 'all' ? 'AND' : 'OR'} </b>`) || 'Add a condition'}</span></div>`).join('');
  }
  function renderRules() {
    const operandControl = (o, side, index, part) => {
      const prefix = `${side} condition ${index + 1} ${part}`;
      const options = Object.entries(AtlasBacktest.operands).map(([type, label]) => `<option value="${type}" ${o.type === type ? 'selected' : ''}>${esc(label)}</option>`).join('');
      const field = o.type === 'constant' ? 'value' : 'period';
      return `<label>${part === 'left' ? 'Indicator / price' : 'Compare with'}<select data-part="${part}" data-property="type" aria-label="${prefix} indicator">${options}</select></label>${o.type === 'constant' || AtlasBacktest.periodTypes.includes(o.type) ? `<label>${field === 'value' ? 'Value' : 'Period'}<input type="number" data-part="${part}" data-property="${field}" aria-label="${prefix} ${field}" value="${esc(o[field] ?? '')}" ${field === 'period' ? 'min="2" max="500" step="1"' : 'step="any"'} required></label>` : '<span class="strategy-rule-spacer"></span>'}`;
    };
    for (const side of ['entry', 'exit']) {
      el(`st-${side}-mode`).value = customConditions[side].mode;
      el(`st-${side}-rules`).innerHTML = customConditions[side].rules.map((r, index) => `<div class="strategy-condition" data-side="${side}" data-index="${index}">${operandControl(r.left, side, index, 'left')}<label>Condition<select data-property="operator" aria-label="${side} condition ${index + 1} comparison">${Object.entries(AtlasBacktest.operators).map(([op, label]) => `<option value="${op}" ${r.operator === op ? 'selected' : ''}>${label}</option>`).join('')}</select></label>${operandControl(r.right, side, index, 'right')}<button type="button" class="secondary" data-remove-condition aria-label="Remove ${side} condition ${index + 1}">×</button></div>`).join('') || '<p class="strategy-rule-help">Add at least one condition before running this strategy.</p>';
      pane.querySelector(`[data-add-condition="${side}"]`).dataset.atLimit = String(customConditions[side].rules.length >= 12);
    }
    configure();
  }
  function rulesChanged() { configure(); invalidate('Conditions changed. Run the backtest to update results.'); }
  el('strategyRuleBuilder').oninput = event => {
    const control = event.target, row = control.closest('[data-side]');
    if (!row || !control.dataset.property) return;
    const r = customConditions[row.dataset.side].rules[Number(row.dataset.index)], property = control.dataset.property;
    if (property === 'operator') r.operator = control.value;
    else {
      const o = r[control.dataset.part];
      if (property === 'type') { o.type = control.value; o.period = Number.isInteger(o.period) ? o.period : 14; o.value = Number.isFinite(o.value) ? o.value : 0; }
      else o[property] = control.value === '' ? null : Number(control.value);
    }
    if (property === 'type') {
      const {side, index} = row.dataset, part = control.dataset.part;
      renderRules(); el(`st-${side}-rules`).querySelector(`[data-index="${index}"] [data-part="${part}"][data-property="type"]`).focus();
    }
    rulesChanged();
  };
  for (const side of ['entry', 'exit']) {
    el(`st-${side}-mode`).onchange = event => { customConditions[side].mode = event.target.value; rulesChanged(); };
    pane.querySelector(`[data-add-condition="${side}"]`).onclick = () => {
      if (customConditions[side].rules.length >= 12) return;
      customConditions[side].rules.push({left: {type: 'close'}, operator: side === 'entry' ? 'gt' : 'lt', right: {type: 'sma', period: 20}});
      renderRules(); rulesChanged(); el(`st-${side}-rules`).lastElementChild.querySelector('select').focus();
    };
  }
  el('strategyRuleBuilder').onclick = event => {
    const button = event.target.closest('[data-remove-condition]'); if (!button) return;
    const row = button.closest('[data-side]'), side = row.dataset.side;
    customConditions[side].rules.splice(Number(row.dataset.index), 1); renderRules(); rulesChanged();
    pane.querySelector(`[data-add-condition="${side}"]`).focus();
  };
  el('strategyCustomize').onclick = () => {
    customConditions = AtlasBacktest.rulesFor(params()); el('st-kind').value = 'custom'; renderRules(); rulesChanged();
  };
  function renderSetups(selectedName = '') {
    el('strategySavedSetup').innerHTML = '<option value="">Choose a saved setup</option>' + Object.keys(setups).sort().map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join('');
    el('strategySavedSetup').value = selectedName; el('strategyDeleteSetup').disabled = !selectedName;
  }
  function storeSetups(message) {
    try { localStorage.setItem(setupsKey, JSON.stringify(setups)); el('strategySetupNotice').textContent = message; }
    catch { el('strategySetupNotice').textContent = 'Browser storage is unavailable; setups are kept for this session only.'; }
  }
  el('strategySaveSetup').onclick = () => {
    const name = el('strategySetupName').value.trim();
    if (!name) { el('strategySetupNotice').textContent = 'Enter a name for this setup.'; el('strategySetupName').focus(); return; }
    if (!el('strategyForm').reportValidity()) return;
    setups[name] = params(); renderSetups(name); storeSetups(`Saved “${name}” in this browser.`);
  };
  el('strategySavedSetup').onchange = event => {
    const name = event.target.value; el('strategyDeleteSetup').disabled = !name;
    if (!Object.hasOwn(setups, name)) return;
    applySettings(setups[name]); el('strategySetupName').value = name; renderRules();
    invalidate('Setup loaded. Run the backtest for the current symbol and range.');
  };
  el('strategyDeleteSetup').onclick = () => { const name = el('strategySavedSetup').value; if (!name) return; delete setups[name]; renderSetups(); storeSetups(`Deleted “${name}”.`); };
  function invalidate(message) {
    result = null; el('strategyResults').hidden = true; el('strategyExport').disabled = true;
    el('strategyError').hidden = true; el('strategyStatus').textContent = message;
  }
  function context() {
    el('strategyContext').textContent = `${selected?.symbol || 'No symbol'} · ${interval} · ${rows.length.toLocaleString()} loaded bars${rows.length ? ` · ${rows[0].date.slice(0,10)} – ${rows.at(-1).date.slice(0,10)}` : ''} · ${selected?.currency || 'quote currency'}`;
  }
  el('strategyForm').addEventListener('input', event => { if (event.target.closest('.strategy-saved')) return; configure(); invalidate('Parameters changed. Run the backtest to update results.'); });
  el('strategyForm').onsubmit = event => {
    event.preventDefault(); invalidate('Running backtest…');
    try {
      if (!selected || !rows.length) throw Error('Load a symbol with price history first.');
      if (el('rangeSummary').textContent.startsWith('Loading')) throw Error('Wait for the chart data to finish loading.');
      const settings = params();
      result = AtlasBacktest.run(rows, settings); resultSymbol = selected.symbol; resultCurrency = selected.currency || ''; resultInterval = interval; tradePage = 0;
      try { localStorage.setItem(key, JSON.stringify(settings)); } catch {}
      const metric = (name, value, cls = '') => `<div><span>${name}</span><strong class="${cls}">${value}</strong></div>`;
      el('strategyMetrics').innerHTML = metric('Net profit', `${fmt(result.netProfit)} ${esc(resultCurrency)}`, result.netProfit >= 0 ? 'positive' : 'negative') + metric('Total return', pct(result.returnPct), result.returnPct >= 0 ? 'positive' : 'negative') + metric('Buy & hold', pct(result.benchmarkPct)) + metric('Max drawdown', `${fmt(result.maxDrawdown)}%`, 'negative') + metric('Closed trades', String(result.trades.length)) + metric('Win rate', result.winRate == null ? '—' : `${fmt(result.winRate)}%`) + metric('Profit factor', result.profitFactor === Infinity ? '∞' : fmt(result.profitFactor)) + metric('Fees paid', fmt(result.fees)) + metric('Exposure', `${fmt(result.exposurePct)}%`);
      el('strategyResults').hidden = false; el('strategyExport').disabled = !result.trades.length;
      el('strategyStatus').textContent = `${names[settings.kind]} · ${result.start} to ${result.end} · ${result.warmup} warmup bars · ${settings.adjusted ? 'Adjusted' : 'Raw'} prices. ${result.trades.length ? 'Test complete.' : 'No trades matched these conditions.'}`;
      renderTrades(); renderPerformance(); drawCurve();
    } catch (error) { invalidate('Test not run. Check the settings or loaded range.'); el('strategyError').hidden = false; el('strategyError').textContent = error.message; }
  };
  function renderTrades() {
    const trades = result?.trades || [], start = tradePage * 50;
    el('strategyTradeRows').innerHTML = trades.slice(start, start + 50).map((t, i) => `<tr><td>${start + i + 1}</td><td>${esc(t.entry)}</td><td>${esc(t.exit)}</td><td>${fmt(t.entryPrice, 4)}</td><td>${fmt(t.exitPrice, 4)}</td><td>${fmt(t.quantity, 4)}</td><td class="${t.pnl >= 0 ? 'positive' : 'negative'}">${fmt(t.pnl)}</td><td>${pct(t.returnPct)}</td><td>${fmt(t.fees)}</td><td>${t.bars}</td><td>${t.reason}</td></tr>`).join('') || '<tr><td colspan="11">No trades. Try a longer range or different parameters.</td></tr>';
    el('strategyTradePage').textContent = trades.length ? `${start + 1}–${Math.min(start + 50, trades.length)} of ${trades.length} trades` : '0 trades';
    el('strategyPrev').disabled = tradePage === 0; el('strategyNext').disabled = start + 50 >= trades.length;
  }
  el('strategyPrev').onclick = () => { tradePage--; renderTrades(); };
  el('strategyNext').onclick = () => { tradePage++; renderTrades(); };
  function renderPerformance() {
    const stats = result.performance;
    const value = (v, digits = 2) => v == null ? '—' : fmt(v, digits);
    const metrics = [
      ['Winning / losing / breakeven trades', `${stats.winners} / ${stats.losers} / ${stats.breakeven}`],
      ['Profit from winning trades', value(stats.grossProfit)],
      ['Loss from losing trades', value(-stats.grossLoss)],
      ['Average winning trade', value(stats.averageWin)],
      ['Average losing trade', value(stats.averageLoss)],
      ['Average net trade', value(stats.averageTrade)],
      ['Best trade', value(stats.bestTrade)],
      ['Worst trade', value(stats.worstTrade)],
      ['Average bars per trade', value(stats.averageBars, 1)]
    ];
    el('strategyPerformance').innerHTML = `<h3>Trade analysis · ${esc(resultCurrency || 'quote currency')}</h3><div class="strategy-table-scroll"><table><tbody>${metrics.map(([label, v]) => `<tr><td>${label}</td><td>${v}</td></tr>`).join('')}</tbody></table></div><h3>Monthly returns</h3><p>Returns use month-end equity, including open positions and trading costs. First and last months may be partial; months without stored bars are omitted. Trade amounts above are net of costs.</p><div class="strategy-table-scroll"><table><thead><tr><th>Month</th><th>Strategy</th><th>Buy &amp; hold</th><th>Ending equity</th></tr></thead><tbody>${result.monthly.map(row => `<tr><td>${esc(row.month)}</td><td class="${row.returnPct >= 0 ? 'positive' : 'negative'}">${pct(row.returnPct)}</td><td>${pct(row.benchmarkPct)}</td><td>${fmt(row.equity)}</td></tr>`).join('')}</tbody></table></div>`;
  }
  const resultTabs = ['Overview', 'Trades', 'Performance'];
  function resultTab(name) {
    for (const n of resultTabs) {
      el(`strategy${n}`).hidden = n !== name;
      el(`strategy${n}Tab`).setAttribute('aria-selected', String(n === name));
      el(`strategy${n}Tab`).tabIndex = n === name ? 0 : -1;
    }
    drawCurve();
  }
  for (const name of resultTabs) el(`strategy${name}Tab`).onclick = () => resultTab(name);
  pane.querySelector('.strategy-result-tabs').onkeydown = event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const current = resultTabs.findIndex(name => el(`strategy${name}Tab`) === document.activeElement);
    const name = event.key === 'Home' ? resultTabs[0] : event.key === 'End' ? resultTabs.at(-1) : resultTabs[(current + (event.key === 'ArrowRight' ? 1 : resultTabs.length - 1)) % resultTabs.length];
    resultTab(name); el(`strategy${name}Tab`).focus();
  };
  function drawCurve(index = -1) {
    if (!result || el('strategyOverview').hidden) return;
    const canvas = el('strategyCurve'), box = canvas.getBoundingClientRect(); if (!box.width) return;
    const w = box.width, h = 240, dpr = devicePixelRatio || 1, c = canvas.getContext('2d'), palette = window.atlasTheme.palette;
    canvas.width = w * dpr; canvas.height = h * dpr; c.scale(dpr, dpr); c.fillStyle = palette.surface; c.fillRect(0, 0, w, h);
    const curve = result.curve, left = 12, right = 86, width = Math.max(1, w - left - right);
    let low = result.parameters.capital, high = low;
    curve.forEach(r => { low = Math.min(low, r.equity, r.benchmark); high = Math.max(high, r.equity, r.benchmark); });
    const span = high - low || high * 0.01, x = i => left + i / Math.max(1, curve.length - 1) * width, y = v => 18 + (high - v) / span * 135;
    c.font = '10px Segoe UI'; c.lineWidth = 1;
    for (let j = 0; j < 4; j++) { const value = high - span * j / 3; c.strokeStyle = palette.grid; c.beginPath(); c.moveTo(left, y(value)); c.lineTo(left + width, y(value)); c.stroke(); c.fillStyle = palette.muted; c.fillText(fmt(value, 0), left + width + 8, y(value) + 4); }
    for (const [field, color] of [['benchmark', '#a78bfa'], ['equity', palette.accent]]) { c.beginPath(); curve.forEach((r, i) => i ? c.lineTo(x(i), y(r[field])) : c.moveTo(x(i), y(r[field]))); c.strokeStyle = color; c.lineWidth = 1.8; c.stroke(); }
    c.beginPath(); c.moveTo(left, 174); curve.forEach((r, i) => c.lineTo(x(i), 174 + (-r.drawdown / (result.maxDrawdown || 1)) * 37)); c.lineTo(left + width, 174); c.closePath(); c.fillStyle = '#f2364535'; c.fill();
    c.fillStyle = palette.muted; c.fillText(`−${fmt(result.maxDrawdown)}%`, left + width + 8, 205); c.fillText(curve[0].date.slice(0,10), left, 232); c.textAlign = 'right'; c.fillText(curve.at(-1).date.slice(0,10), left + width, 232); c.textAlign = 'left';
    if (index >= 0) { const r = curve[index]; c.strokeStyle = palette.muted; c.setLineDash([3,3]); c.beginPath(); c.moveTo(x(index), 10); c.lineTo(x(index), 214); c.stroke(); el('strategyCurveValue').textContent = `${r.date} · Equity ${fmt(r.equity)} · Buy & hold ${fmt(r.benchmark)} ${resultCurrency} · Drawdown ${fmt(r.drawdown)}%`; }
    else el('strategyCurveValue').textContent = `Final equity ${fmt(result.finalEquity)} ${resultCurrency} · Initial capital ${fmt(result.parameters.capital)} ${resultCurrency}`;
  }
  el('strategyCurve').onpointermove = event => {
    if (!result) return; const box = event.currentTarget.getBoundingClientRect();
    drawCurve(Math.max(0, Math.min(result.curve.length - 1, Math.round((event.clientX - box.left - 12) / Math.max(1, box.width - 98) * (result.curve.length - 1)))));
  };
  el('strategyCurve').onpointerleave = () => drawCurve();
  new ResizeObserver(() => drawCurve()).observe(el('strategyCurve'));
  window.addEventListener('atlas-theme-change', () => drawCurve());
  const load = loadHistory;
  loadHistory = async function() { invalidate('Chart data changed. Run a new backtest for this symbol and range.'); await load(); context(); };
  el('strategyExport').onclick = () => {
    if (!result?.trades.length) return;
    const fields = ['entry', 'exit', 'entryPrice', 'exitPrice', 'quantity', 'pnl', 'returnPct', 'fees', 'bars', 'reason'];
    const settingKeys = Object.keys(result.parameters);
    const cell = v => { if (v && typeof v === 'object') v = JSON.stringify(v); return `"${String(typeof v === 'string' && /^[=+\-@\t\r]/.test(v) ? "'" + v : v).replace(/"/g, '""')}"`; };
    const csv = [['symbol', 'currency', 'interval', ...settingKeys, ...fields], ...result.trades.map(t => [resultSymbol, resultCurrency, resultInterval, ...settingKeys.map(k => result.parameters[k]), ...fields.map(k => t[k])])].map(row => row.map(cell).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\uFEFF' + csv], {type: 'text/csv;charset=utf-8'})), a = document.createElement('a');
    a.href = url; a.download = `${resultSymbol}-strategy-trades.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  renderRules(); renderSetups(); configure(); context();
})();
