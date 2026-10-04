// Local stock research: composable screens, saved views, and chart navigation.
(() => {
  const byId = id => document.getElementById(id);
  const html = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
  const storageKey = 'atlas.screener.v1';
  const missingOption = '__missing__';
  const fields = {
    symbol: {label: 'Symbol', type: 'text'}, name: {label: 'Company', type: 'text'},
    market_cap: {label: 'Market cap', type: 'cap', scale: 1e9, unit: 'billions', monetary: true},
    close: {label: 'Price', type: 'price', monetary: true},
    change_1d_pct: {label: '1D change', type: 'percent'},
    return_1w_pct: {label: '1W return', type: 'percent'},
    return_1m_pct: {label: '1M return', type: 'percent'},
    return_3m_pct: {label: '3M return', type: 'percent'},
    return_6m_pct: {label: '6M return', type: 'percent'},
    return_ytd_pct: {label: 'YTD return', type: 'percent'},
    return_1y_pct: {label: '1Y return', type: 'percent'},
    volume: {label: 'Volume', type: 'quantity', scale: 1e6, unit: 'millions'},
    avg_volume20: {label: 'Avg. volume 20D', type: 'quantity', scale: 1e6, unit: 'millions'},
    relative_volume: {label: 'Relative volume', type: 'multiple'},
    avg_turnover20: {label: 'Avg. turnover 20D', type: 'cap', scale: 1e6, unit: 'millions', monetary: true},
    sma20_distance_pct: {label: 'vs. SMA 20', type: 'percent'},
    sma50_distance_pct: {label: 'vs. SMA 50', type: 'percent'},
    sma200_distance_pct: {label: 'vs. SMA 200', type: 'percent'},
    ema20_distance_pct: {label: 'vs. EMA 20', type: 'percent'},
    ema50_distance_pct: {label: 'vs. EMA 50', type: 'percent'},
    macd_pct: {label: 'MACD / price', type: 'percent'},
    macd_signal_pct: {label: 'MACD signal / price', type: 'percent'},
    macd_histogram_pct: {label: 'MACD histogram / price', type: 'percent'},
    adx14: {label: 'ADX 14', type: 'number'},
    stochastic_k: {label: 'Stochastic %K', type: 'number'},
    stochastic_d: {label: 'Stochastic %D', type: 'number'},
    bollinger_position_pct: {label: 'Bollinger position', type: 'percent'},
    bollinger_width_pct: {label: 'Bollinger width', type: 'percent'},
    rsi14: {label: 'RSI 14', type: 'number'}, atr14_pct: {label: 'ATR 14', type: 'percent'},
    volatility20_pct: {label: 'Volatility 20D', type: 'percent'},
    low_52w: {label: '52W low', type: 'price', monetary: true},
    high_52w: {label: '52W high', type: 'price', monetary: true},
    distance_52w_high_pct: {label: 'vs. 52W high', type: 'percent'},
    distance_52w_low_pct: {label: 'vs. 52W low', type: 'percent'},
    range_52w_position_pct: {label: '52W range position', type: 'percent'},
    pe: {label: 'P/E', type: 'number'}, forward_pe: {label: 'Forward P/E', type: 'number'},
    pb: {label: 'Price / book', type: 'number'}, beta: {label: 'Beta', type: 'number'},
    dividend_yield_pct: {label: 'Dividend yield', type: 'percent'},
    revenue_growth_pct: {label: 'Revenue growth', type: 'percent'},
    profit_margin_pct: {label: 'Profit margin', type: 'percent'},
    sector: {label: 'Sector', type: 'text'}, industry: {label: 'Industry', type: 'text'},
    country: {label: 'Country', type: 'text'}, exchange: {label: 'Exchange', type: 'text'},
    currency: {label: 'Currency', type: 'text'}, date: {label: 'Price date', type: 'date'},
    metadata_date: {label: 'Metadata date', type: 'date'},
    metadata_profile_at: {label: 'Profile date', type: 'date'}
  };
  const columnSets = {
    overview: ['symbol', 'name', 'market_cap', 'close', 'change_1d_pct', 'volume', 'relative_volume', 'sector', 'country', 'exchange', 'currency', 'date'],
    performance: ['symbol', 'name', 'close', 'change_1d_pct', 'return_1w_pct', 'return_1m_pct', 'return_3m_pct', 'return_6m_pct', 'return_ytd_pct', 'return_1y_pct', 'distance_52w_high_pct', 'currency', 'date'],
    technicals: ['symbol', 'close', 'sma20_distance_pct', 'sma50_distance_pct', 'sma200_distance_pct', 'rsi14', 'atr14_pct', 'relative_volume', 'avg_volume20', 'volatility20_pct', 'low_52w', 'high_52w', 'currency', 'date'],
    fundamentals: ['symbol', 'name', 'market_cap', 'sector', 'industry', 'country', 'pe', 'forward_pe', 'pb', 'beta', 'dividend_yield_pct', 'revenue_growth_pct', 'profit_margin_pct', 'currency', 'metadata_date', 'metadata_profile_at'],
    trend: ['symbol', 'close', 'ema20_distance_pct', 'ema50_distance_pct', 'macd_pct', 'macd_signal_pct', 'macd_histogram_pct', 'adx14', 'stochastic_k', 'stochastic_d', 'bollinger_position_pct', 'bollinger_width_pct', 'currency', 'date'],
    liquidity: ['symbol', 'name', 'close', 'volume', 'avg_volume20', 'relative_volume', 'avg_turnover20', 'atr14_pct', 'volatility20_pct', 'beta', 'currency', 'date']
  };
  const fieldGroups = {
    'Company': ['symbol', 'name', 'sector', 'industry', 'country', 'exchange', 'currency', 'date', 'metadata_date', 'metadata_profile_at'],
    'Valuation & fundamentals': ['market_cap', 'pe', 'forward_pe', 'pb', 'beta', 'dividend_yield_pct', 'revenue_growth_pct', 'profit_margin_pct'],
    'Price & performance': ['close', 'change_1d_pct', 'return_1w_pct', 'return_1m_pct', 'return_3m_pct', 'return_6m_pct', 'return_ytd_pct', 'return_1y_pct', 'low_52w', 'high_52w', 'distance_52w_high_pct', 'distance_52w_low_pct', 'range_52w_position_pct'],
    'Volume & risk': ['volume', 'avg_volume20', 'relative_volume', 'avg_turnover20', 'atr14_pct', 'volatility20_pct'],
    'Trend & momentum': ['sma20_distance_pct', 'sma50_distance_pct', 'sma200_distance_pct', 'ema20_distance_pct', 'ema50_distance_pct', 'rsi14', 'macd_pct', 'macd_signal_pct', 'macd_histogram_pct', 'adx14', 'stochastic_k', 'stochastic_d', 'bollinger_position_pct', 'bollinger_width_pct']
  };
  const help = {
    return_ytd_pct: 'Return since the final available close of the prior calendar year, based on this stock’s latest price date.',
    avg_turnover20: 'Average daily close × volume over 20 bars, in the reported quote currency. Enter millions.',
    relative_volume: 'Latest daily volume divided by the previous 20 daily volumes, excluding the latest bar.',
    adx14: '14-bar trend strength; ADX does not indicate whether the trend is up or down.',
    stochastic_k: 'Slow stochastic (14, 3, 3), with simple moving averages.',
    stochastic_d: 'Three-bar simple average of slow stochastic %K.',
    bollinger_position_pct: 'Position within 20-bar Bollinger Bands (2 standard deviations): 0% at lower band, 100% at upper band. Can exceed this range.',
    bollinger_width_pct: '20-bar Bollinger Band width divided by its middle band, in percent.',
    range_52w_position_pct: 'Position within the last 252 daily bars: 0% at the low, 100% at the high.',
    macd_pct: '12/26 EMA difference divided by latest adjusted close, in percent.',
    macd_signal_pct: '9-bar MACD signal divided by latest adjusted close, in percent.',
    macd_histogram_pct: 'MACD minus its signal, divided by latest adjusted close, in percent.',
    market_cap: 'Reported market capitalization, in the selected currency. Enter billions.',
    metadata_date: 'Latest metadata update, including bulk provider quotes.',
    metadata_profile_at: 'Last successful detailed company profile fetch. Individual provider fields may be older or unavailable.'
  };
  const operators = {gte: 'At least ≥', lte: 'At most ≤', gt: 'Greater than >', lt: 'Less than <', between: 'Between', eq: 'Equals', missing: 'Is missing', present: 'Is available'};
  const tierBounds = {mega: [200e9, Infinity], large: [10e9, 200e9], mid: [2e9, 10e9], small: [300e6, 2e9], micro: [0, 300e6]};
  const defaults = () => ({search: '', country: '', exchange: '', currency: '', sector: '', industry: '', capTier: '', priceSince: '', onlyHistory: false, onlyFavorites: false, matchMode: 'all', columns: 'overview', customColumns: [...columnSets.overview], pageSize: 50, sortKey: 'market_cap', sortDirection: 'desc', secondarySort: '', secondaryDirection: 'desc', heat: false, compact: false, expanded: false, rules: []});
  const presets = {
    gainers: {label: 'Top gainers', description: 'Positive latest daily change, ranked largest first.', changes: {sortKey: 'change_1d_pct', rules: [{field: 'change_1d_pct', operator: 'gt', value: 0}]}},
    losers: {label: 'Top losers', description: 'Negative latest daily change, ranked lowest first.', changes: {sortKey: 'change_1d_pct', sortDirection: 'asc', rules: [{field: 'change_1d_pct', operator: 'lt', value: 0}]}},
    mega: {label: 'Mega caps', description: 'USD market cap of at least $200 billion.', changes: {currency: 'USD', capTier: 'mega'}},
    momentum: {label: 'Momentum', description: 'Positive 1M return; price above its 50D and 200D averages.', changes: {columns: 'performance', sortKey: 'return_1m_pct', rules: [{field: 'return_1m_pct', operator: 'gt', value: 0}, {field: 'sma50_distance_pct', operator: 'gt', value: 0}, {field: 'sma200_distance_pct', operator: 'gt', value: 0}]}},
    oversold: {label: 'Oversold', description: '14-day RSI below 30.', changes: {columns: 'technicals', sortKey: 'rsi14', sortDirection: 'asc', rules: [{field: 'rsi14', operator: 'lt', value: 30}]}},
    volume: {label: 'Volume surge', description: 'Volume at least twice its 20-day average.', changes: {columns: 'technicals', sortKey: 'relative_volume', rules: [{field: 'relative_volume', operator: 'gte', value: 2}]}},
    value: {label: 'Value & dividends', description: 'Positive P/E up to 20 and dividend yield of at least 2%.', changes: {columns: 'fundamentals', sortKey: 'dividend_yield_pct', rules: [{field: 'pe', operator: 'gt', value: 0}, {field: 'pe', operator: 'lte', value: 20}, {field: 'dividend_yield_pct', operator: 'gte', value: 2}]}},
    growth: {label: 'Profitable growth', description: 'Revenue growth of at least 15% and profit margin of at least 10%.', changes: {columns: 'fundamentals', sortKey: 'revenue_growth_pct', rules: [{field: 'revenue_growth_pct', operator: 'gte', value: 15}, {field: 'profit_margin_pct', operator: 'gte', value: 10}]}},
    highs: {label: 'Near 52W highs', description: 'Price within 5% of its 52-week high and above the 50-day average.', changes: {columns: 'performance', sortKey: 'distance_52w_high_pct', rules: [{field: 'distance_52w_high_pct', operator: 'gte', value: -5}, {field: 'sma50_distance_pct', operator: 'gt', value: 0}]}},
    trend: {label: 'Strong uptrend', description: 'ADX at least 25, positive MACD histogram and price above its 200-day average.', changes: {columns: 'trend', sortKey: 'adx14', rules: [{field: 'adx14', operator: 'gte', value: 25}, {field: 'macd_histogram_pct', operator: 'gt', value: 0}, {field: 'sma200_distance_pct', operator: 'gt', value: 0}]}},
    squeeze: {label: 'Low band width', description: 'Bollinger Band width below 10%, with at least 500,000 shares average daily volume.', changes: {columns: 'trend', sortKey: 'bollinger_width_pct', sortDirection: 'asc', rules: [{field: 'bollinger_width_pct', operator: 'lt', value: 10}, {field: 'avg_volume20', operator: 'gte', value: 0.5}]}},
    liquid: {label: 'Liquid stocks', description: 'USD price at least $5 and average daily turnover of at least $10 million.', changes: {currency: 'USD', columns: 'liquidity', sortKey: 'avg_turnover20', rules: [{field: 'close', operator: 'gte', value: 5}, {field: 'avg_turnover20', operator: 'gte', value: 10}]}}
  };
  let state = defaults(), savedScreens = Object.create(null), data = [], response = null, filtered = [], page = 0;
  let loading = false, loaded = false, loadSequence = 0, ruleSequence = 0, searchTimer;
  const collator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
  const numeric = value => (typeof value === 'number' || typeof value === 'string' && value.trim() !== '') && Number.isFinite(Number(value));
  const visibleColumns = () => state.columns === 'custom' ? state.customColumns : columnSets[state.columns];

  function normalizeState(value) {
    const clean = defaults();
    if (!value || typeof value !== 'object') return clean;
    for (const key of ['search', 'country', 'exchange', 'currency', 'sector', 'industry', 'capTier']) if (typeof value[key] === 'string') clean[key] = value[key];
    if (!Object.hasOwn(tierBounds, clean.capTier)) clean.capTier = '';
    if (Object.hasOwn(columnSets, value.columns) || value.columns === 'custom') clean.columns = value.columns;
    if (Array.isArray(value.customColumns)) clean.customColumns = ['symbol', ...new Set(value.customColumns.filter(key => key !== 'symbol' && Object.hasOwn(fields, key)))];
    if (['20', '50', '100'].includes(String(value.pageSize))) clean.pageSize = Number(value.pageSize);
    if (typeof value.priceSince === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value.priceSince)) clean.priceSince = value.priceSince;
    if (Object.hasOwn(fields, value.sortKey)) clean.sortKey = value.sortKey;
    if (Object.hasOwn(fields, value.secondarySort)) clean.secondarySort = value.secondarySort;
    clean.secondaryDirection = value.secondaryDirection === 'asc' ? 'asc' : 'desc';
    clean.heat = value.heat === true;
    clean.compact = value.compact === true;
    clean.sortDirection = value.sortDirection === 'asc' ? 'asc' : 'desc';
    clean.onlyHistory = value.onlyHistory === true;
    clean.onlyFavorites = value.onlyFavorites === true;
    clean.matchMode = value.matchMode === 'any' ? 'any' : 'all';
    clean.expanded = value.expanded !== false;
    clean.rules = (Array.isArray(value.rules) ? value.rules : []).slice(0, 20).filter(rule => rule && Object.hasOwn(fields, rule.field) && !['text', 'date'].includes(fields[rule.field].type)).map(rule => ({
      id: ++ruleSequence, field: rule.field, operator: Object.hasOwn(operators, rule.operator) ? rule.operator : 'gte',
      value: typeof rule.value === 'number' || typeof rule.value === 'string' ? String(rule.value) : '',
      maximum: typeof rule.maximum === 'number' || typeof rule.maximum === 'string' ? String(rule.maximum) : '',
      currency: typeof rule.currency === 'string' ? rule.currency : 'USD'
    }));
    return clean;
  }

  function restore() {
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || 'null');
      if (!stored) return;
      state = normalizeState(stored.state);
      for (const [name, screen] of Object.entries(stored.screens || {})) if (name && name.length <= 80) savedScreens[name] = normalizeState(screen);
    } catch { /* A missing or invalid saved screen leaves the defaults available. */ }
  }

  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify({state, screens: savedScreens})); }
    catch { announce('This browser cannot save screens. Your current filters still work.'); }
  }

  function announce(message) { byId('screenerNotice').textContent = message; }

  function install() {
    document.querySelector('main > header .header-actions').insertAdjacentHTML('afterbegin', '<button id="toggleScreener" class="secondary" aria-controls="screenerContent">Stock screener</button>');
    document.querySelector('main > header').insertAdjacentHTML('afterend', `
      <section id="stockScreener" class="stock-screener" aria-labelledby="screenerTitle">
        <div class="screener-heading"><div><span class="eyebrow">RESEARCH UNIVERSE</span><h2 id="screenerTitle">Stock screener <span class="screener-metric-count">${Object.values(fields).filter(field => !['text', 'date'].includes(field.type)).length} metrics</span></h2><p>Build your screen across fundamentals, performance, trend and liquidity.</p></div><div class="screener-heading-actions"><button id="refreshScreener" class="secondary">Refresh data</button><button id="collapseScreener" class="secondary" aria-controls="screenerContent">Collapse</button></div></div>
        <div id="screenerContent">
          <div id="screenerCoverage" class="screener-coverage"></div>
          <div class="screener-presets" aria-label="Screen presets"><span>START WITH</span>${Object.entries(presets).map(([id, preset]) => `<button data-screen-preset="${id}" title="${html(preset.description)}">${html(preset.label)}</button>`).join('')}</div>
          <div class="screener-filters">
            <label class="screener-search" for="screenerSearch">Company or symbol<input id="screenerSearch" type="search" placeholder="Search the full stock universe"></label>
            <label for="screenerCountry">Market / country<select id="screenerCountry"></select></label>
            <label for="screenerExchange">Exchange<select id="screenerExchange"></select></label>
            <label for="screenerCurrency">Currency<select id="screenerCurrency"></select></label>
            <label for="screenerSector">Sector<select id="screenerSector"></select></label>
            <label for="screenerIndustry">Industry<select id="screenerIndustry"></select></label>
            <label for="screenerCapTier">Market cap tier (USD)<select id="screenerCapTier"><option value="">All sizes / currencies</option><option value="mega">Mega · $200B+</option><option value="large">Large · $10B–$200B</option><option value="mid">Mid · $2B–$10B</option><option value="small">Small · $300M–$2B</option><option value="micro">Micro · below $300M</option></select></label>
            <label for="screenerPriceSince">Latest price on / after<input id="screenerPriceSince" type="date"></label>
            <label class="screener-check" for="screenerOnlyHistory"><input id="screenerOnlyHistory" type="checkbox">Charts available only</label>
            <label class="screener-check" for="screenerOnlyFavorites"><input id="screenerOnlyFavorites" type="checkbox">Saved stocks only</label>
          </div>
          <div class="screener-rule-box"><div class="screener-rule-heading"><div><h3>Numeric filters</h3><span>Percentages use points: 5 = 5%. Company filters always apply.</span></div><div class="screener-rule-actions"><label for="screenerMatchMode"><span class="screener-sr-only">Combine numeric rules</span><select id="screenerMatchMode"><option value="all">Match all (AND)</option><option value="any">Match any (OR)</option></select></label><button id="addScreenerRule" class="secondary">+ Add rule</button></div></div><div id="screenerRules"></div><p id="screenerRuleError" class="screener-inline-error" role="alert" hidden></p></div>
          <div id="screenerActiveFilters" class="screener-active-filters" aria-label="Active filters"></div>
          <div class="screener-saved"><label for="savedScreenSelect">Saved screens<select id="savedScreenSelect"><option value="">Choose a saved screen</option></select></label><label for="screenName">Screen name<input id="screenName" maxlength="80" placeholder="e.g. Profitable large caps"></label><button id="saveScreener" class="secondary">Save screen</button><button id="deleteScreener" class="secondary" disabled>Delete saved</button><button id="resetScreener" class="secondary">Reset filters</button></div>
          <p id="screenerNotice" class="screener-notice" role="status" aria-live="polite"></p>
          <div id="screenerError" class="error" role="alert" hidden></div>
          <div class="screener-results-heading"><div><strong id="screenerResultCount" role="status" aria-live="polite">Loading stocks…</strong><span id="screenerSortNote"></span></div><div class="screener-export"><label for="screenerExportColumns"><span class="screener-sr-only">CSV columns</span><select id="screenerExportColumns"><option value="visible">Visible columns</option><option value="all">All metrics</option></select></label><button id="exportScreener" class="secondary" disabled>Export matches CSV</button></div></div>
          <div id="screenerColumnSets" class="screener-column-sets" aria-label="Table columns">${[...Object.keys(columnSets), 'custom'].map(id => `<button data-screen-columns="${id}">${id.charAt(0).toUpperCase() + id.slice(1)}</button>`).join('')}</div>
          <details id="screenerColumnsDetails" class="screener-details"><summary>Customize columns <span id="screenerColumnCount"></span></summary><div class="screener-column-picker">${Object.entries(fieldGroups).map(([group, keys]) => `<fieldset><legend>${html(group)}</legend>${keys.map(key => `<label title="${html(help[key] || fields[key].label)}"><input type="checkbox" data-custom-column="${key}" ${key === 'symbol' ? 'disabled checked' : ''}>${html(fields[key].label)}</label>`).join('')}</fieldset>`).join('')}</div></details>
          <div class="screener-table-wrap" tabindex="0" aria-label="Stock results; scroll horizontally for more columns"><table class="screener-table"><caption class="screener-sr-only">Matching stocks. Select a column heading to sort; select a symbol to open its chart.</caption><thead id="screenerTableHead"></thead><tbody id="screenerTableBody"></tbody></table></div>
          <div class="screener-pagination"><span id="screenerPageInfo"></span><div><label class="screener-page-size" for="screenerPageSize">Rows<select id="screenerPageSize"><option>20</option><option selected>50</option><option>100</option></select></label><button id="screenerPrevious" class="secondary">← Previous</button><button id="screenerNext" class="secondary">Next →</button></div></div>
          <p id="screenerDataNote" class="screener-data-note">Missing metrics appear as — and do not match numeric comparisons. Fundamentals and technicals reflect locally stored snapshots.</p>
          <details class="screener-details screener-methodology"><summary>Metric definitions &amp; data coverage</summary><p>Daily stored data. Returns and indicators use adjusted prices where available. No currency conversion is applied. Missing data never passes a numeric comparison; use “Is missing” to find gaps.</p><dl id="screenerMethodology"></dl></details>
        </div>
      </section>`);
    byId('screenerColumnSets').insertAdjacentHTML('beforebegin', `<div class="screener-enhancements"><label>Then sort<select id="screenerSecondarySort" aria-label="Secondary sort"><option value="">Symbol (default)</option>${Object.entries(fields).map(([key, field]) => `<option value="${key}">${html(field.label)}</option>`).join('')}</select></label><select id="screenerSecondaryDirection" aria-label="Secondary sort direction"><option value="desc">Descending</option><option value="asc">Ascending</option></select><label><input type="checkbox" id="screenerHeat">Color values</label><label><input type="checkbox" id="screenerCompact">Compact rows</label><button id="screenerSaveMatches" class="secondary">Save all matches</button><button id="screenerUndoSave" class="secondary" hidden>Undo save</button></div><div id="screenerBreadth" aria-label="Result market breadth"></div>`);
    byId('screenerColumnsDetails').querySelector('summary').insertAdjacentHTML('afterend', '<input id="screenerColumnSearch" type="search" placeholder="Find a column…" aria-label="Find a column"><div class="screener-enhancements"><label>Reorder<select id="screenerMoveColumn" aria-label="Column to move"></select></label><button id="screenerColumnLeft" class="secondary" aria-label="Move column left">← Left</button><button id="screenerColumnRight" class="secondary" aria-label="Move column right">Right →</button></div>');
  }

  function valuesFor(key, rows = data) {
    return [...new Set(rows.map(row => row[key]).filter(value => value != null && value !== '').map(String))].sort(collator.compare);
  }

  function fillOptions(id, key, label, rows = data) {
    const values = valuesFor(key, rows), selectedValue = state[key];
    if (selectedValue && selectedValue !== missingOption && !values.includes(selectedValue)) values.push(selectedValue);
    byId(id).innerHTML = `<option value="">${label}</option>${values.map(value => `<option value="${html(value)}">${html(value)}</option>`).join('')}<option value="${missingOption}">Not available</option>`;
    byId(id).value = selectedValue;
  }

  function refreshFilterOptions() {
    fillOptions('screenerCountry', 'country', 'All markets');
    fillOptions('screenerExchange', 'exchange', 'All exchanges');
    fillOptions('screenerCurrency', 'currency', 'All currencies');
    fillOptions('screenerSector', 'sector', 'All sectors');
    fillOptions('screenerIndustry', 'industry', 'All industries', state.sector ? data.filter(row => matchesChoice(row.sector, state.sector)) : data);
  }

  function populateControls() {
    byId('screenerSearch').value = state.search;
    byId('screenerCapTier').value = state.capTier;
    byId('screenerOnlyHistory').checked = state.onlyHistory;
    byId('screenerOnlyFavorites').checked = state.onlyFavorites;
    byId('screenerPriceSince').value = state.priceSince;
    byId('screenerMatchMode').value = state.matchMode;
    byId('screenerPageSize').value = String(state.pageSize);
    refreshFilterOptions(); renderRules(); renderSavedScreens(); updateExpanded();
  }

  function updateExpanded() {
    byId('screenerContent').hidden = !state.expanded;
    byId('toggleScreener').setAttribute('aria-expanded', String(state.expanded));
    byId('collapseScreener').setAttribute('aria-expanded', String(state.expanded));
    byId('collapseScreener').textContent = state.expanded ? 'Collapse' : 'Expand';
    byId('stockScreener').classList.toggle('is-collapsed', !state.expanded);
  }

  function renderRules() {
    const metricOptions = selected => Object.entries(fieldGroups).map(([group, keys]) => {
      const available = keys.filter(key => !['text', 'date'].includes(fields[key].type));
      return available.length ? `<optgroup label="${html(group)}">${available.map(key => {
        const item = fields[key];
        return `<option value="${key}" ${key === selected ? 'selected' : ''}>${html(item.label)}${item.unit ? ` (${html(item.unit)})` : item.type === 'percent' ? ' (%)' : ''}</option>`;
      }).join('')}</optgroup>` : '';
    }).join('');
    byId('screenerRules').innerHTML = state.rules.length ? state.rules.map((rule, index) => {
      const noValue = ['missing', 'present'].includes(rule.operator), field = fields[rule.field];
      const currencies = [...new Set(['USD', ...valuesFor('currency'), rule.currency])];
      return `<div class="screener-rule" data-screen-rule="${rule.id}"><span class="screener-rule-number">${index ? state.matchMode === 'any' ? 'OR' : 'AND' : 'WHERE'}</span>
        <label title="${html(help[rule.field] || field.label)}"><span class="screener-sr-only">Rule ${index + 1} metric</span><select data-rule-property="field">${metricOptions(rule.field)}</select></label>
        <label><span class="screener-sr-only">Rule ${index + 1} comparison</span><select data-rule-property="operator">${Object.entries(operators).map(([key, label]) => `<option value="${key}" ${key === rule.operator ? 'selected' : ''}>${html(label)}</option>`).join('')}</select></label>
        <label ${noValue ? 'hidden' : ''}><span class="screener-sr-only">Rule ${index + 1} value</span><input data-rule-property="value" type="number" step="any" placeholder="Value" value="${html(rule.value)}"></label>
        <label ${rule.operator !== 'between' ? 'hidden' : ''}><span class="screener-sr-only">Rule ${index + 1} maximum</span><input data-rule-property="maximum" type="number" step="any" placeholder="Maximum" value="${html(rule.maximum)}"></label>
        <label ${!field.monetary || noValue ? 'hidden' : ''}><span class="screener-sr-only">Rule ${index + 1} currency</span><select data-rule-property="currency">${currencies.map(currency => `<option value="${html(currency)}" ${currency === rule.currency ? 'selected' : ''}>${html(currency)}</option>`).join('')}</select></label>
        <button class="screener-remove-rule" data-remove-rule="${rule.id}" aria-label="Remove rule ${index + 1}">×</button></div>`;
    }).join('') : '<p class="screener-no-rules">No numeric rules. Add a rule to combine valuation, performance, volume and technical criteria.</p>';
    byId('addScreenerRule').disabled = state.rules.length >= 20;
  }

  function ruleErrors() {
    return state.rules.flatMap((rule, index) => {
      if (['missing', 'present'].includes(rule.operator)) return [];
      if (!numeric(rule.value) || (rule.operator === 'between' && !numeric(rule.maximum))) return [`Rule ${index + 1} needs a numeric value.`];
      if (rule.operator === 'between' && Number(rule.value) > Number(rule.maximum)) return [`Rule ${index + 1}: minimum must not exceed maximum.`];
      return [];
    });
  }

  function matchesChoice(value, choice) {
    return !choice || (choice === missingOption ? value == null || value === '' : String(value) === choice);
  }

  function matchesRule(row, rule) {
    const value = row[rule.field], field = fields[rule.field];
    if (rule.operator === 'missing') return !numeric(value);
    if (rule.operator === 'present') return numeric(value);
    if (!numeric(value) || (field.monetary && row.currency !== rule.currency)) return false;
    const actual = Number(value), low = Number(rule.value) * (field.scale || 1), high = Number(rule.maximum) * (field.scale || 1);
    if (rule.operator === 'gte') return actual >= low;
    if (rule.operator === 'lte') return actual <= low;
    if (rule.operator === 'gt') return actual > low;
    if (rule.operator === 'lt') return actual < low;
    if (rule.operator === 'eq') return actual === low;
    return actual >= low && actual <= high;
  }

  function hasHistory(row) { return row.has_history === true || (row.has_history == null && numeric(row.close) && !!row.date); }

  function matchingRows() {
    const terms = state.search.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    return data.filter(row => {
      if (state.onlyHistory && !hasHistory(row)) return false;
      if (state.onlyFavorites && !saved.has(row.symbol)) return false;
      if (state.priceSince && (!row.date || row.date < state.priceSince)) return false;
      if (terms.length && !terms.every(term => `${row.symbol} ${row.name || ''}`.toLocaleLowerCase().includes(term))) return false;
      for (const key of ['country', 'exchange', 'currency', 'sector', 'industry']) if (!matchesChoice(row[key], state[key])) return false;
      if (state.capTier) {
        const [low, high] = tierBounds[state.capTier];
        if (row.currency !== 'USD' || !numeric(row.market_cap) || row.market_cap < low || row.market_cap >= high) return false;
      }
      return !state.rules.length || (state.matchMode === 'any' ? state.rules.some(rule => matchesRule(row, rule)) : state.rules.every(rule => matchesRule(row, rule)));
    });
  }

  function compareField(a, b, key, direction) {
    const field = fields[key], av = a[key], bv = b[key];
    const absentA = ['text', 'date'].includes(field.type) ? av == null || av === '' : !numeric(av);
    const absentB = ['text', 'date'].includes(field.type) ? bv == null || bv === '' : !numeric(bv);
    if (absentA !== absentB) return absentA ? 1 : -1;
    if (absentA) return 0;
    if (field.monetary && a.currency !== b.currency) {
      if (a.currency === 'USD') return -1;
      if (b.currency === 'USD') return 1;
      return collator.compare(a.currency || 'ZZZ', b.currency || 'ZZZ');
    }
    const order = ['text', 'date'].includes(field.type) ? collator.compare(String(av), String(bv)) : Number(av) - Number(bv);
    return direction === 'asc' ? order : -order;
  }

  function compareRows(a, b) {
    return compareField(a, b, state.sortKey, state.sortDirection) || (state.secondarySort && state.secondarySort !== state.sortKey ? compareField(a, b, state.secondarySort, state.secondaryDirection) : 0) || collator.compare(a.symbol, b.symbol);
  }

  function number(value, decimals = 2) {
    return Number(value).toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
  }

  function compact(value) {
    const absolute = Math.abs(Number(value));
    for (const [scale, suffix] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) if (absolute >= scale) return `${number(value / scale)}${suffix}`;
    return number(value, 0);
  }

  function dateText(value) {
    if (!value) return '—';
    return String(value).slice(0, 10);
  }

  function renderCell(row, key) {
    const field = fields[key], value = row[key];
    if (key === 'symbol') return `<td class="screener-symbol-cell"><button class="screener-favorite" data-screen-favorite="${html(row.symbol)}" aria-label="${saved.has(row.symbol) ? 'Unsave' : 'Save'} ${html(row.symbol)}" aria-pressed="${saved.has(row.symbol)}">${saved.has(row.symbol) ? '★' : '☆'}</button><button data-screen-symbol="${html(row.symbol)}" ${hasHistory(row) ? '' : 'disabled'} title="${hasHistory(row) ? 'Open daily chart' : 'Price history has not been collected for this stock'}">${html(row.symbol)}</button>${hasHistory(row) ? '' : '<small>No chart yet</small>'}</td>`;
    if (field.type === 'text') return `<td class="screener-text-cell" title="${html(value)}">${html(value || '—')}</td>`;
    if (field.type === 'date') return `<td class="screener-date-cell">${html(dateText(value || (key === 'metadata_date' ? row.fetched_at : null)))}</td>`;
    if (!numeric(value)) return '<td class="screener-missing">—</td>';
    let formatted = number(value), color = '';
    if (field.type === 'cap') formatted = `${compact(value)} <small>${html(row.currency || '')}</small>`;
    if (field.type === 'quantity') formatted = compact(value);
    if (field.type === 'multiple') formatted += '×';
    if (field.type === 'percent') {
      formatted += '%';
      if (/^(change_|return_|sma\d|ema\d|macd|distance_52w_|revenue_growth|profit_margin)/.test(key)) color = Number(value) >= 0 ? 'positive' : 'negative';
    }
    if (field.type === 'price') formatted += ` <small>${html(row.currency || '')}</small>`;
    return `<td class="screener-number-cell ${color}" title="${html(value)}${field.monetary ? ' ' + html(row.currency || '') : ''}">${formatted}</td>`;
  }

  function renderCoverage() {
    if (!response) return;
    const coverage = response.coverage || {};
    const withHistory = coverage.with_history ?? data.filter(hasHistory).length;
    const withCap = coverage.with_market_cap ?? data.filter(row => numeric(row.market_cap)).length;
    const latest = response.data_dates?.latest;
    byId('screenerCoverage').innerHTML = `<div><span>STOCK UNIVERSE</span><strong>${number(data.length, 0)}</strong></div><div><span>CHARTS AVAILABLE</span><strong>${number(withHistory, 0)}</strong></div><div><span>MARKET CAP COVERAGE</span><strong>${number(withCap, 0)} <small>/ ${number(data.length, 0)}</small></strong></div><div><span>LATEST PRICE DATE</span><strong>${html(dateText(latest))}</strong></div>`;
    const firstDate = response.data_dates?.earliest, generated = response.generated_at ? new Date(response.generated_at).toLocaleString() : 'available snapshot';
    const source = response.universe?.scope || 'Locally stored stocks';
    byId('screenerDataNote').textContent = `${source}. Price dates ${dateText(firstDate)} to ${dateText(latest)}. Snapshot generated ${generated}. Missing metrics appear as — and do not match numeric comparisons. Fundamentals may have different refresh dates; see the Fundamentals columns. Monetary values keep their original currency.`;
    const definitions = {...(response.methodology || {})};
    byId('screenerMethodology').innerHTML = Object.entries(definitions).map(([key, value]) => `<dt>${html(key.replace(/_/g, ' '))}</dt><dd>${html(value)}</dd>`).join('');
  }

  function renderActiveFilters() {
    const chips = [];
    const add = (key, label) => chips.push(`<button data-clear-filter="${key}" aria-label="Remove ${html(label)}">${html(label)} <span aria-hidden="true">×</span></button>`);
    if (state.search.trim()) add('search', `Search: ${state.search.trim()}`);
    for (const key of ['country', 'exchange', 'currency', 'sector', 'industry']) if (state[key]) add(key, `${fields[key].label}: ${state[key] === missingOption ? 'Not available' : state[key]}`);
    if (state.capTier) add('capTier', `${state.capTier} cap (USD)`);
    if (state.onlyHistory) add('onlyHistory', 'Charts available');
    if (state.onlyFavorites) add('onlyFavorites', 'Saved stocks');
    if (state.priceSince) add('priceSince', `Price since ${state.priceSince}`);
    for (const rule of state.rules) {
      const field = fields[rule.field], noValue = ['missing', 'present'].includes(rule.operator);
      add(`rule:${rule.id}`, `${field.label} ${operators[rule.operator].toLowerCase()}${noValue ? '' : ` ${rule.value}${rule.operator === 'between' ? ` – ${rule.maximum}` : ''}${field.unit ? ` ${field.unit}` : field.type === 'percent' ? '%' : ''}${field.monetary ? ` ${rule.currency}` : ''}`}`);
    }
    byId('screenerActiveFilters').innerHTML = chips.length ? `<span>${chips.length} ACTIVE${state.rules.length > 1 ? ` · ${state.matchMode === 'any' ? 'ANY' : 'ALL'} NUMERIC RULES` : ''}</span>${chips.join('')}` : '';
  }

  function renderColumnPicker() {
    const columns = visibleColumns();
    byId('screenerColumnCount').textContent = `· ${columns.length} selected`;
    for (const input of byId('screenerColumnsDetails').querySelectorAll('[data-custom-column]')) input.checked = columns.includes(input.dataset.customColumn);
    const previous = byId('screenerMoveColumn').value;
    byId('screenerMoveColumn').innerHTML = columns.filter(key => key !== 'symbol').map(key => `<option value="${key}">${html(fields[key].label)}</option>`).join('');
    if (columns.includes(previous)) byId('screenerMoveColumn').value = previous;
    updateColumnArrows();
  }

  function updateColumnArrows() {
    const columns = visibleColumns(), index = columns.indexOf(byId('screenerMoveColumn').value);
    byId('screenerColumnLeft').disabled = index <= 1;
    byId('screenerColumnRight').disabled = index < 1 || index >= columns.length - 1;
  }

  function renderResults() {
    const focused = document.activeElement;
    const focusKey = ['screenSort', 'screenFavorite', 'screenSymbol'].find(key => focused?.dataset[key]);
    const focusValue = focusKey ? focused.dataset[focusKey] : null;
    const errors = ruleErrors();
    byId('screenerRuleError').hidden = !errors.length;
    byId('screenerRuleError').textContent = errors.join(' ');
    filtered = errors.length ? [] : matchingRows().sort(compareRows);
    const pageSize = state.pageSize;
    page = Math.max(0, Math.min(page, Math.ceil(filtered.length / pageSize) - 1));
    const columns = visibleColumns();
    byId('screenerTableHead').innerHTML = `<tr>${columns.map(key => `<th scope="col" aria-sort="${state.sortKey === key ? state.sortDirection === 'asc' ? 'ascending' : 'descending' : 'none'}"><button data-screen-sort="${key}" title="Sort by ${html(fields[key].label)}${help[key] ? '. ' + html(help[key]) : ''}">${html(fields[key].label)} <span aria-hidden="true">${state.sortKey === key ? state.sortDirection === 'asc' ? '↑' : '↓' : '↕'}</span></button></th>`).join('')}</tr>`;
    const displayed = filtered.slice(page * pageSize, (page + 1) * pageSize);
    byId('screenerTableBody').innerHTML = displayed.map(row => `<tr class="${selected?.symbol === row.symbol ? 'screen-current' : ''}">${columns.map(key => renderCell(row, key)).join('')}</tr>`).join('') || `<tr><td colspan="${columns.length}" class="screener-empty">${loading && !loaded ? 'Loading the stock universe…' : errors.length ? 'Complete the numeric filters to see results.' : loaded && !data.length ? 'No stocks are available in the local database yet.' : 'No stocks match this screen. Remove a rule or reset the filters.'}</td></tr>`;
    byId('screenerResultCount').textContent = `${number(filtered.length, 0)} ${filtered.length === 1 ? 'match' : 'matches'} of ${number(data.length, 0)} stocks`;
    const currencies = new Set(filtered.map(row => row.currency).filter(Boolean));
    const grouped = fields[state.sortKey].monetary && currencies.size > 1;
    byId('screenerSortNote').textContent = `Sorted by ${fields[state.sortKey].label.toLowerCase()} ${state.sortDirection === 'asc' ? '↑' : '↓'}${grouped ? ' · grouped by currency, USD first' : ''}${state.capTier ? ' · USD stocks only' : ''}`;
    if (state.secondarySort && state.secondarySort !== state.sortKey) byId('screenerSortNote').textContent += ` · then ${fields[state.secondarySort].label} ${state.secondaryDirection === 'asc' ? '↑' : '↓'}`;
    byId('screenerPrimarySort').value = state.sortKey;
    byId('screenerPrimaryDirection').value = state.sortDirection;
    byId('screenerSecondarySort').value = state.secondarySort;
    byId('screenerSecondaryDirection').value = state.secondaryDirection;
    byId('screenerSecondaryDirection').disabled = !state.secondarySort;
    byId('screenerHeat').checked = state.heat;
    byId('screenerCompact').checked = state.compact;
    byId('stockScreener').classList.toggle('screen-heat', state.heat);
    byId('stockScreener').classList.toggle('screen-compact', state.compact);
    byId('screenerSaveMatches').disabled = !filtered.length;
    const priced = filtered.filter(row => numeric(row.change_1d_pct));
    const advancing = priced.filter(row => Number(row.change_1d_pct) > 0).length, declining = priced.filter(row => Number(row.change_1d_pct) < 0).length;
    byId('screenerBreadth').innerHTML = `<span class="positive">${advancing.toLocaleString()} advancing</span> · <span class="negative">${declining.toLocaleString()} declining</span> · ${priced.length - advancing - declining} unchanged · ${filtered.length - priced.length} missing 1D change <span> · Per-stock latest dates</span>`;
    byId('screenerPageInfo').textContent = filtered.length ? `${number(page * pageSize + 1, 0)}–${number(Math.min((page + 1) * pageSize, filtered.length), 0)} of ${number(filtered.length, 0)} · ${pageSize} per page` : '0 results';
    byId('screenerPrevious').disabled = page === 0;
    byId('screenerNext').disabled = (page + 1) * pageSize >= filtered.length;
    byId('exportScreener').disabled = !filtered.length || !!errors.length;
    for (const button of byId('screenerColumnSets').querySelectorAll('button')) {
      const active = button.dataset.screenColumns === state.columns;
      button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
    }
    renderActiveFilters(); renderColumnPicker();
    if (focusKey) {
      const replacement = [...byId('stockScreener').querySelectorAll('.screener-table button')].find(button => button.dataset[focusKey] === focusValue && !button.disabled);
      (replacement || byId('stockScreener').querySelector('.screener-table-wrap')).focus({preventScroll: true});
    }
  }

  async function load() {
    if (loading) return;
    const sequence = ++loadSequence;
    loading = true; byId('refreshScreener').disabled = true;
    byId('screenerContent').setAttribute('aria-busy', 'true');
    byId('screenerError').hidden = true;
    renderResults();
    try {
      const result = await api('/api/screener');
      if (sequence !== loadSequence) return;
      if (!Array.isArray(result.rows)) throw Error('The screener returned an invalid stock list.');
      response = result; data = result.rows.filter(row => row && typeof row.symbol === 'string'); loaded = true;
      refreshFilterOptions(); renderCoverage(); renderRules();
    } catch (error) {
      byId('screenerError').textContent = `${loaded ? 'Could not refresh; showing the previous snapshot. ' : 'Could not load the screener. '}${error.message} Use Refresh data to retry.`;
      byId('screenerError').hidden = false;
    } finally {
      loading = false; byId('refreshScreener').disabled = false;
      byId('screenerContent').setAttribute('aria-busy', 'false'); renderResults();
    }
  }

  function changed({rules = false} = {}) {
    page = 0;
    byId('savedScreenSelect').value = '';
    byId('deleteScreener').disabled = true;
    if (rules) renderRules();
    persist(); renderResults();
  }

  function renderSavedScreens(selected = '') {
    byId('savedScreenSelect').innerHTML = '<option value="">Choose a saved screen</option>' + Object.keys(savedScreens).sort(collator.compare).map(name => `<option value="${html(name)}">${html(name)}</option>`).join('');
    byId('savedScreenSelect').value = selected;
    byId('deleteScreener').disabled = !selected;
  }

  function loadSavedScreen(name) {
    if (!Object.hasOwn(savedScreens, name)) return;
    state = normalizeState(savedScreens[name]); state.expanded = true; page = 0;
    populateControls(); byId('screenName').value = name; renderSavedScreens(name);
    persist(); renderResults(); announce(`Loaded “${name}”.`);
    if (!loaded) load();
  }

  function exportCsv() {
    if (!filtered.length || ruleErrors().length) return;
    const allMetrics = byId('screenerExportColumns').value === 'all';
    const keys = [...new Set([...(allMetrics ? Object.keys(fields) : visibleColumns()), 'currency', 'date', 'metadata_date', 'metadata_profile_at', 'has_history'])];
    const cell = value => {
      // Keep company names that start with spreadsheet formula markers as text.
      const safe = typeof value === 'string' && /^[=+\-@\t\r]/.test(value) ? "'" + value : value;
      return `"${String(safe ?? '').replace(/"/g, '""')}"`;
    };
    const csv = '\uFEFF' + [keys.map(key => cell(key)).join(','), ...filtered.map(row => keys.map(key => cell(row[key])).join(','))].join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'})), link = document.createElement('a');
    link.href = url; link.download = `quantstack-screen-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce(`Exported all ${number(filtered.length, 0)} matching stocks with ${allMetrics ? 'all metric' : state.columns} columns.`);
  }

  async function openChart(symbol, button) {
    const row = data.find(item => item.symbol === symbol);
    if (!row || !hasHistory(row)) return;
    button.disabled = true;
    try {
      if (!assets.some(item => item.symbol === symbol)) {
        const currentAssets = await api('/api/symbols');
        if (!Array.isArray(currentAssets)) throw Error('The chart symbol list is unavailable.');
        assets = currentAssets; renderAssets();
      }
      if (!assets.some(item => item.symbol === symbol)) throw Error('This stock is waiting for its chart data to become available.');
      interval = '1d'; period = '1Y';
      for (const control of byId('intervals').querySelectorAll('[data-interval]')) control.classList.toggle('active', control.dataset.interval === interval);
      for (const control of byId('periods').querySelectorAll('[data-period]')) control.classList.toggle('active', control.dataset.period === period);
      await select(symbol);
      if (!byId('error').hidden) throw Error(byId('error').textContent);
      document.querySelector('.overview').scrollIntoView({behavior: 'smooth', block: 'start'});
      renderResults(); announce(`Opened the ${symbol} chart. Use ↑ / ↓ on a symbol to navigate results, then Enter to open.`);
    } catch (error) { announce(`Could not open ${symbol}: ${error.message}`); }
    finally { button.disabled = false; }
  }

  function bind() {
    byId('screenerSecondarySort').onchange = event => { state.secondarySort = event.target.value; changed(); };
    byId('screenerSecondaryDirection').onchange = event => { state.secondaryDirection = event.target.value; changed(); };
    byId('screenerHeat').onchange = event => { state.heat = event.target.checked; persist(); renderResults(); };
    byId('screenerCompact').onchange = event => { state.compact = event.target.checked; persist(); renderResults(); };
    byId('screenerColumnSearch').oninput = event => {
      const term = event.target.value.trim().toLowerCase();
      for (const group of byId('screenerColumnsDetails').querySelectorAll('fieldset')) {
        for (const label of group.querySelectorAll('label')) label.hidden = !label.textContent.toLowerCase().includes(term);
        group.hidden = ![...group.querySelectorAll('label')].some(label => !label.hidden);
      }
    };
    byId('screenerMoveColumn').onchange = updateColumnArrows;
    function moveColumn(direction) {
      const key = byId('screenerMoveColumn').value, columns = [...visibleColumns()], index = columns.indexOf(key), next = index + direction;
      if (index < 1 || next < 1 || next >= columns.length) return;
      [columns[index], columns[next]] = [columns[next], columns[index]];
      state.customColumns = columns; state.columns = 'custom'; changed();
    }
    byId('screenerColumnLeft').onclick = () => moveColumn(-1);
    byId('screenerColumnRight').onclick = () => moveColumn(1);
    let addedFavorites = [];
    function saveWatchlist() {
      try { localStorage.setItem('atlas.saved', JSON.stringify([...saved])); }
      catch { announce('Watchlist changes are available for this session only.'); }
      if (selected) updateSaved(); renderAssets(); renderResults();
    }
    byId('screenerSaveMatches').onclick = () => {
      addedFavorites = filtered.filter(row => !saved.has(row.symbol)).map(row => row.symbol);
      addedFavorites.forEach(symbol => saved.add(symbol)); saveWatchlist();
      byId('screenerUndoSave').hidden = !addedFavorites.length;
      announce(`Added ${addedFavorites.length.toLocaleString()} matching stocks to your watchlist.`);
    };
    byId('screenerUndoSave').onclick = () => {
      addedFavorites.forEach(symbol => saved.delete(symbol)); addedFavorites = []; saveWatchlist();
      byId('screenerUndoSave').hidden = true; announce('Undid the last bulk watchlist addition.');
    };
    byId('screenerTableBody').addEventListener('keydown', event => {
      const button = event.target.closest('[data-screen-symbol]');
      if (!button || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      let index = filtered.findIndex(row => row.symbol === button.dataset.screenSymbol) + step;
      while (index >= 0 && index < filtered.length && !hasHistory(filtered[index])) index += step;
      if (index < 0 || index >= filtered.length) return;
      page = Math.floor(index / state.pageSize); renderResults();
      const next = [...byId('screenerTableBody').querySelectorAll('[data-screen-symbol]')].find(item => item.dataset.screenSymbol === filtered[index].symbol);
      next?.focus({preventScroll: true}); next?.scrollIntoView({block: 'nearest', inline: 'nearest'});
    });
    const toggle = () => { state.expanded = !state.expanded; updateExpanded(); persist(); if (state.expanded && !loaded) load(); };
    byId('toggleScreener').onclick = byId('collapseScreener').onclick = toggle;
    byId('refreshScreener').onclick = load;
    byId('screenerSearch').oninput = event => {
      state.search = event.target.value;
      clearTimeout(searchTimer); searchTimer = setTimeout(() => changed(), 100);
    };
    for (const [id, key] of [['screenerCountry', 'country'], ['screenerExchange', 'exchange'], ['screenerCurrency', 'currency'], ['screenerSector', 'sector'], ['screenerIndustry', 'industry'], ['screenerCapTier', 'capTier']]) {
      byId(id).onchange = event => {
        state[key] = event.target.value;
        if (key === 'sector') { state.industry = ''; refreshFilterOptions(); }
        if (key === 'capTier' && state.capTier) { state.currency = 'USD'; refreshFilterOptions(); }
        if (key === 'currency' && state.currency !== 'USD') { state.capTier = ''; byId('screenerCapTier').value = ''; }
        changed();
      };
    }
    byId('screenerOnlyHistory').onchange = event => { state.onlyHistory = event.target.checked; changed(); };
    byId('screenerOnlyFavorites').onchange = event => { state.onlyFavorites = event.target.checked; changed(); };
    byId('screenerPriceSince').onchange = event => { state.priceSince = event.target.value; changed(); };
    byId('screenerMatchMode').onchange = event => { state.matchMode = event.target.value; changed({rules: true}); };
    byId('screenerPageSize').onchange = event => { state.pageSize = Number(event.target.value); page = 0; persist(); renderResults(); };
    byId('screenerActiveFilters').onclick = event => {
      const button = event.target.closest('[data-clear-filter]');
      if (!button) return;
      const key = button.dataset.clearFilter;
      if (key.startsWith('rule:')) state.rules = state.rules.filter(rule => rule.id !== Number(key.slice(5)));
      else state[key] = defaults()[key];
      if (key === 'sector') state.industry = '';
      if (key === 'currency') state.capTier = '';
      populateControls(); changed();
    };
    byId('addScreenerRule').onclick = () => {
      if (state.rules.length >= 20) return;
      state.rules.push({id: ++ruleSequence, field: 'rsi14', operator: 'between', value: '30', maximum: '70', currency: state.currency && state.currency !== missingOption ? state.currency : 'USD'});
      changed({rules: true});
      byId('screenerRules').lastElementChild?.querySelector('select')?.focus();
    };
    byId('screenerRules').onclick = event => {
      const button = event.target.closest('[data-remove-rule]');
      if (button) { state.rules = state.rules.filter(rule => rule.id !== Number(button.dataset.removeRule)); changed({rules: true}); }
    };
    const editRule = event => {
      const input = event.target.closest('[data-rule-property]');
      if (!input) return;
      const rule = state.rules.find(item => item.id === Number(input.closest('[data-screen-rule]').dataset.screenRule));
      if (!rule) return;
      const property = input.dataset.ruleProperty;
      rule[property] = input.value;
      const rebuild = ['field', 'operator'].includes(property);
      const hadFocus = document.activeElement === input;
      changed({rules: rebuild});
      if (rebuild && hadFocus) byId('screenerRules').querySelector(`[data-screen-rule="${rule.id}"] [data-rule-property="${property}"]`)?.focus({preventScroll: true});
    };
    byId('screenerRules').onchange = editRule;
    byId('screenerRules').oninput = event => { if (event.target.tagName === 'INPUT') editRule(event); };
    byId('stockScreener').addEventListener('click', event => {
      const preset = event.target.closest('[data-screen-preset]');
      if (!preset) return;
      const chosen = presets[preset.dataset.screenPreset];
      state = normalizeState({...defaults(), expanded: state.expanded, heat: state.heat, compact: state.compact, pageSize: state.pageSize, ...chosen.changes}); page = 0;
      populateControls(); persist(); renderResults(); announce(`${chosen.label}: ${chosen.description}`);
    });
    byId('screenerColumnSets').onclick = event => {
      const button = event.target.closest('[data-screen-columns]');
      if (button) {
        state.columns = button.dataset.screenColumns;
        if (state.columns === 'custom') byId('screenerColumnsDetails').open = true;
        changed();
      }
    };
    byId('screenerColumnsDetails').onchange = event => {
      const input = event.target.closest('[data-custom-column]');
      if (!input) return;
      const key = input.dataset.customColumn, columns = [...visibleColumns()];
      state.customColumns = input.checked ? [...new Set([...columns, key])] : columns.filter(item => item !== key || item === 'symbol');
      state.columns = 'custom'; changed();
    };
    byId('screenerTableHead').onclick = event => {
      const button = event.target.closest('[data-screen-sort]');
      if (!button) return;
      const key = button.dataset.screenSort;
      state.sortDirection = state.sortKey === key ? state.sortDirection === 'asc' ? 'desc' : 'asc' : ['text', 'date'].includes(fields[key].type) ? 'asc' : 'desc';
      state.sortKey = key; page = 0; persist(); renderResults();
    };
    byId('screenerTableBody').onclick = event => {
      const favorite = event.target.closest('[data-screen-favorite]');
      if (favorite) {
        const symbol = favorite.dataset.screenFavorite;
        saved.has(symbol) ? saved.delete(symbol) : saved.add(symbol);
        try { localStorage.setItem('atlas.saved', JSON.stringify([...saved])); }
        catch { announce('Saved stocks are available for this session; browser storage is unavailable.'); }
        if (selected) updateSaved();
        renderAssets(); renderResults(); return;
      }
      const button = event.target.closest('[data-screen-symbol]');
      if (button) openChart(button.dataset.screenSymbol, button);
    };
    // The chart and the screener share the same browser-local watchlist.
    byId('save').addEventListener('click', () => renderResults());
    window.addEventListener('watchlist-favorites-changed', () => renderResults());
    window.addEventListener('storage', event => {
      if (event.key !== 'atlas.saved' && event.key !== null) return;
      try {
        const value = JSON.parse(event.newValue || '[]');
        saved = new Set(Array.isArray(value) ? value.filter(item => typeof item === 'string') : []);
        if (selected) updateSaved();
        renderAssets(); renderResults();
      } catch { /* Ignore invalid watchlist data written by another tab. */ }
    });
    byId('screenerPrevious').onclick = () => { page--; renderResults(); };
    byId('screenerNext').onclick = () => { page++; renderResults(); };
    byId('exportScreener').onclick = exportCsv;
    byId('resetScreener').onclick = () => { state = {...defaults(), expanded: state.expanded, heat: state.heat, compact: state.compact, pageSize: state.pageSize}; page = 0; populateControls(); persist(); renderResults(); announce('Filters reset. All available stocks are shown.'); };
    byId('savedScreenSelect').onchange = event => loadSavedScreen(event.target.value);
    byId('saveScreener').onclick = () => {
      const name = byId('screenName').value.trim();
      if (!name) { announce('Enter a name for this screen.'); byId('screenName').focus(); return; }
      if (ruleErrors().length) { announce('Complete the numeric filters before saving this screen.'); return; }
      savedScreens[name] = normalizeState(state); persist(); renderSavedScreens(name); announce(`Saved “${name}” in this browser.`);
    };
    byId('deleteScreener').onclick = () => {
      const name = byId('savedScreenSelect').value;
      if (!name) return;
      delete savedScreens[name]; persist(); renderSavedScreens(); announce(`Deleted the saved screen “${name}”. Current filters are unchanged.`);
    };
  }

  restore(); install();
  byId('stockScreener').querySelector('.screener-enhancements').insertAdjacentHTML('afterbegin', `<label>Sort by<select id="screenerPrimarySort" aria-label="Primary sort">${Object.entries(fields).map(([key, field]) => `<option value="${key}">${html(field.label)}</option>`).join('')}</select></label><select id="screenerPrimaryDirection" aria-label="Primary sort direction"><option value="desc">Descending</option><option value="asc">Ascending</option></select>`);
  byId('screenerPrimarySort').onchange = event => { state.sortKey = event.target.value; changed(); };
  byId('screenerPrimaryDirection').onchange = event => { state.sortDirection = event.target.value; changed(); };
  bind(); populateControls(); renderResults();
  if (state.expanded) load();
})();
