// Static-host adapter. Loaded only by the generated hosted site.
window.ATLAS_STATIC = true;
const historyCache = new Map();
let unavailableHistory = new Set();
const snapshotResource = path => window.ATLAS_DATA_BASE ? new URL(path, new URL(window.ATLAS_DATA_BASE, document.baseURI)).href : path;
function aggregateBars(rows, interval) {
    const grouped = new Map();
    for (const row of rows) {
        const day = new Date(`${row.date}T00:00:00Z`);
        let key, date;
        if (interval === '1w') {
            day.setUTCDate(day.getUTCDate() - ((day.getUTCDay() + 6) % 7));
            date = day.toISOString().slice(0, 10);
            key = date;
        } else {
            key = row.date.slice(0, 7);
            date = `${key}-01`;
        }
        const bar = grouped.get(key);
        if (!bar) grouped.set(key, {...row, date});
        else {
            bar.high = Math.max(bar.high, row.high);
            bar.low = Math.min(bar.low, row.low);
            bar.close = row.close;
            bar.adjusted_close = row.adjusted_close;
            bar.volume += row.volume;
            bar.dividends = (bar.dividends || 0) + (row.dividends || 0);
            if (row.splits) bar.splits = bar.splits ? bar.splits * row.splits : row.splits;
        }
    }
    return [...grouped.values()];
}
window.atlasApi = async function(url) {
    const route = new URL(url, 'http://local');
    if (route.pathname === '/api/symbols') {
        const response = await fetch(snapshotResource('./symbols.json'), {cache:'no-cache'});
        if (!response.ok) throw Error('The asset catalog is unavailable. Please try refreshing.');
        historyCache.clear();
        const assets = await response.json();
        unavailableHistory = new Set(assets.filter(asset => asset.has_data === false).map(asset => asset.symbol));
        return assets;
    }
    if (route.pathname === '/api/screener') {
        const response = await fetch(snapshotResource('./screener.json'), {cache:'no-cache'});
        if (!response.ok) throw Error('The stock screener snapshot is unavailable.');
        return response.json();
    }
    const symbol = route.searchParams.get('symbol');
    const interval = route.searchParams.get('interval') || '1d';
    const start = route.searchParams.get('start') || '0001-01-01';
    const end = route.searchParams.get('end') || '9999-12-31';
    if (start > end) throw Error('Start date must be before the end date.');
    if (unavailableHistory.has(symbol)) return [];
    if (!historyCache.has(symbol)) {
        const response = await fetch(snapshotResource('./prices/' + encodeURIComponent(symbol) + '.json.gz'));
        if (!response.ok) throw Error('Price history is unavailable for this asset.');
        const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
        const data = await new Response(stream).json();
        const records = data.rows.map(row=>Object.fromEntries(data.columns.map((key,i)=>[key,row[i]])));
        // Limit browser memory while moving through a large universe.
        if (historyCache.size >= 5) historyCache.delete(historyCache.keys().next().value);
        historyCache.set(symbol, records);
    }
    const rows = historyCache.get(symbol).filter(row=>row.date>=start && row.date<=end);
    return interval === '1w' || interval === '1mo' ? aggregateBars(rows, interval) : rows;
};
