// Deterministic, long-only backtests. Signals observe closes; orders fill next open.
(() => {
  'use strict';
  const defaults = {kind: 'sma', fast: 10, slow: 30, lookback: 14, lower: 30, upper: 70, capital: 10000, allocation: 100, commission: 0.1, slippage: 0.05, stop: 0, target: 0, adjusted: true};
  const catalog = {
    sma: {label: 'SMA trend', family: 'Trend', controls: ['ma'], description: 'Enter when fast SMA is above slow SMA; exit when it is below.'},
    ema: {label: 'EMA trend', family: 'Trend', controls: ['ma'], description: 'Enter when fast EMA is above slow EMA; exit when it is below.'},
    sma_cross: {label: 'SMA crossover', family: 'Trend', controls: ['ma'], description: 'Enter when fast SMA crosses above slow SMA; exit on a cross below.'},
    ema_cross: {label: 'EMA crossover', family: 'Trend', controls: ['ma'], description: 'Enter when fast EMA crosses above slow EMA; exit on a cross below.'},
    triple_ema: {label: 'Triple EMA alignment', family: 'Trend', controls: [], description: 'Enter when EMA 5 > EMA 20 > EMA 50; exit when either pair reverses. Customize to change the three periods.'},
    golden_cross: {label: 'Golden / death cross', family: 'Trend', controls: [], description: 'Enter when SMA 50 crosses above SMA 200; exit on a cross below. Requires at least 203 loaded bars. Customize to change periods.'},
    price_ema: {label: 'Price / EMA crossover', family: 'Trend', controls: ['lookback'], description: 'Enter when close crosses above the lookback EMA; exit when close crosses below it.'},
    rsi: {label: 'RSI reversion', family: 'Reversion', controls: ['lookback', 'rsi'], description: 'Enter when Wilder RSI is below the lower threshold; exit above the upper threshold.'},
    rsi_recovery: {label: 'RSI recovery', family: 'Reversion', controls: ['lookback', 'rsi'], description: 'Enter when RSI crosses above the lower threshold; exit when it crosses below the upper threshold.'},
    bollinger: {label: 'Bollinger reversion', family: 'Reversion', controls: ['lookback'], description: 'Enter below the lower Bollinger Band (two population standard deviations); exit above its middle SMA.'},
    stochastic: {label: 'Stochastic recovery', family: 'Reversion', controls: ['lookback'], description: 'Enter when unsmoothed stochastic %K crosses above 20; exit when it crosses below 80. Customize to edit thresholds.'},
    williams: {label: 'Williams %R recovery', family: 'Reversion', controls: ['lookback'], description: 'Enter when Williams %R crosses above -80; exit when it crosses below -20. Customize to edit thresholds.'},
    cci: {label: 'CCI recovery', family: 'Reversion', controls: ['lookback'], description: 'Enter when CCI crosses above -100; exit when it crosses below +100. Uses typical price and mean absolute deviation.'},
    macd: {label: 'MACD crossover', family: 'Momentum', controls: [], description: 'Enter when MACD (12/26 EMA) crosses above its 9-bar EMA signal; exit on a cross below.'},
    macd_zero: {label: 'MACD zero-line cross', family: 'Momentum', controls: [], description: 'Enter when MACD (12/26 EMA) crosses above zero; exit when it crosses below zero.'},
    rsi_momentum: {label: 'RSI momentum', family: 'Momentum', controls: ['lookback', 'rsi'], description: 'Enter when RSI is above the upper threshold; exit when it is below the lower threshold.'},
    roc: {label: 'Rate-of-change momentum', family: 'Momentum', controls: ['lookback'], description: 'Enter when percent rate of change crosses above zero; exit on a cross below. The lookback is measured in bars.'},
    breakout: {label: 'Channel breakout', family: 'Breakout', controls: ['lookback'], description: 'Enter above the prior channel high; exit below the prior channel low. The current bar is excluded from both levels.'},
    channel_mid: {label: 'Channel breakout / midpoint exit', family: 'Breakout', controls: ['lookback'], description: 'Enter above the prior channel high; exit below its midpoint. Channel levels exclude the current bar.'},
    bollinger_breakout: {label: 'Bollinger breakout', family: 'Breakout', controls: ['lookback'], description: 'Enter when close crosses above the upper Bollinger Band; exit when close falls below its middle SMA.'}
  };
  const operands = {close: 'Close', open: 'Open', high: 'High', low: 'Low', sma: 'SMA', ema: 'EMA', rsi: 'RSI', macd: 'MACD (12/26)', macdSignal: 'MACD signal (9)', bbUpper: 'Bollinger upper (2σ)', bbLower: 'Bollinger lower (2σ)', channelHigh: 'Prior channel high', channelLow: 'Prior channel low', constant: 'Number'};
  const operators = {gt: 'Above', lt: 'Below', gte: 'At or above', lte: 'At or below', crossUp: 'Crosses above', crossDown: 'Crosses below'};
  const periodTypes = ['sma', 'ema', 'rsi', 'bbUpper', 'bbLower', 'channelHigh', 'channelLow'];
  Object.assign(operands, {roc: 'Rate of change %', stochastic: 'Fast stochastic %K', williams: 'Williams %R', cci: 'CCI', channelMid: 'Prior channel midpoint'});
  periodTypes.push('roc', 'stochastic', 'williams', 'cci', 'channelMid');
  const operand = (type, period = 14) => ({type, period});
  const rule = (left, operator, right) => ({left, operator, right});
  function rulesFor(p) {
    if (p.kind === 'custom') return structuredClone(p.conditions);
    let entry, exit;
    if (['sma', 'ema', 'sma_cross', 'ema_cross'].includes(p.kind)) {
      const type = p.kind.startsWith('sma') ? 'sma' : 'ema', crossing = p.kind.endsWith('_cross');
      entry = rule(operand(type, p.fast), crossing ? 'crossUp' : 'gt', operand(type, p.slow));
      exit = rule(operand(type, p.fast), crossing ? 'crossDown' : 'lt', operand(type, p.slow));
    } else if (p.kind === 'rsi') {
      entry = rule(operand('rsi', p.lookback), 'lt', {type: 'constant', value: p.lower});
      exit = rule(operand('rsi', p.lookback), 'gt', {type: 'constant', value: p.upper});
    } else if (p.kind === 'breakout') {
      entry = rule(operand('close'), 'gt', operand('channelHigh', p.lookback));
      exit = rule(operand('close'), 'lt', operand('channelLow', p.lookback));
    } else if (p.kind === 'macd') {
      entry = rule(operand('macd'), 'crossUp', operand('macdSignal'));
      exit = rule(operand('macd'), 'crossDown', operand('macdSignal'));
    } else if (p.kind === 'bollinger') {
      entry = rule(operand('close'), 'lt', operand('bbLower', p.lookback));
      exit = rule(operand('close'), 'gt', operand('sma', p.lookback));
    } else if (p.kind === 'triple_ema') {
      return {entry: {mode: 'all', rules: [rule(operand('ema', 5), 'gt', operand('ema', 20)), rule(operand('ema', 20), 'gt', operand('ema', 50))]}, exit: {mode: 'any', rules: [rule(operand('ema', 5), 'lt', operand('ema', 20)), rule(operand('ema', 20), 'lt', operand('ema', 50))]}};
    } else if (p.kind === 'golden_cross') {
      entry = rule(operand('sma', 50), 'crossUp', operand('sma', 200));
      exit = rule(operand('sma', 50), 'crossDown', operand('sma', 200));
    } else if (p.kind === 'price_ema') {
      entry = rule(operand('close'), 'crossUp', operand('ema', p.lookback));
      exit = rule(operand('close'), 'crossDown', operand('ema', p.lookback));
    } else if (p.kind === 'rsi_recovery' || p.kind === 'rsi_momentum') {
      const recovery = p.kind === 'rsi_recovery';
      entry = rule(operand('rsi', p.lookback), recovery ? 'crossUp' : 'gt', {type: 'constant', value: recovery ? p.lower : p.upper});
      exit = rule(operand('rsi', p.lookback), recovery ? 'crossDown' : 'lt', {type: 'constant', value: recovery ? p.upper : p.lower});
    } else if (['stochastic', 'williams', 'cci', 'roc', 'macd_zero'].includes(p.kind)) {
      const type = p.kind === 'macd_zero' ? 'macd' : p.kind;
      const levels = {stochastic: [20, 80], williams: [-80, -20], cci: [-100, 100], roc: [0, 0], macd: [0, 0]}[type];
      entry = rule(operand(type, p.lookback), 'crossUp', {type: 'constant', value: levels[0]});
      exit = rule(operand(type, p.lookback), 'crossDown', {type: 'constant', value: levels[1]});
    } else if (p.kind === 'channel_mid') {
      entry = rule(operand('close'), 'gt', operand('channelHigh', p.lookback));
      exit = rule(operand('close'), 'lt', operand('channelMid', p.lookback));
    } else if (p.kind === 'bollinger_breakout') {
      entry = rule(operand('close'), 'crossUp', operand('bbUpper', p.lookback));
      exit = rule(operand('close'), 'lt', operand('sma', p.lookback));
    } else throw Error('Choose a supported strategy.');
    return {entry: {mode: 'all', rules: [entry]}, exit: {mode: 'any', rules: [exit]}};
  }
  function run(source, options = {}) {
    const p = {...defaults, ...options};
    const controls = catalog[p.kind]?.controls || [], ma = controls.includes('ma'), hasRsi = controls.includes('rsi');
    const periods = ma ? ['fast', 'slow'] : controls.includes('lookback') ? ['lookback'] : [];
    for (const key of periods) if (!Number.isInteger(p[key]) || p[key] < 2 || p[key] > 500) throw Error('Periods must be whole numbers from 2 to 500.');
    if (ma && p.fast >= p.slow) throw Error('The fast period must be smaller than the slow period.');
    for (const key of ['capital', 'allocation', 'commission', 'slippage', 'stop', 'target', ...(hasRsi ? ['lower', 'upper'] : [])]) if (!Number.isFinite(p[key])) throw Error('Enter finite numbers for all settings.');
    if (p.capital <= 0 || p.capital > 1e12 || p.allocation <= 0 || p.allocation > 100) throw Error('Capital must be positive (up to 1 trillion); position size must be 0–100%.');
    if (p.commission < 0 || p.commission > 10 || p.slippage < 0 || p.slippage > 10) throw Error('Fees and slippage must be between 0% and 10%.');
    if (p.stop < 0 || p.stop >= 100 || p.target < 0 || p.target > 1000) throw Error('Stop must be 0–99.99%; target must be 0–1000%.');
    if (hasRsi && (p.lower < 0 || p.upper > 100 || p.lower >= p.upper)) throw Error('RSI lower threshold must be below the upper threshold, within 0–100.');
    const conditions = rulesFor(p);
    for (const side of ['entry', 'exit']) {
      const group = conditions?.[side];
      if (!group || !['all', 'any'].includes(group.mode) || !Array.isArray(group.rules) || !group.rules.length || group.rules.length > 12) throw Error(`Set 1–12 ${side} conditions and choose All or Any.`);
      for (const item of group.rules) {
        if (!item || !Object.hasOwn(operators, item.operator)) throw Error(`Choose a valid ${side} comparison.`);
        for (const value of [item.left, item.right]) {
          if (!value || !Object.hasOwn(operands, value.type)) throw Error(`Choose valid ${side} indicators.`);
          if (value.type === 'constant' && !Number.isFinite(value.value)) throw Error(`Enter a finite number for each ${side} threshold.`);
          if (periodTypes.includes(value.type) && (!Number.isInteger(value.period) || value.period < 2 || value.period > 500)) throw Error('Indicator periods must be whole numbers from 2 to 500.');
        }
      }
    }
    p.conditions = conditions;
    const bars = source.map((row, i) => {
      if (!row.date || !Number.isFinite(Date.parse(row.date)) || (i && Date.parse(row.date) <= Date.parse(source[i - 1].date))) throw Error('Bars must have unique dates in ascending order.');
      if (!['open', 'high', 'low', 'close'].every(k => Number.isFinite(row[k]) && row[k] > 0) || row.high < Math.max(row.open, row.close, row.low) || row.low > Math.min(row.open, row.close)) throw Error('The loaded range contains incomplete or invalid OHLC bars. Choose another range.');
      if (p.adjusted && (!Number.isFinite(row.adjusted_close) || row.adjusted_close <= 0)) throw Error('Adjusted prices are missing. Use raw prices or choose another range.');
      const ratio = p.adjusted ? row.adjusted_close / row.close : 1;
      const adjusted = {date: row.date, open: row.open * ratio, high: row.high * ratio, low: row.low * ratio, close: row.close * ratio};
      if (!['open', 'high', 'low', 'close'].every(key => Number.isFinite(adjusted[key]) && adjusted[key] > 0)) throw Error('Adjusted OHLC values are outside the supported numeric range.');
      return adjusted;
    });
    const firstValid = o => o.type === 'macd' ? 25 : o.type === 'macdSignal' ? 33 : ['rsi', 'roc', 'channelHigh', 'channelLow', 'channelMid'].includes(o.type) ? o.period : periodTypes.includes(o.type) ? o.period - 1 : 0;
    const warmup = 1 + Math.max(...['entry', 'exit'].flatMap(side => conditions[side].rules.map(r => Math.max(firstValid(r.left), firstValid(r.right)) + (r.operator.startsWith('cross') ? 1 : 0))));
    if (bars.length < warmup + 2) throw Error(`Load at least ${warmup + 2} bars for these periods. Try a longer chart range.`);
    const closes = bars.map(b => b.close);
    function average(values, n, exponential = false) {
      let value = null, window = [], sum = 0;
      return values.map(v => {
        if (!Number.isFinite(v)) { value = null; window = []; sum = 0; return null; }
        window.push(v); sum += v; if (window.length > n) sum -= window.shift();
        if (window.length < n) return null;
        value = !exponential || value == null ? sum / n : value + 2 / (n + 1) * (v - value);
        return value;
      });
    }
    const cache = new Map();
    function series(o) {
      const key = JSON.stringify(o); if (cache.has(key)) return cache.get(key);
      let values;
      if (o.type === 'constant') values = bars.map(() => o.value);
      else if (['open', 'high', 'low', 'close'].includes(o.type)) values = bars.map(b => b[o.type]);
      else if (['sma', 'ema'].includes(o.type)) values = average(closes, o.period, o.type === 'ema');
      else if (o.type === 'rsi') {
        let gain = 0, loss = 0;
        values = closes.map((v, i) => {
          if (!i) return null;
          const delta = v - closes[i - 1], up = Math.max(0, delta), down = Math.max(0, -delta), n = o.period;
          if (i <= n) { gain += up / n; loss += down / n; }
          else { gain = (gain * (n - 1) + up) / n; loss = (loss * (n - 1) + down) / n; }
          return i < n ? null : loss === 0 ? gain === 0 ? 50 : 100 : 100 - 100 / (1 + gain / loss);
        });
      } else if (o.type === 'macd') {
        const fast = average(closes, 12, true), slow = average(closes, 26, true);
        values = fast.map((v, i) => slow[i] == null ? null : v - slow[i]);
      } else if (o.type === 'macdSignal') values = average(series({type: 'macd'}), 9, true);
      else if (o.type.startsWith('bb')) {
        const means = average(closes, o.period);
        values = means.map((mean, i) => mean == null ? null : mean + (o.type === 'bbUpper' ? 2 : -2) * Math.sqrt(closes.slice(i - o.period + 1, i + 1).reduce((sum, v) => sum + (v - mean) ** 2, 0) / o.period));
      } else if (o.type === 'roc') values = closes.map((v, i) => i < o.period ? null : (v / closes[i - o.period] - 1) * 100);
      else if (o.type === 'stochastic' || o.type === 'williams') {
        values = bars.map((b, i) => {
          if (i < o.period - 1) return null;
          const window = bars.slice(i - o.period + 1, i + 1), high = Math.max(...window.map(r => r.high)), low = Math.min(...window.map(r => r.low));
          if (high === low) return null;
          return o.type === 'stochastic' ? 100 * (b.close - low) / (high - low) : -100 * (high - b.close) / (high - low);
        });
      } else if (o.type === 'cci') {
        const typical = bars.map(b => (b.high + b.low + b.close) / 3), means = average(typical, o.period);
        values = means.map((mean, i) => {
          if (mean == null) return null;
          const deviation = typical.slice(i - o.period + 1, i + 1).reduce((sum, v) => sum + Math.abs(v - mean), 0) / o.period;
          return deviation === 0 ? 0 : (typical[i] - mean) / (.015 * deviation);
        });
      } else values = bars.map((_, i) => {
        if (i < o.period) return null;
        const window = bars.slice(i - o.period, i), high = Math.max(...window.map(b => b.high)), low = Math.min(...window.map(b => b.low));
        return o.type === 'channelHigh' ? high : o.type === 'channelLow' ? low : (high + low) / 2;
      });
      cache.set(key, values); return values;
    }
    const compiled = Object.fromEntries(['entry', 'exit'].map(side => [side, conditions[side].rules.map(r => ({...r, left: series(r.left), right: series(r.right)}))]));
    function matches(side, i) {
      const checks = compiled[side].map(r => {
        const a = r.left[i], b = r.right[i];
        if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
        if (r.operator === 'gt') return a > b;
        if (r.operator === 'lt') return a < b;
        if (r.operator === 'gte') return a >= b;
        if (r.operator === 'lte') return a <= b;
        const prevA = r.left[i - 1], prevB = r.right[i - 1];
        if (!Number.isFinite(prevA) || !Number.isFinite(prevB)) return false;
        return r.operator === 'crossUp' ? a > b && prevA <= prevB : a < b && prevA >= prevB;
      });
      return conditions[side].mode === 'all' ? checks.every(Boolean) : checks.some(Boolean);
    }
    const fee = p.commission / 100, slip = p.slippage / 100;
    let cash = p.capital, position = null, pending = null, fees = 0, exposed = 0, peak = p.capital, maxDrawdown = 0;
    const trades = [], curve = [];
    function closePosition(rawPrice, bar, index, reason) {
      const price = rawPrice * (1 - slip), proceeds = position.quantity * price, exitFee = proceeds * fee;
      cash += proceeds - exitFee; fees += exitFee;
      const pnl = proceeds - exitFee - position.cost;
      trades.push({entry: position.date, exit: bar.date, entryPrice: position.price, exitPrice: price, quantity: position.quantity, pnl, returnPct: pnl / position.cost * 100, fees: position.fee + exitFee, bars: index - position.index + 1, reason});
      position = null;
    }
    // Both portfolios start at the first executable open after indicator warmup.
    const start = warmup, benchmarkPrice = bars[start].open * (1 + slip);
    const benchmarkQuantity = p.capital / (benchmarkPrice * (1 + fee));
    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      if (i >= start) {
        if (position && pending === 'exit') closePosition(bar.open, bar, i, 'Signal');
        if (!position && pending === 'enter') {
          const budget = cash * p.allocation / 100, price = bar.open * (1 + slip), quantity = budget / (price * (1 + fee)), entryFee = quantity * price * fee;
          cash -= budget; fees += entryFee;
          position = {date: bar.date, index: i, price, quantity, fee: entryFee, cost: budget};
        }
        if (position) {
          exposed++;
          const stop = p.stop ? position.price * (1 - p.stop / 100) : null;
          const target = p.target ? position.price * (1 + p.target / 100) : null;
          // Opening gaps are observed first. For ambiguous intrabar touches, stop wins.
          if (stop && bar.open <= stop) closePosition(bar.open, bar, i, 'Stop gap');
          else if (target && bar.open >= target) closePosition(bar.open, bar, i, 'Target gap');
          else if (stop && bar.low <= stop) closePosition(stop, bar, i, 'Stop loss');
          else if (target && bar.high >= target) closePosition(target, bar, i, 'Take profit');
        }
        if (i === bars.length - 1 && position) closePosition(bar.close, bar, i, 'End of test');
        const equity = cash + (position ? position.quantity * bar.close : 0);
        peak = Math.max(peak, equity); const drawdown = (equity / peak - 1) * 100;
        maxDrawdown = Math.max(maxDrawdown, -drawdown);
        const benchmark = benchmarkQuantity * bar.close * (i === bars.length - 1 ? (1 - slip) * (1 - fee) : 1);
        curve.push({date: bar.date, equity, benchmark, drawdown});
      }
      pending = null;
      if (i < warmup - 1 || i === bars.length - 1) continue;
      if (position) { if (matches('exit', i)) pending = 'exit'; }
      else if (matches('entry', i)) pending = 'enter';
    }
    const grossProfit = trades.reduce((v, t) => v + Math.max(0, t.pnl), 0), grossLoss = -trades.reduce((v, t) => v + Math.min(0, t.pnl), 0);
    const winners = trades.filter(t => t.pnl > 0), losers = trades.filter(t => t.pnl < 0);
    const performance = {
      grossProfit, grossLoss, winners: winners.length, losers: losers.length,
      breakeven: trades.length - winners.length - losers.length,
      averageWin: winners.length ? grossProfit / winners.length : null,
      averageLoss: losers.length ? -grossLoss / losers.length : null,
      averageTrade: trades.length ? (cash - p.capital) / trades.length : null,
      bestTrade: trades.length ? trades.reduce((best, t) => Math.max(best, t.pnl), -Infinity) : null,
      worstTrade: trades.length ? trades.reduce((worst, t) => Math.min(worst, t.pnl), Infinity) : null,
      averageBars: trades.length ? trades.reduce((sum, t) => sum + t.bars, 0) / trades.length : null
    };
    // Each observed month compounds from the previous observed month's close.
    // The first period starts at initial capital and includes entry costs.
    const monthly = [];
    let previousEquity = p.capital, previousBenchmark = p.capital;
    for (let i = 0; i < curve.length; i++) {
      const row = curve[i], month = row.date.slice(0, 7);
      if (i + 1 < curve.length && curve[i + 1].date.slice(0, 7) === month) continue;
      monthly.push({month, equity: row.equity, returnPct: (row.equity / previousEquity - 1) * 100, benchmarkPct: (row.benchmark / previousBenchmark - 1) * 100});
      previousEquity = row.equity; previousBenchmark = row.benchmark;
    }
    return {parameters: p, curve, trades, performance, monthly, bars: bars.length, warmup: start, start: bars[start].date, end: bars.at(-1).date, finalEquity: cash, netProfit: cash - p.capital, returnPct: (cash / p.capital - 1) * 100, benchmarkPct: (curve.at(-1).benchmark / p.capital - 1) * 100, maxDrawdown, fees, exposurePct: exposed / curve.length * 100, winRate: trades.length ? trades.filter(t => t.pnl > 0).length / trades.length * 100 : null, profitFactor: grossLoss ? grossProfit / grossLoss : grossProfit ? Infinity : null};
  }
  window.AtlasBacktest = {run, defaults, rulesFor, operands, operators, periodTypes, catalog};
})();
