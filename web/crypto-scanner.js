(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  document.querySelectorAll('[data-site-theme]').forEach(el => window.atlasTheme?.bind(el));
  $('destination').onchange = e => { if (e.target.value) location.href = e.target.value; };
  let result, controller;
  const short = value => value ? `${value.slice(0, 8)}…${value.slice(-6)}` : 'Unavailable';
  function link(value, kind) { const a = document.createElement('a'); a.textContent = short(value); if (new RegExp(kind === 'tx' ? '^0x[0-9a-fA-F]{64}$' : '^0x[0-9a-fA-F]{40}$').test(value)) { a.href = `https://eth.blockscout.com/${kind}/${value}`; a.target = '_blank'; a.rel = 'noopener noreferrer'; } return a; }
  function renderRows() {
    const records = result.activity.filter(row => ($('feed').value === 'all' || row.feed === $('feed').value) && (!$('matches').checked || row.match));
    $('rows').replaceChildren(); $('count').textContent = `(${records.length} records)`; $('empty').hidden = records.length > 0;
    for (const row of records) {
      const tr = document.createElement('tr');
      for (const value of [new Date(row.timestamp * 1000).toLocaleString(), row.feed === 'internal' ? `Internal ${row.call_type || 'call'}` : row.feed === 'tokens' ? 'Token transfer' : 'Transaction', row.direction, `${row.amount ?? 'Unknown'} ${row.asset}`, link(row.counterpart, 'address'), row.status, link(row.hash, 'tx')]) { const td = document.createElement('td'); value instanceof Node ? td.append(value) : td.textContent = value; tr.append(td); }
      if (row.match) { const small = document.createElement('small'); small.textContent = row.match; tr.children[4].append(small); }
      $('rows').append(tr);
    }
  }
  $('feed').onchange = renderRows; $('matches').onchange = renderRows;
  $('cancel').onclick = () => controller?.abort();
  $('scanForm').onsubmit = async event => {
    event.preventDefault(); controller?.abort(); controller = new AbortController();
    $('results').hidden = true; result = null; $('scan').disabled = true; $('cancel').hidden = false;
    $('message').textContent = 'Scanning three activity feeds and the current balance…';
    try {
      const query = new URLSearchParams({address: $('address').value.trim(), days: $('days').value, limit: $('limit').value});
      const response = await fetch(`/api/crypto/scan?${query}`, {signal: controller.signal});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Scan could not complete.'); result = data;
      const matched = data.activity.filter(row => row.match), completed = matched.filter(row => row.status === 'successful' || row.status === 'recorded'), failed = matched.filter(row => row.status === 'failed');
      $('finding').textContent = data.known_contract ? 'This address is a known Tornado Cash contract' : completed.length ? 'Direct Tornado Cash activity found' : matched.length ? 'Tornado Cash attempts or unconfirmed activity found' : 'No direct Tornado Cash matches in scanned records';
      $('findingDetail').textContent = `${completed.length} successful or recorded matches · ${failed.length} failed attempts · ${matched.length - completed.length - failed.length} unknown-status matches. Incoming transfers do not establish intent. This is a bounded check, not a complete historical or indirect-exposure assessment.`;
      if (data.coverage.every(item => item.state === 'unavailable') && !data.known_contract) {
        $('finding').textContent = 'Activity check unavailable';
        $('findingDetail').textContent = 'The explorer did not return activity feeds. No conclusion about Tornado Cash interactions can be drawn. Please retry later.';
      }
      $('coverage').replaceChildren(); for (const item of data.coverage) { const tag = document.createElement('span'); tag.textContent = `${item.feed}: ${item.state}${item.capped ? ' · record cap reached' : ''} · ${item.count} checked${item.note ? ' — ' + item.note : ''}`; $('coverage').append(tag); }
      $('addressLink').textContent = data.address; $('addressLink').href = `https://eth.blockscout.com/address/${data.address}`;
      $('stats').replaceChildren(); for (const [label, value] of [['Current ETH balance', data.balance ?? 'Unavailable'], ['Activity in selected window', String(data.activity.length)], ['Latest returned activity', data.activity.length ? new Date(data.activity[0].timestamp * 1000).toLocaleString() : 'No records returned']]) { const card = document.createElement('div'); card.className = 'stat'; const small = document.createElement('small'); small.textContent = label; const strong = document.createElement('strong'); strong.textContent = value; card.append(small, strong); $('stats').append(card); }
      $('message').textContent = `Scanned ${new Date(data.scanned_at).toLocaleString()} · ${data.days}-day window · up to ${data.limit} records per feed. ${data.coverage.some(c => c.state !== 'window scanned') ? 'Coverage is limited; review the feed statuses below.' : 'Subject to explorer indexing.'}`;
      $('results').hidden = false; renderRows();
    } catch (error) { $('message').textContent = error.name === 'AbortError' ? 'Scan canceled.' : `Unable to scan: ${error.message}`; }
    finally { $('scan').disabled = false; $('cancel').hidden = true; }
  };
  $('export').onclick = () => { if (!result) return; const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'})); const a = document.createElement('a'); a.href = url; a.download = `ethereum-${result.address}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
})();
