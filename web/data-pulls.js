// One place for on-demand pulls, saved request presets and collection management.
document.addEventListener('DOMContentLoaded', () => {
  const get = id => document.getElementById(id);
  const button = document.createElement('button');
  button.id = 'pullData'; button.textContent = 'Pull data'; button.className = 'secondary';
  document.querySelector('.terminal-actions').prepend(button);
  const dialog = document.createElement('dialog');
  dialog.id = 'pullDialog'; dialog.setAttribute('aria-labelledby', 'pullTitle');
  dialog.innerHTML = `<form id="pullForm"><div class="dialog-head"><h2 id="pullTitle">Pull market data</h2><button type="button" id="pullClose">Close</button></div>
    <p>Requested pulls run before scheduled downloads. An active download finishes first.</p>
    <div class="pull-fields"><label>Symbol<input id="pullSymbol" required maxlength="30"></label>
    <label>Asset type<select id="pullKind"><option value="stocks">Stock / ETF</option><option value="crypto">Crypto (USD)</option></select></label>
    <label>Bar interval<select id="pullInterval"><option value="1m">1 minute</option><option value="5m">5 minutes</option><option value="15m">15 minutes</option><option value="60m" selected>1 hour</option><option value="1d">1 day</option></select></label>
    <label>Range<select id="pullRange"><option value="relative">Last N days</option><option value="custom">Custom dates (UTC)</option></select></label>
    <label id="pullDaysLabel">Days<input id="pullDays" type="number" min="1" max="729" step="1" value="7" required></label>
    <label id="pullStartLabel" hidden>Start date<input id="pullStart" type="date"></label>
    <label id="pullEndLabel" hidden>End date (inclusive)<input id="pullEnd" type="date"></label></div>
    <p id="pullLimit" class="muted"></p><div class="pull-presets"><label>Favorite requests<select id="pullFavorites"></select></label><button type="button" id="pullApply">Apply</button><button type="button" id="pullDelete">Delete favorite</button></div>
    <div class="pull-presets"><label>Favorite name<input id="pullName" maxlength="60" placeholder="Weekly hourly review"></label><button type="button" id="pullSave">Save favorite</button></div>
    <p class="muted">Favorites save the range and interval in this browser and use whichever symbol you choose.</p>
    <p id="pullMessage" role="status" aria-live="polite"></p><div class="dialog-actions"><button type="button" id="pullSchedule">Manage scheduled symbols</button><button type="submit" id="pullSubmit">Queue priority pull</button></div></form>
    <section><div class="dialog-head"><h3>Request queue</h3><button id="pullRefresh">Refresh status</button></div><p>Completed requests are available using the chart’s Refresh control.</p><div id="pullRequests" role="log"></div></section>`;
  document.body.append(dialog);
  const key = 'quantstack.pullFavorites.v1', limits = {'1m':7,'5m':59,'15m':59,'60m':729,'1d':36500};
  let favorites = [{name:'Last 7 days · 1 hour',range:'relative',days:7,interval:'60m'}, {name:'Last 30 days · daily',range:'relative',days:30,interval:'1d'}];
  try { const stored=JSON.parse(localStorage.getItem(key)); if(Array.isArray(stored)) favorites=stored.filter(p=>p && typeof p.name==='string' && limits[p.interval] && ['relative','custom'].includes(p.range)).slice(0,50); } catch {}
  let available = false, busy = false, timer;
  const endpoint = new URL('./api/pull-queue', location.href);
  function fields() { return {range:get('pullRange').value, days:Number(get('pullDays').value), interval:get('pullInterval').value,start:get('pullStart').value,end:get('pullEnd').value}; }
  function rangeUI() {
    const custom=get('pullRange').value==='custom';
    get('pullDaysLabel').hidden=custom;get('pullDays').disabled=custom;
    for(const part of ['Start','End']) {get('pull'+part+'Label').hidden=!custom;get('pull'+part).disabled=!custom;get('pull'+part).required=custom;}
    get('pullDays').max=limits[get('pullInterval').value];
    get('pullLimit').textContent=`Available lookback: up to ${limits[get('pullInterval').value]} days. Only completed bars are stored; provider availability may vary.`;
  }
  function renderFavorites() {
    get('pullFavorites').replaceChildren(...favorites.map((p,i)=>new Option(p.name,String(i))));
    get('pullApply').disabled=get('pullDelete').disabled=!favorites.length;
  }
  function saveFavorites(next) {
    try {localStorage.setItem(key,JSON.stringify(next));favorites=next;renderFavorites();get('pullMessage').textContent='Favorites saved.';}
    catch {get('pullMessage').textContent='Browser storage is unavailable; favorites could not be saved.';}
  }
  async function request(options) {
    const response=await fetch(endpoint,options),data=await response.json().catch(()=>({}));
    if(!response.ok) throw Error(data.error || 'Data pulls are available in the local app. Hosted snapshots are read-only.');
    return data;
  }
  async function refresh() {
    try {
      const data=await request();available=true;
      get('pullRequests').replaceChildren(...data.requests.map(item=>{
        const row=document.createElement('p');
        row.textContent=`#${item.id} · ${item.symbol} · ${item.interval==='60m'?'1h':item.interval} · ${item.start.slice(0,10)} to ${item.end.slice(0,10)} · ${item.status}${item.rows!=null?' · '+item.rows+' bars':''}${item.error?' · '+item.error:''}`;
        return row;
      }));
      if(!data.requests.length)get('pullRequests').textContent='No requests yet.';
    } catch(error) {available=false;get('pullMessage').textContent=error.message;}
    get('pullSubmit').disabled=!available||busy;
    clearTimeout(timer);if(dialog.open)timer=setTimeout(refresh,4000);
  }
  button.onclick=()=>{
    get('pullSymbol').value=selected?.symbol||'';get('pullKind').value=selected?.symbol?.endsWith('-USD')?'crypto':'stocks';
    get('pullMessage').textContent='';get('pullSubmit').disabled=true;dialog.showModal();refresh();
  };
  get('pullClose').onclick=()=>dialog.close();dialog.addEventListener('close',()=>clearTimeout(timer));
  get('pullRange').onchange=get('pullInterval').onchange=rangeUI;
  get('pullRefresh').onclick=refresh;
  get('pullApply').onclick=()=>{const p=favorites[Number(get('pullFavorites').value)];if(!p)return;for(const name of ['range','days','interval','start','end'])get('pull'+name[0].toUpperCase()+name.slice(1)).value=p[name]??'';rangeUI();};
  get('pullSave').onclick=()=>{
    if(!get('pullForm').reportValidity())return;
    const name=get('pullName').value.trim();if(!name){get('pullMessage').textContent='Enter a favorite name.';return;}
    const next=favorites.filter(p=>p.name!==name);if(next.length>=50){get('pullMessage').textContent='Delete a favorite before adding another (50 maximum).';return;}
    saveFavorites([...next,{name,...fields()}]);
  };
  get('pullDelete').onclick=()=>saveFavorites(favorites.filter((_,i)=>i!==Number(get('pullFavorites').value)));
  get('pullSchedule').onclick=()=>{dialog.close();get('schedulerButton').click();};
  get('pullForm').onsubmit=async event=>{
    event.preventDefault();if(busy||!available)return;busy=true;get('pullSubmit').disabled=true;
    try {const data=await request({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...fields(),symbol:get('pullSymbol').value,kind:get('pullKind').value})});get('pullMessage').textContent=`Request #${data.id} queued ahead of scheduled pulls.`;}
    catch(error){get('pullMessage').textContent=error.message;}
    finally {busy=false;await refresh();}
  };
  rangeUI();renderFavorites();
});
