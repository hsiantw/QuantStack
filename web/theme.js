// Apply before first paint; UI is installed once the workspace is ready.
(() => {
  const key = 'atlas.theme', system = matchMedia('(prefers-color-scheme: dark)');
  const palettes = {
    light: {name:'Light',dark:false,surface:'#ffffff',raised:'#f7f9fc',hover:'#edf2ff',grid:'#e9edf2',text:'#20242c',muted:'#637086',accent:'#2962ff',border:'#e3e5e9'},
    dark: {name:'Dark',dark:true,surface:'#131722',raised:'#1b2230',hover:'#24334e',grid:'#282f3e',text:'#dce2ed',muted:'#9aa7bd',accent:'#7299ff',border:'#2b3445'},
    midnight: {name:'Midnight',dark:true,surface:'#0b1020',raised:'#141c32',hover:'#232c4b',grid:'#202b44',text:'#e2e8ff',muted:'#9aa9ce',accent:'#a597ff',border:'#29344f'},
    forest: {name:'Forest',dark:true,surface:'#101d1b',raised:'#172925',hover:'#243e35',grid:'#284039',text:'#e1eee7',muted:'#9db5a9',accent:'#6ed6aa',border:'#2b433b'},
    paper: {name:'Paper',dark:false,surface:'#faf6ed',raised:'#f1ebdf',hover:'#e9e0ce',grid:'#e6decd',text:'#3b352a',muted:'#746857',accent:'#966229',border:'#ddd3bf'}
  };
  let preference = 'system';
  try { preference = localStorage.getItem(key) || 'system'; } catch {}
  function apply(value, persist = false) {
    preference = (value === 'system' || Object.hasOwn(palettes, value)) ? value : 'system';
    const resolved = preference === 'system' ? system.matches ? 'dark' : 'light' : preference;
    const palette = palettes[resolved], root = document.documentElement;
    root.dataset.theme = palette.dark ? 'dark' : 'light';
    root.dataset.palette = resolved;
    root.style.colorScheme = palette.dark ? 'dark' : 'light';
    root.dataset.prepTheme = palette.dark ? 'dark' : 'light';
    let browserColor = document.querySelector('meta[name="theme-color"]');
    if (!browserColor) { browserColor = document.createElement('meta'); browserColor.name = 'theme-color'; document.head.append(browserColor); }
    browserColor.content = palette.surface;
    const values = {surface:palette.surface,'surface-raised':palette.raised,'surface-hover':palette.hover,text:palette.text,'text-muted':palette.muted,border:palette.border,accent:palette.accent,
      'atlas-surface':palette.surface,'atlas-raised':palette.raised,'atlas-hover':palette.hover,'atlas-text':palette.text,'atlas-muted':palette.muted,'atlas-border':palette.border,'atlas-accent':palette.accent,
      ink:palette.text,muted:palette.muted,line:palette.border,bg:palette.surface,
      panel:palette.raised,raised:palette.raised,page:palette.raised,heading:palette.text,
      amber:palette.accent,focus:palette.accent,'on-accent':palette.dark?'#101722':'#ffffff'};
    for (const [name,color] of Object.entries(values)) root.style.setProperty('--'+name,color);
    window.atlasTheme = {palette: palettes[resolved], palettes, preference, apply, bind};
    document.querySelectorAll('[data-site-theme], #workspaceTheme, #toolTheme').forEach(control => { control.value = preference; });
    if (persist) try { localStorage.setItem(key, preference); } catch {}
    if (typeof draw === 'function') requestAnimationFrame(() => draw());
    window.dispatchEvent(new Event('atlas-theme-change'));
  }
  function bind(control) {
    if (!control) return;
    control.dataset.siteTheme = '';
    control.title = 'Applies to every QuantStack page in this browser';
    control.innerHTML = '<option value="system">System</option>' + Object.entries(palettes).map(([id,p]) => `<option value="${id}">${p.name}</option>`).join('');
    control.value = preference;
    control.onchange = event => apply(event.target.value, true);
  }
  apply(preference);
  system.addEventListener('change', () => { if (preference === 'system') apply(preference); });
  window.addEventListener('storage', event => { if (event.key === key || event.key === null) apply(event.newValue || 'system'); });
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-site-theme]').forEach(bind);
    const dock = document.getElementById('workspaceSettingsDock');
    if (!dock) return;
    const label = document.createElement('label'); label.className = 'workspace-theme';
    label.innerHTML = '<span aria-hidden="true">◐</span><select id="workspaceTheme" aria-label="Workspace theme"><option value="system">System theme</option><option value="light">Light</option><option value="dark">Dark</option><option value="midnight">Midnight</option><option value="forest">Forest</option><option value="paper">Paper</option></select>';
    dock.append(label);
    document.getElementById('workspaceTheme').value = preference;
    document.getElementById('workspaceTheme').onchange = event => apply(event.target.value, true);
  });
})();

// One shared pair of lights follows the active scroll area, including wide tables.
// Event delegation avoids observing or scanning the thousands of screener rows.
(() => {
  function installScrollLights() {
    const root = document.documentElement;
    const vertical = document.createElement('div'), horizontal = document.createElement('div');
    for (const light of [vertical, horizontal]) {
      light.className = 'scroll-light'; light.setAttribute('aria-hidden', 'true');
      document.body.append(light);
    }
    vertical.dataset.scrollLight = 'vertical'; horizontal.dataset.scrollLight = 'horizontal';
    root.classList.add('scroll-light-ready');
    let active = document.scrollingElement, frame = 0, idle = 0, scrolling = false;
    function draw() {
      frame = 0;
      if (!active?.isConnected) active = document.scrollingElement;
      const page = active === document.scrollingElement;
      const rect = page ? {left:0, top:0, right:innerWidth, bottom:innerHeight} : active.getBoundingClientRect();
      const left = Math.max(0, rect.left), top = Math.max(0, rect.top);
      const right = Math.min(innerWidth, rect.right), bottom = Math.min(innerHeight, rect.bottom);
      const width = Math.max(0, right-left), height = Math.max(0, bottom-top);
      const extentY = active.scrollHeight - active.clientHeight, extentX = active.scrollWidth - active.clientWidth;
      const showY = extentY > 1 && height > 24, showX = extentX > 1 && width > 24;
      const lengthY = Math.min(110, height * .3), lengthX = Math.min(110, width * .3);
      const y = top + (height-lengthY) * Math.max(0, Math.min(1, active.scrollTop / (extentY || 1)));
      const x = left + (width-lengthX) * Math.max(0, Math.min(1, Math.abs(active.scrollLeft) / (extentX || 1)));
      Object.assign(vertical.style, {left:`${right-13}px`, top:`${y}px`, width:'18px', height:`${lengthY}px`});
      Object.assign(horizontal.style, {left:`${x}px`, top:`${bottom-13}px`, width:`${lengthX}px`, height:'18px'});
      vertical.classList.toggle('is-visible', showY); horizontal.classList.toggle('is-visible', showX);
      for (const light of [vertical,horizontal]) light.classList.toggle('is-scrolling', scrolling);
    }
    function schedule() { if (!frame) frame = requestAnimationFrame(draw); }
    document.addEventListener('scroll', event => {
      const target = event.target === document ? document.scrollingElement : event.target;
      if (!(target instanceof Element)) return;
      active = target; scrolling = true; clearTimeout(idle); schedule();
      idle = setTimeout(() => { scrolling = false; active = document.scrollingElement; schedule(); }, 900);
    }, {capture:true, passive:true});
    window.addEventListener('resize', schedule, {passive:true});
    window.addEventListener('load', schedule, {once:true});
    // Observe only page geometry, not every content mutation or individual row.
    if (window.ResizeObserver) new ResizeObserver(schedule).observe(document.body);
    schedule();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', installScrollLights, {once:true});
  else installScrollLights();
})();

// Shared destination search: local navigation only, no extra market-data requests.
document.addEventListener('DOMContentLoaded', () => {
  const destinations = [
    ['Home','/','Start and explore'],['Charts','/workspace/','Symbols, drawings and indicators'],
    ['Stock screener','/screener','Filters and saved screens'],['Market overview','/markets','Market movers and watchlists'],
    ['Sector rotation','/rotation','Sector momentum'],['Company fundamentals','/fundamentals','Company research'],
    ['Strategy tester','/strategy','Backtesting'],['Strategy comparison','/compare','Compare results'],
    ['Portfolio','/portfolio','Holdings and allocation'],['Returns & risk','/risk','Drawdowns and returns'],
    ['Options lab','/options','Pricing and payoffs'],['Pairs analysis','/pairs','Spreads and correlations'],
    ['Forecast lab','/forecast','Forecast evaluation'],['Markov analysis','/markov','Market regimes'],
    ['Monte Carlo simulations','/brownian','Possible price paths'],['Volume & liquidity','/liquidity','Participation and turnover'],
    ['Price history','/data','Stored prices'],['Crypto address scanner','/crypto-scanner','Ethereum and Tornado Cash checks'],
    ['Idea notebook','/ideas','Ideas and notes'],['Research journal','/journal','Symbol research'],
    ['Interview prep','/workspace/interview-prep.html','Buyer Fraud practice'],['Explore the system','/workspace/architecture.html','Project architecture'],
    ['Account','/account','Preferences and saved work']
  ];
  const dialog = document.createElement('dialog'); dialog.id = 'siteNavigator';
  dialog.setAttribute('aria-labelledby','siteNavigatorTitle');
  dialog.innerHTML = '<div class="navigator-heading"><h2 id="siteNavigatorTitle">Go to</h2><button type="button" id="navigatorClose" aria-label="Close navigation">Esc</button></div><label class="navigator-search">Find a page or tool<input id="navigatorQuery" type="search" autocomplete="off" placeholder="Search charts, screener, research…"></label><div id="navigatorResults" class="navigator-results"></div><p id="navigatorCount" role="status"></p><footer>↑ ↓ to browse · Enter to open · Esc to close</footer>';
  document.body.append(dialog);
  const query=dialog.querySelector('input'), results=dialog.querySelector('#navigatorResults');
  let opener;
  function render() {
    results.replaceChildren();
    const matches=destinations.filter(row=>row.join(' ').toLowerCase().includes(query.value.trim().toLowerCase()));
    for (const [name,path,description] of matches) {
      const link=document.createElement('a');link.href=path;
      const label=document.createElement('strong');label.textContent=name;
      const detail=document.createElement('span');detail.textContent=description;
      link.append(label,detail);
      if(location.pathname.replace(/\/$/,'')===path.replace(/\/$/,'')){link.setAttribute('aria-current','page');}
      results.append(link);
    }
    dialog.querySelector('#navigatorCount').textContent=matches.length?`${matches.length} destinations`:'No matching tools. Try “charts” or “research”.';
  }
  function open() {
    if(document.querySelector('dialog[open]')) return;
    opener=document.activeElement; query.value='';render();dialog.showModal();query.focus();
  }
  dialog.querySelector('#navigatorClose').onclick=()=>dialog.close();
  dialog.addEventListener('close',()=>opener?.isConnected&&opener.focus());
  dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
  query.oninput=render;
  dialog.addEventListener('keydown',event=>{
    if(event.key==='Escape'){event.stopPropagation();return;}
    const links=[...results.querySelectorAll('a')], index=links.indexOf(document.activeElement);
    if(event.key==='ArrowDown'||event.key==='ArrowUp'){
      event.preventDefault();if(!links.length)return;
      links[(index+(event.key==='ArrowDown'?1:-1)+links.length)%links.length].focus();
    }else if(event.key==='Enter'&&document.activeElement===query){event.preventDefault();links[0]?.click();}
  });
  const chartWorkspace = document.body.classList.contains('workspace') && !document.body.dataset.toolPage;
  document.addEventListener('keydown',event=>{
    const editing=event.target.closest?.('input,textarea,select,[contenteditable="true"],.ProseMirror');
    if(chartWorkspace){
      if(!editing && event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.key.toLowerCase()==='k'){event.preventDefault();open();}
      return;
    }
    if(((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k')||(!editing&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&event.key==='/')){event.preventDefault();open();}
  });
  const button=document.createElement('button');button.type='button';button.className='site-navigator-trigger';button.setAttribute('aria-label','Search pages and tools');button.setAttribute('aria-haspopup','dialog');button.innerHTML=chartWorkspace?'<span>Tools</span><kbd>Alt K</kbd>':'<span>Go to</span><kbd>Ctrl K</kbd>';button.title=chartWorkspace?'Search pages and tools (Alt+K)':'Search pages and tools (Ctrl+K)';button.onclick=open;
  // Tool headers are built by their page script; defer insertion until that exists.
  const mount=()=>{
    const header=chartWorkspace?document.querySelector('.header-actions'):document.querySelector('.tool-header-actions,.site-header,.topbar .header-tools,body > header nav,body > header,.prep-toolbar');
    if(header)header.prepend(button);else{button.classList.add('navigator-floating');document.body.append(button);}
  };
  requestAnimationFrame(mount);
});
