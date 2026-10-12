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
