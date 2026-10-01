// Apply before first paint; UI is installed once the workspace is ready.
(() => {
  const key = 'atlas.theme', system = matchMedia('(prefers-color-scheme: dark)');
  const palettes = {
    light: {surface: '#ffffff', grid: '#e9edf2', text: '#20242c', muted: '#727d90', accent: '#2962ff'},
    dark: {surface: '#131722', grid: '#282f3e', text: '#dce2ed', muted: '#9aa7bd', accent: '#7299ff'}
  };
  let preference = 'system';
  try { preference = localStorage.getItem(key) || 'system'; } catch {}
  function apply(value, persist = false) {
    preference = ['light', 'dark', 'system'].includes(value) ? value : 'system';
    const resolved = preference === 'system' ? system.matches ? 'dark' : 'light' : preference;
    document.documentElement.dataset.theme = resolved;
    window.atlasTheme = {palette: palettes[resolved], preference, apply};
    const control = document.getElementById('workspaceTheme'); if (control) control.value = preference;
    if (persist) try { localStorage.setItem(key, preference); } catch {}
    if (typeof draw === 'function') requestAnimationFrame(() => draw());
    window.dispatchEvent(new Event('atlas-theme-change'));
  }
  apply(preference);
  system.addEventListener('change', () => { if (preference === 'system') apply(preference); });
  window.addEventListener('storage', event => { if (event.key === key || event.key === null) apply(event.newValue || 'system'); });
  document.addEventListener('DOMContentLoaded', () => {
    const label = document.createElement('label'); label.className = 'workspace-theme';
    label.innerHTML = '<span aria-hidden="true">◐</span><select id="workspaceTheme" aria-label="Workspace theme"><option value="system">System theme</option><option value="light">Light theme</option><option value="dark">Dark theme</option></select>';
    document.querySelector('.terminal-actions').append(label);
    document.getElementById('workspaceTheme').value = preference;
    document.getElementById('workspaceTheme').onchange = event => apply(event.target.value, true);
  });
})();
