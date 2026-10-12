(() => {
  'use strict';
  const toggle = document.getElementById('exploreToggle');
  const menu = document.getElementById('destinationMenu');
  const chart = (key) => key ? '/' + key : '/workspace/';
  const groups = {
    Products: [
      ['↗', 'Charts', chart(), 'Your complete chart workspace'],
      ['◎', 'Stock screener', chart('screener')],
      ['⌁', 'Strategy tester', chart('strategy')],
      ['▥', 'Portfolios', chart('portfolio')],
      ['◇', 'Options lab', chart('options')],
      ['◴', 'Returns & risk', chart('risk')]
    ],
    Markets: [
      ['◉', 'Market overview', chart('markets')],
      ['↻', 'Sector rotation', chart('rotation')],
      ['▤', 'Company fundamentals', chart('fundamentals')],
      ['▥', 'Volume & liquidity', chart('liquidity')],
      ['≋', 'Stored market data', chart('data')]
    ],
    Research: [
      ['◎', 'Crypto address scanner', chart('crypto-scanner')],
      ['✎', 'Idea notebook', chart('ideas')],
      ['▤', 'Research journal', chart('journal')],
      ['⌁', 'Pairs analysis', chart('pairs')],
      ['↗', 'Forecast lab', chart('forecast')],
      ['◇', 'Markov analysis', chart('markov')],
      ['≋', 'Monte Carlo simulations', chart('brownian')],
      ['▥', 'Strategy comparison', chart('compare')]
    ],
    More: [
      ['♙', 'Your account', '/account'],
      ['→', 'Sign in', '/login'],
      ['◇', 'Project architecture', '/workspace/architecture.html'],
      ['▤', 'Interview preparation', '/workspace/interview-prep.html']
    ]
  };
  function close(focus = false) {
    menu.hidden = true; toggle.setAttribute('aria-expanded', 'false');
    if (focus) toggle.focus();
  }
  function render(group) {
    menu.replaceChildren();
    if (group) {
      const head = document.createElement('div'); head.className = 'menu-title';
      const back = document.createElement('button'); back.textContent = '‹'; back.setAttribute('aria-label', 'Back to destinations'); back.onclick = () => { render(); menu.querySelector('button').focus(); };
      const title = document.createElement('strong'); title.textContent = group;
      const exit = document.createElement('button'); exit.textContent = '×'; exit.setAttribute('aria-label', 'Close navigation'); exit.onclick = () => close(true);
      head.append(back, title, exit); menu.append(head);
      const label = document.createElement('div'); label.className = 'menu-label'; label.textContent = group === 'More' ? 'YOUR QUANTSTACK' : 'EXPLORE ' + group.toUpperCase(); menu.append(label);
      for (const [icon, title, href, description] of groups[group]) {
        const link = document.createElement('a'); link.className = 'menu-row'; link.href = href;
        const glyph = document.createElement('span'); glyph.className = 'menu-icon'; glyph.textContent = icon; glyph.setAttribute('aria-hidden', 'true');
        const text = document.createElement('span'); text.textContent = title;
        if (description) { const small = document.createElement('small'); small.textContent = description; text.append(small); }
        link.append(glyph, text); menu.append(link);
      }
    } else {
      for (const [group, icon] of [['Products', '◇'], ['Markets', '◉'], ['Research', '⌁'], ['More', '•••']]) {
        const button = document.createElement('button'); button.className = 'menu-row';
        button.innerHTML = `<span class="menu-icon" aria-hidden="true">${icon}</span><span>${group}</span><span class="arrow" aria-hidden="true">›</span>`;
        button.onclick = () => { render(group); menu.querySelector('button').focus(); }; menu.append(button);
      }
    }
  }
  toggle.onclick = () => {
    if (!menu.hidden) return close();
    render(); menu.hidden = false; toggle.setAttribute('aria-expanded', 'true');
  };
  toggle.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown') { event.preventDefault(); if (menu.hidden) toggle.click(); menu.querySelector('button').focus(); }
  });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !menu.hidden) close(true); });
  document.addEventListener('click', event => { if (!event.composedPath().includes(toggle.parentElement)) close(); });
  document.addEventListener('focusin', event => { if (!event.target.closest('.destination')) close(); });
  const search = document.getElementById('symbolSearch');
  search.form.addEventListener('submit', event => {
    search.value = search.value.trim().toUpperCase();
    if (!search.value) { event.preventDefault(); search.focus(); }
  });
})();
