/* Source reference: 765eba42baccc390ebdde44fb515c6b227c1b859.
 * Diagram relationships are editorial descriptions of this inspected revision. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id), data = window.QUANTSTACK_ARCHITECTURE;
  const n = (id, title, subtitle, kind, column, row, detail, files = []) => ({id, title, subtitle, kind, column, row, detail, files});
  const diagrams = [
    {id:'overview', label:'System overview', title:'From market data to workspace', description:'The active application, its external inputs and the two market-data paths into the chart.',
      notes:['The gateway exposes the chart under /workspace/. Its dashboard backend listens on a private loopback port.', 'Database and hosted-snapshot market paths are alternatives chosen at server startup. Account routes are separate.', 'The preserved Streamlit application is outside this active request path.'],
      nodes:[
        n('yahoo','Yahoo Finance','Price history and metadata','source',0,0,'Daily and hourly prices arrive through yfinance. Universe expansion can enrich metadata.', ['market_data.py','refresh_hourly.py','expand_stocks.py']),
        n('collect','Collectors','Daily, hourly and priority pulls','process',1,0,'market_data.py, refresh_hourly.py and refresh_all.py write normalized prices. pull_queue.py serves requested ranges.', ['market_data.py','refresh_hourly.py','refresh_all.py','pull_queue.py']),
        n('market','Market database','data/market.sqlite','store',2,0,'Shared local market history, collection state, request activity and captured liquidations.', ['market_data.py']),
        n('binance','Binance USD-M','Public force-order stream','source',0,1,'Partial venue-specific liquidation events; not complete exchange-wide liquidation totals.', ['liquidations.py']),
        n('liq','Local liquidation capture','Loopback launch only','process',1,1,'The unified server starts the collector only when its host is a loopback address.', ['serve_quantstack.py','liquidations.py']),
        n('dashboard','Dashboard backend','Private loopback HTTP server','process',2,1,'Serves catalog, history, indicators, screener and local controls from dashboard.Handler.', ['dashboard.py','screener.py']),
        n('finviz','Finviz','Sector performance snapshot','source',0,2,'sector_rotation.py reads the public Group Screener performance view.', ['sector_rotation.py']),
        n('sector','Sector rotation','300-second memory cache','process',1,2,'Parses and ranks sector returns. The live panel belongs to the database-backed dashboard.', ['sector_rotation.py']),
        n('gateway','Unified gateway','serve_quantstack.py','process',2,2,'aiohttp routes /workspace/ to the private dashboard or serves verified snapshot files.', ['serve_quantstack.py']),
        n('release','Published market data','GitHub release + manifest','source',0,3,'The tracked manifest pins a data-only archive URL and SHA-256 checksum.', ['market-snapshot.json','publish_site.py']),
        n('snapshot','Verified daily files','Snapshot-mode alternative','store',1,3,'prepare_snapshot.py installs catalogs and compressed daily histories in a versioned directory.', ['prepare_snapshot.py']),
        n('browser','Browser workspace','web/index.html','process',2,3,'Canvas charts, drawings, notebook and analysis tools. The browser receives current UI assets in both modes.', ['web/index.html','web/app.js'])],
      edges:[['yahoo','collect','history'],['collect','market','write'],['binance','liq','events'],['liq','market','persist'],['market','dashboard','read'],['finviz','sector','fetch'],['sector','dashboard','sector API'],['dashboard','gateway','database mode'],['release','snapshot','verify/install'],['snapshot','gateway','snapshot mode'],['gateway','browser','UI + data']]},
    {id:'modes',label:'Serving modes',title:'One gateway, two market-data modes',description:'Mode selection occurs once at startup. It depends on stored prices, not the hostname.',
      notes:['has_market_data() checks for rows in prices or intraday_prices; SQLite errors select snapshot mode.', 'Snapshot mode injects static-data.js. The browser fetches files instead of calling the dashboard market API.', 'Account routes are registered before the catch-all router and can operate in either mode, subject to account configuration.'],
      nodes:[
        n('start','Start server','serve(host, port)','process',1,0,'Starts the priority worker and checks the market database.', ['serve_quantstack.py']),
        n('dbmode','Stored prices found','Database mode','process',0,1,'The gateway strips /workspace and proxies to dashboard.Handler.', ['serve_quantstack.py','dashboard.py']),
        n('staticmode','No stored prices','Snapshot mode','process',2,1,'prepare() obtains the release before the gateway begins serving the snapshot.', ['serve_quantstack.py','prepare_snapshot.py']),
        n('api','Market API','/workspace/api/*','process',0,2,'Supports server history, screener, indicators and local controls. Local controls have additional access checks.', ['dashboard.py']),
        n('static','Static data adapter','web/static-data.js','process',2,2,'Loads symbols.json, screener.json and prices/*.json.gz, decompresses prices and aggregates weekly/monthly bars.', ['web/static-data.js']),
        n('account','Independent account API','/workspace/api/account/*','process',1,2,'This explicit route precedes both mode branches. Origin checks and durable storage configuration still apply.', ['accounts.py']),
        n('ui','Shared chart workspace','Current web/ assets','process',1,3,'A restart is required to change serving mode after provisioning stored market prices.', ['web/index.html','serve_quantstack.py'])],
      edges:[['start','dbmode','rows exist'],['start','staticmode','no rows / error'],['dbmode','api','proxy'],['staticmode','static','files'],['api','ui','JSON / CSV'],['static','ui','daily bars'],['account','ui','account state']]},
    {id:'collection',label:'Collection pipeline',title:'Collection, priorities and shared history',description:'How scheduled refreshes and requested ranges reach the same local market database.',
      notes:['Collectors coordinate through data/ingestion.lock. Queue state lives in a separate SQLite file.', 'refresh_all.py interleaves popular/configured assets and US cap/volume leaders. refresh_priority.py promotes due assets while preserving rotation.', 'Current scheduled intraday collection routes to refresh_hourly.py. Retained minute-exchange collector helpers are not automatically scheduled live feeds.'],
      nodes:[
        n('config','Configured universe','config.json + discovery','source',0,0,'Symbols and resource settings seed daily and hourly refreshes.', ['config.json','market_data.py','expand_stocks.py']),
        n('priority','Due-symbol ordering','refresh_priority.py','process',1,0,'Reads data/refresh-priority.json and gives due priority assets a bounded fast lane.', ['refresh_priority.py']),
        n('daily','Daily / full refresh','market_data.py, refresh_all.py','process',2,0,'Normalizes completed daily candles, maintains state and supports full/incremental catch-up.', ['market_data.py','refresh_all.py']),
        n('request','Requested range','Local chart data-pull controls','process',0,1,'The UI submits a validated local same-origin pull request.', ['web/data-pulls.js','dashboard.py']),
        n('queue','Durable pull queue','data/pull-queue.sqlite','store',1,1,'requests records move through queued, running, complete or failed states.', ['pull_queue.py']),
        n('hourly','Hourly / queue workers','refresh_hourly.py + pull_queue.py','process',2,1,'Scheduled refreshes and the daemon worker drain requests while coordinating on the ingestion lock.', ['refresh_hourly.py','pull_queue.py']),
        n('provider','Yahoo Finance','yfinance history / metadata','source',0,2,'Provider calls return prices and exchange metadata; rate-limit handling and progress reporting belong to collectors.', ['market_data.py','refresh_hourly.py']),
        n('db','Normalized market history','data/market.sqlite','store',2,2,'Shared daily and intraday history. Market readers and snapshot builders consume this database.', ['market_data.py']),
        n('reports','Collection reports','data/*refresh*.json and logs','store',1,3,'Reports describe collection outcomes and progress, not proof that Windows tasks are installed.', ['refresh_all.py','refresh_hourly.py','local_scheduler.py'])],
      edges:[['config','priority','universe'],['priority','daily','ordered jobs'],['priority','hourly','due jobs'],['request','queue','enqueue'],['queue','hourly','drain'],['provider','daily','daily bars'],['provider','hourly','intraday bars'],['daily','db','persist'],['hourly','db','persist'],['daily','reports','daily outcome'],['hourly','reports','hourly outcome']]},
    {id:'browser',label:'Browser modules',title:'Chart UI and analysis execution',description:'Plain JavaScript, native canvas and browser workers; the editor has its own build boundary.',
      notes:['index.html loads ordered classic scripts that share chart state. There is no separate frontend application server in this runtime.', 'Markov, Brownian and research modules create workers. research-worker.js imports both research-engine.js and strategy-engine.js.', 'Node/esbuild rebuilds the notebook bundle; Render serves the checked-in result.'],
      nodes:[
        n('html','Workspace document','web/index.html','source',0,0,'Loads app, feature scripts, theme and styles in the established order.', ['web/index.html']),
        n('chart','Canvas and layout','app.js + workspace.js','process',1,0,'Loads data, draws the chart, manages workspace layout and coordinates feature panels.', ['web/app.js','web/workspace.js']),
        n('tools','Chart interaction','Scale, drawings, context menu','process',2,0,'Drawing objects and undo, chart scales, comparisons, settings and export.', ['web/drawings.js','web/chart-scale.js','web/terminal.js']),
        n('ui','Analysis panels','Strategy, risk, research','process',0,1,'Strategy and risk run browser calculations; research routes heavier work to a worker.', ['web/strategy.js','web/risk.js','web/research.js']),
        n('workers','Analysis workers','Markov / Brownian / research','process',1,1,'Worker entrypoints keep the corresponding numerical work separate from chart interaction.', ['web/markov-worker.js','web/brownian-worker.js','web/research-worker.js']),
        n('engines','Numerical engines','Browser calculation modules','process',2,1,'Research worker imports research and strategy engines; Markov and Brownian workers import their matching engines.', ['web/research-engine.js','web/strategy-engine.js','web/markov-engine.js','web/brownian-engine.js','web/risk-engine.js']),
        n('editor','Editor source + packages','editor/','source',0,2,'Tiptap/ProseMirror and DOMPurify are pinned and bundled locally.', ['editor/package.json','editor/note-editor.js']),
        n('bundle','Editor build','editor/build.mjs → bundle','process',1,2,'esbuild emits web/note-editor.js and dependency licenses are kept with the build source.', ['editor/build.mjs','web/note-editor.js']),
        n('notes','Notebook and browser state','notes.js + localStorage','store',2,2,'Notes, drawings, watchlists and preferences persist in the browser; account save/load is explicit.', ['web/notes.js','web/accounts.js'])],
      edges:[['html','chart','load'],['chart','tools','shared state'],['chart','ui','history / selection'],['ui','workers','research jobs'],['workers','engines','imports'],['editor','bundle','bundle'],['bundle','notes','rich text'],['tools','notes','saved workspace']]},
    {id:'storage',label:'Storage boundaries',title:'Different stores, different lifecycles',description:'Market data, account state, browser work and generated snapshots are distinct.',
      notes:['The market schema defines 10 base tables; expand_stocks.py adds stock_metadata and stock_enrichment.', 'The new account store is data/accounts.sqlite, not the preserved root users.db.', 'The Git tree excludes ignored runtime data; chart-snapshots/ contains bounded checkpoints, not the full database.'],
      nodes:[
        n('market','Market database','data/market.sqlite','store',0,0,'Daily and intraday bars, states, attempts, membership, exchange prices, liquidations and API-request logs. Metadata expansion adds two tables.', ['market_data.py','expand_stocks.py']),
        n('queue','Queue database','data/pull-queue.sqlite','store',1,0,'A requests table stores local data-pull jobs separately from market history.', ['pull_queue.py']),
        n('account','Account database','data/accounts.sqlite','store',2,0,'accounts stores credential hashes, workspace JSON and revision. sessions stores token hashes; auth_limits stores throttle counters.', ['accounts.py']),
        n('browser','Browser storage','localStorage: atlas.*','store',0,1,'Per-origin notes, settings, drawings and watchlists. quantstack.* keys hold account bookkeeping.', ['web/accounts.js','web/notes.js']),
        n('saved','Explicit workspace snapshot','Versioned account save/load','process',1,1,'The user chooses when browser work is uploaded or replaced with the saved account copy.', ['web/accounts.js','accounts.py']),
        n('legacy','Preserved legacy database','users.db','legacy',2,1,'Tracked legacy file. The current gateway does not serve it or use it as the new account store.', ['QuantStack-main/LEGACY.md']),
        n('daily','Hosted daily snapshot','catalogs + prices/*.json.gz','store',0,2,'Daily prices and public market metadata are packaged; account databases and UI files are excluded from the data release.', ['build_snapshot.py','prepare_snapshot.py']),
        n('csv','Bounded Git checkpoints','chart-snapshots/','store',1,2,'32 CSV buckets and a manifest: latest 30 daily and 48 hourly bars per symbol.', ['snapshot_charts.py']),
        n('git','Committed tree','Source + tracked artifacts','store',2,2,'The inspected revision includes legacy files, bounded CSVs and a tracked Python environment. Full runtime data is outside this root hash.', ['.gitignore','Auto-Commit.ps1'])],
      edges:[['queue','market','workers persist prices'],['browser','saved','save / load'],['saved','account','API'],['market','daily','daily export'],['market','csv','bounded export'],['csv','git','checkpoint commit']]},
    {id:'accounts',label:'Account save / load',title:'Explicit personal-work synchronization',description:'Workspace snapshots travel separately from market prices.',
      notes:['Account requests use /workspace/api/account/{action}. Market serving mode does not change this explicit route.', 'POST requests require the expected origin and JSON; account payloads are limited to 2 MB.', 'Public account operation needs HTTPS/origin configuration and durable storage. render.yaml does not itself provision a durable account disk.'],
      nodes:[
        n('work','Browser workspace','Notes / drawings / settings','store',0,0,'Work persists in this browser until the user explicitly saves an account snapshot.', ['web/notes.js','web/drawings.js','web/accounts.js']),
        n('user','User chooses Save or Load','web/accounts.js','process',1,0,'The UI gathers atlas.* keys, flushes notebook edits, and tracks the server revision.', ['web/accounts.js']),
        n('route','Gateway account route','/workspace/api/account/*','process',2,0,'install_accounts() registers the route before the gateway catch-all.', ['serve_quantstack.py','accounts.py']),
        n('checks','Request and session checks','Origin, size, token, revision','process',2,1,'Accounts uses HttpOnly cookies and server-side token hashes. Concurrent saves are guarded by revisions.', ['accounts.py']),
        n('db','Account storage','accounts / sessions / auth_limits','store',1,1,'Workspace JSON and its revision are stored on the accounts row; session and throttle records are separate tables.', ['accounts.py']),
        n('restore','Load saved browser work','Explicit replacement','process',0,1,'Loading restores the saved workspace into browser storage; it does not fetch a private copy of market prices.', ['web/accounts.js']),
        n('public','Public server configuration','HTTPS origin + durable DB path','legacy',1,2,'QUANTSTACK_PUBLIC_ORIGIN and QUANTSTACK_ACCOUNTS_DB configure the account service. External hosting configuration was not inspected.', ['HOSTING.md','accounts.py','render.yaml'])],
      edges:[['work','user','explicit action'],['user','route','GET / POST'],['route','checks','validate'],['checks','db','read / write'],['db','restore','saved snapshot'],['restore','work','apply'],['public','db','configured path']]},
    {id:'automation',label:'Scheduled automation',title:'Collection and Git checkpoints are separate jobs',description:'These are task definitions in source, not a report of installed or running Windows tasks.',
      notes:['Install-Schedule.ps1 defines daily collection at 09:00 machine-local time with a four-hour limit.', 'Two installers use the same MarketData-IntradaySync task name and -Force; the last installer run determines its definition.', 'Auto-Commit.ps1 stages bounded snapshots and tracked changes. It pushes only when invoked with -Push.'],
      nodes:[
        n('daily','Daily task definition','Install-Schedule.ps1','source',0,0,'09:00 machine-local; Refresh-And-Publish.ps1; four-hour execution limit.', ['Install-Schedule.ps1']),
        n('wrapper','Refresh and publish','Refresh-And-Publish.ps1','process',1,0,'Runs refresh_all.py --daily-only --workers 2 and publishes newly collected data via a dedicated checkout.', ['Refresh-And-Publish.ps1']),
        n('publish','Daily dataset release','Clean deployment checkout','process',2,0,'The default sibling checkout is market-data-render-deploy. Publisher commits only the manifest there.', ['publish_site.py']),
        n('hourly','Hourly task definitions','Two alternative installers','source',0,1,'Install-Intraday-Schedule.ps1 uses market_data.py intraday (15-minute limit); Install-Update-Automation.ps1 uses refresh_hourly.py (55-minute limit).', ['Install-Intraday-Schedule.ps1','Install-Update-Automation.ps1']),
        n('refresh','Hourly refresh','refresh_hourly.py','process',1,1,'Both task entrypoints reach the hourly collector; profiles control resource usage.', ['refresh_hourly.py','market_data.py']),
        n('db','Market history','data/market.sqlite','store',2,1,'Collection writes data. A Git commit alone does not refresh market prices.', ['market_data.py']),
        n('checkpoint','Daily checkpoint definition','09:30 Asia/Taipei','source',0,2,'Install-Update-Automation.ps1 translates the Taipei schedule into the machine local timezone.', ['Install-Update-Automation.ps1']),
        n('commit','Auto-Commit.ps1','Snapshot export + local commit','process',1,2,'Uses an exclusive lock and skips pre-staged work or in-progress Git operations.', ['Auto-Commit.ps1','snapshot_charts.py']),
        n('push','Optional push','origin / current branch','legacy',2,2,'Only -Push enables remote pushing. Rejected pushes leave commits intact; no force push or automatic merge.', ['Auto-Commit.ps1'])],
      edges:[['daily','wrapper','daily'],['wrapper','publish','publish'],['wrapper','db','daily bars'],['hourly','refresh','hourly'],['refresh','db','persist'],['checkpoint','commit','daily'],['db','commit','bounded CSVs'],['commit','push','only -Push']]},
    {id:'deployment',label:'Publication & Render',title:'Data release and source deployment meet at the manifest',description:'Public market data is packaged separately from the website source.',
      notes:['build_site.py produces a portable preview archive. build_snapshot.py selects only permitted market-data files for release.', 'publish_site.py verifies the uploaded release digest and updates market-snapshot.json with the actual asset URL.', 'Render runs pip install and prepare_snapshot.py, then starts the unified server. A successful Git push alone is not proof of a completed deploy.'],
      nodes:[
        n('db','Local market database','data/market.sqlite','store',0,0,'The export source for daily histories and screener catalogs.', ['build_site.py']),
        n('portable','Portable site build','build_site.py → site.zip','process',1,0,'Copies the browser UI and writes symbols, screener, snapshot metadata and compressed daily prices.', ['build_site.py']),
        n('data','Data-only archive','build_snapshot.py','process',2,0,'Filters allowed data members, computes SHA-256 and creates market-data-<hash16>.zip.', ['build_snapshot.py']),
        n('github','GitHub market-data release','Uploaded ZIP asset','source',2,1,'publish_site.py uses github_cli.py and existing Git authentication to upload and inspect the release asset.', ['publish_site.py','github_cli.py']),
        n('manifest','Tracked manifest','market-snapshot.json','store',1,1,'Pins the URL and checksum. Optional --deployment-worktree uses a clean checkout and restricts unpublished changes.', ['publish_site.py','market-snapshot.json']),
        n('main','GitHub main','Configured deployment branch','source',0,1,'The manifest or source changes can trigger Render auto-deployment when enabled in the service.', ['HOSTING.md','render.yaml']),
        n('render','Render build','Dependencies + prepare_snapshot.py','process',0,2,'The checked-in build installs requirements.txt and then verifies/installs the snapshot.', ['render.yaml','prepare_snapshot.py']),
        n('verified','Installed snapshot','SHA-256 checked files','store',1,2,'Validates checksum, archive paths, required catalogs and size; installs a version directory with a completion marker.', ['prepare_snapshot.py']),
        n('serve','Website process','serve_quantstack.py --host 0.0.0.0','process',2,2,'Uses PORT. Snapshot market mode is selected if the server has no populated local market database.', ['serve_quantstack.py','render.yaml'])],
      edges:[['db','portable','read daily data'],['portable','data','filter ZIP'],['data','github','upload'],['github','manifest','asset URL / hash'],['manifest','main','commit / push'],['main','render','auto-deploy'],['render','verified','download / verify'],['github','verified','archive bytes'],['verified','serve','same-origin files']]},
    {id:'merkle',label:'Git Merkle tree',title:'The content-addressed project snapshot',description:'Actual object IDs from the inspected commit. Root and child hashes identify committed bytes.',
      notes:['Git uses SHA-1 here. A tree hashes encoded child modes, names and object IDs; a blob hashes its Git header and content.', 'The root covers 14,978 tracked files, including 14,717 files in .venv/. Ignored runtime data and uncommitted changes are outside it.', 'The complete TSV includes 16,989 file/directory entries. ZIP SHA-256 verification is a separate deployment mechanism.'],
      nodes:[
        n('commit','Commit 765eba4','765eba42baccc390…','source',1,0,'Commit: '+data.commit+'. Commits reference a root tree, parent history and metadata.'),
        n('root','Root tree 4a66ce7','14,978 files · 2,011 trees','store',1,1,'Root tree: '+data.tree+'. Tree count includes the root and counts directory entries, not unique deduplicated objects.'),
        n('web','web/ · 39 files','Tree 6164a17d89c6…','store',0,2,'6164a17d89c6bbf5ffdf3eb15781d9c06af1783a — browser application and styles.'),
        n('other','Other root entries','101 root files + 5 directories','store',1,2,'editor/, docs/, chart-snapshots/, QuantStack-main/ and .streamlit/, plus 101 root-file entries. Every object ID is in the TSV.'),
        n('venv','.venv/ · 14,717 files','Tree 05e0d28cddce…','legacy',2,2,'05e0d28cddcea86ed649d90608e4fc14c8e1c8e4 — a Python environment already tracked despite the current ignore rule.'),
        n('index','index.html blob','c43172abb0f059f0…','store',0,3,'c43172abb0f059f0c4e98e6391c4a262e4e22e6e', ['web/index.html']),
        n('app','app.js blob','d44a6f250bae2ddc…','store',1,3,'d44a6f250bae2ddca11cd8db12a878444d7ee3b9', ['web/app.js']),
        n('terminal','terminal.js blob','5f373c715089079f…','store',2,3,'5f373c715089079fb5eb7e816ba86147599cae67', ['web/terminal.js'])],
      edges:[['commit','root','tree reference'],['root','web','directory'],['root','other','entries'],['root','venv','directory'],['web','index','blob'],['web','app','blob'],['web','terminal','blob']]}
  ];
  const NS = 'http://www.w3.org/2000/svg';
  let current = diagrams[0], zoom = 1, selectedNode = null, fitMode = false;
  // Only link when this exact inspected blob also exists at the published revision.
  const sourceURL = path => data.publishedFiles.includes(path) ? `https://github.com/hsiantw/QuantStack/blob/${data.publishedCommit}/${path.split('/').map(encodeURIComponent).join('/')}` : null;
  $('revisionLink').href = '#merkle';
  $('revisionLink').title = 'Inspect the captured local commit and its verified tree hash';
  const systemTheme = matchMedia('(prefers-color-scheme: dark)');
  let theme = 'system';
  try { theme = localStorage.getItem('atlas.architecture.theme') || 'system'; } catch {}
  if (!['system','light','dark'].includes(theme)) theme = 'system';
  function applyTheme() { document.documentElement.dataset.theme = theme === 'system' ? systemTheme.matches ? 'dark' : 'light' : theme; }
  applyTheme(); $('mapTheme').value = theme;
  $('mapTheme').onchange = event => { theme = event.target.value; try { localStorage.setItem('atlas.architecture.theme', theme); } catch {} applyTheme(); renderDiagram(); };
  systemTheme.addEventListener('change', () => { if (theme === 'system') { applyTheme(); renderDiagram(); } });
  function element(tag, attributes = {}, text) { const node = document.createElementNS(NS, tag); Object.entries(attributes).forEach(([key,value]) => node.setAttribute(key, value)); if (text !== undefined) node.textContent = text; return node; }
  function inspect(node) {
    selectedNode = node.id; $('nodeTitle').textContent = node.title; $('nodeDescription').textContent = node.detail;
    $('nodeSources').replaceChildren(...node.files.map(file => {
      const url=sourceURL(file), item=document.createElement(url?'a':'span');
      item.textContent=file+(url?' ↗':' · local snapshot');
      if(url){item.href=url;item.target='_blank';item.rel='noopener';item.title='Identical blob at published commit '+data.publishedCommit.slice(0,7);}
      return item;
    }));
    document.querySelectorAll('.diagram-node').forEach(item => { const active = item.dataset.node === node.id; item.classList.toggle('selected', active); item.setAttribute('aria-pressed', String(active)); });
  }
  function renderDiagram() {
    const colors = getComputedStyle(document.documentElement), color = name => colors.getPropertyValue('--' + name).trim();
    const height = (Math.max(...current.nodes.map(node => node.row)) + 1) * 142 + 36;
    const svg = element('svg', {xmlns:NS, viewBox:`0 0 900 ${height}`, role:'group', 'aria-labelledby':'svgTitle svgDescription'});
    svg.append(element('title', {id:'svgTitle'}, current.title), element('desc', {id:'svgDescription'}, current.description + ' Select a component for details. A text version of connections follows the diagram.'));
    svg.append(element('rect', {width:900,height,fill:color('surface')}));
    const defs = element('defs'), marker = element('marker', {id:'mapArrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});
    marker.append(element('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color('edge')})); defs.append(marker); svg.append(defs);
    const positions = new Map(current.nodes.map(node => [node.id,{x:30+node.column*294,y:30+node.row*142}]));
    for (const [from,to,label] of current.edges) {
      const a=positions.get(from), b=positions.get(to), same=a.x===b.x;
      const down=b.y>a.y, right=b.x>a.x;
      const x1=same?a.x+126:a.x+(right?252:0), y1=same?a.y+(down?78:0):a.y+39;
      const x2=same?b.x+126:b.x+(right?0:252), y2=same?b.y+(down?0:78):b.y+39;
      const middleX=(x1+x2)/2, middleY=(y1+y2)/2;
      const path=same?`M${x1} ${y1} L${x2} ${y2}`:`M${x1} ${y1} C${middleX} ${y1},${middleX} ${y2},${x2} ${y2}`;
      svg.append(element('path',{d:path,fill:'none',stroke:color('edge'),'stroke-width':1.4,'marker-end':'url(#mapArrow)'}));
      const text=element('text',{x:middleX+(same?8:0),y:middleY-7,'text-anchor':same?'start':'middle',fill:color('muted'),'font-family':'Segoe UI, sans-serif','font-size':9,'paint-order':'stroke',stroke:color('surface'),'stroke-width':4,'stroke-linejoin':'round'},label);svg.append(text);
    }
    for (const node of current.nodes) {
      const {x,y}=positions.get(node.id), group=element('g',{class:'diagram-node',tabindex:0,role:'button','aria-label':node.title+': '+node.subtitle,'aria-pressed':String(node.id===selectedNode),'data-node':node.id});
      if (node.id===selectedNode) group.classList.add('selected');
      const fill=color(node.kind==='process'?'soft':node.kind), stroke=color(node.kind==='process'?'border':node.kind+'-stroke');
      group.append(element('rect',{x,y,width:252,height:78,rx:8,fill,stroke,'stroke-width':1}));
      group.append(element('text',{x:x+16,y:y+29,fill:color('ink'),'font-family':'Segoe UI, sans-serif','font-size':12,'font-weight':600},node.title));
      group.append(element('text',{x:x+16,y:y+50,fill:color('muted'),'font-family':'Segoe UI, sans-serif','font-size':10},node.subtitle));
      group.append(element('title',{},node.detail));
      group.onclick=()=>inspect(node); group.onkeydown=event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();inspect(node);}}; svg.append(group);
    }
    $('diagramCanvas').replaceChildren(svg); applyZoom();
  }
  function applyZoom() { $('diagramCanvas').style.width = `${zoom*100}%`; $('diagramCanvas').querySelector('svg').style.minWidth=fitMode?'0px':''; $('zoomLabel').value = `${Math.round(zoom*100)}%`; $('zoomOut').disabled=zoom<=.75; $('zoomIn').disabled=zoom>=2; }
  $('zoomIn').onclick=()=>{zoom=Math.min(2,zoom+.25);applyZoom();};
  $('zoomOut').onclick=()=>{zoom=Math.max(.75,zoom-.25);applyZoom();};
  $('zoomFit').onclick=()=>{zoom=1;fitMode=true;applyZoom();$('diagramViewport').scrollTo(0,0);};
  function selectDiagram(id) {
    current=diagrams.find(d=>d.id===id)||diagrams[0];zoom=1;selectedNode=null;fitMode=false;
    $('diagramNumber').textContent=`VIEW ${String(diagrams.indexOf(current)+1).padStart(2,'0')} / ${diagrams.length}`;
    $('diagramTitle').textContent=current.title;$('diagramDescription').textContent=current.description;
    document.querySelectorAll('[data-diagram]').forEach(link=>{link.setAttribute('aria-current',String(link.dataset.diagram===current.id));});
    $('diagramNotes').replaceChildren(...current.notes.map(note=>{const li=document.createElement('li');li.textContent=note;return li;}));
    $('connectionList').replaceChildren(...current.edges.map(([a,b,label])=>{const li=document.createElement('li');li.textContent=`${current.nodes.find(n=>n.id===a).title} → ${current.nodes.find(n=>n.id===b).title}: ${label}.`;return li;}));
    renderDiagram();inspect(current.nodes[0]);$('diagramViewport').scrollTo(0,0);
  }
  for(const [index,diagram] of diagrams.entries()) {const link=document.createElement('a');link.href='#'+diagram.id;link.dataset.diagram=diagram.id;const num=document.createElement('span');num.textContent=String(index+1).padStart(2,'0');link.append(num,document.createTextNode(diagram.label));$('mapNav').append(link);}
  window.addEventListener('hashchange',()=>{if(diagrams.some(d=>d.id===location.hash.slice(1)))selectDiagram(location.hash.slice(1));});
  $('mapNav').addEventListener('click',event=>{const link=event.target.closest('a');if(!link)return;if(location.hash==='#'+link.dataset.diagram)selectDiagram(link.dataset.diagram);document.querySelector('.diagram-card').scrollIntoView({block:'start'});});
  $('diagramCount').textContent=diagrams.length;selectDiagram(location.hash.slice(1));
  $('downloadDiagram').onclick=()=>{
    const clone=$('diagramCanvas').querySelector('svg').cloneNode(true);clone.setAttribute('width','900');clone.setAttribute('height',clone.viewBox.baseVal.height);
    clone.querySelectorAll('[tabindex]').forEach(node=>{node.removeAttribute('tabindex');node.removeAttribute('role');node.removeAttribute('aria-pressed');});
    const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'}));
    const link=document.createElement('a');link.href=url;link.download=`quantstack-${current.id}-${data.commit.slice(0,7)}.svg`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  let chosenFile=null;
  function selectFile(file) {
    chosenFile=file.path;$('fileTitle').textContent=file.path;$('fileDescription').textContent='Committed file at source revision '+data.commit.slice(0,7)+'.';
    $('fileMetadata').replaceChildren();
    for(const [label,value] of [['Blob SHA-1',file.object_id],['File mode',file.mode],['Logical size',Number(file.bytes).toLocaleString()+' bytes']]){const term=document.createElement('dt'),definition=document.createElement('dd');term.textContent=label;definition.textContent=value;$('fileMetadata').append(term,definition);}
    const url=sourceURL(file.path);$('fileSource').hidden=!url;
    if(url){$('fileSource').href=url;$('fileSource').target='_blank';$('fileSource').rel='noopener';$('fileSource').textContent='View identical blob at '+data.publishedCommit.slice(0,7)+' ↗';}
    else $('fileDescription').textContent+=' This inspected version is local; its exact blob ID is recorded below.';
    $('fileTree').querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.path===chosenFile)));
  }
  function renderFiles(){
    const query=$('fileSearch').value.trim().toLowerCase(),files=data.files.filter(file=>file.path.toLowerCase().includes(query));
    $('fileCount').textContent=`${files.length} / ${data.files.length}`;const fragment=document.createDocumentFragment();let group=null;
    for(const file of files){const directory=file.path.includes('/')?file.path.split('/')[0]+'/':'Root files';if(directory!==group){const title=document.createElement('div');title.className='file-group';title.textContent=directory;fragment.append(title);group=directory;}const button=document.createElement('button');button.type='button';button.dataset.path=file.path;button.textContent=file.path;button.setAttribute('aria-pressed',String(file.path===chosenFile));button.onclick=()=>selectFile(file);fragment.append(button);}
    if(!files.length){const empty=document.createElement('p');empty.className='empty';empty.textContent='No matching files. Try a module name such as accounts or market_data.';fragment.append(empty);}
    $('fileTree').replaceChildren(fragment);
  }
  $('fileSearch').oninput=renderFiles;renderFiles();
})();
