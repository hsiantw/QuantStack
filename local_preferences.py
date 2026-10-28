"""Automatic, loopback-only preference backups, separate from account snapshots."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from aiohttp import web
from local_scheduler import local_request

KEYS = frozenset({
    'atlas.saved', 'atlas.watchlists.v1', 'atlas.timeframeFavorites.v1',
    'atlas.home.shortcuts.v1',
    'atlas.strategy.v1', 'atlas.strategy.setups.v1',
    'atlas.theme', 'atlas.chart.session', 'atlas.chartScale', 'atlas.terminal',
    'atlas.appearance.presets.v1', 'atlas.studies.v1',
    'atlas.workspace.watchlistWidth', 'atlas.workspace.dockHeight',
    'atlas.workspace.toolsExpanded', 'atlas.workspace.rightToolbar.v1',
})
LIMIT = 2_000_000


class Preferences:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS preferences (
                profile TEXT, key TEXT, value TEXT, updated_at INTEGER,
                PRIMARY KEY(profile,key))''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def read(self, profile):
        with closing(self.connect()) as db:
            return dict(db.execute('SELECT key,value FROM preferences WHERE profile=?', (profile,)))

    def patch(self, profile, changes):
        if not isinstance(changes, dict) or any(k not in KEYS or (v is not None and not isinstance(v, str)) for k,v in changes.items()):
            raise ValueError('Invalid preference keys or values.')
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            values = dict(db.execute('SELECT key,value FROM preferences WHERE profile=?', (profile,)))
            values.update(changes)
            if len(json.dumps(values).encode()) > LIMIT:
                raise ValueError('Local preferences exceed the 2 MB limit.')
            # NULL tombstones prevent removed preferences being reimported by older tabs.
            db.executemany('''INSERT INTO preferences VALUES (?,?,?,?)
                ON CONFLICT(profile,key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at''',
                [(profile,k,v,int(time.time())) for k,v in changes.items()])


def install_preferences(app, accounts, accounts_path):
    store = Preferences(Path(accounts_path).with_name('local-preferences.sqlite'))
    script = Path(__file__).parent / 'web' / 'local-preferences.js'
    headers = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}

    async def profile(request):
        session = await asyncio.to_thread(accounts.session, request.cookies.get('quantstack_session', ''))
        return 'account:' + session['user']['id'] if session['user'] else 'local'

    def allowed(request, bootstrap=False):
        if not local_request(request.remote, request.host, request.headers.get('Origin')):
            return False
        site = request.headers.get('Sec-Fetch-Site')
        return site == 'same-origin' if bootstrap else site not in ('cross-site', 'same-site')

    async def bootstrap(request):
        if not allowed(request, True):
            # Hosted/nonlocal pages retain their existing browser-only behavior.
            return web.Response(text='/* Local preference backup unavailable on this origin. */',
                                content_type='text/javascript', headers=headers)
        current = await profile(request)
        values = await asyncio.to_thread(store.read, current)
        initial = json.dumps({'profile': current, 'values': values, 'keys': sorted(KEYS)})
        return web.Response(text='window.quantLocalPreferenceSeed=' + initial + ';\n' + script.read_text(encoding='utf-8'),
                            content_type='text/javascript', headers=headers)

    async def endpoint(request):
        if not allowed(request):
            return web.json_response({'error':'Local preferences are available on this PC only.'}, status=403, headers=headers)
        current = await profile(request)
        if request.method == 'POST':
            if request.headers.get('Origin') != f'{request.scheme}://{request.host}' or request.content_type != 'application/json':
                return web.json_response({'error':'Same-origin JSON required.'}, status=403, headers=headers)
            body = bytearray()
            async for chunk in request.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > LIMIT:
                    return web.json_response({'error':'Preference request too large.'}, status=413, headers=headers)
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError('Expected preference changes.')
                if payload.get('profile') != current:
                    return web.json_response({'error':'Account changed. Reload before saving preferences.'}, status=409, headers=headers)
                await asyncio.to_thread(store.patch, current, payload.get('changes'))
            except (ValueError, UnicodeError) as exc:
                return web.json_response({'error':str(exc)}, status=400, headers=headers)
            return web.json_response({'profile':current,'saved':True}, headers=headers)
        return web.json_response({'profile':current,'values':await asyncio.to_thread(store.read,current)}, headers=headers)

    app.router.add_get('/workspace/api/account/local-preferences.js', bootstrap)
    app.router.add_get('/workspace/api/account/local-preferences', endpoint)
    app.router.add_post('/workspace/api/account/local-preferences', endpoint)
