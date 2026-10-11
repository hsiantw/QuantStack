"""Serve the unified chart workspace without a Streamlit process."""
import argparse
import asyncio
import json
from contextlib import closing
import os
from pathlib import Path
import re
import sqlite3
import threading
from http.server import ThreadingHTTPServer

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from dashboard import Handler
from prepare_snapshot import data_file, prepare
from snapshot_catalog import snapshot_catalog

ROOT = Path(__file__).resolve().parent
DASHBOARD = web.AppKey('dashboard', str)
SNAPSHOT_PATH = web.AppKey('snapshot_path', Path)
CLIENT = web.AppKey('client', ClientSession)
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade'}


def forwarded_headers(headers):
    excluded = HOP_HEADERS | {h.strip().lower() for h in headers.get('Connection', '').split(',')}
    return [(key, value) for key, value in headers.items() if key.lower() not in excluded]


def has_market_data(path):
    if not path.is_file():
        return False
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
            return any(connection.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone()
                       for table in ('prices', 'intraday_prices'))
    except sqlite3.Error:
        return False


async def snapshot(request):
    """Serve the current UI and release data entirely from QuantStack."""
    if request.method not in ('GET', 'HEAD'):
        raise web.HTTPMethodNotAllowed(request.method, ['GET', 'HEAD'])
    name = request.path.removeprefix('/workspace/')
    if not name or name == 'index.html':
        index = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8-sig')
        index = index.replace('<script src="./app.js">',
                              '<script src="./static-data.js"></script><script src="./app.js">')
        return web.Response(text=index, content_type='text/html')
    if name == 'interview-prep.html':
        return web.FileResponse(ROOT / 'web' / name, headers={'X-Content-Type-Options': 'nosniff'})
    if re.fullmatch(r'[a-z-]+\.(js|css)', name) and (ROOT / 'web' / name).is_file():
        return web.FileResponse(ROOT / 'web' / name)
    if not data_file(name):
        raise web.HTTPNotFound()
    path = request.app[SNAPSHOT_PATH] / name
    if not path.is_file():
        raise web.HTTPNotFound()
    if name == 'symbols.json':
        assets = json.loads(path.read_text(encoding='utf-8'))
        return web.json_response(snapshot_catalog(assets, ROOT / 'config.json'),
                                 headers={'Cache-Control': 'no-cache'})
    return web.FileResponse(path, headers={'X-Content-Type-Options': 'nosniff'})


async def proxy(request):
    if request.path in ('/workspace/api/local-scheduler', '/workspace/api/pull-queue'):
        from local_scheduler import local_request
        if not local_request(request.remote, request.host, request.headers.get('Origin')):
            raise web.HTTPForbidden(text='Collector settings are available locally only.')
    if request.path == '/' and request.method in ('GET', 'HEAD'):
        target = '/workspace/'
        query = request.query.copy()
        query.popall('research', None)
        if query:
            from urllib.parse import urlencode
            target += '?' + urlencode(list(query.items()))
        raise web.HTTPFound(target)
    workspace = request.path == '/workspace' or request.path.startswith('/workspace/')
    if request.path == '/workspace':
        raise web.HTTPPermanentRedirect('/workspace/')
    if workspace and request.app[SNAPSHOT_PATH]:
        return await snapshot(request)
    if not workspace:
        raise web.HTTPNotFound(text='This application now uses the chart workspace.')
    upstream = request.app[DASHBOARD]
    path = request.raw_path[len('/workspace'):] if workspace else request.raw_path
    url = upstream + path
    session = request.app[CLIENT]
    headers = forwarded_headers(request.headers)
    try:
        body = request.content.iter_chunked(64 * 1024) if request.can_read_body else None
        async with session.request(request.method, url, headers=headers, data=body,
                                   allow_redirects=False) as response:
            result = web.StreamResponse(status=response.status, headers=forwarded_headers(response.headers))
            await result.prepare(request)
            async for chunk in response.content.iter_chunked(64 * 1024):
                await result.write(chunk)
            await result.write_eof()
            return result
    except (ClientError, OSError, asyncio.TimeoutError) as exc:
        raise web.HTTPBadGateway(text='QuantStack is starting. Please retry.') from exc


def create_app(dashboard_url, snapshot_dir=None, liquidation_symbols=(), accounts_path=None):
    app = web.Application(client_max_size=200 * 1024**2)
    app[DASHBOARD] = dashboard_url
    app[SNAPSHOT_PATH] = snapshot_dir

    async def client_context(application):
        async with ClientSession(auto_decompress=False, timeout=ClientTimeout(total=None, sock_connect=10)) as session:
            application[CLIENT] = session
            yield

    app.cleanup_ctx.append(client_context)
    if liquidation_symbols:
        async def liquidation_context(application):
            from liquidations import collect_force_orders
            task = asyncio.create_task(collect_force_orders(
                application[CLIENT], ROOT / 'data' / 'market.sqlite', liquidation_symbols))
            try:
                yield
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        app.cleanup_ctx.append(liquidation_context)
    from accounts import install_accounts
    install_accounts(app, accounts_path)
    app.router.add_route('*', '/{path:.*}', proxy)
    return app


async def serve(host, port):
    from pull_queue import start_worker
    start_worker()
    use_snapshot = not has_market_data(ROOT / 'data' / 'market.sqlite')
    snapshot_dir = await asyncio.to_thread(prepare) if use_snapshot else None
    liquidation_symbols = ()
    if host in ('127.0.0.1', 'localhost', '::1'):
        import json
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
        liquidation_symbols = tuple(symbol for symbol in config.get('symbols', [])
                                    if isinstance(symbol, str) and symbol.endswith('-USD'))
    dashboard = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=dashboard.serve_forever, daemon=True).start()
    runner = None
    try:
        app = create_app(f'http://127.0.0.1:{dashboard.server_port}', snapshot_dir,
                         liquidation_symbols)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        print(f'QuantStack ready at http://{host}:{port}', flush=True)
        await asyncio.Event().wait()
    finally:
        if runner:
            await runner.cleanup()
        await asyncio.to_thread(dashboard.shutdown)
        dashboard.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8501')))
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.host, args.port))
    except KeyboardInterrupt:
        pass
