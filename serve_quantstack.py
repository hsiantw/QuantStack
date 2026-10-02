"""Serve the Streamlit app and market workspace on one public port."""
import argparse
import asyncio
import contextlib
import os
from pathlib import Path
import re
import socket
import sqlite3
import sys
import threading
from http.server import ThreadingHTTPServer

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

from dashboard import Handler
from prepare_snapshot import data_file, prepare

ROOT = Path(__file__).resolve().parent
STREAMLIT = web.AppKey('streamlit', str)
DASHBOARD = web.AppKey('dashboard', str)
SNAPSHOT_PATH = web.AppKey('snapshot_path', Path)
CLIENT = web.AppKey('client', ClientSession)
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade'}


def forwarded_headers(headers):
    excluded = HOP_HEADERS | {h.strip().lower() for h in headers.get('Connection', '').split(',')}
    return [(key, value) for key, value in headers.items() if key.lower() not in excluded]


async def relay(source, destination):
    async for message in source:
        if message.type == WSMsgType.TEXT:
            await destination.send_str(message.data)
        elif message.type == WSMsgType.BINARY:
            await destination.send_bytes(message.data)
        elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            break


def has_market_data(path):
    if not path.is_file():
        return False
    try:
        with contextlib.closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
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
    if re.fullmatch(r'[a-z-]+\.(js|css)', name) and (ROOT / 'web' / name).is_file():
        return web.FileResponse(ROOT / 'web' / name)
    if not data_file(name):
        raise web.HTTPNotFound()
    path = request.app[SNAPSHOT_PATH] / name
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={'X-Content-Type-Options': 'nosniff'})


async def proxy(request):
    if request.path == '/' and request.method in ('GET', 'HEAD') and request.query.get('research') != '1':
        target = '/workspace/'
        if request.query_string:
            target += '?' + request.query_string
        raise web.HTTPFound(target)
    workspace = request.path == '/workspace' or request.path.startswith('/workspace/')
    if request.path == '/workspace':
        raise web.HTTPPermanentRedirect('/workspace/')
    if workspace and request.app[SNAPSHOT_PATH]:
        return await snapshot(request)
    upstream = request.app[DASHBOARD] if workspace else request.app[STREAMLIT]
    path = request.raw_path[len('/workspace'):] if workspace else request.raw_path
    url = upstream + path
    session = request.app[CLIENT]
    headers = forwarded_headers(request.headers)
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        # Reject cross-origin browser connections before forwarding Streamlit sessions.
        origin = request.headers.get('Origin')
        if origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != request.host:
                raise web.HTTPForbidden(text='WebSocket origin does not match this site.')
        protocols = [p.strip() for p in request.headers.get('Sec-WebSocket-Protocol', '').split(',') if p.strip()]
        headers = [(k, v) for k, v in headers if not k.lower().startswith('sec-websocket-')]
        try:
            async with session.ws_connect(url, headers=headers, protocols=protocols,
                                          max_msg_size=200 * 1024**2) as backend:
                frontend = web.WebSocketResponse(protocols=[backend.protocol] if backend.protocol else (),
                                                  max_msg_size=200 * 1024**2)
                await frontend.prepare(request)
                tasks = [asyncio.create_task(relay(frontend, backend)),
                         asyncio.create_task(relay(backend, frontend))]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await frontend.close()
                return frontend
        except (ClientError, OSError, asyncio.TimeoutError) as exc:
            raise web.HTTPBadGateway(text='QuantStack is starting. Please retry.') from exc
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


def create_app(streamlit_url, dashboard_url, snapshot_dir=None):
    app = web.Application(client_max_size=200 * 1024**2)
    app[STREAMLIT], app[DASHBOARD] = streamlit_url, dashboard_url
    app[SNAPSHOT_PATH] = snapshot_dir

    async def client_context(application):
        async with ClientSession(auto_decompress=False, timeout=ClientTimeout(total=None, sock_connect=10)) as session:
            application[CLIENT] = session
            yield

    app.cleanup_ctx.append(client_context)
    app.router.add_route('*', '/{path:.*}', proxy)
    return app


async def serve(host, port):
    use_snapshot = not has_market_data(ROOT / 'data' / 'market.sqlite')
    snapshot_dir = await asyncio.to_thread(prepare) if use_snapshot else None
    dashboard = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=dashboard.serve_forever, daemon=True).start()
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        streamlit_port = reservation.getsockname()[1]
    environment = dict(os.environ, QUANTSTACK_WORKSPACE_URL='/workspace/',
                       QUANTSTACK_WORKSPACE_MODE='snapshot' if use_snapshot else 'stored')
    process = None
    runner = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'QuantStack-main' / 'app.py'),
            '--server.address=127.0.0.1', f'--server.port={streamlit_port}',
            '--server.headless=true', '--server.enableStaticServing=true', '--server.enableCORS=false',
            '--server.enableXsrfProtection=true', '--browser.gatherUsageStats=false',
            cwd=ROOT, env=environment)
        streamlit_url = f'http://127.0.0.1:{streamlit_port}'
        async with ClientSession() as session:
            for _ in range(120):
                if process.returncode is not None:
                    raise RuntimeError('Streamlit exited during startup.')
                try:
                    async with session.get(streamlit_url + '/_stcore/health') as response:
                        if response.status == 200:
                            break
                except (ClientError, OSError):
                    pass
                await asyncio.sleep(.5)
            else:
                raise RuntimeError('Streamlit did not become ready within 60 seconds.')
        app = create_app(streamlit_url, f'http://127.0.0.1:{dashboard.server_port}', snapshot_dir)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        print(f'QuantStack ready at http://{host}:{port}', flush=True)
        await process.wait()
        raise RuntimeError(f'Streamlit stopped (exit {process.returncode}).')
    finally:
        if runner:
            await runner.cleanup()
        if process and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
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
