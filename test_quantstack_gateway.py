"""Integration tests for HTTP, websocket, and hosted-snapshot routing."""
import gzip
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, WSServerHandshakeError, web
from aiohttp.test_utils import TestServer

from serve_quantstack import create_app, has_market_data


class MarketDataDetectionTests(unittest.TestCase):
    def test_partial_schema_and_empty_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'market.sqlite'
            self.assertFalse(has_market_data(path))
            self.assertFalse(path.exists())
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE intraday_prices (close REAL)')
                self.assertFalse(has_market_data(path))
                db.execute('INSERT INTO intraday_prices VALUES (10)')
                db.commit()
            self.assertTrue(has_market_data(path))


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def echo(request):
            if request.path == '/_stcore/stream':
                socket = web.WebSocketResponse(protocols=('streamlit',))
                await socket.prepare(request)
                async for message in socket:
                    if isinstance(message.data, bytes):
                        await socket.send_bytes(message.data)
                    else:
                        await socket.send_str(message.data)
                return socket
            if request.path == '/symbols.json':
                return web.json_response({'cookie': request.headers.get('Cookie'), 'auth': request.headers.get('Authorization')})
            if request.path.startswith('/prices/'):
                return web.Response(body=gzip.compress(b'{"rows": []}'), content_type='application/gzip')
            if request.path == '/missing.json':
                raise web.HTTPNotFound()
            return web.json_response({'path': request.path_qs, 'body': (await request.read()).decode(),
                                      'cookie': request.headers.get('Cookie')},
                                     headers={'Set-Cookie': 'session=test; Path=/; HttpOnly'})

        backend = web.Application()
        backend.router.add_route('*', '/{path:.*}', echo)
        self.backend = TestServer(backend)
        await self.backend.start_server()
        url = str(self.backend.make_url('')).rstrip('/')
        self.gateway = TestServer(create_app(url))
        await self.gateway.start_server()
        self.client = ClientSession()
        self.temporary = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        await self.client.close()
        await self.gateway.close()
        await self.backend.close()
        self.temporary.cleanup()

    async def test_legacy_runtime_is_not_served(self):
        for path in ('/_stcore/health', '/_stcore/stream', '/portfolio_manager'):
            async with self.client.get(self.gateway.make_url(path)) as response:
                self.assertEqual(response.status, 404)
        async with self.client.post(self.gateway.make_url('/_stcore/upload_file'), data=b'old upload') as response:
            self.assertEqual(response.status, 404)

    async def test_workspace_prefix_and_query(self):
        async with self.client.get(self.gateway.make_url('/workspace/api/history?symbol=AAPL')) as response:
            self.assertEqual((await response.json())['path'], '/api/history?symbol=AAPL')
        async with self.client.get(self.gateway.make_url('/workspace'), allow_redirects=False) as response:
            self.assertEqual(response.status, 308)
            self.assertEqual(response.headers['Location'], '/workspace/')

    async def test_old_research_link_redirects_into_chart(self):
        for path, target in [('/?research=1', '/workspace/'),
                             ('/?research=1&symbol=MSFT&interval=1d', '/workspace/?symbol=MSFT&interval=1d')]:
            async with self.client.get(self.gateway.make_url(path), allow_redirects=False) as response:
                self.assertEqual(response.status, 302)
                self.assertEqual(response.headers['Location'], target)

    async def test_local_settings_reject_remote_host_and_origin(self):
        for headers in ({'Host': 'public.example'}, {'Origin': 'https://public.example'}):
            async with self.client.post(self.gateway.make_url('/workspace/api/local-scheduler'),
                                        headers=headers, json={'symbols': 'MSFT'}) as response:
                self.assertEqual(response.status, 403)

    async def use_snapshot(self):
        await self.gateway.close()
        url = str(self.backend.make_url('')).rstrip('/')
        directory = Path(self.temporary.name)
        (directory / 'prices').mkdir(exist_ok=True)
        (directory / 'symbols.json').write_text('[{"symbol":"AAPL"}]')
        (directory / 'prices' / 'AAPL.json.gz').write_bytes(gzip.compress(b'{"rows": []}'))
        self.gateway = TestServer(create_app(url, snapshot_dir=directory))
        await self.gateway.start_server()

    async def test_snapshot_serves_current_ui_and_local_prices(self):
        # Enrollment is a user setting; keep this fixture independent of it.
        from snapshot_catalog import snapshot_catalog
        config = Path(self.temporary.name) / 'configured.json'
        config.write_text(json.dumps({'symbols': ['SOL-USD']}))
        configured = patch('serve_quantstack.snapshot_catalog',
                           side_effect=lambda assets, _: snapshot_catalog(assets, config))
        configured.start()
        self.addCleanup(configured.stop)
        await self.use_snapshot()
        async with self.client.get(self.gateway.make_url('/workspace/')) as response:
            html = await response.text()
            self.assertIn('QuantStack', html)
            self.assertIn('src="./static-data.js"', html)
            self.assertLess(html.index('src="./static-data.js"'), html.index('src="./app.js"'))
        async with self.client.get(self.gateway.make_url('/workspace/symbols.json'),
                                    headers={'Cookie': 'private=test', 'Authorization': 'Bearer private'}) as response:
            assets = {asset['symbol']: asset for asset in await response.json()}
            self.assertEqual(assets['AAPL'], {'symbol': 'AAPL'})
            self.assertIn('SOL-USD', assets)
            self.assertFalse(assets['SOL-USD']['has_data'])
            self.assertEqual(assets['SOL-USD']['kind'], 'Crypto')
        async with self.client.get(self.gateway.make_url('/workspace/prices/AAPL.json.gz')) as response:
            self.assertEqual(json.loads(gzip.decompress(await response.read())), {'rows': []})

    async def test_snapshot_does_not_expose_project_files(self):
        await self.use_snapshot()
        for path in ('users.db', 'config.json', 'api/history', 'prices/subdir/file.json.gz'):
            async with self.client.get(self.gateway.make_url('/workspace/' + path)) as response:
                self.assertEqual(response.status, 404, path)
        async with self.client.get(self.gateway.make_url('/workspace/markov-worker.js')) as response:
            self.assertEqual(response.status, 200)
            self.assertIn('importScripts', await response.text())
        for name in ('research.js', 'notes.js', 'note-editor.js', 'research.css', 'research-worker.js', 'research-engine.js', 'risk.js'):
            async with self.client.get(self.gateway.make_url('/workspace/' + name)) as response:
                self.assertEqual(response.status, 200, name)


if __name__ == '__main__':
    unittest.main()
