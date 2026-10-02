"""Integration tests for HTTP, websocket, and hosted-snapshot routing."""
import gzip
import json
import unittest

from aiohttp import ClientSession, WSServerHandshakeError, web
from aiohttp.test_utils import TestServer

from serve_quantstack import create_app


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
        self.gateway = TestServer(create_app(url, url))
        await self.gateway.start_server()
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.gateway.close()
        await self.backend.close()

    async def test_streamlit_upload_and_cookies(self):
        async with self.client.post(self.gateway.make_url('/_stcore/upload_file?a=1'),
                                    data=b'uploaded content', headers={'Cookie': 'session=example'}) as response:
            self.assertEqual(response.status, 200)
            self.assertIn('session=test', response.headers['Set-Cookie'])
            self.assertEqual(await response.json(), {'path': '/_stcore/upload_file?a=1',
                             'body': 'uploaded content', 'cookie': 'session=example'})

    async def test_workspace_prefix_and_query(self):
        async with self.client.get(self.gateway.make_url('/workspace/api/history?symbol=AAPL')) as response:
            self.assertEqual((await response.json())['path'], '/api/history?symbol=AAPL')
        async with self.client.get(self.gateway.make_url('/workspace'), allow_redirects=False) as response:
            self.assertEqual(response.status, 308)
            self.assertEqual(response.headers['Location'], '/workspace/')

    async def test_streamlit_binary_websocket_and_subprotocol(self):
        async with self.client.ws_connect(self.gateway.make_url('/_stcore/stream'),
                                           protocols=['streamlit'], origin=str(self.gateway.make_url('')).rstrip('/')) as socket:
            self.assertEqual(socket.protocol, 'streamlit')
            await socket.send_bytes(b'protobuf payload')
            self.assertEqual((await socket.receive(timeout=3)).data, b'protobuf payload')
            await socket.send_str('rerun')
            self.assertEqual((await socket.receive(timeout=3)).data, 'rerun')

    async def test_reject_cross_origin_websocket(self):
        with self.assertRaises(WSServerHandshakeError) as raised:
            await self.client.ws_connect(self.gateway.make_url('/_stcore/stream'), origin='https://unrelated.example')
        self.assertEqual(raised.exception.status, 403)

    async def use_snapshot(self):
        await self.gateway.close()
        url = str(self.backend.make_url('')).rstrip('/')
        self.gateway = TestServer(create_app(url, url, snapshot_url=url))
        await self.gateway.start_server()

    async def test_snapshot_serves_current_ui_without_forwarding_credentials(self):
        await self.use_snapshot()
        async with self.client.get(self.gateway.make_url('/workspace/')) as response:
            html = await response.text()
            self.assertIn('QuantStack', html)
            self.assertIn('src="./static-data.js"', html)
            self.assertLess(html.index('src="./static-data.js"'), html.index('src="./app.js"'))
        async with self.client.get(self.gateway.make_url('/workspace/symbols.json'),
                                    headers={'Cookie': 'private=test', 'Authorization': 'Bearer private'}) as response:
            self.assertEqual(await response.json(), {'cookie': None, 'auth': None})
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


if __name__ == '__main__':
    unittest.main()
