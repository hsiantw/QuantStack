"""Account isolation, cookie sessions, request protections and revision conflicts."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from aiohttp import ClientSession, CookieJar, web
from aiohttp.test_utils import TestServer

from accounts import Accounts, COOKIE, LIMIT, install_accounts


class AccountTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'accounts.sqlite'
        app = web.Application(client_max_size=3_000_000)
        install_accounts(app, self.path)
        self.server = TestServer(app)
        await self.server.start_server()
        self.origin = str(self.server.make_url('')).rstrip('/')
        self.clients = []
        self.a = self.client()
        self.b = self.client()

    def client(self):
        client = ClientSession(cookie_jar=CookieJar(unsafe=True))
        self.clients.append(client)
        return client

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        await self.server.close()
        self.temp.cleanup()

    async def request(self, client, action, data=None, origin=None):
        url = self.server.make_url('/workspace/api/account/' + action)
        if data is None:
            response = await client.get(url)
        else:
            response = await client.post(url, json=data, headers={'Origin': origin or self.origin})
        self.assertEqual(response.headers.get('Cache-Control'), 'no-store')
        return response, await response.json()

    async def register(self, client, username):
        response, data = await self.request(client, 'register', {'username': username, 'password': 'long-test-password'})
        self.assertEqual(response.status, 200, data)
        cookie = response.cookies[COOKIE]
        self.assertTrue(cookie['httponly'])
        self.assertEqual(cookie['samesite'], 'Strict')
        self.assertEqual(cookie['path'], '/workspace/api/account')
        return data['user']

    async def test_isolation_save_load_relogin_and_logout(self):
        alice = await self.register(self.a, 'Alice')
        bob = await self.register(self.b, 'Bob')
        entries = {'atlas.notes.TEST': 'Private research', 'atlas.drawings.TEST.1d': '[]', 'atlas.theme': 'dark'}
        response, saved = await self.request(self.a, 'workspace', {'account_id': alice['id'], 'revision': 0, 'entries': entries})
        self.assertEqual(response.status, 200)
        self.assertEqual(saved['user']['revision'], 1)
        _, separate = await self.request(self.b, 'workspace')
        self.assertEqual(separate['entries'], {})
        response, _ = await self.request(self.b, 'workspace', {'account_id': alice['id'], 'revision': 0, 'entries': entries})
        self.assertEqual(response.status, 409)
        other_device = self.client()
        response, _ = await self.request(other_device, 'workspace')
        self.assertEqual(response.status, 401)
        response, _ = await self.request(other_device, 'login', {'username': 'ALICE', 'password': 'long-test-password'})
        self.assertEqual(response.status, 200)
        _, loaded = await self.request(other_device, 'workspace')
        self.assertEqual(loaded['entries'], entries)
        # The database contains no plaintext password or session token.
        cookie = other_device.cookie_jar.filter_cookies(self.server.make_url('/workspace/api/account'))[COOKIE].value
        raw = self.path.read_bytes()
        self.assertNotIn(b'long-test-password', raw)
        self.assertNotIn(cookie.encode(), raw)
        self.assertEqual(Accounts(self.path).workspace(cookie)['entries'], entries)
        response, _ = await self.request(other_device, 'logout', {'account_id': alice['id']})
        self.assertEqual(response.status, 200)
        response, _ = await self.request(other_device, 'workspace')
        self.assertEqual(response.status, 401)
        self.assertEqual((await self.request(self.b, 'session'))[1]['user']['id'], bob['id'])

    async def test_csrf_conflicts_validation_expiration_and_limits(self):
        alice = await self.register(self.a, 'Alice')
        payload = {'account_id': alice['id'], 'revision': 0, 'entries': {'atlas.notes.TEST': 'Original'}}
        response, _ = await self.request(self.a, 'workspace', payload, origin='https://evil.invalid')
        self.assertEqual(response.status, 403)
        self.assertEqual((await self.request(self.a, 'workspace', payload))[0].status, 200)
        response, _ = await self.request(self.a, 'workspace', payload)
        self.assertEqual(response.status, 409)
        payload.update(revision=1, entries={'other-app.password': 'not allowed'})
        self.assertEqual((await self.request(self.a, 'workspace', payload))[0].status, 400)
        payload['entries'] = {'atlas.notes.TEST': 'x' * LIMIT}
        self.assertEqual((await self.request(self.a, 'workspace', payload))[0].status, 413)
        self.assertEqual((await self.request(self.a, 'workspace'))[1]['entries']['atlas.notes.TEST'], 'Original')
        response, _ = await self.request(self.b, 'login', {'username': 'Alice', 'password': 'incorrect-password'})
        self.assertEqual(response.status, 401)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('UPDATE sessions SET expires=0')
            db.execute('UPDATE auth_limits SET count=30')
        self.assertEqual((await self.request(self.a, 'workspace'))[0].status, 401)
        response, _ = await self.request(self.b, 'login', {'username': 'Alice', 'password': 'long-test-password'})
        self.assertEqual(response.status, 429)

    async def test_password_length_boundaries(self):
        for password in ('short12', 'x' * 129):
            response, _ = await self.request(self.a, 'register',
                {'username': 'Boundary', 'password': password})
            self.assertEqual(response.status, 400)
        for username, password in (('Eight', 'testpass'), ('Maximum', 'x' * 128)):
            payload = {'username': username, 'password': password}
            response, data = await self.request(self.a, 'register', payload)
            self.assertEqual(response.status, 200, data)
            response, data = await self.request(self.b, 'login', payload)
            self.assertEqual(response.status, 200, data)

    async def test_public_origin_secure_cookie(self):
        app = web.Application()
        install_accounts(app, self.path, 'https://quant.example')
        server = TestServer(app)
        await server.start_server()
        try:
            response = await self.a.post(server.make_url('/workspace/api/account/register'),
                json={'username': 'Public', 'password': 'long-test-password'}, headers={'Origin': 'https://quant.example'})
            self.assertEqual(response.status, 200)
            self.assertTrue(response.cookies[COOKIE]['secure'])
        finally:
            await server.close()


if __name__ == '__main__':
    unittest.main()
