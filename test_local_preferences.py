import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import ClientSession, CookieJar
from aiohttp.test_utils import TestServer
from accounts import Accounts
from local_preferences import Preferences
from serve_quantstack import create_app


class PreferenceStoreTests(unittest.TestCase):
    def test_restart_patch_tombstone_and_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prefs.sqlite'
            store=Preferences(path)
            store.patch('local',{'atlas.saved':'["BTC-USD"]','atlas.theme':'dark'})
            store.patch('account:one',{'atlas.saved':'["ETH-USD"]'})
            store=Preferences(path)
            store.patch('local',{'atlas.theme':'paper'})
            self.assertEqual(store.read('local')['atlas.saved'],'["BTC-USD"]')
            self.assertEqual(store.read('account:one')['atlas.saved'],'["ETH-USD"]')
            store.patch('local',{'atlas.saved':None})
            self.assertIsNone(Preferences(path).read('local')['atlas.saved'])
            for invalid in ({'password':'secret'},{'atlas.theme':42},[],{'atlas.theme':'x'*2_000_001}):
                with self.assertRaises(ValueError): store.patch('local',invalid)
            self.assertEqual(store.read('local')['atlas.theme'],'paper')


class PreferenceAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'accounts.sqlite'
        self.server=TestServer(create_app('http://127.0.0.1:1',accounts_path=self.path))
        await self.server.start_server()
        self.base=str(self.server.make_url('')).rstrip('/')
        self.url=self.base+'/workspace/api/account/local-preferences'
        self.client=ClientSession(cookie_jar=CookieJar(unsafe=True))

    async def asyncTearDown(self):
        await self.client.close();await self.server.close();self.temp.cleanup()

    async def test_origin_checks_and_account_isolation(self):
        payload={'profile':'local','changes':{'atlas.saved':'["BTC-USD"]'}}
        async with self.client.post(self.url,json=payload,headers={'Origin':'https://example.com'}) as r:
            self.assertEqual(r.status,403)
        async with self.client.post(self.url,json=payload,headers={'Origin':self.base}) as r:
            self.assertEqual(r.status,200)
        async with self.client.get(self.url+'.js',headers={'Sec-Fetch-Site':'cross-site'}) as r:
            self.assertNotIn('BTC-USD',await r.text())
        async with self.client.get(self.url+'.js',headers={'Sec-Fetch-Site':'same-origin'}) as r:
            self.assertIn('BTC-USD',await r.text())
            self.assertEqual(r.headers['Cache-Control'],'no-store')
        user,token=Accounts(self.path).authenticate({'username':'prefuser','password':'long-password'},'test',True)
        self.client.cookie_jar.update_cookies({'quantstack_session':token})
        async with self.client.get(self.url) as r:
            state=await r.json();self.assertEqual(state['profile'],'account:'+user['id']);self.assertEqual(state['values'],{})
        async with self.client.post(self.url,json=payload,headers={'Origin':self.base}) as r:
            self.assertEqual(r.status,409)


if __name__=='__main__': unittest.main()
