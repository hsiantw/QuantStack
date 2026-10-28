"""Use an isolated local gateway/database to check persistence through real restarts."""
import asyncio
import json
from pathlib import Path
import tempfile

from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright
from serve_quantstack import create_app


async def run():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'accounts.sqlite'
        server=TestServer(create_app('http://127.0.0.1:8501/workspace',accounts_path=path))
        await server.start_server()
        port=server.port
        url=str(server.make_url('/workspace/?symbol=BTC-USD&interval=1m&period=MAX'))
        async with async_playwright() as p:
            browser=await p.chromium.launch(channel='msedge',headless=True)
            context=await browser.new_context()
            page=await context.new_page()
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            watchlist=json.dumps({'lists':[{'id':'list-persistence','name':'My BTC list','symbols':['BTC-USD']}],
                                  'active':'list-persistence','labels':{'BTC-USD':'blue'},'color':'all'})
            await context.add_init_script('localStorage.setItem("atlas.watchlists.v1",'+json.dumps(watchlist)+');')
            await page.goto(url)
            await page.wait_for_function("window.QuantLocalPreferences && typeof rows!=='undefined' && rows.length>0")
            await page.evaluate("async()=>{localStorage.setItem('atlas.saved','[\"BTC-USD\"]');localStorage.setItem('atlas.timeframeFavorites.v1','[\"1m\",\"30m\"]');await QuantLocalPreferences.flush()}")
            data=await page.evaluate("async()=>(await fetch('/workspace/api/account/local-preferences')).json()")
            assert data['values']['atlas.watchlists.v1']==watchlist,data
            assert data['values']['atlas.saved']=='["BTC-USD"]'
            await context.close()
            # Restart the app with the same disk database; use an empty browser profile.
            await server.close()
            server=TestServer(create_app('http://127.0.0.1:8501/workspace',accounts_path=path),port=port)
            await server.start_server()
            context=await browser.new_context()
            page=await context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
            await page.goto(url)
            await page.wait_for_function("typeof rows!=='undefined' && rows.length>0")
            assert await page.evaluate("localStorage.getItem('atlas.watchlists.v1')")==watchlist
            assert await page.locator('#watchlistSelect').input_value()=='list-persistence'
            assert await page.locator('#intervals [data-interval="30m"]').count()==1
            # An offline save stays in the browser and retries when the server returns.
            await page.route('**/api/account/local-preferences',lambda route:route.abort(),times=1)
            await page.evaluate("async()=>{localStorage.setItem('atlas.theme','paper');await QuantLocalPreferences.flush()}")
            await page.wait_for_function("document.getElementById('localPreferenceStatus').textContent==='Saved on this PC'",timeout=10000)
            data=await page.evaluate("async()=>(await fetch('/workspace/api/account/local-preferences')).json()")
            assert data['values']['atlas.theme']=='paper'
            # A second tab can change another setting without erasing this tab's list.
            second=await context.new_page();await second.goto(url)
            await second.evaluate("async()=>{localStorage.setItem('atlas.workspace.watchlistWidth','320');await QuantLocalPreferences.flush()}")
            await page.evaluate("async()=>{localStorage.removeItem('atlas.saved');await QuantLocalPreferences.flush()}")
            data=await page.evaluate("async()=>(await fetch('/workspace/api/account/local-preferences')).json()")
            assert data['values']['atlas.saved'] is None
            assert data['values']['atlas.watchlists.v1']==watchlist
            assert data['values']['atlas.workspace.watchlistWidth']=='320'
            await second.close()
            page.on('dialog',lambda dialog:dialog.accept())
            await page.locator('#accountButton').click()
            await page.locator('#accountUsername').fill('preference-user')
            await page.locator('#accountPassword').fill('preference-test-password')
            await page.locator('#accountRegister').click()
            await page.wait_for_function("localStorage.getItem('quantstack.localPreferenceProfile').startsWith('account:')")
            account_list=watchlist.replace('My BTC list','Account BTC list')
            await page.evaluate("async value=>{localStorage.setItem('atlas.watchlists.v1',value);await QuantLocalPreferences.flush()}",account_list)
            await page.locator('#accountLogout').click()
            await page.wait_for_function("localStorage.getItem('quantstack.localPreferenceProfile')==='local'")
            assert await page.evaluate("localStorage.getItem('atlas.watchlists.v1')")==watchlist
            await page.locator('#accountButton').click()
            await page.locator('#accountUsername').fill('preference-user')
            await page.locator('#accountPassword').fill('preference-test-password')
            await page.locator('#accountLogin').click()
            await page.wait_for_function("localStorage.getItem('quantstack.localPreferenceProfile').startsWith('account:')")
            assert await page.evaluate("localStorage.getItem('atlas.watchlists.v1')")==account_list
            assert not errors,errors
            await context.close();await browser.close()
        await server.close()
        print('PASS: migration, app restart, fresh browser restore, favorites, retry, tab saves, deletions, account logout/login restoration')


if __name__=='__main__': asyncio.run(run())
