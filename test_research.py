"""Independent numerical and invalid-input checks for native research tools."""
import math
import unittest
from pathlib import Path
from datetime import date, timedelta
from playwright.sync_api import sync_playwright


def bars(n=260, scale=1):
    price=100*scale
    result=[]
    for i in range(n):
        previous=price
        price*=math.exp(.001+.01*math.sin(i*.8))
        result.append(dict(date=(date(2025,1,1)+timedelta(days=i)).isoformat(), open=previous,
                           close=price, adjusted_close=price, high=max(previous,price)+1,
                           low=min(previous,price)-1, volume=1000+i))
    return result


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=sync_playwright().start()
        cls.browser=cls.p.chromium.launch(channel='msedge',headless=True)
        cls.page=cls.browser.new_page()
        cls.page.add_script_tag(path=str(Path('web/research-engine.js').resolve()))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.p.stop()

    def engine(self, kind, *args):
        return self.page.evaluate('({kind,args})=>AtlasResearch[kind](...args)',dict(kind=kind,args=args))

    def test_option_reference_and_put_call_parity(self):
        p=dict(spot=100,strike=100,days=365,volatility=.2,rate=.05,dividend=0)
        call=self.engine('option',p)
        put=self.engine('option',dict(p,type='put'))
        self.assertAlmostEqual(call['price'],10.4506,places=3)
        self.assertAlmostEqual(call['price']-put['price'],100-100*math.exp(-.05),places=6)
        self.assertGreater(call['gamma'],0)
        self.assertGreater(call['vega'],0)
        self.assertLess(call['theta'],0)
        bump=.01
        finite_delta=(self.engine('option',dict(p,spot=100+bump))['price']-self.engine('option',dict(p,spot=100-bump))['price'])/(2*bump)
        self.assertAlmostEqual(call['delta'],finite_delta,places=4)

    def test_portfolio_alignment_and_growth(self):
        a=bars();b=bars(scale=2)[10:]
        r=self.engine('portfolio',[a,b],[.4,.6])
        self.assertEqual(r['count'],250)
        self.assertAlmostEqual(r['return'],a[-1]['close']/a[10]['close']-1)
        self.assertAlmostEqual(r['correlations'][0][1],1)
        self.assertLessEqual(r['maxDrawdown'],0)

    def test_allocation_is_long_only_and_normalized(self):
        a=bars();b=bars()
        for i,r in enumerate(b):
            r['adjusted_close']=100*math.exp(.0004*i+.03*math.sin(i))
        for method in ['equal','inverse','minimum']:
            w=self.engine('allocation',[a,b],method)
            self.assertAlmostEqual(sum(w),1,places=6)
            self.assertTrue(all(0<=x<=1 for x in w))
        optimized=self.engine('portfolio',[a,b],self.engine('allocation',[a,b],'minimum'))
        equal=self.engine('portfolio',[a,b],[.5,.5])
        c=optimized['covariance'];w=self.engine('allocation',[a,b],'minimum')
        self.assertLessEqual(sum(w[i]*w[j]*c[i][j] for i in range(2) for j in range(2)),sum(sum(r) for r in equal['covariance'])/4+1e-10)

    def test_pairs_known_log_hedge_ratio(self):
        a=bars();b=bars()
        for r in a:r['adjusted_close']=r['adjusted_close']**2
        r=self.engine('pairs',a,b,60)
        self.assertAlmostEqual(r['beta'],2,places=8)
        self.assertIsNone(r['z'])

    def test_volume_baseline_and_bucket_total(self):
        a=bars(30)
        for r in a:r['volume']=100
        a[-1]['volume']=200
        result=self.engine('liquidity',a)
        self.assertEqual(result['relativeVolume'],2)
        self.assertEqual(sum(r['volume'] for r in result['profile']),3100)

    def test_forecast_validation_has_no_future_leak(self):
        a=bars();before=self.engine('forecast',a,10)
        a[-1]['adjusted_close']*=1.5
        after=self.engine('forecast',a,10)
        self.assertEqual(before['validation'][0],after['validation'][0])
        self.assertEqual(before['validation'][-1]['predicted'],after['validation'][-1]['predicted'])
        self.assertNotEqual(before['validation'][-1]['actual'],after['validation'][-1]['actual'])

    def test_invalid_and_missing_inputs(self):
        for kind,args in [('portfolio',[[bars(),bars()],[.8,.8]]),
                          ('option',[dict(spot=100,strike=100,days=0,volatility=.2)]),
                          ('forecast',[bars(30),20]), ('liquidity',[bars(10)])]:
            result=self.page.evaluate('({kind,args})=>{try{AtlasResearch[kind](...args);return false}catch{return true}}',dict(kind=kind,args=args))
            self.assertTrue(result,kind)
        a=bars();a[2]['adjusted_close']=None
        with self.assertRaises(Exception):self.engine('prices',a)


if __name__=='__main__': unittest.main()
