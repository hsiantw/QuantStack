"""Independent numerical checks for geometric Brownian motion in headless Edge."""
import math
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright


class BrownianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel='msedge', headless=True)
        cls.page = cls.browser.new_page()
        cls.page.add_script_tag(path=str(Path('web/brownian-engine.js').resolve()))
        cls.page.evaluate('''() => {
          window.make = rs => {let price=100;return [0,...rs].map((r,i)=>{price*=Math.exp(r);return {date:new Date(Date.UTC(2020,0,i+1)).toISOString(),close:price,adjusted_close:price};});};
          window.fixture=make(Array.from({length:300},(_,i)=>Math.sin(i)*.02));
          window.run = p => AtlasBrownian.run(fixture,{horizon:20,paths:100,...p});
        }''')

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def test_calibration_sample_variance_and_drift(self):
        result = self.page.evaluate('''() => AtlasBrownian.calibrate(make(Array.from({length:20},(_,i)=>i%2?.02:-.01)),AtlasBrownian.options({window:20}))''')
        mean = .005
        variance = 20 * .015**2 / 19
        self.assertAlmostEqual(result['logMean'], mean)
        self.assertAlmostEqual(result['variance'], variance)
        self.assertAlmostEqual(result['mu'], mean + variance / 2)
        self.assertEqual(result['count'], 20)

    def test_exact_zero_volatility_paths(self):
        r = self.page.evaluate("run({mode:'manual',drift:.1,volatility:0})")
        expected = r['data']['price'] * math.exp(.001 * 20)
        for key in ('p05', 'p25', 'median', 'p75', 'p95', 'mean'):
            self.assertAlmostEqual(r['fan'][-1][key], expected, places=9)
        self.assertEqual(r['fan'][-1]['lossProbability'], 0)

    def test_zero_drift_mean_and_median_distinction(self):
        r = self.page.evaluate("run({mode:'zero'})")
        self.assertEqual(r['mu'], 0)
        self.assertAlmostEqual(r['theory']['mean'], r['data']['price'])
        self.assertLess(r['theory']['median'], r['theory']['mean'])

    def test_monte_carlo_matches_closed_form(self):
        r = self.page.evaluate("run({mode:'manual',drift:.1,volatility:3,paths:10000,seed:72})")
        expected = r['data']['price'] * math.exp(.001 * 20)
        self.assertLess(abs(r['fan'][-1]['mean'] / expected - 1), .01)
        center = (.001 - .03**2 / 2) * 20
        spread = .03 * math.sqrt(20)
        expected_loss = .5 * (1 + math.erf(-center / spread / math.sqrt(2)))
        self.assertAlmostEqual(r['theory']['lossProbability'], expected_loss, places=6)
        self.assertLess(abs(r['fan'][-1]['lossProbability'] - expected_loss), .02)
        for f in r['fan']:
            self.assertTrue(0 < f['p05'] <= f['p25'] <= f['median'] <= f['p75'] <= f['p95'])

    def test_seed_reproducibility_and_variation(self):
        self.assertTrue(self.page.evaluate('JSON.stringify(run({seed:0})) === JSON.stringify(run({seed:0}))'))
        self.assertTrue(self.page.evaluate('JSON.stringify(run({seed:0}).fan) !== JSON.stringify(run({seed:1}).fan)'))

    def test_reject_invalid_inputs_and_prices(self):
        self.page.evaluate('''() => {
          const reject = fn => {let failed=false;try{fn();}catch{failed=true;}if(!failed)throw Error('Invalid input accepted');};
          for(const p of [{horizon:0},{paths:10001},{seed:-1},{volatility:NaN},{window:19},{adjusted:'true'}])reject(()=>run(p));
          reject(()=>AtlasBrownian.run(fixture.slice(0,20)));
          const bad=structuredClone(fixture);bad[290].adjusted_close=null;reject(()=>AtlasBrownian.run(bad));
          AtlasBrownian.run(bad,{adjusted:false,paths:100,horizon:1});
          bad[290].adjusted_close=100;bad[290].date=bad[289].date;reject(()=>AtlasBrownian.run(bad));
        }''')


if __name__ == '__main__':
    unittest.main()
