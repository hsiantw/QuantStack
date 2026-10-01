"""Numerical and causal invariants of the browser Markov engine (headless Edge)."""
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright


class MarkovTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel='msedge', headless=True)
        cls.page = cls.browser.new_page()
        cls.page.add_script_tag(path=str(Path('web/markov-engine.js').resolve()))
        cls.page.evaluate('''() => {
          window.assert = (ok, message) => { if (!ok) throw Error(message); };
          window.near = (actual, expected, tolerance=1e-10) => assert(Math.abs(actual-expected)<tolerance, `${actual} != ${expected}`);
          window.make = (returns) => {
            let price=100;
            return [0,...returns].map((r,i) => {price*=Math.exp(r);return {date:new Date(Date.UTC(2020,0,i+1)).toISOString(),close:price,adjusted_close:price};});
          };
          window.fixture = make(Array.from({length:300},(_,i)=>Math.sin(i*.7)*.025+Math.cos(i*.13)*.012));
          window.base = {paths:100,horizon:10};
          window.reject = (fn, pattern='') => {let error=null;try {fn();} catch(e) {error=e;} assert(error && error.message.includes(pattern),'Expected rejection: '+pattern);};
        }''')

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def check(self, code):
        self.page.evaluate('() => {' + code + '}')

    def test_hand_counted_transitions_and_smoothing(self):
        self.check('''
          const c=AtlasMarkov.countsFor([0,1,0,0,2,1],3);
          assert(JSON.stringify(c)===JSON.stringify([[1,1,1],[1,0,0],[0,1,0]]),'Adjacent counts');
          const p=AtlasMarkov.probabilities(c,.5);
          near(p[0][0],1/3);near(p[1][0],.6);near(p[1][1],.2);
          p.forEach(row=>near(row.reduce((a,b)=>a+b,0),1));
          const empty=AtlasMarkov.probabilities([[0,0,0],[1,0,0],[0,1,0]],0);near(empty[0][0],1/3);
        ''')

    def test_probability_propagation_by_hand(self):
        self.check('''
          const p=[[.8,.2],[.3,.7]];
          const one=AtlasMarkov.step([1,0],p),two=AtlasMarkov.step(one,p);
          near(one[1],.2);near(two[0],.7);near(two[1],.3);
        ''')

    def test_stationary_distribution_and_residual(self):
        self.check('''
          const c=AtlasMarkov.structure([[.8,.2],[.3,.7]]);
          near(c.stationary[0],.6);near(c.stationary[1],.4);near(c.residual,0);
          assert(c.irreducible && c.periods[0]===1,'Irreducible and aperiodic');
        ''')

    def test_periodic_chain(self):
        self.check('''
          const c=AtlasMarkov.structure([[0,1,0],[0,0,1],[1,0,0]]);
          assert(c.periods[0]===3,'Three-cycle period');c.stationary.forEach(x=>near(x,1/3));
          const two=AtlasMarkov.structure([[0,1],[1,0]]);assert(two.periods[0]===2,'Two-cycle period');
        ''')

    def test_ill_conditioned_stationary_system_is_not_misclassified(self):
        self.check('''
          const c=AtlasMarkov.structure([[1-1e-14,1e-14],[1e-14,1-1e-14]]);
          assert(c.closed.length===1 && c.irreducible,'A numerically difficult system still has one closed class');
          assert(c.stationary===null || c.residual<1e-10,'Do not report an inaccurate stationary solution');
        ''')

    def test_intraday_timestamps_and_single_bar_horizon(self):
        self.check('''
          const hourly=structuredClone(fixture);
          hourly.forEach((b,i)=>b.date=new Date(Date.UTC(2026,0,1)+i*3600000).toISOString());
          const r=AtlasMarkov.run(hourly,{...base,horizon:1});
          assert(r.forecasts.length===1&&r.simulation.fan.length===1,'Single horizon');
          assert(r.history.at(-1).date===hourly.at(-1).date,'Hourly timestamps retained');
          r.forecasts[0].probabilities.forEach((x,i)=>near(x,r.matrix[r.current][i]));
        ''')

    def test_reducible_and_absorbing_chains(self):
        self.check('''
          const multiple=AtlasMarkov.structure([[1,0,0],[0,1,0],[.5,.5,0]]);
          assert(multiple.stationary===null && multiple.closed.length===2,'No unique stationary distribution');
          const single=AtlasMarkov.structure([[1,0],[.5,.5]]);
          near(single.stationary[0],1);near(single.stationary[1],0);assert(!single.irreducible,'Transient state');
        ''')

    def test_fixed_boundaries_and_quantile_ties(self):
        self.check('''
          const e=[-.01,.01];
          assert(AtlasMarkov.classify(-.02,e,'fixed')===0,'Down');
          assert(AtlasMarkov.classify(-.01,e,'fixed')===1,'Lower equality neutral');
          assert(AtlasMarkov.classify(.01,e,'fixed')===1,'Upper equality neutral');
          assert(AtlasMarkov.classify(.02,e,'fixed')===2,'Up');
          assert(AtlasMarkov.classify(0,[0,0],'quantile')===0,'Ties lower');
        ''')

    def test_wilson_intervals(self):
        self.check('''
          assert(AtlasMarkov.wilson(0,0)===null,'No sample');
          const middle=AtlasMarkov.wilson(50,100);near(middle[0],.4038315303659956);near(middle[1],.5961684696340044);
          const empty=AtlasMarkov.wilson(0,10);near(empty[0],0);near(empty[1],.2775327998628892);
          const all=AtlasMarkov.wilson(10,10);near(all[1],1);
        ''')

    def test_validation_is_score_before_update(self):
        self.check('''
          const states=[0,1,0,1,2,2],v=AtlasMarkov.validate(states,4,3,{alpha:1,validation:'expanding'});
          // Training transitions: 0->1 twice, 1->0 once. First held-out origin is 1.
          near(v.predictions[0].probabilities[0],.5);near(v.predictions[0].probabilities[2],.25);
          near(v.predictions[0].baseline[2],1/7);
          // The newly observed state 2 has no outgoing transition before its first prediction.
          v.predictions[1].probabilities.forEach(x=>near(x,1/3));
          near(v.predictions[1].baseline[2],2/8);
        ''')

    def test_frozen_validation(self):
        self.check('''
          const v=AtlasMarkov.validate([0,1,0,1,2,2],4,3,{alpha:1,validation:'frozen'});
          near(v.predictions[0].baseline[2],1/7);near(v.predictions[1].baseline[2],1/7);
        ''')

    def test_scores_by_hand(self):
        self.check('''
          const v=AtlasMarkov.validate([0,1,0,1],3,2,{alpha:0,validation:'frozen'});
          near(v.model.brier,0);near(v.model.logLoss,0);near(v.model.accuracy,1);
          near(v.baseline.brier,8/9);near(v.baseline.logLoss,Math.log(3));near(v.skill,1);
          assert(v.confusion[1][1]===1,'Confusion orientation');
          const impossible=AtlasMarkov.validate([0,0,0,1],3,2,{alpha:0,validation:'frozen'});
          near(impossible.model.brier,2);near(impossible.model.logLoss,-Math.log(1e-15));
        ''')

    def test_future_changes_do_not_change_earlier_validation(self):
        self.check('''
          const original=AtlasMarkov.run(fixture,base),changed=structuredClone(fixture);
          for(let i=270;i<changed.length;i++) {changed[i].close*=1.2;changed[i].adjusted_close*=1.2;}
          const later=AtlasMarkov.run(changed,base);
          assert(JSON.stringify(original.edges)===JSON.stringify(later.edges),'Edges fitted only to training');
          assert(JSON.stringify(original.validation.predictions.filter(x=>x.index<269))===JSON.stringify(later.validation.predictions.filter(x=>x.index<269)),'No future leakage');
          assert(JSON.stringify(original.matrix)!==JSON.stringify(later.matrix),'Final estimates refit full window');
        ''')

    def test_state_and_forecast_probability_invariants(self):
        self.check('''
          for(const states of [3,5]) for(const alpha of [0,.5,10]) {
            const r=AtlasMarkov.run(fixture,{...base,states,alpha});
            assert(r.counts.flat().reduce((a,b)=>a+b,0)===r.data.returns-1,'Transition total');
            assert(r.summary.reduce((a,s)=>a+s.count,0)===r.data.returns,'Occupancy total');
            for(const row of [...r.matrix,...r.forecasts.map(f=>f.probabilities)]) {near(row.reduce((a,b)=>a+b,0),1);assert(row.every(x=>x>=0&&x<=1),'Probability bounds');}
            r.forecasts.forEach((f,i)=>f.hit.forEach((x,j)=>{assert(x>=0&&x<=1,'Hit bounds');if(i)assert(x+1e-12>=r.forecasts[i-1].hit[j],'Hit probability monotone');}));
            near(r.forecasts.at(-1).hit[r.current],1);
            assert(!JSON.stringify(r).includes('null,null,null'),'No numerical gaps in probability vectors');
          }
        ''')

    def test_reproducible_simulations_and_ordered_percentiles(self):
        self.check('''
          const a=AtlasMarkov.run(fixture,{...base,seed:7}),b=AtlasMarkov.run(fixture,{...base,seed:7}),c=AtlasMarkov.run(fixture,{...base,seed:8});
          assert(JSON.stringify(a.simulation)===JSON.stringify(b.simulation),'Seed reproducibility');
          assert(JSON.stringify(a.simulation)!==JSON.stringify(c.simulation),'Different seed changes paths');
          assert(JSON.stringify(a.matrix)===JSON.stringify(c.matrix),'Seed does not affect fitting');
          a.simulation.fan.forEach(f=>{const values=[f.p05,f.p25,f.median,f.p75,f.p95];assert(values.every((x,i)=>x>=-100&&(!i||x>=values[i-1])),'Ordered return percentiles');});
          assert(a.simulation.drawdownP05<=a.simulation.drawdownMedian&&a.simulation.drawdownMedian<=0,'Drawdown sign and order');
        ''')

    def test_absorbing_flat_history_and_empty_states(self):
        self.check('''
          const flat=make(Array(100).fill(0));
          const r=AtlasMarkov.run(flat,{...base,mode:'fixed',threshold:.5,alpha:0});
          assert(r.current===1&&r.streak===100,'Flat state and run');near(r.matrix[1][1],1);
          assert(r.summary[1].dwell===null,'Infinite dwell sentinel');
          assert(r.simulation.available,'Unreachable empty states do not prevent simulation');
          r.simulation.fan.forEach(f=>{near(f.median,0);near(f.lossProbability,0);});
          const smoothed=AtlasMarkov.run(flat,{...base,mode:'fixed',alpha:.5});
          assert(!smoothed.simulation.available,'Reachable empty states disable simulations');
          const tied=AtlasMarkov.run(flat,base);assert(tied.warnings.some(x=>x.includes('Repeated quantile')),'Quantile ties disclosed');
        ''')

    def test_hit_probability_by_enumerating_paths(self):
        self.check('''
          const r=AtlasMarkov.run(fixture,{...base,horizon:2}),target=(r.current+1)%3,p=r.matrix;
          let hit=p[r.current][target];
          for(let j=0;j<3;j++) if(j!==target)hit+=p[r.current][j]*p[j][target];
          near(r.forecasts[1].hit[target],hit);
        ''')

    def test_basis_and_window(self):
        self.check('''
          const bars=structuredClone(fixture);bars.forEach((b,i)=>b.close=i%2?50:100);
          const adjusted=AtlasMarkov.run(bars,{...base,window:100}),raw=AtlasMarkov.run(bars,{...base,window:100,adjusted:false});
          assert(adjusted.data.returns===100&&adjusted.history.length===100,'Window uses preceding bar');
          assert(adjusted.data.start===bars.at(-101).date,'Window start');
          near(raw.history[0].logReturn,Math.log(bars.at(-100).close/bars.at(-101).close));
          assert(JSON.stringify(raw.edges)!==JSON.stringify(adjusted.edges),'Price basis honored');
          const shifted=structuredClone(fixture);shifted[0].close=null;shifted[0].adjusted_close=null;
          AtlasMarkov.run(shifted,{...base,window:100});
        ''')

    def test_malformed_data_rejected(self):
        self.check('''
          reject(()=>AtlasMarkov.run([],base),'61 price bars');
          for(const value of [null,0,-1,NaN,Infinity,'100']) {const bad=structuredClone(fixture);bad[100].adjusted_close=value;reject(()=>AtlasMarkov.run(bad,base),'close');}
          for(const value of [fixture[99].date,'invalid']) {const bad=structuredClone(fixture);bad[100].date=value;reject(()=>AtlasMarkov.run(bad,base),'timestamps');}
          reject(()=>AtlasMarkov.run(fixture.slice().reverse(),base),'timestamps');
          const missing=structuredClone(fixture);delete missing[100].adjusted_close;reject(()=>AtlasMarkov.run(missing,base),'adjusted');
        ''')

    def test_invalid_configuration_rejected(self):
        self.check('''
          for(const options of [{states:4},{horizon:0},{horizon:251},{window:20},{train:100},{alpha:-1},{paths:10001},{seed:-1},{mode:'invalid'},{validation:'invalid'},{adjusted:'true'},{horizon:1.2},{alpha:null}])reject(()=>AtlasMarkov.run(fixture,{...base,...options}));
        ''')

    def test_snapshot_does_not_mutate_input(self):
        self.check('''
          const bars=structuredClone(fixture),before=JSON.stringify(bars),settings={...base};
          const r=AtlasMarkov.run(bars,settings);settings.horizon=200;
          assert(JSON.stringify(bars)===before,'Inputs untouched');assert(r.parameters.horizon===10,'Settings snapshot');
        ''')


if __name__ == '__main__':
    unittest.main()
