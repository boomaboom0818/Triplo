from itertools import product
from types import SimpleNamespace
from random import Random
import math
import pytest
from poker.cfr import TraversalLimit
from scripts.average_mis_probe import BalancedAverage,Probe,balance_weight
from poker.trajectory import TrajectoryMCCFR
from poker.cfr import Holdem
from poker.train_cfr import train
from scripts import average_mis_probe as probe


class Toy:
    def __init__(self,actors,paid,h=()):self.actors=actors;self.paid=paid;self.h=h
    def terminal(self):return len(self.h)==len(self.actors)
    def info(self):return self.actors[len(self.h)],str(self.h),('call','fold')
    def child(self,a):return Toy(self.actors,self.paid,self.h+(('call','fold').index(a),))
    @property
    def game(self):
        return SimpleNamespace(observe=lambda:SimpleNamespace(legal=('call','fold'),call=self.paid,street=1,active=(True,True,True),player=self.actors[len(self.h)]))


class RNG:
    def __init__(self,path):self.path=iter(path);self.p=1.;self.draws=0
    def choices(self,actions,weights):
        j=next(self.path);self.p*=weights[j]/sum(weights);self.draws+=1;return [actions[j]]


def exhaustive(solver,state,rows,reach=1.):
    if state.terminal():return
    actor,key,actions=state.info();strategy=solver.strategy(key,actions)
    if actor==0:rows[key]=[reach*p for p in strategy]
    for a,p in zip(actions,strategy):exhaustive(solver,state.child(a),rows,reach*(p if actor==0 else 1.))


@pytest.mark.parametrize('actors',[(0,1,2,0),(1,0,2,0)])
@pytest.mark.parametrize('root',[[1,3],[0,3]])
@pytest.mark.parametrize('paid',[0,1])
def test_exact_marginal_mixture_expectation_with_repeated_owner_and_zero_support(actors,root,paid):
    ref=BalancedAverage();ref.regrets={str(()):root};expected={};exhaustive(ref,Toy(actors,paid),expected)
    actual={};total=0.
    for path in product(range(2),repeat=5):
        s=BalancedAverage();s.regrets=ref.regrets;s.rng=RNG(path);s.trajectory_average(Toy(actors,paid),0)
        q=s.rng.p;total+=q
        assert s.rng.draws==5 and s.max_weight_ratio<=2+1e-12
        for key,row in s.sums.items():
            dest=actual.setdefault(key,[0.,0.])
            for j,x in enumerate(row):dest[j]+=q*x
    assert total==pytest.approx(1)
    for key,row in expected.items():assert actual.get(key,[0.,0.])==pytest.approx(row)


@pytest.mark.parametrize('loga,logb',[(0,0),(-5,-math.inf),(-600,-5),(-25,-30)])
def test_pointwise_bound_uses_external_prefix_marginal(loga,logb):
    weight,ratio=balance_weight(loga,logb,.125)
    assert weight<=2*.125*math.exp(-loga)*(1+1e-12)
    assert ratio<=2+1e-12


def test_nonfinite_second_moment_stops_without_clipping():
    with pytest.raises(TraversalLimit,match='nonfinite'):balance_weight(-800,-900,1.)
    solver=BalancedAverage();before=solver.rng.getstate()
    with pytest.raises(RuntimeError,match='read-only'):solver.step(lambda seed:Toy((0,1,2,0),1),3,max_nodes=1)
    assert solver.rng.getstate()==before and solver.regrets=={} and solver.sums=={} and solver.iterations==0


def test_instrumented_baseline_is_exact_original_sampler():
    state=Holdem(9900000,6,'texture-buckets-v2');owner=state.info()[0]
    a=Probe();b=TrajectoryMCCFR();a.rng=Random(123);b.rng=Random(123)
    a.trajectory_average(state,owner);b.trajectory_average(state,owner)
    assert a.sums==b.sums and a.nodes==b.nodes and a.rng.getstate()==b.rng.getstate()


def test_readonly_probe_preserves_checkpoint_bytes_and_validates_inputs(tmp_path,monkeypatch):
    parents=[tmp_path/'a.json',tmp_path/'b.json']
    for i,p in enumerate(parents):train(1,p,game='holdem',seed=11+i*18)
    original=[p.read_bytes() for p in parents]
    monkeypatch.setattr(probe,'PARENTS',parents);monkeypatch.setattr(probe,'REPLICATES',2)
    monkeypatch.setattr(probe,'available_memory',lambda:10*1024**3)
    monkeypatch.setattr(probe.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    import time,json
    output=tmp_path/'probe.json';probe.main(output,time.time()+30)
    result=json.loads(output.read_text());manifest=json.loads(output.with_suffix('.manifest.json').read_text())
    assert result['status']=='completed' and len(result['fixtures'])==4 and result['read_only']
    assert original==[p.read_bytes() for p in parents]
    probe.verify(manifest)
    with parents[0].open('a') as h:h.write(' ')
    with pytest.raises(ValueError,match='frozen'):probe.verify(manifest)
