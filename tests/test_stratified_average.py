from itertools import product
from types import SimpleNamespace
import copy,math
import pytest
from poker.cfr import TraversalLimit
from poker.stratified_average import StratifiedMCCFR,prefix_weight,fork_allowed


class Toy:
    def __init__(self,actors=(0,1,2,0),paid=1,h=()):self.actors=actors;self.paid=paid;self.h=h
    def terminal(self):return len(self.h)==len(self.actors)
    def info(self):return self.actors[len(self.h)],str(self.h),('call','fold')
    def child(self,a):return Toy(self.actors,self.paid,self.h+(('call','fold').index(a),))
    def returns(self):return [1.,-1.,0.]
    @property
    def game(self):return SimpleNamespace(observe=lambda:SimpleNamespace(legal=('call','fold'),call=self.paid,street=1,active=(True,True,True),player=self.actors[len(self.h)]))


class RNG:
    def __init__(self,path):self.path=iter(path);self.p=1.
    def choices(self,actions,weights):
        j=next(self.path);self.p*=weights[j]/sum(weights);return [actions[j]]


def exact(s,state,out,reach=1.):
    if state.terminal():return
    actor,key,actions=state.info();strategy=s.strategy(key,actions)
    if actor==0:out[key]=[reach*p for p in strategy]
    for action,p in zip(actions,strategy):exact(s,state.child(action),out,reach*(p if actor==0 else 1.))


@pytest.mark.parametrize('actors',[(0,1,2,0),(1,0,2,0)])
@pytest.mark.parametrize('paid',[0,1])
@pytest.mark.parametrize('root',[[1.,3.],[0.,3.]])
def test_fixed_component_expectation_matches_original_target(actors,paid,root):
    ref=StratifiedMCCFR();ref.regrets={str(()):root};expected={};exact(ref,Toy(actors,paid),expected)
    actual={}
    for mode,share in [(0,.75),(1,.25)]:
        total=0.
        for path in product(range(2),repeat=4):
            s=StratifiedMCCFR();s.regrets=ref.regrets;s.rng=RNG(path);s.average_path(Toy(actors,paid),0,mode)
            q=s.rng.p;total+=q
            for key,row in s.sums.items():
                dest=actual.setdefault(key,[0.,0.])
                for j,x in enumerate(row):dest[j]+=share*q*x
        assert total==pytest.approx(1.)
    for key,row in expected.items():assert actual.get(key,[0.,0.])==pytest.approx(row)


def test_bound_nonfinite_stop_and_fixed_schedule():
    for a,b in [(0,0),(-10,-math.inf),(-600,-5),(-20,-30)]:
        weight,ratio=prefix_weight(a,b,.125);assert ratio<=4/3+1e-12 and weight<=4/3*.125*math.exp(-a)*(1+1e-12)
    with pytest.raises(TraversalLimit,match='nonfinite'):prefix_weight(-800,-900,1.)
    with pytest.raises(ValueError,match='multiple'):StratifiedMCCFR(samples=7)
    s=StratifiedMCCFR();s.step(lambda seed:Toy(),3)
    assert s.diagnostics['averaging-component-v4:call']==18 and s.diagnostics['averaging-component-v4:guarded']==6
    assert s.traversal_stats['max_average_to_call_weight_ratio']<=4/3+1e-12


def test_failed_average_phase_rolls_back_every_learning_counter_and_rng():
    class FailsInAverage(StratifiedMCCFR):
        def average_path(self,*a,**kw):
            super().average_path(*a,**kw);raise TraversalLimit('deliberate average failure')
    s=FailsInAverage();before=copy.deepcopy((s.regrets,s.sums,s.nodes,s.iterations,s.rng.getstate(),s.diagnostics,s.traversal_stats,s.epoch_weight_moments))
    with pytest.raises(TraversalLimit,match='deliberate'):s.step(lambda seed:Toy(),3)
    assert before==(s.regrets,s.sums,s.nodes,s.iterations,s.rng.getstate(),s.diagnostics,s.traversal_stats,s.epoch_weight_moments)
