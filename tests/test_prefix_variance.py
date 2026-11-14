from itertools import product
import pytest
from poker.cfr import TraversalLimit
from scripts.prefix_variance import PrefixMCCFR


class Toy:
    def __init__(self,h=()):self.h=h
    def terminal(self):return len(self.h)==3
    def info(self):return (0,1,0)[len(self.h)],str(self.h),(0,1)
    def child(self,a):return Toy(self.h+(a,))
    def returns(self):return [(2*self.h[0]-1)*(3*self.h[2]+1)+self.h[1],0]


class RNG:
    def __init__(self,path):self.path=iter(path);self.p=1.
    def choices(self,actions,weights):
        j=next(self.path);self.p*=weights[j]/sum(weights);return [actions[j]]


def exact(solver,state,delta,reach=1.):
    if state.terminal():return state.returns()[0]
    actor,key,actions=state.info();strategy=solver.strategy(key,actions)
    values=[exact(solver,state.child(a),delta,reach*(p if actor else 1)) for a,p in zip(actions,strategy)]
    value=sum(p*v for p,v in zip(strategy,values))
    if actor==0:delta[key]=[reach*(v-value) for v in values]
    return value


@pytest.mark.parametrize('depth,draws',[(0,3),(1,4),(2,2)])
@pytest.mark.parametrize('root',[[1,3],[0,3]])
def test_prefix_estimates_exact_regret_expectation(depth,draws,root):
    reference=PrefixMCCFR(depth=depth);reference.regrets={str(()):root,str((0,)):[2,1]}
    expected={};value=exact(reference,Toy(),expected);actual={};mean=0;p=0
    for path in product(range(2),repeat=draws):
        s=PrefixMCCFR(depth=depth);s.regrets=reference.regrets;s.rng=RNG(path);delta={}
        v=s.outcome(Toy(),0,delta);q=s.rng.p;mean+=q*v;p+=q
        for key,row in delta.items():
            dest=actual.setdefault(key,[0.,0.])
            for j,x in enumerate(row):dest[j]+=q*x
    assert p==pytest.approx(1);assert mean==pytest.approx(value)
    for key in expected:assert actual[key]==pytest.approx(expected[key])


def test_prefix_budget_rolls_back_and_depth_is_bounded():
    s=PrefixMCCFR(depth=2);before=s.rng.getstate()
    with pytest.raises(TraversalLimit):s.step(lambda seed:Toy(),2,max_nodes=2)
    assert s.rng.getstate()==before and s.regrets=={} and s.sums=={} and s.iterations==0
    with pytest.raises(ValueError):PrefixMCCFR(depth=3)
