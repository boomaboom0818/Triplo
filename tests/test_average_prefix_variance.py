from itertools import product
import pytest
from scripts.average_prefix_variance import AveragePrefix


class Toy:
    def __init__(self,h=()):self.h=h
    def terminal(self):return len(self.h)==4
    def info(self):return (0,1,2,0)[len(self.h)],str(self.h),('call','fold')
    def child(self,a):return Toy(self.h+(('call','fold').index(a),))


class RNG:
    def __init__(self,path):self.path=iter(path);self.p=1.
    def choices(self,actions,weights):
        j=next(self.path);self.p*=weights[j]/sum(weights);return [actions[j]]


def exact(solver,state,rows,reach=1.):
    if state.terminal():return
    actor,key,actions=state.info();strategy=solver.strategy(key,actions)
    if actor==0:rows[key]=[reach*p for p in strategy]
    for a,p in zip(actions,strategy):exact(solver,state.child(a),rows,reach*(p if actor==0 else 1))


@pytest.mark.parametrize('depth,draws',[(0,4),(1,5),(2,5)])
@pytest.mark.parametrize('root',[[1,3],[0,3]])
def test_external_prefix_average_exact_expectation(depth,draws,root):
    ref=AveragePrefix(depth);ref.regrets={str(()):root};expected={};exact(ref,Toy(),expected)
    actual={};total=0
    for path in product(range(2),repeat=draws):
        s=AveragePrefix(depth);s.regrets=ref.regrets;s.rng=RNG(path);s.trajectory_average(Toy(),0);q=s.rng.p;total+=q
        for key,row in s.sums.items():
            dest=actual.setdefault(key,[0.,0.])
            for j,x in enumerate(row):dest[j]+=q*x
    assert total==pytest.approx(1)
    for key in expected:assert actual[key]==pytest.approx(expected[key])
