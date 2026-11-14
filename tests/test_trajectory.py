"""Exact expectation tests cover earlier-own and later-own reach corrections."""
from itertools import product
import json
import pytest
from poker.cfr import MCCFR,normalize,TraversalLimit
from poker.trajectory import TrajectoryMCCFR,ALGORITHM,AVERAGING,fork_allowed
from poker.train_cfr import train
from poker.config import load_checkpoint


class Toy:
    def __init__(self,h=()):self.h=h
    def terminal(self):return len(self.h)==4
    def info(self):return (0,1,0,2)[len(self.h)],str(self.h),('call','fold')
    def child(self,a):return Toy(self.h+(('call','fold').index(a),))
    def returns(self):return [(2*self.h[0]-1)*(3*self.h[2]+1)+self.h[1]-2*self.h[3],0,0]


class PathRNG:
    def __init__(self,path):self.path=iter(path);self.probability=1.0
    def choices(self,actions,weights):
        i=next(self.path);self.probability*=weights[i]/sum(weights)
        return [actions[i]]


def exact_regrets(solver,state,owner,delta,opponent_reach=1):
    if state.terminal():return state.returns()[owner]
    actor,key,actions=state.info();strategy=solver.strategy(key,actions)
    values=[exact_regrets(solver,state.child(a),owner,delta,opponent_reach*(1 if actor==owner else p)) for a,p in zip(actions,strategy)]
    value=sum(p*v for p,v in zip(strategy,values))
    if actor==owner:delta[key]=[opponent_reach*(v-value) for v in values]
    return value


def exact_averages(solver,state,owner,rows,own_reach=1):
    if state.terminal():return
    actor,key,actions=state.info();strategy=solver.strategy(key,actions)
    if actor==owner:rows[key]=[own_reach*p for p in strategy]
    for a,p in zip(actions,strategy):exact_averages(solver,state.child(a),owner,rows,own_reach*(p if actor==owner else 1))


@pytest.mark.parametrize('root_weights',[[1,3],[0,3]])
def test_exact_sampled_regret_expectation(root_weights):
    reference=TrajectoryMCCFR();reference.regrets={str(()):root_weights,str((0,)):[2,1],str((1,)):[3,1]}
    expected={};value=exact_regrets(reference,Toy(),0,expected)
    actual={};sampled_value=0;total=0
    for path in product(range(2),repeat=4):
        solver=TrajectoryMCCFR();solver.regrets=reference.regrets;solver.rng=PathRNG(path);delta={}
        v=solver.outcome(Toy(),0,delta);q=solver.rng.probability;sampled_value+=q*v;total+=q
        for key,row in delta.items():
            dest=actual.setdefault(key,[0,0])
            for i,x in enumerate(row):dest[i]+=q*x
    assert total==pytest.approx(1);assert sampled_value==pytest.approx(value)
    for key in expected:assert actual[key]==pytest.approx(expected[key])


def test_exact_multiplayer_average_expectation_with_repeated_owner():
    reference=TrajectoryMCCFR();reference.regrets={str(()):[1,3],str((0,)):[2,1]}
    expected={};exact_averages(reference,Toy(),0,expected);actual={};total=0
    for path in product(range(2),repeat=4):
        solver=TrajectoryMCCFR();solver.regrets=reference.regrets;solver.rng=PathRNG(path)
        solver.trajectory_average(Toy(),0);q=solver.rng.probability;total+=q
        for key,row in solver.sums.items():
            dest=actual.setdefault(key,[0,0])
            for i,x in enumerate(row):dest[i]+=q*x
    assert total==pytest.approx(1)
    for key in expected:assert actual[key]==pytest.approx(expected[key])


def test_sample_count_normalizes_update_scale(monkeypatch):
    solver=TrajectoryMCCFR(samples=8);weights=[]
    monkeypatch.setattr(solver,'outcome',lambda state,owner,delta,scale:weights.append(('r',owner,scale)))
    monkeypatch.setattr(solver,'trajectory_average',lambda state,owner,weight:weights.append(('a',owner,weight)))
    solver.step(lambda seed:None,3)
    for kind in ('r','a'):
        for player in range(3):assert sum(w for k,p,w in weights if k==kind and p==player)==1


def test_trajectory_rollback_and_exact_resume(tmp_path):
    opts=dict(traversal='trajectory-v2',averaging_sampler=AVERAGING,trajectory_samples=2)
    full=train(20,tmp_path/'full',**opts)
    p=tmp_path/'split';train(7,p,**opts);split=train(20,p,resume=p,**opts)
    for field in ('regrets','sums','rng','iterations','nodes','diagnostics','average_weight_moments','traversal_stats'):assert full[field]==split[field]
    solver=TrajectoryMCCFR();rng=solver.rng.getstate()
    with pytest.raises(TraversalLimit):solver.step(lambda seed:Toy(),3,max_nodes=5)
    assert solver.rng.getstate()==rng and solver.regrets=={} and solver.sums=={} and solver.nodes==0
    assert solver.traversal_stats['iterations']==0 and solver.traversal_stats['total_nodes']==0


def test_explicit_version_fork_preserves_original_learning(tmp_path):
    parent=tmp_path/'parent';base=dict(game='holdem',averaging_sampler='call-mixture-v1',max_iteration_nodes=500000)
    before=train(1,parent,**base);frozen=parent.read_bytes()
    opts={**base,'traversal':'trajectory-v2','trajectory_samples':2,'averaging_sampler':AVERAGING}
    with pytest.raises(ValueError,match='incompatible'):train(2,tmp_path/'implicit',resume=parent,**opts)
    after=train(2,tmp_path/'fork',resume=parent,fork_traversal=True,**{**opts,'max_iteration_nodes':1})
    assert parent.read_bytes()==frozen and after['algorithm']==ALGORITHM
    assert after['recipe_fork']['iterations']==1 and after['recipe_fork']['parent_algorithm']==before['algorithm']
    for key in ('regrets','sums','archive','rng','iterations','nodes'):assert after[key]==before[key]
    assert load_checkpoint(tmp_path/'fork')['algorithm']==ALGORITHM
    with pytest.raises(ValueError,match='new output'):train(2,parent,resume=parent,fork_traversal=True,**opts)
    bad={**after,'config':{**after['config'],'trajectory_samples':True}}
    p=tmp_path/'bad';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='metadata'):load_checkpoint(p)


def test_trajectory_kuhn_value():
    from itertools import permutations
    from poker.cfr import Kuhn
    solver=TrajectoryMCCFR(7,samples=2)
    for _ in range(12000):solver.step(Kuhn,2)
    def value(state):
        if state.terminal():return state.returns()[0]
        _,key,actions=state.info()
        return sum(p*value(state.child(a)) for p,a in zip(normalize(solver.sums[key]),actions))
    result=sum(value(Kuhn(0,c)) for c in permutations(range(3),2))/6
    assert abs(result+1/18)<.025


def test_evaluator_uses_new_checkpoint_policy_and_records_coverage(tmp_path):
    from poker.evaluate import evaluate
    p=tmp_path/'candidate'
    train(1,p,game='holdem',traversal='trajectory-v2',averaging_sampler=AVERAGING,trajectory_samples=1)
    report=evaluate(p,blocks=2,seed=8300000,copies=1)
    assert report['algorithm']==ALGORITHM
    assert all(row['candidate_decisions']>0 and row['coverage_counts']['decisions']>0 for row in report['populations'].values())
