import copy
import math
import pytest

from poker.cfr import Holdem, TraversalLimit
from poker.config import CFR_ALGORITHMS
from poker.future_linear_average import FutureLinearAverageMCCFR
from poker.cross_role_average import CrossRoleAverageMCCFR, Prefix, cross_weight, ALGORITHM
from tests.test_adaptive_average import Replay, NeedChoice
from tests.test_joint_average import exact, AliasToy
from tests.test_stratified_average import Toy


@pytest.mark.parametrize('kind',[Toy,AliasToy])
@pytest.mark.parametrize('root',[[1.,3.],[0.,3.]])
@pytest.mark.parametrize('depth',[1,2])
@pytest.mark.parametrize('guarded',[False,True])
@pytest.mark.parametrize('clock',[1,50])
def test_all_source_expectation_matches_each_owners_exact_target(kind,root,depth,guarded,clock):
    state = kind((0,1,0,2,0),0)
    ref = CrossRoleAverageMCCFR(players=3,depth=depth,samples=1)
    ref.regrets = {state.info()[1]:root}
    expected = {}
    for owner in range(3):
        exact(ref,state,expected,owner)
    actual = {}
    for owner in range(3):
        for source in ('regret','average'):
            pending=[()];total=0.
            while pending:
                path=pending.pop()
                s=CrossRoleAverageMCCFR(players=3,depth=depth,samples=1)
                s.regrets=ref.regrets;s.opponents={j:'guarded-v2' for j in range(3)} if guarded else {};s.rng=Replay(path)
                s.iterations=clock-1
                try:
                    if source=='regret':s.outcome(state,owner,{})
                    else:s.trajectory_average(state,owner)
                except NeedChoice as choice:
                    pending.extend(path+(j,) for j,w in enumerate(choice.weights) if w>0)
                    continue
                total+=s.rng.p
                for key,row in s.sums.items():
                    dest=actual.setdefault(key,[0.]*len(row))
                    for j,x in enumerate(row):dest[j]+=s.rng.p*x
            assert total==pytest.approx(1.)
    assert set(actual)<=set(expected)
    for key,row in expected.items():assert actual.get(key,[0.,0.])==pytest.approx([clock*v for v in row],rel=1e-10,abs=1e-10)


@pytest.mark.parametrize('opponents',[{}, {i:'guarded-v2' for i in range(6)}])
def test_real_six_role_regret_rng_nodes_and_preserved_means(opponents):
    a=FutureLinearAverageMCCFR(samples=1,average_origin=10)
    b=CrossRoleAverageMCCFR(samples=1,average_origin=10)
    a.iterations=b.iterations=59
    a.sums={'inherited':[2.,5.]};b.sums=copy.deepcopy(a.sums)
    factory=lambda seed:Holdem(seed,6,'texture-buckets-v2')
    a.step(factory,6,max_nodes=100000,opponents=opponents)
    b.step(factory,6,max_nodes=100000,opponents=opponents)
    assert a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate() and a.nodes==b.nodes
    assert b.sums['inherited']==[2.,5.]
    assert b.diagnostics['cross-role-v9:reused-other-role']>0
    assert b.traversal_stats['max_joint_average_to_original_weight_ratio']<=1+1e-12
    assert ALGORITHM not in CFR_ALGORITHMS


def test_weight_support_boundary_and_transactional_rollback():
    prefix=Prefix.root(3).child(0,0.,.25,.8,.5,2)
    assert cross_weight(prefix,0,1.)==(0.,0.)
    assert cross_weight(prefix,1,1.)[0]>0
    tiny=Prefix((-1000.,0.),(0.,0.),(0.,0.),(1,0))
    with pytest.raises(TraversalLimit,match='underflowed positive'):cross_weight(tiny,0,1.)
    for actor in range(3):
        p=Prefix.root(3).child(actor,.2,.6,.4,.35,1)
        weight,ratio=cross_weight(p,actor,.125)
        assert ratio<=1 and weight<=.125*math.exp(p.own[actor]-p.average[actor])
    s=CrossRoleAverageMCCFR(players=3,samples=1)
    fields=('regrets','sums','nodes','iterations','diagnostics','traversal_stats','epoch_weight_moments','average_origin')
    before=copy.deepcopy({k:getattr(s,k) for k in fields});rng=s.rng.getstate()
    with pytest.raises(TraversalLimit):s.step(lambda seed:Toy(),3,max_nodes=4)
    assert before=={k:getattr(s,k) for k in fields} and rng==s.rng.getstate()
    a=copy.deepcopy(s);b=copy.deepcopy(s)
    for _ in range(2):a.step(lambda seed:Toy(),3)
    b.step(lambda seed:Toy(),3);b=copy.deepcopy(b);b.step(lambda seed:Toy(),3)
    assert a.sums==b.sums and a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate()


def test_engineering_batch_contains_every_role_and_preserves_control_signature():
    import time
    from scripts.cross_role_probe import full_batch
    class SixToy(Toy):
        def child(self,a):return SixToy(self.actors,self.paid,self.h+(('call','fold').index(a),))
        def returns(self):return [1.,-1.,0.,0.,0.,0.]
    state=SixToy((0,1,2,3,4,5,0),0)
    a=FutureLinearAverageMCCFR();b=CrossRoleAverageMCCFR()
    a.iterations=b.iterations=2000
    signature_a,roles_a=full_batch(a,state,time.time()+30)
    signature_b,roles_b=full_batch(b,state,time.time()+30)
    assert signature_a==signature_b
    assert [r['owner'] for r in roles_a]==[r['owner'] for r in roles_b]==list(range(6))
    assert [r['nodes'] for r in roles_a]==[r['nodes'] for r in roles_b]
    assert sum(r['nodes'] for r in roles_b)==b.nodes
    assert not a.regrets and not b.regrets  # diagnostic deltas are never committed
