import copy
import pytest
from poker.cfr import TraversalLimit,Holdem
from poker.joint_average import JointAverageMCCFR
from poker.future_linear_average import FutureLinearAverageMCCFR,ALGORITHM
from poker.config import CFR_ALGORITHMS
from tests.test_stratified_average import Toy


@pytest.mark.parametrize('clock',[1,50,2000])
@pytest.mark.parametrize('zero',[False,True])
def test_weighted_sources_match_exact_scalar_target(clock,zero):
    for source in ('regret','average'):
        a=JointAverageMCCFR(samples=1);b=FutureLinearAverageMCCFR(samples=1)
        a.iterations=b.iterations=clock-1
        state=Toy((0,1,0,2,0),0)
        if zero:a.regrets=b.regrets={state.info()[1]:[0.,3.]}
        if source=='regret':
            da,db={},{};a.outcome(state,0,da);b.outcome(state,0,db);assert da==db
        else:a.trajectory_average(state,0);b.trajectory_average(state,0)
        assert a.rng.getstate()==b.rng.getstate() and a.nodes==b.nodes
        assert a.sums.keys()==b.sums.keys()
        for key,row in a.sums.items():assert b.sums[key]==pytest.approx([clock*v for v in row],rel=1e-12)


def test_full_step_preserves_regret_rng_nodes_old_means_and_rollback():
    a=JointAverageMCCFR(samples=1);b=FutureLinearAverageMCCFR(samples=1,average_origin=10)
    a.iterations=b.iterations=59
    a.sums={'inherited':[2.,5.]};b.sums=copy.deepcopy(a.sums)
    factory=lambda seed:Holdem(seed,6,'texture-buckets-v2')
    a.step(factory,6,max_nodes=100000);b.step(factory,6,max_nodes=100000)
    assert a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate() and a.nodes==b.nodes
    assert b.sums['inherited']==[2.,5.]
    for key,row in a.sums.items():
        if key!='inherited':assert b.sums[key]==pytest.approx([50*v for v in row],rel=1e-12)
    fields=('regrets','sums','rng','nodes','iterations','diagnostics','traversal_stats','epoch_weight_moments','average_origin')
    before=copy.deepcopy({k:b.rng.getstate() if k=='rng' else getattr(b,k) for k in fields})
    with pytest.raises(TraversalLimit):b.step(factory,6,max_nodes=1)
    assert before=={k:b.rng.getstate() if k=='rng' else getattr(b,k) for k in fields}
    assert b.average_multiplier==51


def test_clock_rejects_invalid_boundary_and_split_steps_are_exact():
    with pytest.raises(ValueError):FutureLinearAverageMCCFR(average_origin=True)
    bad=FutureLinearAverageMCCFR(average_origin=10)
    with pytest.raises(ValueError,match='precedes fork'):bad.step(lambda seed:Toy(),3)
    a=FutureLinearAverageMCCFR(samples=1);b=copy.deepcopy(a)
    factory=lambda seed:Toy()
    a.step(factory,3);a.step(factory,3)
    b.step(factory,3);b=copy.deepcopy(b);b.step(factory,3)
    assert a.sums==b.sums and a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate()
    assert a.average_multiplier==b.average_multiplier==3


def test_engineering_peak_uses_actual_lifetime_moment_schema():
    from scripts.future_linear_probe import Candidate,batch,maximum_average_weight
    s=Candidate(samples=8);s.iterations=1999
    batch(s,Holdem(11300000,6,'texture-buckets-v2'),0)
    assert s.average_weight_moments and all(k.startswith('street=') for k in s.average_weight_moments)
    assert maximum_average_weight(s)==max(row['maximum'] for row in s.average_weight_moments.values())>0
