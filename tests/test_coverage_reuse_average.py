import copy
import pytest
from poker.cfr import Holdem,TraversalLimit
from poker.config import CFR_ALGORITHMS
from poker.future_linear_average import FutureLinearAverageMCCFR
from poker.coverage_reuse_average import CoverageReuseMCCFR,ALGORITHM
from tests.test_adaptive_average import Replay,NeedChoice
from tests.test_joint_average import exact,AliasToy
from tests.test_stratified_average import Toy


@pytest.mark.parametrize('kind',[Toy,AliasToy])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('mask',['none','partial','all'])
def test_frozen_coverage_partition_preserves_exact_all_owner_target(kind,zero,mask):
    state=kind((0,1,0,2,0),0);ref=CoverageReuseMCCFR(players=3,samples=1)
    ref.regrets={state.info()[1]:[0.,3.] if zero else [1.,3.]};expected={}
    for owner in range(3):exact(ref,state,expected,owner)
    reference={key:[2.,5.] for i,key in enumerate(sorted(expected)) if mask=='all' or mask=='partial' and i%2==0}
    actual={}
    for owner in range(3):
        for source in ('regret','average'):
            pending=[()];total=0.
            while pending:
                path=pending.pop();s=CoverageReuseMCCFR(players=3,samples=1,coverage_reference=reference)
                s.regrets=ref.regrets;s.iterations=49;s.opponents={i:'guarded-v2' for i in range(3)};s.rng=Replay(path)
                try:
                    if source=='regret':s.outcome(state,owner,{})
                    else:s.trajectory_average(state,owner)
                except NeedChoice as choice:
                    pending.extend(path+(i,) for i,w in enumerate(choice.weights) if w>0);continue
                total+=s.rng.p
                for key,row in s.sums.items():
                    dst=actual.setdefault(key,[0.]*len(row))
                    for i,x in enumerate(row):dst[i]+=s.rng.p*x
            assert total==pytest.approx(1.)
    assert set(actual)<=set(expected)
    for key,row in expected.items():assert actual.get(key,[0.,0.])==pytest.approx([50*v for v in row],rel=1e-10,abs=1e-10)


def test_six_role_step_preserves_covered_updates_regrets_rng_nodes_and_old_learning():
    # Revisit the same deal so the test actually exercises pre-existing rows.
    factory=lambda seed:Holdem(77,6,'texture-buckets-v2')
    a=FutureLinearAverageMCCFR(samples=1);a.step(factory,6,max_nodes=100000)
    b=CoverageReuseMCCFR(samples=1)
    for name in ('regrets','sums','iterations','nodes','diagnostics','traversal_stats','average_weight_moments','epoch_weight_moments'):
        setattr(b,name,copy.deepcopy(getattr(a,name)))
    b.rng.setstate(a.rng.getstate());before=copy.deepcopy(b.sums)
    opponents={i:'guarded-v2' for i in range(6)}
    a.step(factory,6,max_nodes=100000,opponents=opponents);b.step(factory,6,max_nodes=100000,opponents=opponents)
    assert a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate() and a.nodes==b.nodes
    for key,row in before.items():
        if any(row):assert b.sums[key]==a.sums[key]
        assert all(after>=old for after,old in zip(b.sums[key],row))
    assert b.diagnostics['coverage-reuse-v10:covered-foreign-omitted']>0
    assert ALGORITHM not in CFR_ALGORITHMS
    fields=('regrets','sums','iterations','nodes','diagnostics','traversal_stats','average_origin','epoch_weight_moments')
    snapshot=copy.deepcopy({k:getattr(b,k) for k in fields});rng=b.rng.getstate()
    with pytest.raises(TraversalLimit):b.step(factory,6,max_nodes=1)
    assert snapshot=={k:getattr(b,k) for k in fields} and rng==b.rng.getstate()


def test_reference_does_not_switch_on_pending_updates_and_split_steps_are_exact():
    a=CoverageReuseMCCFR(players=3,samples=1);b=copy.deepcopy(a)
    for _ in range(2):a.step(lambda seed:Toy(),3)
    b.step(lambda seed:Toy(),3);b=copy.deepcopy(b);b.step(lambda seed:Toy(),3)
    assert a.sums==b.sums and a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate()
    assert a.coverage_reference is None and b.coverage_reference is None


def test_readonly_driver_keeps_parent_reference_when_clearing_probe_deltas():
    from scripts.coverage_reuse_probe import Candidate
    parent=dict(sums={'known':[2.,5.]},regrets={},abstraction='texture-buckets-v2')
    solver=Candidate();solver.setup(parent)
    solver.sums={}
    assert solver.coverage_reference is parent['sums']
    assert solver.coverage_reference=={'known':[2.,5.]}
