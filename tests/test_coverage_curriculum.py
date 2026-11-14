from types import SimpleNamespace
import pytest
from poker.population import profile_strategy,population_fork_allowed
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from poker.prefix import PrefixMCCFR
from poker.cfr import Holdem
from scripts.coverage_population_probe import guarded_strategy,ReachProbe


def test_production_guarded_profile_matches_probe_and_public_boundary():
    for legal in [('call',),('call',4,8),('call','fold'),('call','fold',4,8)]:
        for price in (0,2):
            o=SimpleNamespace(legal=legal,call=price,hole=('As','Ah'))
            weights=profile_strategy(o,'guarded-v2');assert weights==guarded_strategy(o)
            o.hole=('2s','3h');assert weights==profile_strategy(o,'guarded-v2')
    a=PrefixMCCFR(depth=2);a.opponents={i:'guarded-v2' for i in range(6)}
    b=ReachProbe(True);b.reference_sums={};left={};right={}
    a.outcome(Holdem(9),3,left);b.outcome(Holdem(9),3,right)
    assert left==right and a.nodes==b.nodes and a.rng.getstate()==b.rng.getstate()


def test_guarded_fork_only_changes_population():
    old=dict(seed=11,traversal='prefix-v3',regret_prefix_depth=2)
    new={**old,'opponent_population':'guarded-tables-v2'}
    assert population_fork_allowed(old,new)
    assert not population_fork_allowed(old,{**new,'seed':29})
    assert not population_fork_allowed(old,{**new,'regret_prefix_depth':1})
    assert not population_fork_allowed(new,old)


def test_guarded_curriculum_fork_rollback_exact_resume_and_diagnostic_boundary(tmp_path):
    opts=dict(game='holdem',traversal='trajectory-v2',averaging_sampler=AVERAGING,trajectory_samples=1,archive_every=2,archive_size=2,population_probability=.25)
    base=tmp_path/'base';train(1,base,**opts)
    source=tmp_path/'source';opts['traversal']='prefix-v3'
    before=train(2,source,resume=base,fork_traversal=True,**opts);frozen=source.read_bytes()
    new={**opts,'opponent_population':'guarded-tables-v2'}
    with pytest.raises(ValueError,match='incompatible'):train(3,tmp_path/'implicit',resume=source,**new)
    stopped=train(3,tmp_path/'stopped',resume=source,fork_population=True,**{**new,'max_iteration_nodes':1})
    for field in ('regrets','sums','archive','rng','iterations','nodes','diagnostics','average_weight_moments'):assert stopped[field]==before[field]
    assert source.read_bytes()==frozen and stopped['traversal_stats']['iterations']==0 and stopped['epoch_weight_moments']=={}
    assert stopped['traversal_stats_started_at']==2 and stopped['recipe_fork']['previous_recipe_fork']==before['recipe_fork']
    full=train(10,tmp_path/'full',resume=source,fork_population=True,**new)
    p=tmp_path/'split';train(5,p,resume=source,fork_population=True,**new);after=train(10,p,resume=p,**new)
    for field in ('regrets','sums','archive','rng','iterations','nodes','diagnostics','average_weight_moments','traversal_stats','epoch_weight_moments'):assert full[field]==after[field]
    assert after['traversal_stats']['iterations']==8 and after['traversal_stats_started_at']==2
    counts=[after['diagnostics'].get(f'population:guarded-tables-v2:{mode}_iterations',0) for mode in ('guarded','historical')]
    assert sum(counts)==8 and all(n>0 for n in counts)
    from poker.config import load_checkpoint
    assert load_checkpoint(p)['config']['opponent_population']=='guarded-tables-v2'
    from poker.evaluate import evaluate
    assert evaluate(p,2,9500000,1)['populations']['mixed']['candidate_decisions']>0


def test_curriculum_driver_inputs_and_fresh_evidence_are_explicit(monkeypatch):
    from scripts import trajectory_pilot as d
    from scripts.coverage_curriculum import configure
    for name in ('ROOT','PARENTS','EXPECTED_BASES','ADDITIONS','ALGORITHM','TRAVERSAL','PREFIX_DEPTH','POPULATION','FORK_FLAG','REQUIRE_TIGHT_IMPROVEMENT','fork_allowed','CRITERIA','SCREEN_SEED','CONFIRM_SEED','ADDITIONAL_SOURCES'):
        monkeypatch.setattr(d,name,getattr(d,name))
    configure()
    assert d.EXPECTED_BASES==[4606,4850] and d.ADDITIONS==[50,150,300]
    assert d.POPULATION=='guarded-tables-v2' and d.FORK_FLAG=='--fork-population'
    assert d.REQUIRE_TIGHT_IMPROVEMENT
    assert (d.SCREEN_SEED,d.CONFIRM_SEED)==(9500000,9600000)
    assert d.VALIDATION not in d.PARENTS and all(op not in d.PARENTS for op in d.ROBUSTNESS)


@pytest.mark.parametrize('decisions,fraction,old,passes',[(200,.05,.01,True),(199,.90,.01,False),(200,.04,.01,False),(200,.05,.04,False),(200,.07,.05,True)])
def test_new_curriculum_requires_coverage_sample_floor_and_real_gain(decisions,fraction,old,passes):
    from scripts.trajectory_pilot import tight_coverage_improvement
    result=tight_coverage_improvement({'tight':dict(decisions=decisions,fraction=fraction)},{'tight':dict(decisions=200,fraction=old)})
    assert result['passed'] is passes
