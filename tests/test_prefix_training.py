import json
import pytest
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from poker.prefix import ALGORITHM
from poker.config import load_checkpoint


def test_explicit_prefix_fork_resume_preserves_all_learning_and_diagnostic_boundary(tmp_path):
    opts=dict(game='holdem',traversal='trajectory-v2',averaging_sampler=AVERAGING,trajectory_samples=1)
    source=tmp_path/'parent';before=train(1,source,**opts);original=source.read_bytes()
    new={**opts,'traversal':'prefix-v3','regret_prefix_depth':2}
    with pytest.raises(ValueError,match='incompatible'):train(2,tmp_path/'implicit',resume=source,**new)
    stopped=train(2,tmp_path/'stopped',resume=source,fork_traversal=True,**{**new,'max_iteration_nodes':1})
    assert source.read_bytes()==original
    for key in ('regrets','sums','rng','archive','iterations','nodes','diagnostics','average_weight_moments'):assert stopped[key]==before[key]
    assert stopped['traversal_stats']['iterations']==0 and stopped['epoch_weight_moments']=={} and stopped['traversal_stats_started_at']==1
    full=train(4,tmp_path/'full',resume=source,fork_traversal=True,**new)
    split=tmp_path/'split';train(2,split,resume=source,fork_traversal=True,**new);after=train(4,split,resume=split,**new)
    for key in ('regrets','sums','rng','archive','iterations','nodes','diagnostics','average_weight_moments','traversal_stats','epoch_weight_moments'):assert full[key]==after[key]
    assert after['algorithm']==ALGORITHM and after['traversal_stats']['iterations']==3
    assert after['traversal_stats_started_at']==1 and after['recipe_fork']['parent_algorithm']==before['algorithm']
    assert load_checkpoint(split)['algorithm']==ALGORITHM
    bad={**after,'config':{**after['config'],'regret_prefix_depth':True}}
    p=tmp_path/'bad';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='prefix checkpoint'):load_checkpoint(p)
    from poker.evaluate import evaluate
    report=evaluate(split,2,8500000,1)
    assert report['algorithm']==ALGORITHM and report['populations']['mixed']['candidate_decisions']>0


def test_prefix_pilot_configuration_is_explicit_and_bounded(monkeypatch):
    from scripts import trajectory_pilot as driver
    from scripts.prefix_pilot import configure
    for name in ('ROOT','PARENTS','EXPECTED_BASES','ADDITIONS','ALGORITHM','TRAVERSAL','PREFIX_DEPTH','fork_allowed','CRITERIA','SCREEN_SEED','CONFIRM_SEED','ADDITIONAL_SOURCES'):
        monkeypatch.setattr(driver,name,getattr(driver,name))
    configure()
    assert driver.ALGORITHM==ALGORITHM and driver.ADDITIONS==[50,200,500] and driver.EXPECTED_BASES==[4106,4350]
    assert (driver.SCREEN_SEED,driver.CONFIRM_SEED)==(8500000,8600000) and driver.PREFIX_DEPTH==2
