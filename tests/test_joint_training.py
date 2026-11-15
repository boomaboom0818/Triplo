"""Joint-estimator checkpoint compatibility and exact fork/resume preservation."""
import json
import pytest
from poker.train_cfr import train
from poker.trajectory import AVERAGING as OLD_AVERAGING
from poker.joint_average import ALGORITHM,AVERAGING
from poker.config import load_checkpoint


def test_joint_fork_preserves_every_parent_value_and_exact_resumed_learning(tmp_path):
    opts=dict(game='holdem',seed=11,traversal='prefix-v3',averaging_sampler=OLD_AVERAGING,trajectory_samples=1,
        opponent_population='guarded-tables-v2',population_probability=.25,archive_every=2,archive_size=2,regret_prefix_depth=2)
    source=tmp_path/'parent';before=train(1,source,**opts);original=source.read_bytes()
    new={**opts,'traversal':'joint-average-v7','averaging_sampler':AVERAGING}
    with pytest.raises(ValueError,match='incompatible'):train(2,tmp_path/'implicit',resume=source,**new)
    stopped=train(2,tmp_path/'stopped',resume=source,fork_traversal=True,**{**new,'max_iteration_nodes':1})
    assert source.read_bytes()==original
    for key in ('regrets','sums','rng','archive','iterations','nodes','diagnostics','average_weight_moments','diagnostics_started_at','weight_diagnostics_started_at'):assert stopped[key]==before[key]
    assert stopped['algorithm']==ALGORITHM and stopped['traversal_stats']['iterations']==0 and stopped['epoch_weight_moments']=={}
    assert stopped['joint_average_diagnostics_started_at']==1 and stopped['average_estimator_schema']=='joint-inclusion-v7'
    full=train(3,tmp_path/'full',resume=source,fork_traversal=True,**new)
    split=tmp_path/'split';train(2,split,resume=source,fork_traversal=True,**new);after=train(3,split,resume=split,**new)
    for key in ('regrets','sums','rng','archive','iterations','nodes','diagnostics','average_weight_moments','traversal_stats','epoch_weight_moments','recipe_fork'):assert full[key]==after[key]
    assert after['recipe_fork']['parent_algorithm']==before['algorithm'] and after['recipe_fork']['previous_recipe_fork']==before['recipe_fork']
    assert after['joint_average_diagnostics_started_at']==1
    assert after['diagnostics']['joint-traversal-v7:regret-trees']==12 and after['diagnostics']['joint-traversal-v7:average-paths']==12
    assert after['traversal_stats']['max_joint_average_to_original_weight_ratio']<=1+1e-12
    assert source.read_bytes()==original and load_checkpoint(split)['algorithm']==ALGORITHM
    for changes in ({'averaging_sampler':OLD_AVERAGING},{'trajectory_samples':True},{'regret_prefix_depth':True}):
        bad={**after,'config':{**after['config'],**changes}};path=tmp_path/'bad';path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):load_checkpoint(path)
    bad={**after,'joint_average_diagnostics_started_at':True};path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_checkpoint(path)
    bad={**after,'traversal_stats':{**after['traversal_stats'],'max_joint_average_to_original_weight_ratio':1.5}};path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_checkpoint(path)
    with pytest.raises(ValueError,match='joint averaging diagnostics'):train(4,tmp_path/'bad-resume',resume=path,**new)
    from poker.evaluate import evaluate
    report=evaluate(split,2,17,1)
    assert report['algorithm']==ALGORITHM and report['populations']['mixed']['candidate_decisions']>0
