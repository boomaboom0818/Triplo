"""Real disk fork/resume/provenance tests for the future-linear clock."""
import json
import pytest
from poker.train_cfr import train
from poker.config import load_checkpoint
from poker.future_linear_average import ALGORITHM,AVERAGING


def test_preserved_fork_exact_disk_resume_and_evaluation(tmp_path):
    old=dict(game='holdem',seed=11,traversal='joint-average-v7',averaging_sampler='regret-path-mis-average-v7',trajectory_samples=1,
        opponent_population='guarded-tables-v2',population_probability=.25,archive_every=2,archive_size=2,regret_prefix_depth=2)
    parent=tmp_path/'parent';before=train(1,parent,**old);original=parent.read_bytes()
    new={**old,'traversal':'future-linear-average-v8','averaging_sampler':AVERAGING}
    with pytest.raises(ValueError,match='incompatible'):train(2,tmp_path/'implicit',resume=parent,**new)
    failed=train(2,tmp_path/'failed',resume=parent,fork_traversal=True,**{**new,'max_iteration_nodes':1})
    for key in ('regrets','sums','rng','archive','iterations','nodes','diagnostics','average_weight_moments','weight_diagnostics_started_at','diagnostics_started_at'):
        assert failed[key]==before[key]
    assert failed['future_average_origin']==1 and failed['future_average_completed_iterations']==0
    assert failed['joint_average_diagnostics_started_at']==before['joint_average_diagnostics_started_at']
    full=train(3,tmp_path/'full',resume=parent,fork_traversal=True,**new)
    split=tmp_path/'split';train(2,split,resume=parent,fork_traversal=True,**new)
    after=train(3,split,resume=split,**new)
    for key in ('regrets','sums','rng','archive','iterations','nodes','diagnostics','average_weight_moments','traversal_stats','epoch_weight_moments','recipe_fork','future_average_origin','future_average_completed_iterations'):
        assert full[key]==after[key]
    assert parent.read_bytes()==original
    assert after['future_average_origin']==1 and after['future_average_completed_iterations']==2
    assert after['recipe_fork']['previous_recipe_fork']==before['recipe_fork']
    assert load_checkpoint(split)['algorithm']==ALGORITHM
    assert all(after['sums'][key][j]>=v for key,row in before['sums'].items() for j,v in enumerate(row))
    from poker.evaluate import evaluate
    report=evaluate(split,2,19,1)
    assert report['algorithm']==ALGORITHM and report['populations']['mixed']['candidate_decisions']>0
    for changes in ({'future_average_origin':True},{'future_average_origin':2},{'future_average_completed_iterations':0},{'future_average_weighting_schema':'uniform'}, {'joint_average_diagnostics_started_at':True}):
        bad={**after,**changes};path=tmp_path/'bad';path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):load_checkpoint(path)
        with pytest.raises(ValueError):train(4,tmp_path/'bad-resume',resume=path,**new)
    with pytest.raises(ValueError):train(2,tmp_path/'wrong',resume=parent,fork_traversal=True,**{**new,'trajectory_samples':2})
