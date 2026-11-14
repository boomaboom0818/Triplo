from types import SimpleNamespace
import pytest
from poker.population import profile_strategy,population_fork_allowed
from poker.train_cfr import train
from poker.abstraction import VERSION_V2


def test_profiles_respect_legal_actions_and_full_support():
    for style in ('passive-v1','pressure-v1'):
        for legal in [('call',),('call','fold'),('call',2,4),('call','fold',2,4)]:
            weights=profile_strategy(SimpleNamespace(legal=legal),style)
            assert sum(weights)==pytest.approx(1)
            assert len(weights)==len(legal) and all(x>0 for x in weights)


def test_fork_allows_only_explicit_population_change():
    before={'seed':11,'abstraction':VERSION_V2}
    after={**before,'opponent_population':'diverse-v1'}
    assert population_fork_allowed(before,after)
    assert not population_fork_allowed(before,{**after,'seed':29})
    assert not population_fork_allowed(after,before)
    assert not population_fork_allowed(before,before)


def test_diverse_exact_resume(tmp_path):
    opts=dict(game='holdem',abstraction=VERSION_V2,averaging_sampler='call-mixture-v1',opponent_population='diverse-v1',max_iteration_nodes=500000)
    full=train(2,tmp_path/'full',**opts)
    path=tmp_path/'split';train(1,path,**opts);split=train(2,path,resume=path,**opts)
    for field in ('sums','regrets','rng','diagnostics','average_weight_moments'):assert full[field]==split[field]
    assert any(k.startswith('population:') for k in full['diagnostics'])


def test_explicit_fork_preserves_learning_on_interrupted_iteration(tmp_path):
    opts=dict(game='holdem',abstraction=VERSION_V2,averaging_sampler='call-mixture-v1',max_iteration_nodes=500000)
    parent=tmp_path/'parent';before=train(1,parent,**opts)
    frozen=parent.read_bytes()
    with pytest.raises(ValueError,match='incompatible'):
        train(2,tmp_path/'implicit',resume=parent,opponent_population='diverse-v1',**opts)
    after=train(2,tmp_path/'fork',resume=parent,opponent_population='diverse-v1',fork_population=True,**{**opts,'max_iteration_nodes':1})
    assert parent.read_bytes()==frozen
    assert after['recipe_fork']['parent_config']==before['config']
    assert after['stop_reason']=='iteration node budget exceeded'
    for field in ('sums','regrets','archive','rng','iterations','nodes'):assert after[field]==before[field]
    with pytest.raises(ValueError,match='new output'):
        train(2,parent,resume=parent,opponent_population='diverse-v1',fork_population=True,**opts)
