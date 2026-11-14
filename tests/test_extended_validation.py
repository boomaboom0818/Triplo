import pytest
from scripts.extended_validation_check import validator_config


def test_validator_recipe_preserves_original_protections():
    saved=dict(iterations=250,config=dict(seed=900001,population_probability=0,abstraction='texture-buckets-v1',archive_size=2),limits=dict(max_information_sets=250000,max_iteration_nodes=500000))
    cfg=validator_config(saved)
    assert cfg['iterations']==750
    assert cfg['max_information_sets']==250000
    assert cfg['max_iteration_nodes']==500000
    assert cfg['population_probability']==0
    assert cfg['averaging_sampler']=='uniform'
    saved['config']['seed']=11
    with pytest.raises(ValueError):validator_config(saved)
