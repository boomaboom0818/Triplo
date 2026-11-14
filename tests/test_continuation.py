import pytest
from scripts import continuation_control as runner


def test_resume_rejects_learning_change_and_budget_failure(monkeypatch):
    config=dict(game='holdem',players=6,seed=11,abstraction='texture-buckets-v2',population_probability=.25,archive_every=100,archive_size=2,iterations=1000)
    saved=dict(config={k:v for k,v in config.items() if k!='iterations'},iterations=987,stop_reason='run time budget reached')
    monkeypatch.setattr(runner,'load_checkpoint',lambda *a,**kw:saved)
    assert runner.checkpoint_iterations('unused',config)==987
    saved['stop_reason']='information_set_budget'
    with pytest.raises(ValueError,match='investigation'):runner.checkpoint_iterations('unused',config)
    saved['stop_reason']='run time budget reached'
    saved['config']['seed']=29
    with pytest.raises(ValueError,match='configuration'):runner.checkpoint_iterations('unused',config)
