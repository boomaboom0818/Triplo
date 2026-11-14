from random import Random
import pytest
from scripts import population_pilot as pilot


def test_diverse_resume_does_not_reapply_fork(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.write_text('original');child=tmp_path/'child'
    assert pilot.recovery_source(child,parent,'diverse')==(parent,False,True)
    child.touch()
    original={'config':{'seed':11},'iterations':2000}
    saved={'config':{'seed':11,'opponent_population':'diverse-v1'},'iterations':2050,'stop_reason':'interrupted','rng':Random(1).getstate(),'recipe_fork':{'parent_sha256':pilot.digest(parent)}}
    monkeypatch.setattr(pilot,'load_checkpoint',lambda p,**kw:original if p==parent else saved)
    assert pilot.recovery_source(child,parent,'diverse')==(child,False,False)
    saved.update(iterations=2100,stop_reason='target_reached')
    assert pilot.recovery_source(child,parent,'diverse')==(child,True,False)
    saved['recipe_fork']['parent_sha256']='wrong'
    with pytest.raises(ValueError,match='lineage'):pilot.recovery_source(child,parent,'diverse')


def test_control_blocks_resource_failure_and_stalled_timeout(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.write_text('original');child=tmp_path/'child';child.touch()
    original={'config':{'seed':11},'iterations':2000}
    saved={'config':{'seed':11},'iterations':2050,'stop_reason':'information_set_budget','rng':Random(1).getstate()}
    monkeypatch.setattr(pilot,'load_checkpoint',lambda p,**kw:original if p==parent else saved)
    with pytest.raises(ValueError,match='investigation'):pilot.recovery_source(child,parent,'control')
    saved.update(iterations=2000,stop_reason='run time budget reached')
    with pytest.raises(ValueError,match='no iteration progress'):pilot.recovery_source(child,parent,'control')
