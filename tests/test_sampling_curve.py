import json
from random import Random
import pytest
from scripts import sampling_curve as curve


def test_resume_preserves_completed_and_rejects_budget_stop(tmp_path,monkeypatch):
    parent=tmp_path/'parent';path=tmp_path/'child';path.touch()
    data={'config':{'seed':11},'iterations':250,'rng':Random(1).getstate()}
    child=dict(data,iterations=400,stop_reason='run time budget reached')
    monkeypatch.setattr(curve,'load_checkpoint',lambda p,**kw:data if p==parent else child)
    assert curve.resume_checkpoint(path,parent,500)==(path,False)
    child.update(iterations=500,stop_reason='target_reached')
    assert curve.resume_checkpoint(path,parent,500)==(path,True)
    child.update(iterations=400,stop_reason='information_set_budget')
    with pytest.raises(ValueError,match='investigation'):curve.resume_checkpoint(path,parent,500)
    child.update(config={'seed':29})
    with pytest.raises(ValueError,match='mismatch'):curve.resume_checkpoint(path,parent,500)


def test_invalid_evaluation_never_published(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('{}')
    validation=tmp_path/'validation';validation.write_text('{}')
    monkeypatch.setattr(curve,'VALIDATION',validation)
    monkeypatch.setattr(curve,'run',lambda args,p:p.write_text('incomplete output'))
    output=tmp_path/'report.json'
    with pytest.raises(json.JSONDecodeError):curve.evaluate_cached(['poker.evaluate','--checkpoint',str(checkpoint)],output)
    assert not output.exists()
    assert (tmp_path/'report.json.pending').exists()
