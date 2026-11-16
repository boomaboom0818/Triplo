from copy import deepcopy
import pytest
from scripts.recent_average import derive


def checkpoint(iterations,row,visits):
    return dict(iterations=iterations,abstraction='history-compact-v1',menu='lean-menu-v1',players=6,stack_bb=100,
                sums={'key':row},visits={'key':visits})


def test_recent_policy_uses_only_new_mass_and_new_support_without_mutation():
    old=checkpoint(100,[900.,100.],1000);end=checkpoint(200,[900.,200.],1064)
    original=deepcopy(end)
    policy,stats=derive(end,old)
    assert policy['sums']['key']==[0.,1.] and policy['visits']['key']==64
    assert end==original and stats['replaced']==1
    blended,_=derive(end,old,.5)
    assert blended['sums']['key']==pytest.approx([.5*900/1100,.5*200/1100+.5])
    end['visits']['key']=1063
    sparse,_=derive(end,old)
    assert sparse['sums']['key']==end['sums']['key']


def test_rejects_noncontinuations_and_action_changes():
    old=checkpoint(100,[10.,20.],100)
    for row,visits in (([9.,30.],164),([20.,30.],99),([20.],164),([float('nan'),30.],164)):
        with pytest.raises(ValueError):derive(checkpoint(200,row,visits),old)
    with pytest.raises(ValueError):derive(old,old)


def test_new_rows_and_zero_delta_keep_valid_support():
    old = checkpoint(100, [10., 20.], 100)
    end = checkpoint(200, [10., 20.], 164)
    end['sums']['new'] = [20., 40.]
    end['visits']['new'] = 64
    policy, stats = derive(end, old)
    assert policy['sums']['key'] == [10., 20.]
    assert policy['sums']['new'] == pytest.approx([1/3, 2/3])
    assert stats['replaced'] == stats['new_rows'] == 1
    assert stats['inherited'] == 1


def test_rejects_transformed_inputs_invalid_counts_and_missing_parent_rows():
    old = checkpoint(100, [10., 20.], 100)
    end = checkpoint(200, [20., 40.], 164)
    transformed, _ = derive(end, old)
    with pytest.raises(ValueError, match='raw accumulated'):
        derive(transformed, old)
    for count in (164.5, True, -1):
        with pytest.raises(ValueError, match='visit count'):
            derive(checkpoint(200, [20., 40.], count), old)
    del end['sums']['key']
    with pytest.raises(ValueError, match='parent rows missing'):
        derive(end, old)


def test_export_preserves_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    from scripts import recent_average
    import json

    old = checkpoint(100, [10., 20.], 100)
    end = checkpoint(200, [20., 40.], 164)
    parent, completed, out = [tmp_path / name for name in ('parent', 'completed', 'out')]
    parent.write_text(json.dumps(old))
    completed.write_text(json.dumps(end))
    monkeypatch.setattr(recent_average, 'load', lambda p: json.loads(p.read_text()))
    recent_average.run(completed, parent, out)
    result = json.loads(out.read_text())
    assert result['inference_transform']['parent_sha256'] == recent_average.digest(parent)
    assert json.loads(completed.read_text()) == end
    with pytest.raises(ValueError, match='preserve'):
        recent_average.run(completed, parent, out)
    assert set(tmp_path.iterdir()) == {parent, completed, out}
