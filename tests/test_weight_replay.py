import time
import pytest
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from scripts.diagnose_weights import diagnose


def test_direct_epoch_replay_preserves_source_and_keeps_weight_groups(tmp_path):
    parent=tmp_path/'source'
    before=train(1,parent,game='holdem',traversal='trajectory-v2',averaging_sampler=AVERAGING,trajectory_samples=1)
    original=parent.read_bytes();output=tmp_path/'diagnostic.json'
    report=diagnose(parent,output,2,time.time()+30,20)
    assert parent.read_bytes()==original and report['source_iterations']==before['iterations']
    assert report['read_only'] and report['discarded_local_learning'] and report['simulated_complete_iterations']==2
    assert any(k.startswith('average:street=') for k in report['moments'])
    assert any(k.startswith('regret_prefix:street=') for k in report['moments'])
    assert all(1<=row['concentration_proxy']<=row['updates']+1e-9 for row in report['moments'].values())
    assert not (tmp_path/'source.previous').exists()
    with pytest.raises(ValueError,match='preserve'):diagnose(parent,output,2,time.time()+30)
