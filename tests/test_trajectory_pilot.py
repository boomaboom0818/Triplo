import pytest
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from poker.config import load_checkpoint
from scripts.trajectory_pilot import verified_resume,config


def test_pilot_resume_requires_version_lineage_and_safe_stop(tmp_path):
    parent=tmp_path/'parent';before=train(1,parent,game='holdem',averaging_sampler='call-mixture-v1')
    target=3;p=tmp_path/'fork';cfg=config(before,target)
    opts={k:v for k,v in cfg.items() if k!='schema'}
    train(path=p,resume=parent,fork_traversal=True,**{**opts,'max_iteration_nodes':1})
    expected={**before['config'],'traversal':'trajectory-v2','trajectory_samples':8,'averaging_sampler':AVERAGING}
    with pytest.raises(ValueError,match='limit mismatch'):verified_resume(p,parent,target,expected)
    p.unlink();train(path=p,resume=parent,fork_traversal=True,**opts)
    assert verified_resume(p,parent,target,expected)==(p,True)
    assert load_checkpoint(p)['iterations']==target
    with pytest.raises(ValueError,match='mismatch'):verified_resume(p,parent,target,{**expected,'seed':29})
