import json
import pytest
from poker.cfr import MCCFR,Holdem
from poker.trajectory import TrajectoryMCCFR,AVERAGING
from poker.prefix import PrefixMCCFR
from poker.train_cfr import train


@pytest.mark.parametrize('base',[TrajectoryMCCFR,PrefixMCCFR])
def test_actual_sampling_labels_preserve_learning_rng_and_historical_keys(base):
    class LegacyLabels(base):
        def record_visit(self,state,traversal,owner,sampling=None):
            return MCCFR.record_visit(self,state,traversal,owner)
    old=LegacyLabels(11,samples=1);new=base(11,samples=1)
    old.diagnostics['historic:row']=7;new.diagnostics['historic:row']=7
    for _ in range(2):
        old.step(Holdem,6,max_nodes=500000);new.step(Holdem,6,max_nodes=500000)
    for name in ('regrets','sums','nodes','iterations','average_weight_moments','traversal_stats'):assert getattr(old,name)==getattr(new,name)
    assert old.rng.getstate()==new.rng.getstate() and new.diagnostics['historic:row']==7
    if hasattr(new,'epoch_weight_moments'):assert old.epoch_weight_moments==new.epoch_weight_moments
    assert any(k.startswith('sampling-v2:regret:') and k.endswith(':sampled') for k in new.diagnostics)
    assert all(k.endswith(':sampled') for k in new.diagnostics if k.startswith('sampling-v2:average:'))
    assert any(k.endswith(':enumerated') for k in new.diagnostics if k.startswith('sampling-v2:regret:'))==(base is PrefixMCCFR)


def test_diagnostic_boundary_survives_resume(tmp_path):
    opts=dict(game='holdem',traversal='prefix-v3',averaging_sampler=AVERAGING,trajectory_samples=1)
    p=tmp_path/'model';before=train(1,p,**opts);after=train(2,p,resume=p,**opts)
    assert before['sampling_diagnostics_started_at']==after['sampling_diagnostics_started_at']==0
    assert after['sampling_diagnostics_schema']=='actual-actions-v2'


def test_legacy_checkpoint_gets_explicit_new_boundary_without_relabeling_history(tmp_path):
    opts=dict(game='holdem',traversal='prefix-v3',averaging_sampler=AVERAGING,trajectory_samples=1)
    p=tmp_path/'legacy';saved=train(1,p,**opts)
    saved.pop('sampling_diagnostics_schema');saved.pop('sampling_diagnostics_started_at')
    saved['diagnostics']={'regret:street=0:active=6:enumerated':17}
    p.write_text(json.dumps(saved));after=train(2,p,resume=p,**opts)
    full=train(2,tmp_path/'full',**opts)
    for field in ('regrets','sums','rng','nodes','iterations'):assert after[field]==full[field]
    assert after['sampling_diagnostics_started_at']==1
    assert after['diagnostics']['regret:street=0:active=6:enumerated']==17
