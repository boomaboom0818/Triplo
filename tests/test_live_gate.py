from copy import deepcopy
import json
import pytest
from scripts.live_gate import assess, SCENARIOS, evaluate
from poker.history_cfr import ParallelHistory


def report(value, identity='candidate', fallback=False):
    return dict(seed=99,deals=10,min_visits=64,fallback='equity-v1',incumbent_sha256='champion',
                checkpoint_sha256=identity,fallback_only=fallback,scenarios=SCENARIOS,source_sha256={'source':'hash'},
                cases={name:dict(blocks=[value]*10,decisions=100,learned=40,postflop_decisions=40,
                                postflop_learned=10,p99_seconds=.01) for name,*_ in SCENARIOS})


def test_promotion_requires_both_controls_and_every_condition():
    champion=report(5,'champion');fallback=report(6,None,True)
    c=report(10)
    assert assess(c,champion,fallback)['promote']
    assert not assess(c,champion,fallback,confirmation=False)['promote']
    assert not assess(report(5),champion,fallback)['promote']
    weak=deepcopy(c);weak['cases'][SCENARIOS[0][0]]['blocks']=[-100]*10
    assert not assess(weak,champion,fallback)['checks']['no_material_case_regression']
    for case in c['cases'].values():case['learned']=0;case['postflop_learned']=0
    assert not assess(c,champion,fallback)['promote']


def test_gate_rejects_unpaired_or_truncated_reports():
    c=report(10);i=report(5,'champion');f=report(6,None,True)
    i['seed']=98
    with pytest.raises(ValueError):assess(c,i,f)
    i['seed']=99;i['cases'][SCENARIOS[0][0]]['blocks'].pop()
    with pytest.raises(ValueError):assess(c,i,f)


def test_end_to_end_gate_uses_exact_short_handed_seat_counts(tmp_path):
    path=tmp_path/'empty.json';ParallelHistory().export_policy(path)
    out=tmp_path/'result.json'
    r=evaluate(path,path,out,98423,2)
    assert json.loads(out.read_text())['checkpoint_sha256']==r['checkpoint_sha256']
    for name,players,*_ in SCENARIOS:
        assert r['cases'][name]['hands']==2*players
        assert r['cases'][name]['decisions']>0
    with pytest.raises(ValueError):evaluate(path,path,out,98423,2)
