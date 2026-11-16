from copy import deepcopy
import pytest
from tests.test_live_gate import report
from scripts.aggressive_campaign import readiness
from scripts.live_gate import assess


def test_readiness_requires_aggressive_gain_over_both_controls():
    c=report(10);i=report(5,'champion');f=report(6,None,True)
    assert readiness(c,i,f,.025)['ready_for_confirmation']
    for name in c['cases']:
        if 'aggressive' in name:c['cases'][name]['blocks']=[5]*10
    r=readiness(c,i,f,.025)
    assert not r['ready_for_confirmation']
    assert r['aggressive_advantage']['fallback']['mean']<0


def test_smaller_error_budget_tightens_bounds():
    c=report(10);i=report(5,'champion');f=report(6,None,True)
    for case in c['cases'].values():case['blocks']=[8,9,10,11,12]*2
    assert assess(c,i,f,family_alpha=.025)['aggregate']['vs_incumbent']['lower'] < assess(c,i,f)['aggregate']['vs_incumbent']['lower']
    with pytest.raises(ValueError):assess(c,i,f,family_alpha=0)
