"""Range evidence and live routing regressions, independent of experimental guards."""
from copy import deepcopy

import pytest

from poker.equity_policy import EquityPolicy, holding_range, preflop_weight
from poker.fast_eval import CARD_ID
from poker.history_cfr import HistoryPolicy
from scripts.openpoker_client import decision
from tests.test_openpoker import snapshot


@pytest.mark.parametrize('holding,strength', [
    (('As','2s'), .25), (('As','8s'), .25),
    (('Ks','Qs'), .65), (('As','Ad'), .65),
    (('Qs','Qd'), .25), (('As','Qs'), .12),
])
@pytest.mark.parametrize('pressure', [1, 2])
def test_range_pair_weight_uses_paired_rank(holding, strength, pressure):
    board=tuple(CARD_ID[c] for c in ('Kh','8d','2c'))
    pairs, cumulative, _=holding_range(tuple(sorted(board)), board, False, pressure)
    hand=tuple(sorted(CARD_ID[c] for c in holding))
    i=pairs.index(hand)
    weight=cumulative[i]-(cumulative[i-1] if i else 0)
    assert weight == pytest.approx(preflop_weight(*hand,False)*strength**pressure)


def test_log_identifies_learned_rules_and_safeguard():
    snap=snapshot()
    policy=HistoryPolicy(dict(sums={},visits={},stack_bb=100))
    policy.fallback=EquityPolicy(32)
    snap['hero']['hole_cards']=['Ks','Qs']
    payload=decision(snap,policy)
    assert decision.last['decision_source']=='preflop_rules'
    assert 'decision_source' not in payload
    observation=deepcopy(decision.last['observation'])
    # Capture the exact live key, including legal sizes and normalized stacks.
    from poker.game import Observation
    o=Observation(**{k:tuple(v) if isinstance(v,list) else v for k,v in observation.items()})
    key=policy.key(o)
    policy.sums[key]=[1.]+[0.]*(len(o.legal)-1)
    policy.visits[key]=64
    decision(snap,policy,history_complete=True)
    assert decision.last['decision_source']=='learned_policy'
    snap['hero']['hole_cards']=['As','Ah']
    decision(snap,policy,history_complete=True)
    assert decision.last['decision_source']=='poker_safeguard'


def test_log_identifies_equity_and_safety_fallback():
    snap=snapshot()
    snap['board']=['Kh','8d','2c']
    snap['street']='flop'
    snap['hero']['hole_cards']=['As','2s']
    policy=HistoryPolicy(dict(sums={},visits={},stack_bb=100))
    policy.fallback=EquityPolicy(32)
    decision(snap,policy)
    assert decision.last['decision_source']=='equity_fallback'
    assert 'equity_fallback' in decision.last
    from poker.cfr import CFRPolicy
    decision(snap,CFRPolicy({},abstraction='texture-buckets-v2'))
    assert decision.last['decision_source']=='safety_fallback'


def test_equity_raise_stays_within_bounded_legal_menu(monkeypatch):
    from dataclasses import replace
    from tests.test_equity_policy import observation
    import poker.equity_policy as module
    monkeypatch.setattr(module, 'estimate_equity', lambda *args: (.95, 0.))
    o=observation(('As','Ad'), ('Kh','8d','2c'), call=0)
    # Weak baseline sizing leaves the equity branch to select the bet.
    policy=EquityPolicy(32)
    monkeypatch.setattr(policy.preflop, 'act', lambda _: 'call')
    o=replace(o, legal=('call',50,200,10000))
    assert policy.act(o)==200
    assert policy.act(replace(o,legal=('call',10000)))=='call'
