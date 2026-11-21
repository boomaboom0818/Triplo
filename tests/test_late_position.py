from copy import deepcopy
import json

import pytest

from poker.equity_pricing import make_fallback
from poker.history_cfr import HistoryPolicy
from poker.late_position import FoldEvidence, LatePositionSteals
from scripts.fit_late_position import fit, record_open
from scripts.openpoker_client import decision


FOLDS = ((2, 2, 0, 'fold', 3, 2), (3, 3, 0, 'fold', 3, 2), (4, 4, 0, 'fold', 3, 2))
CUTOFF = '2026-11-20T00:00:00+00:00'


def snapshot():
    names = ('SB', 'BB', 'UTG', 'HJ', 'CO', 'Triplo')
    return dict(hand_id='h1', actor_seat=5, dealer_seat=5, big_blind=20, pot=30, board=[],
                seats=[dict(seat=i, name=n, in_hand=True, folded=i in (2, 3, 4), bet=b, stack=2000-b)
                       for i, (n, b) in enumerate(zip(names, (10, 20, 0, 0, 0, 0)))],
                hero=dict(seat=5, hole_cards=['Qh', '9s'], turn_token='t1',
                          valid_actions=[dict(action='fold'), dict(action='call', amount=20),
                                         dict(action='raise', min=40, max=2000)]))


def policy(evidence=None, report=False):
    p = HistoryPolicy(dict(sums={}, visits={}, stack_bb=100))
    p.fallback = make_fallback('equity-guarded-v3')
    if evidence is not None:
        p.late_position_steals = LatePositionSteals('Triplo', {(5, ('SB', 'BB')): evidence})
        p.late_position_report = report
    return p


def wire_action(a):
    return {k: v for k, v in a.items() if k != 'client_action_id'}


def test_overlay_changes_only_a_baseline_fold_and_keeps_exact_wire_amount():
    s = snapshot()
    baseline = decision(s, policy(), FOLDS, True)
    assert baseline['action'] == 'fold'
    a = decision(s, policy(FoldEvidence((29, 30), (15, 15))), FOLDS, True)
    assert wire_action(a) == dict(wire_action(baseline), action='raise', amount=50)
    assert decision.last['decision_source'] == 'late_position_steal'
    assert decision.last['late_position_steal']['baseline_source'] == 'preflop_rules'
    assert decision.last['late_position_steal']['applied'] is True
    assert 'late_position_steal' not in a


def test_report_mode_keeps_learned_action_and_logs_proposal():
    s = snapshot()
    p = policy(FoldEvidence((30, 30), (15, 15)), report=True)
    p.key = lambda _: 'known'
    p.sums['known'], p.visits['known'] = [0, 1, 0, 0], 64
    assert decision(s, p, FOLDS, True)['action'] == 'fold'
    assert decision.last['covered'] is True
    assert decision.last['decision_source'] == 'learned_policy'
    assert decision.last['late_position_steal']['applied'] is False
    p.late_position_report = False
    assert decision(s, p, FOLDS, True)['action'] == 'raise'
    assert decision.last['covered'] is False
    assert decision.last['late_position_steal']['baseline_covered'] is True


@pytest.mark.parametrize('evidence', [FoldEvidence((29, 29), (15, 15)), FoldEvidence((30, 30), (14, 14)),
                                     FoldEvidence((30, 30), (9, 15)), FoldEvidence((20, 30), (15, 15))])
def test_insufficient_or_nonreplicating_evidence_leaves_v3(evidence):
    assert decision(snapshot(), policy(evidence), FOLDS, True)['action'] == 'fold'
    assert 'late_position_steal' not in decision.last


@pytest.mark.parametrize('case', ['unknown history', 'wrong names', 'wrong hero', 'inexact size', 'short stack',
                                  'limper', 'previous raise', 'not six-max', 'weak cards', 'postflop'])
def test_out_of_scope_has_identical_wire_and_diagnostics(case):
    s, history, verified = snapshot(), FOLDS, True
    if case == 'unknown history': verified = False
    if case == 'wrong names': s['seats'][0]['name'] = 'Another SB'
    if case == 'wrong hero': s['seats'][5]['name'] = 'Another hero'
    if case == 'inexact size': s['hero']['valid_actions'][-1]['min'] = 51
    if case == 'short stack': s['seats'][0]['stack'] = 20
    if case == 'limper': history = FOLDS[:2] + ((4, 4, 0, 'call', 3, 2),)
    if case == 'previous raise': history = FOLDS[:2] + ((4, 4, 0, 5, 3, 2),)
    if case == 'not six-max': s['seats'].pop(2)
    if case == 'weak cards': s['hero']['hole_cards'] = ['3h', '7s']
    if case == 'postflop': s['board'] = ['2c', '4d', 'As']
    v3 = wire_action(decision(s, policy(), history, verified))
    diagnostics = deepcopy(decision.last)
    overlay = wire_action(decision(s, policy(FoldEvidence((30, 30), (15, 15))), history, verified))
    assert v3 == overlay and diagnostics == decision.last


def test_overlay_does_not_replace_a_learned_limp_or_raise():
    p = policy(FoldEvidence((30, 30), (15, 15)))
    p.key = lambda _: 'known'; p.visits['known'] = 64
    for row, expected in (([1, 0, 0, 0], 'call'), ([0, 0, 1, 0], 'raise')):
        p.sums['known'] = row
        assert decision(snapshot(), p, FOLDS, True)['action'] == expected
        assert 'late_position_steal' not in decision.last


@pytest.mark.parametrize('pos,behind,history,hole', [
    (4, ('BTN', 'SB', 'BB'), FOLDS[:2], ('Kh', 'Ts')),
    (0, ('BB',), FOLDS + ((5, 5, 0, 'fold', 3, 2),), ('Kh', 'Ts')),
])
def test_cutoff_and_small_blind_use_their_own_remaining_lineup(pos, behind, history, hole):
    s = snapshot()
    s['actor_seat'] = s['hero']['seat'] = pos
    s['hero']['hole_cards'] = hole
    s['hero']['valid_actions'][1]['amount'] = 10 if pos == 0 else 20
    for i, seat in enumerate(s['seats']):
        seat['folded'] = i in {a[1] for a in history}
        if i == 5:
            seat['name'] = 'BTN'
        if i == pos:
            seat['name'] = 'Triplo'
    p = policy(FoldEvidence((30, 30), (15, 15)))
    p.late_position_steals.evidence[(pos, behind)] = FoldEvidence((30, 30), (15, 15))
    a = decision(s, p, history, True)
    assert a['action'] == 'raise' and a['amount'] == 50
    assert decision.last['late_position_steal']['required'] == (0.6014 if pos == 0 else 0.655)


def evidence_hand(hid, day, fold=True):
    s = snapshot()
    return dict(hand_id=hid, num_players=6, big_blind=20, small_blind=10, dealer_seat=5,
                started_at=f'2026-11-{day}T12:00:00+00:00', ended_at=f'2026-11-{day}T12:01:00+00:00',
                players=[dict(seat=p['seat'], bot_name=p['name'], stack_start=2000) for p in s['seats']],
                board=[] if fold else ['2c', '4d', 'As'], action_log=[
                    dict(seat=seat, sequence_in_hand=i+1, action=action, amount=50 if action=='raise' else 0,
                         street=('showdown' if fold else 'flop') if seat==1 else 'preflop')
                    for i,(seat,action) in enumerate(zip((2,3,4,5,0,1), ('fold','fold','fold','raise','fold', 'fold' if fold else 'call')))])


def test_fit_keeps_losses_and_uses_disjoint_time_windows(tmp_path):
    a = evidence_hand('a', 19)
    b = evidence_hand('b', 20, fold=False)
    c = evidence_hand('c', 19); c['ended_at'] = CUTOFF  # spans the cutoff
    packed = fit([b, a, a, c], 'Triplo', CUTOFF)
    assert packed['lineups'] == [dict(position=5, behind=['SB', 'BB'], train=[1, 1], holdout=[0, 1])]
    path = tmp_path/'stats.json'; path.write_text(json.dumps(packed))
    loaded = LatePositionSteals.load(path)
    assert loaded.evidence[(5, ('SB', 'BB'))].lower() is None
    b['hand_id'] = 'a'
    with pytest.raises(ValueError, match='conflicting'):
        fit([a, b], 'Triplo', CUTOFF)


def test_incomplete_or_different_size_evidence_is_not_counted():
    h = evidence_hand('a', 19)
    assert record_open(h, 'Triplo') == ((5, ('SB','BB')), True)
    h['action_log'].pop()
    assert record_open(h, 'Triplo') is None
    h = evidence_hand('b', 19)
    h['action_log'][3]['amount'] = 49
    assert record_open(h, 'Triplo') is None
