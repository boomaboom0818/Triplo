from dataclasses import replace

import pytest
from poker.equity_policy import PressureGuard
from poker.fast_game import FastGame
from poker.history_cfr import HistoryPolicy, observation_key


def observation(**changes):
    return replace(FastGame(1, lean=True).observe(), **changes)


def test_preflop_guard_requires_repeated_pressure_and_weak_hand():
    guard = PressureGuard()
    events = ((0, 2, 0, 6, 3, 2), (1, 3, 0, 20, 9, 6))
    weak = observation(hole=('Ks', '8s'), call=20, pot=29,
                       legal=('fold', 'call', 60), history=events)
    assert guard.act(weak, 60) == 'fold'
    assert guard.act(replace(weak, history=events[:1]), 60) == 60
    assert guard.act(replace(weak, hole=('As', 'Ah')), 60) == 60
    assert guard.act(replace(weak, call=0), 60) == 60


def test_postflop_guard_preserves_free_actions_folds_and_supported_continuations(monkeypatch):
    guard = PressureGuard()
    o = observation(street=1, board=('9h', 'Jc', '3c'), hole=('As', 'Qd'),
                    call=20, pot=40, legal=('fold', 'call', 60))
    monkeypatch.setattr(guard.equity, 'act', lambda o: 'fold')
    assert guard.act(o, 'call') == 'fold'
    assert guard.act(o, 60) == 'fold'
    assert guard.act(o, 'fold') == 'fold'
    assert guard.act(replace(o, call=0), 60) == 60
    assert guard.act(replace(o, legal=('call',)), 'call') == 'call'
    monkeypatch.setattr(guard.equity, 'act', lambda o: 'call')
    assert guard.act(o, 60) == 60


def test_guard_is_opt_in_and_overrides_are_not_counted_as_learned():
    events = ((0, 2, 0, 6, 3, 2), (1, 3, 0, 20, 9, 6))
    o = observation(hole=('Ks', '8s'), call=20, pot=29,
                    legal=('fold', 'call', 60), history=events)
    key = observation_key(o)
    saved = dict(sums={key: [0., 0., 100.]}, visits={key: 100}, stack_bb=100)
    original = HistoryPolicy(saved)
    guarded = HistoryPolicy({**saved, 'decision_guard': PressureGuard.VERSION})
    assert original.act(o, history_complete=True) == 60
    assert guarded.act(o, history_complete=True) == 'fold'
    assert guarded.covered == 0 and guarded.pressure_overrides == 1
    with pytest.raises(ValueError, match='unknown history decision guard'):
        HistoryPolicy({**saved, 'decision_guard': 'typo'})


def test_audited_aq_river_raise_is_vetoed_and_nuts_still_take_precedence():
    history = ((2, 2, 0, 'fold', 3, 2), (3, 3, 0, 'call', 3, 2),
               (4, 4, 0, 'fold', 5, 2), (5, 5, 0, 'fold', 5, 2),
               (0, 0, 0, 7, 5, 1), (1, 1, 0, 'fold', 11, 5),
               (3, 3, 0, 'call', 11, 5), (0, 0, 1, 8, 16, 0),
               (3, 3, 1, 'call', 24, 8), (0, 0, 2, 32, 32, 0),
               (3, 3, 2, 'call', 64, 32), (0, 0, 3, 96, 96, 0))
    river = observation(position=3, street=3, hole=('Ad', 'Qs'),
                        board=('9h', 'Jc', '3c', 'Ac', '7s'), call=96, pot=192,
                        legal=('call', 'fold', 153), active=(True, False, False, True, False, False),
                        history=history)
    key = observation_key(river)
    policy = HistoryPolicy(dict(sums={key: [0., 0., 100.]}, visits={key: 100},
                                stack_bb=100, decision_guard=PressureGuard.VERSION))
    assert policy.act(river, history_complete=True) == 'fold'
    assert policy.pressure_overrides == 1
    nuts = replace(river, hole=('Kc', 'Qc'))
    policy.sums[observation_key(nuts)] = [0., 100., 0.]
    policy.visits[observation_key(nuts)] = 100
    assert policy.act(nuts, history_complete=True) == 153
    assert policy.pressure_overrides == 1
