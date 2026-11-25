"""Triplo v9 --open-translation: off-menu opens answered from the nearest trained spot."""
from dataclasses import replace

import pytest

from poker.fast_game import FastGame
from poker.history_cfr import HistoryPolicy, compact_observation_key
from poker.open_translation import back, translate


def spot(actions, hole=('Jc', 'Tc'), stacks=None):
    """Observation after ``actions`` (fold/call/raise-to) from UTG, as live history records it."""
    g = FastGame(3, button=5, stacks=stacks or [500] * 6, lean=True)
    for a in actions:
        pot, call, _ = g.menu()
        o = g.observe()
        g.history.append((o.player, o.position, o.street, a, pot, call))
        g.act(a, checked=False)
    return replace(g.observe(), hole=hole)


def test_bigger_open_becomes_the_trained_two_and_a_half_bb_spot():
    real = spot([7])                                   # UTG opens 3.5 BB, HJ to act
    assert (real.pot, real.call, real.legal) == (10, 7, ('call', 'fold', 12, 17))
    t = translate(real)
    trained = spot([5])                                # the same hand after a trained 2.5 BB open
    for field in ('position', 'pot', 'call', 'legal', 'stacks', 'active', 'history', 'hole'):
        assert getattr(t, field) == getattr(trained, field), field
    assert compact_observation_key(t) == compact_observation_key(trained)
    assert compact_observation_key(real) != compact_observation_key(trained)


def test_cold_callers_and_blinds_are_translated_with_the_open():
    real = spot([6, 'call', 'fold', 'fold'])           # UTG 3 BB, HJ calls, CO/BTN fold, SB to act
    t = translate(real)
    trained = spot([5, 'call', 'fold', 'fold'])
    assert t.position == 0 and (t.pot, t.call, t.legal) == (trained.pot, trained.call, trained.legal)
    assert t.history == trained.history and t.stacks == trained.stacks


@pytest.mark.parametrize('actions', [
    [5],                                               # trained size: nothing to translate
    [4],
    ['call', 7],                                       # limper first: an isolation raise
    [7, 18],                                           # a 3-bet
])
def test_only_single_off_menu_opens_are_translated(actions):
    assert translate(spot(actions)) is None


def test_max_open_and_hero_already_acted():
    assert translate(spot([7]), max_open=6) is None
    assert translate(spot([6]), max_open=6) is not None
    # SB completes, BB raises big: SB acts again, so this is not a first decision.
    o = spot(['fold', 'fold', 'fold', 'fold', 'call', 9])
    assert o.position == 0 and translate(o) is None


def test_all_in_somebody_is_left_alone():
    o = spot([7], stacks=[500, 500, 500, 500, 500, 500])
    assert translate(replace(o, stacks=o.stacks[:4] + (0,) + o.stacks[5:])) is None


def test_back_keeps_menu_places_and_prices():
    real = spot([7])
    t = translate(real)
    assert [back(a, t, real) for a in t.legal] == ['call', 'fold', 12, 17]
    # Menus of different length (an all-in on the real side only) map by pot fraction.
    shove = replace(real, legal=real.legal + (40,))
    assert back(13, t, shove) == 17 and back(9, t, shove) == 12


class Rules:
    def __init__(self):
        self.calls = 0

    def act(self, o):
        self.calls += 1
        return 'fold' if 'fold' in o.legal else 'call'


def policy_for(o, row=(0., 0., 0., 1.), visits=100, **setting):
    t = translate(o)
    key = compact_observation_key(t)
    p = HistoryPolicy(dict(sums={key: list(row)}, visits={key: visits}, stack_bb=100,
                           abstraction='history-compact-v1'), seed=1)
    p.fallback = Rules()
    if setting:
        p.open_translation = setting
    return p


def test_off_by_default_is_the_rules_decision():
    o = spot([7])
    p = policy_for(o)
    assert p.open_translation is None
    assert p.act(o, history_complete=True) == 'fold' and p.fallback.calls == 1 and p.translated == 0


def test_on_uses_the_translated_learned_row_and_maps_back():
    o = spot([7])
    p = policy_for(o, max_open=None)
    assert p.act(o, history_complete=True) == 17       # learned pot 3-bet, at the real pot size
    assert p.translated == 1 and p.covered == 0 and p.fallback.calls == 0
    # Unverified history never uses learned rows, translated or not.
    assert p.act(o, history_complete=False) == 'fold' and p.translated == 1


def test_on_needs_enough_visits_and_respects_options():
    o = spot([7])
    assert policy_for(o, visits=10, max_open=None).act(o, history_complete=True) == 'fold'
    assert policy_for(o, max_open=6).act(o, history_complete=True) == 'fold'
    p = policy_for(o, max_open=None, fish_only=True)
    p.fallback.fish_seats = set()
    assert p.act(o, history_complete=True) == 'fold'
    p.fallback.fish_seats = {2}                        # the UTG opener is a fish
    assert p.act(o, history_complete=True) == 17
    assert policy_for(o, max_open=None, keep=('call',)).act(o, history_complete=True) == 'fold'
    assert policy_for(o, (1., 0., 0., 0.), max_open=None, keep=('call',)).act(o, history_complete=True) == 'call'


def test_a_learned_row_for_the_real_spot_wins():
    o = spot([7])
    p = policy_for(o, max_open=None)
    key = compact_observation_key(o)
    p.sums[key], p.visits[key] = [1., 0., 0., 0.], 100
    assert p.act(o, history_complete=True) == 'call' and p.covered == 1 and p.translated == 0


def test_live_client_labels_translated_decisions():
    from poker.game import Observation
    from scripts.openpoker_client import decision
    snap = dict(hand_id='h1', actor_seat=3, dealer_seat=5, big_blind=20, pot=100, board=[],
                seats=[dict(seat=i, in_hand=True, folded=False, stack=s, bet=b)
                       for i, (s, b) in enumerate([(1990, 10), (1980, 20), (1930, 70), (2000, 0), (2000, 0), (2000, 0)])],
                hero=dict(seat=3, hole_cards=['Jc', 'Tc'], turn_token='t1',
                          valid_actions=[dict(action='fold'), dict(action='call', amount=70),
                                         dict(action='raise', min=120, max=2000)]))
    history = ((2, 2, 0, 7, 3, 2),)
    p = HistoryPolicy(dict(sums={}, visits={}, stack_bb=100, abstraction='history-compact-v1'), seed=1)
    p.fallback = Rules()
    decision(snap, p, history, history_complete=True)
    assert decision.last['decision_source'] == 'rule_fallback'
    obs = decision.last['observation']
    obs = Observation(**{k: tuple(tuple(h) for h in v) if k == 'history' else tuple(v) if isinstance(v, list) else v
                         for k, v in obs.items()})
    key = compact_observation_key(translate(obs))
    p.sums[key], p.visits[key] = [0., 0., 0., 1.], 100
    p.open_translation = dict(max_open=None)
    wire = decision(snap, p, history, history_complete=True)
    assert decision.last['decision_source'] == 'open_translation'
    assert wire['action'] == 'raise' and wire['amount'] == 170      # pot 3-bet over the real 3.5 BB open
