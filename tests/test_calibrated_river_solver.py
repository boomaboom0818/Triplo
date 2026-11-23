from dataclasses import replace
from itertools import combinations
import sys

import pytest

np = pytest.importorskip('numpy')

from poker.calibrated_river_solver import ComboPayoffs, propose, solve_tree
from poker.equity_pricing import GuardedEquityPolicyV3
from poker.history_cfr import HistoryPolicy, VERSION
from poker.river_solver_fallback import RiverSolverFallback
from scripts.openpoker_client import decision
from poker.game import Observation


def spot(*, facing=False):
    hist = [(p, p, 0, 'fold', 3, 2) for p in range(2, 6)]
    hist += [(0, 0, 0, 'call', 3, 1), (1, 1, 0, 'call', 4, 0),
             (0, 0, 1, 'call', 4, 0), (1, 1, 1, 50, 4, 0), (0, 0, 1, 'call', 54, 50),
             (0, 0, 2, 'call', 104, 0), (1, 1, 2, 'call', 104, 0)]
    if facing:
        hist += [(0, 0, 3, 'call', 104, 0), (1, 1, 3, 104, 104, 0)]
    return Observation(0, 0, ('As', 'Kd'), ('Kc', '7c', '5h', '2s', '9d'),
                       (300, 196 if facing else 300, 300, 300, 300, 300),
                       (True, True, False, False, False, False), 208 if facing else 104,
                       104 if facing else 0, 3, ('call', 'fold', 208) if facing else ('call', 52, 104), tuple(hist))



def test_payoffs_match_combo_enumeration_with_card_collisions_and_ties():
    hands = tuple(combinations(range(47), 2))
    rng = np.random.default_rng(42)
    rank = rng.integers(0, 200, len(hands))
    mass = rng.random(len(hands))
    a, b = np.array(hands).T
    compatible = (a[:, None] != a) & (a[:, None] != b) & (b[:, None] != a) & (b[:, None] != b)
    showdown = compatible * np.sign(rank[:, None] - rank)
    payoffs = ComboPayoffs(hands, rank)
    np.testing.assert_allclose(payoffs.compatible(mass), compatible @ mass)
    np.testing.assert_allclose(payoffs.showdown(mass), showdown @ mass)


def test_solver_calls_nuts_and_checks_losing_hand_when_opponent_always_has_nuts():
    payoffs = ComboPayoffs(((0, 1), (2, 3)), [0, 1])
    oop_open, ip_response, ip_check_to, oop_response = solve_tree(
        payoffs, np.array([1., 0.]), np.array([0., 1.]), .5)
    assert oop_open[1, 0] < .05
    assert ip_response[1, 1] > .95
    assert oop_response[1, 0] < .05
    for strategy in (oop_open, ip_response, ip_check_to, oop_response):
        np.testing.assert_allclose(strategy.sum(axis=0), 1.)


def test_solver_bounds_eligibility_and_keeps_action_legal():
    opening = spot()
    facing = spot(facing=True)
    for o, node in ((opening, 0), (facing, 3)):
        action, details = propose(o, 'call')
        assert action in o.legal
        assert details['node'] == node
        assert 0 <= details['probability'] <= 1
        assert propose(o, 'call') == (action, details)
    assert propose(replace(facing, history=()), 'call') is None
    assert propose(replace(facing, pot=999), 'call') is None
    assert propose(replace(facing, active=(True, True, True, False, False, False)), 'call') is None
    assert propose(facing, 208) is None


def test_client_solver_requires_verified_history(monkeypatch):
    o = spot(facing=True)
    p = HistoryPolicy(dict(stack_bb=100, abstraction=VERSION, visits={}, sums={}))
    p.fallback = RiverSolverFallback()
    monkeypatch.setattr(GuardedEquityPolicyV3, 'act', lambda *_: 'call')
    snapshot = dict(hand_id='h', actor_seat=0, dealer_seat=5, big_blind=2, board=o.board, pot=o.pot,
                    hero=dict(seat=0, turn_token='token', hole_cards=o.hole,
                              valid_actions=[dict(action='fold'), dict(action='call', amount=o.call)]),
                    seats=[dict(seat=i, name=str(i), folded=not active, in_hand=True,
                                stack=o.stacks[i], bet=o.call if i == 1 else 0)
                           for i, active in enumerate(o.active)])
    decision(snapshot, p, o.history, history_complete=True)
    assert decision.last['equity_fallback']['river_solver']['node'] == 3
    assert decision(snapshot, p, o.history, history_complete=False)['action'] == 'call'
    assert 'equity_fallback' not in decision.last


@pytest.mark.parametrize(('flags', 'buy_in', 'solver'), [
    ([], 1000, True),
    (['--no-river-solver'], 1000, False),
    (['--buy-in', '2500'], 2500, False),
    (['--buy-in', '5000', '--river-solver'], 5000, True),
])
def test_v5_cli_presets(tmp_path, monkeypatch, flags, buy_in, solver):
    from poker.history_cfr import save
    from poker.sampled_cfr import SampledCFR
    from scripts import openpoker_client

    checkpoint = tmp_path / 'history.json'
    save(checkpoint, SampledCFR(6, 1), 1, 20)
    monkeypatch.setenv('OPEN_POKER_API_KEY', 'test-placeholder')
    monkeypatch.setattr(sys, 'argv', ['client', '--checkpoint', str(checkpoint),
                                      '--fallback', 'equity-guarded-v5', *flags])
    seen = []

    async def fake_run(args, policy, _key):
        seen.append((args.buy_in, policy.fallback.VERSION, policy.fallback.solve_river))

    monkeypatch.setattr(openpoker_client, 'run', fake_run)
    openpoker_client.main()
    assert seen == [(buy_in, 'equity-guarded-v5', solver)]
