from dataclasses import asdict

import pytest

from poker import stack_press
from poker.equity_policy import GuardedEquityPolicy
from poker.fast_eval import NAMES
from poker.history_cfr import HistoryPolicy, COMPACT_VERSION
from poker.live_curriculum import root_game
from scripts import live_gate
from scripts.openpoker_client import decision


def empty_policy(press=False, fallback=None):
    p = HistoryPolicy(dict(sums={}, visits={}, stack_bb=100, abstraction=COMPACT_VERSION))
    p.fallback = fallback or GuardedEquityPolicy(32, stack_press=press)
    return p


def spy_keys(p):
    keys, original = [], p.key
    p.key = lambda o: (keys.append(original(o)), keys[-1])[1]
    return keys


def repro_snapshot():
    blinds = {0: 10, 1: 20}
    return dict(hand_id='offline', actor_seat=2, dealer_seat=5, big_blind=20, pot=30, board=[],
                seats=[dict(seat=i, in_hand=True, folded=False, bet=blinds.get(i, 0),
                            stack=(5000 if i == 2 else 1000) - blinds.get(i, 0)) for i in range(6)],
                hero=dict(seat=2, turn_token='offline', hole_cards=['As', 'Ks'],
                          valid_actions=[dict(action='call', amount=20), dict(action='fold'),
                                         dict(action='raise', min=40, max=5000)]))


def flop_game(stacks_bb=(500,) * 6):
    """Everyone limps; the small blind is first to act on the flop."""
    game = root_game(7, 6, list(stacks_bb))
    while not game.street:
        game.act('call')
    return game


def snapshot_from(game):
    """The OpenPoker snapshot of a FastGame state (big blind 2: one wire chip per unit)."""
    i = game.actors[0]
    o = game.observe()
    top = max(game.bets)
    valid = [dict(action='check') if not o.call else dict(action='call', amount=o.call)]
    if 'fold' in o.legal:
        valid.append(dict(action='fold'))
    if game.can_raise():
        hi = game.stacks[i] + game.bets[i]
        valid.append(dict(action='raise', min=min(hi, max(game.raise_size, 2) + top), max=hi))
    return dict(hand_id='h', actor_seat=i, dealer_seat=5, big_blind=2, pot=sum(game.paid),
                board=[NAMES[c] for c in game.board],
                seats=[dict(seat=j, in_hand=True, folded=not game.active[j], bet=game.bets[j],
                            stack=game.stacks[j]) for j in range(6)],
                hero=dict(seat=i, turn_token='t', hole_cards=[NAMES[c] for c in game.hole[i]],
                          valid_actions=valid))


def test_learned_state_is_unchanged_by_stack_press_preflop_repro():
    p = empty_policy()
    keys = spy_keys(p)
    decision(repro_snapshot(), p, history_complete=True)
    k = keys[-1]
    p.sums[k] = [0., 0., 0., 1.]; p.visits[k] = 64
    seen = {}
    for press in (False, True):
        p.fallback.stack_press = press
        decision(repro_snapshot(), p, history_complete=True)
        seen[press] = (keys[-1], decision.last['observation']['legal'], decision.last['decision_source'])
    assert seen[True] == seen[False]
    assert seen[True][0] == k and seen[True][2] == 'learned_policy'


def test_learned_postflop_state_keeps_key_when_press_sizes_exist():
    game = flop_game()
    p = empty_policy()
    keys = spy_keys(p)
    decision(snapshot_from(game), p, history_complete=True, public_history=game.history)
    k = keys[-1]
    legal = decision.last['observation']['legal']
    p.sums[k] = [1.] + [0.] * (len(legal) - 1); p.visits[k] = 64
    p.fallback.stack_press = True
    decision(snapshot_from(game), p, history_complete=True, public_history=game.history)
    assert keys[-1] == k and decision.last['observation']['legal'] == legal
    assert decision.last['decision_source'] == 'learned_policy'
    assert decision.last['stack_press_sizes']  # offered, but never to the learned lookup


class Spy(GuardedEquityPolicy):
    def __init__(self):
        super().__init__(32, stack_press=True)
        self.menus = []

    def act(self, o):
        self.menus.append(o.legal)
        return o.legal[-1]


def test_fallback_miss_receives_stack_press_sizes():
    game = flop_game()
    spy = Spy()
    p = empty_policy(fallback=spy)
    message = decision(snapshot_from(game), p, history_complete=True, public_history=game.history)
    trained = decision.last['observation']['legal']
    extra = decision.last['stack_press_sizes']
    assert extra and not set(extra) & set(trained)
    assert spy.menus[-1] == trained + tuple(extra)
    assert decision.last['decision_source'] == 'equity_fallback'
    assert message['action'] == 'raise' and message['amount'] == extra[-1]
    # A wrapper must not survive the decision.
    assert p.fallback is spy


def test_effective_stack_not_hero_stack_gates_press_sizes():
    # Hero is 2500 BB deep but every opponent has 50 BB: nothing to press.
    game = flop_game((50, 50, 50, 50, 50, 50))
    snap = snapshot_from(game)
    hero = snap['actor_seat']
    snap['seats'][hero]['stack'] = 5000
    snap['hero']['valid_actions'][-1]['max'] = 5000 + snap['seats'][hero]['bet']
    spy = Spy()
    decision(snap, empty_policy(fallback=spy), history_complete=True, public_history=game.history)
    assert 'stack_press_sizes' not in decision.last
    assert spy.menus[-1] == decision.last['observation']['legal']
    off = Spy(); off.stack_press = False
    decision(snap, empty_policy(fallback=off), history_complete=True, public_history=game.history)
    assert off.menus[-1] == spy.menus[-1]
    assert stack_press.targets(0, 12, 0, 5000, [(0, 100)] * 5) == ()
    assert stack_press.targets(0, 12, 0, 5000, [(0, 100), (0, 600)]) == (24, 300)


def test_gate_and_live_adapter_build_identical_menus_keys_and_press_sizes():
    for stacks in ((500,) * 6, (100,) * 6, (50, 100, 100, 100, 200, 500)):
        game = flop_game(stacks)
        o = game.observe()
        gate_spy, live_spy = Spy(), Spy()
        gate, live = empty_policy(fallback=gate_spy), empty_policy(fallback=live_spy)
        gate_keys, live_keys = spy_keys(gate), spy_keys(live)
        gate_action = live_gate.hero_decision(gate, game, o)
        decision(snapshot_from(game), live, history_complete=True, public_history=game.history)
        assert decision.last['observation'] == asdict(o)
        assert set(gate_keys) == set(live_keys) and len(set(live_keys)) == 1  # adapter also looks up once itself
        assert gate_spy.menus == live_spy.menus
        assert tuple(decision.last.get('stack_press_sizes', ())) == live_gate.press_sizes(gate, game)
        assert decision.last['amount'] == (gate_action if isinstance(gate_action, int) else None)
        # The gate can apply an off-menu press size and keeps a consistent ledger.
        live_gate.play(game, o, gate_action)
        assert game.history[-1][3] == gate_action


def test_gate_stack_press_hero_runs_end_to_end(tmp_path):
    from poker.history_cfr import ParallelHistory
    path = tmp_path / 'empty.json'; ParallelHistory().export_policy(path)
    r = live_gate.evaluate(path, path, tmp_path / 'r.json', 98423, 2, hero_fallback='equity-guarded-v1+stack-press')
    assert all(case['decisions'] > 0 for case in r['cases'].values())
