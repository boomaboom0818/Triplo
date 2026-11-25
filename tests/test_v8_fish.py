"""Tests for Triplo v8 --fish-exploit: no_code model, leaderboard fish seats, exploits, client wiring."""
from dataclasses import replace
from random import Random

import pytest

from poker import equity_pricing
from poker.equity_pricing import VERSION_V5, make_fallback
from poker.fast_game import BIG_BLIND, FastGame
from poker.fish_preflop import FishPreflop, hand_rank
from poker.game import Observation
from poker.nocode_model import NoCodeFish
from poker.opponent_strength import OpponentField

from tests.test_v6_anti_exploit import V5_FAST_FLAGS, _run_client

FISH = dict(fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', .6))
# Hero (position 5) opened, villain (position 1) called; villain checks the turn to hero.
TURN = Observation(
    player=5, position=5, hole=('Kc', 'Qd'), board=('Kh', '9d', '2c', '5s'),
    stacks=(500, 470, 500, 500, 500, 470),
    active=(False, True, False, False, False, True),
    pot=60, call=0, street=2, legal=('call', 30, 60, 470),
    history=((5, 5, 0, 6, 3, 2), (1, 1, 0, 'call', 9, 4), (1, 1, 1, 'call', 12, 0), (5, 5, 1, 12, 12, 0),
             (1, 1, 1, 'call', 24, 12), (1, 1, 2, 'call', 60, 0)),
)


def _equity(monkeypatch, value):
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (value, 0.02))


def test_nocode_model_plays_legal_hands_at_live_frequencies():
    fish = NoCodeFish(seed=3)
    opens = unopened = 0
    for seed in range(150):
        rng = Random(seed)
        game = FastGame(seed, button=5, stacks=[rng.choice((40, 80, 120)) * BIG_BLIND for _ in range(6)], lean=0)
        while game.live:
            o = game.observe()
            action = fish.act(o, game)
            raises = [a for a in o.legal if isinstance(a, int)]
            assert action in ('call', 'fold') or (raises and raises[0] <= action <= raises[-1])
            assert action != 'fold' or 'fold' in o.legal
            if not o.street and not any(h[2] == 0 and isinstance(h[3], int) for h in o.history) and o.position != 1:
                unopened += 1
                opens += isinstance(action, int)
            game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
            game.act(action, checked=False)
    assert .2 < opens / unopened < .4  # live: 22-46% by position


def test_opponent_field_is_weak_matches_strong_check():
    rows = [
        dict(bot_name='fish', bot_kind='no_code', hands_played=900, chip_balance=1000, chips_at_table=0, rebuys=3),
        dict(bot_name='loser', bot_kind='self_hosted', hands_played=900, chip_balance=500, chips_at_table=0, rebuys=4),
        dict(bot_name='apex', bot_kind='self_hosted', hands_played=900, chip_balance=60000, chips_at_table=0, rebuys=0),
        dict(bot_name='reg', bot_kind='self_hosted', hands_played=900, chip_balance=9000, chips_at_table=0, rebuys=0),
    ]
    field = OpponentField(rows)
    assert field.is_weak('fish') and field.is_weak('loser')
    assert not field.is_weak('apex') and not field.is_weak('reg') and not field.is_weak('unknown')
    assert set(field.strong_check('hero', ['fish', 'loser', 'reg']).weak) == {'fish', 'loser'}


def test_fish_value_sizing_only_against_leaderboard_fish(monkeypatch):
    _equity(monkeypatch, 0.8)
    river = replace(TURN, street=3, board=TURN.board + ('3h',), legal=('call', 30, 60, 75, 470),
                    history=TURN.history + ((5, 5, 2, 'call', 60, 0), (1, 1, 3, 'call', 60, 0)))
    fb = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False, **FISH)
    fb.profiles = {1: (0.05, 0.4)}
    monkeypatch.setattr(fb, '_act_v5', lambda o: 30)              # the v7 decision: a half-pot value bet
    fb.fish_seats = {1}
    assert fb.act(river) == 75 and fb.last.get('fish_size')      # 1.25 pot on the river
    fb.fish_seats = set()                                         # leaderboard says not a fish
    assert fb.act(river) == 30                                    # a sticky profile alone never triggers
    fb.fish_seats = None                                          # no leaderboard: the profile decides
    assert fb.act(river) == 75
    fb.fish_seats = {1}
    assert fb.act(replace(river, hole=('7c', '4d'))) == 30        # no top pair: never resized


def test_fish_reraise_with_top_pair_not_weak_pair(monkeypatch):
    _equity(monkeypatch, 0.6)
    raised = replace(TURN, pot=150, call=60, legal=('call', 'fold', 180, 260, 470),
                     history=TURN.history + ((5, 5, 2, 30, 60, 0), (1, 1, 2, 90, 90, 30)))
    fb = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False, **FISH)
    fb.profiles = {1: (0.05, 0.4)}
    fb.fish_seats = {1}
    action = fb.act(raised)
    assert isinstance(action, int) and fb.last.get('fish_raise')
    weak = replace(raised, hole=('9c', '8c'))
    assert not isinstance(fb.act(weak), int)
    fb.fish_seats = set()
    assert not isinstance(fb.act(raised), int) or not fb.last.get('fish_raise')


class _Inner:
    def __init__(self, fish_seats):
        self.fallback = make_fallback(VERSION_V5, solve_river=False, **FISH)
        self.fallback.fish_seats = fish_seats
        self.calls = 0

    def act(self, o, **kwargs):
        self.calls += 1
        return 'inner'


def test_fish_preflop_isolates_fish_opens_and_delegates_otherwise():
    opened = Observation(player=5, position=5, hole=('Ah', 'Kh'), board=(), stacks=(199, 198, 195, 200, 200, 200),
                         active=(True, True, True, True, True, True), pot=8, call=5, street=0,
                         legal=('call', 'fold', 13, 21, 200),
                         history=((2, 2, 0, 5, 3, 2), (3, 3, 0, 'fold', 8, 5), (4, 4, 0, 'fold', 8, 5)))
    assert hand_rank(('Ah', 'Kh')) < .05
    layer = FishPreflop(_Inner({0, 1, 2, 3, 4}), steal=False)
    action = layer.act(opened)
    assert isinstance(action, int) and layer.overrides == 1
    assert layer.act(replace(opened, hole=('7c', '2d'))) == 'fold'
    reg_open = FishPreflop(_Inner({0, 1, 3, 4}), steal=False)            # opener (2) is not a fish
    assert reg_open.act(opened) == 'inner' and reg_open.overrides == 0
    unopened = replace(opened, call=2, history=())
    assert layer.act(unopened) == 'inner'                               # steal off: no unopened override
    postflop = replace(opened, street=1, board=('2c', '3d', '9s'))
    assert layer.act(postflop) == 'inner'


def test_client_fish_exploit_flag_wiring_and_requirements(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_client(tmp_path, monkeypatch, *V5_FAST_FLAGS, '--fish-exploit')  # needs --avoid-strong
    with pytest.raises(SystemExit):
        _run_client(tmp_path, monkeypatch, '--fallback', 'equity-guarded-v3', '--avoid-strong', 'report',
                    '--fish-exploit')
    fb = _run_client(tmp_path, monkeypatch, *V5_FAST_FLAGS, '--avoid-strong', 'report', '--fish-exploit')
    assert fb.fish_size == FISH['fish_size'] and fb.fish_raise == FISH['fish_raise']
    plain = _run_client(tmp_path, monkeypatch, *V5_FAST_FLAGS)
    assert plain.fish_size is None and plain.fish_raise is None and plain.fish_seats is None


def test_client_opponent_profiles_sets_fish_seats_from_lookup():
    from scripts.openpoker_client import opponent_profiles
    fb = make_fallback(VERSION_V5, solve_river=False)
    fb.fish_lookup = lambda name: name.startswith('fish')
    seats = [dict(name='fish_a', folded=False), dict(name='reg', folded=False), dict(name='hero', folded=False),
             dict(name='fish_b', folded=True), dict(name='fish_c', folded=False), dict(name='x', folded=False)]
    opponent_profiles(fb, seats, 0, 2)
    assert fb.fish_seats == {0, 4}
