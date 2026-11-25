"""Tests for Triplo v9: A/B metric harness, fish open sizing, raise-not-limp, fish flop stabs,
the fish value cap, client wiring, and decision identity with every v9 option off."""
from dataclasses import replace
import json

import pytest

from poker import ab_metric as M
from poker import equity_pricing
from poker.equity_pricing import VERSION_V5, make_fallback
from poker.fish_preflop import FishPreflop
from poker.game import Observation

from tests.test_v6_anti_exploit import V5_FAST_FLAGS, _run_client
from tests.test_v8_fish import FISH, TURN

UNOPENED = Observation(player=2, position=2, hole=('Ah', 'Kh'), board=(), stacks=(499, 498, 500, 500, 500, 500),
                       active=(True,) * 6, pot=3, call=2, street=0, legal=('call', 'fold', 4, 5), history=())
# Hero (5) and a fish (1) see a heads-up flop; the fish checks to hero.
FLOP = Observation(player=5, position=5, hole=('7c', '6d'), board=('Kh', '9s', '2c'),
                   stacks=(500, 494, 500, 500, 500, 494),
                   active=(False, True, False, False, False, True), pot=13, call=0, street=1,
                   legal=('call', 6, 13),
                   history=((2, 2, 0, 'fold', 3, 2), (3, 3, 0, 'fold', 3, 2), (4, 4, 0, 'fold', 3, 2),
                            (5, 5, 0, 6, 3, 2), (0, 0, 0, 'fold', 9, 5), (1, 1, 0, 'call', 9, 4),
                            (1, 1, 1, 'call', 13, 0)))


class _Inner:
    """Stand-in learned policy: returns a fixed preflop action; owns a real v5 fallback."""

    def __init__(self, action, fish_seats):
        self.fallback = make_fallback(VERSION_V5, solve_river=False, **FISH)
        self.fallback.fish_seats = fish_seats
        self.action = action

    def act(self, o, **kwargs):
        return self.action


def _equity(monkeypatch, value):
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (value, 0.02))


# -- A/B metric harness ------------------------------------------------------------------------

def test_metric_caps_pairs_and_window_shares():
    nets = [100., -3., 2., -1., 0.5, -60.]
    assert M.estimate(nets, 25)['bb100'] == pytest.approx(100 * (25 - 3 + 2 - 1 + .5 - 25) / 6, abs=.01)
    assert M.estimate(nets)['n'] == 6
    d = M.paired([1., 2., 3.], [2., 3., 4.])
    assert d['bb100'] == 100 and d['ci95'] == [100, 100]
    with pytest.raises(ValueError):
        M.paired([1.], [1., 2.])
    assert M.welch([1., 2., 3.], [2., 3., 4.])['bb100'] == 100
    # Windows: two 10-minute windows, one positive.
    w = M.window_share([(0, 5.), (100, -1.), (700, -2.)])
    assert w['windows'] == 2 and w['positive_share'] == .5
    assert M.block_share([1, 1, -5, 1], 2) == dict(blocks=2, positive_share=.5)
    assert M.paired_block_share([0, 0, 0, 0], [1, -2, 0, 0], 2)['arm_ahead_share'] == .25
    # Hands needed scale with variance; capping cuts them.
    assert M.hands_for(25, 35) > M.hands_for(12.4, 35) > M.hands_for(7.4, 35)
    assert 1500 < M.hands_for(7.4, 35) < 2000


def test_metric_big_pot_breakdown_sums_to_total():
    nets, pots = [1., -2., 60., -40., 3.], [3., 12., 130., 90., 5.]
    by = M.by_pot(nets, pots)
    assert sum(v['contribution_bb100'] for v in by.values()) == pytest.approx(M.estimate(nets)['bb100'], abs=.05)
    big = M.big_pot_summary(nets, pots)
    assert big['hands'] == 2 and big['contribution_bb100'] == pytest.approx(400)
    paired = M.paired_by_pot(nets, [n + 1 for n in nets], pots, pots)
    assert sum(v['contribution_bb100'] for v in paired.values()) == pytest.approx(100, abs=.05)


def test_ab_report_reads_session_log(tmp_path):
    from scripts.ab_report import load, report
    log = tmp_path / 'session.jsonl'
    rows = []
    stacks = 5000
    for k in range(6):
        new = stacks + (100 if k % 2 else -40)
        rows.append(dict(time=1000 + 120 * k, type='hand_result', seat=2,
                         result=dict(table_id='t', hand_id=f'h{k}', total_pot=200, final_stacks={'2': new},
                                     payouts=[dict(seat=2, amount=200)] if k % 2 else [])))
        stacks = new
    log.write_text('\n'.join(json.dumps(r) for r in rows))
    rep = report(load([log], 20, 5000), [('a', 1000, 1400), ('b', 1400, None)])
    assert rep['arms']['a']['uncapped']['n'] == 3 and rep['arms']['b']['uncapped']['n'] == 2
    assert 'b-a' in rep['differences']


# -- #2 bigger opens against fish --------------------------------------------------------------

def test_fish_open_resizes_only_when_everyone_behind_is_a_fish():
    layer = FishPreflop(_Inner(5, {0, 1, 3, 4, 5}), steal=False, open_bb=3.5)
    assert layer.act(UNOPENED) == 7 and layer.resized == 1 and layer.overrides == 0
    assert layer.last_open == dict(baseline=5, action=7, limpers=0, fish_behind=5, others=5)
    reg_behind = FishPreflop(_Inner(5, {0, 1, 3, 4}), steal=False, open_bb=3.5)
    assert reg_behind.act(UNOPENED) == 5 and reg_behind.resized == 0
    majority = FishPreflop(_Inner(5, {0, 1, 3, 4}), steal=False, open_bb=3.5, open_scope='majority')
    assert majority.act(UNOPENED) == 7
    for action in ('call', 'fold'):
        assert FishPreflop(_Inner(action, {0, 1, 3, 4, 5}), steal=False, open_bb=3.5).act(UNOPENED) == action
    # Never shrinks a bigger learned raise; capped by the stack.
    assert FishPreflop(_Inner(12, {0, 1, 3, 4, 5}), steal=False, open_bb=3.5).act(UNOPENED) == 12
    short = replace(UNOPENED, stacks=(499, 498, 5, 500, 500, 500), legal=('call', 'fold', 4, 5))
    assert FishPreflop(_Inner(5, {0, 1, 3, 4, 5}), steal=False, open_bb=3.5).act(short) == 5
    # Postflop and raised pots are untouched.
    assert FishPreflop(_Inner(13, {1}), steal=False, open_bb=3.5).act(FLOP) == 13


def test_fish_open_adds_a_big_blind_per_fish_limper():
    limped = replace(UNOPENED, position=4, player=4, pot=5, legal=('call', 'fold', 4, 7, 9),
                     history=((2, 2, 0, 'call', 3, 2), (3, 3, 0, 'fold', 5, 2)))
    layer = FishPreflop(_Inner(7, {0, 1, 2, 5}), steal=False, open_bb=3.5)
    assert layer.act(limped) == 9                                       # 3.5 + 1 BB = 9 chips
    assert FishPreflop(_Inner(7, {0, 1, 2, 5}), steal=False, open_bb=3.5, open_limper_bb=.5).act(limped) == 8
    assert FishPreflop(_Inner(7, {0, 1, 5}), steal=False, open_bb=3.5).act(limped) == 7   # limper is a regular


def test_raise_not_limp_converts_top_hand_limps_only():
    layer = FishPreflop(_Inner('call', {0, 1, 3, 4, 5}), steal=False, limp_raise=.25)
    assert layer.act(UNOPENED) == 5 and layer.limp_raises == 1          # AKs: the menu's 2.5 BB open
    assert layer.act(replace(UNOPENED, hole=('7c', '2d'))) == 'call'
    check = replace(UNOPENED, position=1, player=1, call=0, legal=('call', 4, 5),
                    history=tuple((i, i, 0, 'fold' if i != 0 else 'call', 3, 2) for i in (2, 3, 4, 5, 0)))
    assert layer.act(check) == 'call'                                   # the big blind's free check is not a limp
    fish_only = FishPreflop(_Inner('call', {0, 1, 3, 4}), steal=False, limp_raise=.25, limp_raise_scope='fish')
    assert fish_only.act(UNOPENED) == 'call'
    both = FishPreflop(_Inner('call', {0, 1, 3, 4, 5}), steal=False, limp_raise=.25, open_bb=3.5)
    assert both.act(UNOPENED) == 7


def test_preflop_layer_defaults_are_v8():
    inner = _Inner(5, {0, 1, 3, 4, 5})
    layer = FishPreflop(inner, steal=False)
    assert layer.open_bb is None and layer.limp_raise is None
    assert layer.act(UNOPENED) == 5 and layer.resized == layer.limp_raises == 0


# -- #4 flop stabs ------------------------------------------------------------------------------

def _flop_fallback(monkeypatch, fish_seats, **options):
    fb = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False, **FISH, **options)
    fb.profiles = {1: (0.05, 0.4)}
    fb.fish_seats = fish_seats
    monkeypatch.setattr(fb, '_act_v5', lambda o: 'call')                  # the v8 decision: check
    return fb


def test_fish_stab_bets_exact_fraction_when_a_fish_checks_to_us(monkeypatch):
    _equity(monkeypatch, 0.3)
    fb = _flop_fallback(monkeypatch, {1}, fish_stab=(.33, 'all'))
    assert fb.act(FLOP) == 4 and fb.last.get('fish_stab')                # round(13 * .33), off the menu
    assert _flop_fallback(monkeypatch, {1}, fish_stab=(.5, 'all')).act(FLOP) == 6
    assert _flop_fallback(monkeypatch, set(), fish_stab=(.33, 'all')).act(FLOP) == 'call'     # a regular
    assert _flop_fallback(monkeypatch, {1}).act(FLOP) == 'call'                               # option off
    first = replace(FLOP, history=FLOP.history[:-1])                      # nobody checked to us yet
    assert _flop_fallback(monkeypatch, {1}, fish_stab=(.33, 'all')).act(first) == 'call'
    paired = replace(FLOP, hole=('9c', '6d'))
    assert _flop_fallback(monkeypatch, {1}, fish_stab=(.33, 'nopair')).act(paired) == 'call'
    assert _flop_fallback(monkeypatch, {1}, fish_stab=(.33, 'nopair')).act(FLOP) == 4
    turn = replace(TURN)                                                  # flop only
    assert _flop_fallback(monkeypatch, {1}, fish_stab=(.33, 'all')).act(turn) == 'call'


# -- #3 value cap against regulars -------------------------------------------------------------

def test_fish_value_cap_blocks_overbets_while_a_regular_remains(monkeypatch):
    big = replace(TURN, legal=('call', 30, 45, 60, 90, 120, 470))
    fb = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False, **FISH,
                       fish_value_cap=.75)
    fb.profiles = {1: (0.05, 1.9)}
    sizes = []

    def fake_size(self, o, proposed, press, all_sticky, strong, usable):
        sizes.append(tuple(a for a in o.legal if isinstance(a, int)))
        return max(a for a in o.legal if isinstance(a, int))              # pretend EV loves the biggest size

    monkeypatch.setattr(equity_pricing.GuardedEquityPolicyV4, '_deep_value_size', fake_size)
    fb.fish_seats = set()                                                 # villain is a regular
    assert fb._deep_value_size(big, 470, True, True, True, .9) == 45      # 0.75 * 60 pot
    assert fb.last['fish_value_cap']['proposed'] == 470 and sizes[-1] == (30, 45)
    fb.fish_seats = {1}                                                   # villain is a fish: overbets stay
    assert fb._deep_value_size(big, 470, True, True, True, .9) == 470
    fb.fish_seats = None                                                  # no leaderboard: v8 behaviour
    assert fb._deep_value_size(big, 470, True, True, True, .9) == 470
    off = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False, **FISH)
    off.fish_seats = set()
    assert off._deep_value_size(big, 470, True, True, True, .9) == 470
    # A jam that is at most pot-sized is still allowed against a regular.
    short = replace(big, legal=('call', 30, 45, 55))
    fb.fish_seats = set()
    assert fb._deep_value_size(short, 55, True, True, True, .9) == 55


# -- client wiring ------------------------------------------------------------------------------

def _run_client_policy(tmp_path, monkeypatch, *flags):
    import sys
    from scripts import openpoker_client
    from tests.test_v6_anti_exploit import _client_checkpoint
    monkeypatch.chdir(tmp_path)
    observed = {}

    async def fake_run(args, policy, key):
        observed['policy'] = policy

    monkeypatch.setattr(openpoker_client, 'run', fake_run)
    monkeypatch.setenv('OPEN_POKER_API_KEY', 'test-placeholder')
    monkeypatch.setattr(sys, 'argv', ['client', '--checkpoint', str(_client_checkpoint(tmp_path)), *flags])
    openpoker_client.main()
    return observed['policy']


FISH_FLAGS = (*V5_FAST_FLAGS, '--avoid-strong', 'report', '--fish-exploit')


def test_client_v9_flags_wire_into_the_fish_layer_and_fallback(tmp_path, monkeypatch):
    policy = _run_client_policy(tmp_path, monkeypatch, *FISH_FLAGS, '--fish-open', '3.5', '--fish-stab', '.33',
                                '--fish-value-cap', '.75', '--raise-not-limp', '.25')
    assert isinstance(policy, FishPreflop) and policy.steal is False
    assert policy.open_bb == 3.5 and policy.open_limper_bb == 1.0 and policy.limp_raise == .25
    assert policy.fallback.fish_stab == (.33, 'all') and policy.fallback.fish_value_cap == .75
    assert policy.fallback.fish_size == FISH['fish_size']
    v8 = _run_client_policy(tmp_path, monkeypatch, *FISH_FLAGS)
    assert v8.open_bb is None and v8.limp_raise is None
    assert v8.fallback.fish_stab is None and v8.fallback.fish_value_cap is None


def test_client_v9_flags_require_fish_exploit(tmp_path, monkeypatch):
    for flag, value in (('--fish-open', '3.5'), ('--fish-stab', '.33'), ('--fish-value-cap', '.75'),
                        ('--raise-not-limp', '.25')):
        with pytest.raises(SystemExit):
            _run_client(tmp_path, monkeypatch, *V5_FAST_FLAGS, '--avoid-strong', 'report', flag, value)
    with pytest.raises(SystemExit):
        _run_client(tmp_path, monkeypatch, *FISH_FLAGS, '--fish-open', '9')


def test_client_sends_off_menu_raise_amounts():
    from tests.test_openpoker import snapshot
    from scripts.openpoker_client import decision

    class Policy:
        abstraction = 'texture-buckets-v2'

        class Rows:
            def get(self, key):
                return [1]
        sums = Rows()

        def __init__(self, amount):
            self.amount = amount

        def act(self, o):
            return self.amount

    sent = decision(snapshot(), Policy(7))                                # 3.5 BB, not on the menu
    assert sent['action'] == 'raise' and sent['amount'] == 70
    assert decision(snapshot(), Policy(5))['amount'] == 50                # a menu size is unchanged
    assert decision(snapshot(), Policy(900))['amount'] == 2000            # clamped to the server maximum
