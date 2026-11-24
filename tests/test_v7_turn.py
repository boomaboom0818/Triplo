"""Tests for the Triplo v7 opt-in turn switches: turn_defend and turn_probe."""
from dataclasses import replace

import pytest

from poker import equity_policy, equity_pricing
from poker.equity_pricing import GuardedEquityPolicyV4, VERSION_V5, make_fallback
from poker.game import Observation

from tests.test_v6_anti_exploit import V5_FAST_FLAGS, V5_FLAGS, _run_client

# Hero (position 5) opened, villain (position 1) called; both checked the flop.
CHECKED_FLOP = ((5, 5, 0, 6, 3, 2), (1, 1, 0, 'call', 9, 4),
                (1, 1, 1, 'call', 20, 0), (5, 5, 1, 'call', 20, 0))
PROBE = Observation(
    player=5, position=5, hole=('9h', '8c'), board=('Kh', '9d', '2c', '5s'),
    stacks=(500, 478, 500, 500, 500, 494),
    active=(False, True, False, False, False, True),
    pot=36, call=16, street=2, legal=('call', 'fold', 32, 48, 494),
    history=CHECKED_FLOP + ((1, 1, 2, 16, 20, 0),),
)
CHECKED_TO_HERO = replace(PROBE, pot=20, call=0, legal=('call', 4, 10, 13, 20, 40, 494),
                          history=CHECKED_FLOP + ((1, 1, 2, 'call', 20, 0),))


def _equity(monkeypatch, value):
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (value, 0.02))
    monkeypatch.setattr(equity_policy, 'estimate_equity', lambda *a, **kw: (value, 0.02))


def test_turn_defend_calls_probes_with_pairs_but_not_barrels_air_or_big_bets(monkeypatch):
    _equity(monkeypatch, 0.36)
    base = GuardedEquityPolicyV4(64, deep_value=True)
    defend = GuardedEquityPolicyV4(64, deep_value=True, turn_defend=True)
    assert base.act(PROBE) == 'fold'
    assert defend.act(PROBE) == 'call'
    assert defend.last.get('turn_defend') is True

    barrel = replace(PROBE, pot=56, call=20, history=(
        (5, 5, 0, 6, 3, 2), (1, 1, 0, 'call', 9, 4), (1, 1, 1, 8, 20, 0),
        (5, 5, 1, 'call', 28, 8), (1, 1, 2, 20, 36, 0)))
    assert defend.act(barrel) == 'fold'  # the bettor also bet the flop
    assert defend.act(replace(PROBE, hole=('7c', '3d'))) == 'fold'  # no pair, no draw
    assert defend.act(replace(PROBE, pot=40, call=20)) == 'fold'  # over 0.85 pot
    river = replace(PROBE, street=3, board=('Kh', '9d', '2c', '5s', '4d'),
                    history=PROBE.history[:-1] + ((1, 1, 2, 'call', 20, 0), (5, 5, 2, 'call', 20, 0),
                                                  (1, 1, 3, 16, 20, 0)))
    assert defend.act(river) == base.act(river)  # river unchanged


def test_turn_probe_bets_draws_after_checked_flop_and_respects_gates(monkeypatch):
    _equity(monkeypatch, 0.25)
    draw = replace(CHECKED_TO_HERO, hole=('Qc', 'Td'))  # gutshot to K-Q-J-T-9
    assert GuardedEquityPolicyV4(64, deep_value=True).act(draw) == 'call'
    probe = GuardedEquityPolicyV4(64, deep_value=True, turn_probe=True)
    assert probe.act(draw) == 13  # nearest legal size to 2/3 pot
    assert probe.last.get('turn_probe') is True

    assert probe.act(replace(draw, hole=('9c', '3d'))) == 'call'  # pairs keep showdown value
    flop_bet = replace(draw, history=((5, 5, 0, 6, 3, 2), (1, 1, 0, 'call', 9, 4), (1, 1, 1, 8, 20, 0),
                                      (5, 5, 1, 'call', 28, 8), (1, 1, 2, 'call', 36, 0)))
    assert probe.act(flop_bet) == 'call'  # not a checked flop
    sticky = GuardedEquityPolicyV4(64, deep_value=True, turn_probe=True, profiles={1: (0.08, 0.03)})
    assert sticky.act(draw) == 'call'
    multiway = replace(draw, active=(False, True, True, False, False, True),
                       stacks=(500, 478, 494, 500, 500, 494))
    assert probe.act(multiway) == 'call'


def test_turn_probe_air_is_mixed_and_draws_always_bet(monkeypatch):
    _equity(monkeypatch, 0.15)
    probe = GuardedEquityPolicyV4(64, deep_value=True, turn_probe=True)
    air = [('7c', '3d'), ('7d', '3h'), ('8c', '3s'), ('8d', '4h'), ('Jc', '3h'), ('Js', '4c'),
           ('Tc', '3h'), ('Td', '4s'), ('6c', '3h'), ('7h', '4c'), ('8h', '3c'), ('Jd', '4d')]
    bets = sum(isinstance(probe.act(replace(CHECKED_TO_HERO, hole=h)), int) for h in air)
    assert 0 < bets < len(air)


def test_v6_tight_path_gets_the_same_turn_switches(monkeypatch):
    _equity(monkeypatch, 0.36)
    base = make_fallback(VERSION_V5, solve_river=False)
    fb = make_fallback(VERSION_V5, solve_river=False, turn_defend=True, turn_probe=True)
    assert not fb.deep_value
    assert base.act(PROBE) == 'fold'
    assert fb.act(PROBE) == 'call'
    _equity(monkeypatch, 0.25)
    draw = replace(CHECKED_TO_HERO, hole=('Qc', 'Td'))
    assert base.act(draw) == 'call'
    assert fb.act(draw) == 13


def test_make_fallback_and_client_wire_v7_switches(tmp_path, monkeypatch):
    default_fb = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False)
    assert not default_fb.turn_defend and not default_fb.turn_probe
    fb = _run_client(tmp_path, monkeypatch, *V5_FAST_FLAGS, '--defend-3bet', '--flop-peel',
                     '--turn-defend', '--turn-probe')
    assert fb.turn_defend is True and fb.turn_probe is True and fb.flop_peel is True
    fb = _run_client(tmp_path, monkeypatch, *V5_FLAGS, '--turn-defend')
    assert fb.turn_defend is True and not fb.turn_probe and not fb.deep_value
    with pytest.raises(SystemExit):
        _run_client(tmp_path, monkeypatch, '--fallback', 'equity-guarded-v3', '--turn-defend')
    with pytest.raises(SystemExit):
        _run_client(tmp_path, monkeypatch, '--fallback', 'equity-guarded-v3', '--turn-probe')


def test_v7_crossplay_builds_match_live_command_lines(tmp_path):
    from poker import history_cfr
    from scripts.triplo_v7_crossplay import BOTS, build
    from tests.test_v6_anti_exploit import _client_checkpoint
    saved = history_cfr.load(_client_checkpoint(tmp_path))
    built = {tag: build(tag, saved) for tag in BOTS}
    for tag, policy in built.items():
        fb = policy.fallback
        assert fb.defend_3bet and fb.flop_peel and not fb.turn_probe
        assert fb.turn_defend is tag.startswith('v7')
        fast = tag.endswith('fast')
        assert fb.deep_value is fast and fb.stack_press is fast
        assert policy.learned_postflop is not fast
