"""Tests for ``equity-guarded-v4`` (fair-share multiway postflop scaling)."""
import json
import sys
from dataclasses import replace

import pytest

from poker import equity_pricing, history_cfr
from poker.equity_pricing import (
    FALLBACKS,
    VERSION_V4,
    GuardedEquityPolicyV3,
    GuardedEquityPolicyV4,
    make_fallback,
    multiway_thresholds,
)
from poker.game import Observation
from poker.live_curriculum import root_game
from poker.opponent_stats import OpponentStats
from poker.rule_opponents import RulePolicy
from scripts import openpoker_client
from tests.test_late_position import FOLDS, snapshot


def test_v4_registry_and_multiway_thresholds():
    assert VERSION_V4 in FALLBACKS
    v4 = make_fallback(VERSION_V4, stack_press=True)
    assert isinstance(v4, GuardedEquityPolicyV4)
    assert isinstance(v4, GuardedEquityPolicyV3)
    assert v4.RANGE_MODEL == 'v3'
    assert v4.stack_press is True

    # Fair-share thresholds strictly decrease as table size grows from 3 to 6 players.
    prev = (1.0, 1.0, 1.0)
    for opponents in (2, 3, 4, 5):
        base, val, big = multiway_thresholds(opponents)
        assert base == pytest.approx(1.0 / (opponents + 1))
        assert base < prev[0] and val < prev[1] and big < prev[2]
        assert base < val < big
        prev = (base, val, big)


def test_v4_multiway_unlocks_half_pot_and_preserves_safeguards(monkeypatch):
    # 5-way flop (4 opponents): fair-share base_thresh = 0.20; v3 requires 0.40.
    o = Observation(
        player=0,
        position=0,
        hole=('Ah', 'Qd'),
        board=('As', '9c', '4d'),
        stacks=(200,) * 6,
        active=(True, True, True, True, True, False),
        pot=20,
        call=0,
        street=1,
        legal=('call', 10, 20, 40),
        history=(),
    )
    # Usable = 0.36 * 0.90 - 0.02 = 0.304 -> in [0.20, 0.40): v3 checks, v4 bets half-pot (raises[0] = 10).
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (0.36, 0.02))
    monkeypatch.setattr('poker.equity_policy.estimate_equity', lambda *a, **kw: (0.36, 0.02))
    v3, v4 = GuardedEquityPolicyV3(64), GuardedEquityPolicyV4(64)
    assert v3.act(o) == 'call'
    assert v4.act(o) == 10

    # Above 0.40 usable (0.50 * 0.90 - 0.02 = 0.43): v4 keeps v3's full baseline size (raises[1] = 20).
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (0.50, 0.02))
    monkeypatch.setattr('poker.equity_policy.estimate_equity', lambda *a, **kw: (0.50, 0.02))
    assert v3.act(o) == 20
    assert v4.act(o) == 20

    # Safeguards stay intact: board_plays on the river still never bets.
    board_plays = replace(o, hole=('2h', '3d'), board=('As', 'Kd', 'Qc', 'Js', 'Th'), street=3)
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (0.36, 0.02))
    assert v4.act(board_plays) == 'call'

    # River >100 BB one-pair fold safeguard still triggers in multiway pots.
    huge_river = replace(
        o,
        board=('As', '9c', '4d', '7s', '2h'),
        street=3,
        call=250,
        pot=300,
        legal=('call', 'fold'),
    )
    monkeypatch.setattr(equity_pricing, 'estimate_equity', lambda *a, **kw: (0.60, 0.02))
    assert v4.act(huge_river) == 'fold'


def test_v4_matches_v3_verbatim_preflop_and_heads_up():
    v3, v4 = GuardedEquityPolicyV3(64), GuardedEquityPolicyV4(64)
    checked_pre = checked_hu = 0
    for seed in range(60):
        game = root_game(seed, 6, [100] * 6)
        bots = [RulePolicy(('tight-v1', 'aggressive-v1', 'station-v1')[i % 3]) for i in range(6)]
        while game.live:
            o = game.observe()
            if o.street == 0:
                assert v4.act(o) == v3.act(o)
                checked_pre += 1
            elif sum(o.active) - 1 == 1:
                assert v4.act(o) == v3.act(o)
                checked_hu += 1
            game.act(bots[o.position].act(o))
    assert checked_pre > 100 and checked_hu > 20


def test_openpoker_client_supports_v4_with_opponent_stats_and_steals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps(dict(
        schema=1,
        algorithm=history_cfr.ALGORITHM,
        abstraction=history_cfr.VERSION,
        menu=history_cfr.LEAN_MENU,
        players=6,
        stack_bb=100,
        iterations=0,
        sums={},
        visits={},
    )))
    stats_path = tmp_path / 'opponent-stats.json'
    OpponentStats().save(stats_path)
    steals_path = tmp_path / 'runs/bot/late-position-folds.json'
    steals_path.parent.mkdir(parents=True)
    steals_path.write_text(json.dumps(dict(
        schema=1,
        hero='Triplo',
        lineups=[dict(position=5, behind=['SB', 'BB'], train=[30, 30], holdout=[15, 15])],
    )))

    observed = {}

    async def fake_run(args, policy, key):
        observed['fallback'] = type(policy.fallback)
        observed['has_stats'] = getattr(policy.fallback, 'opponent_stats', None) is not None
        observed['has_steals'] = getattr(policy.fallback, 'late_position_evidence', None) is not None
        observed['action'] = openpoker_client.decision(snapshot(), policy, FOLDS, True)['action']

    monkeypatch.setattr(openpoker_client, 'run', fake_run)
    monkeypatch.setenv('OPEN_POKER_API_KEY', 'test-placeholder')
    monkeypatch.setattr(sys, 'argv', [
        'client',
        '--checkpoint', str(checkpoint),
        '--fallback', VERSION_V4,
        '--opponent-stats', str(stats_path),
    ])
    openpoker_client.main()
    assert observed == dict(
        fallback=GuardedEquityPolicyV4,
        has_stats=True,
        has_steals=True,
        action='raise',
    )
