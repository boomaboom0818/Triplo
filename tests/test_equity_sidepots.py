"""equity-guarded-v2: side-pot call pricing and opponent-aware value sizing."""
import subprocess
import types
from dataclasses import replace
from pathlib import Path

import pytest

from poker.equity_policy import EquityPolicy, GuardedEquityPolicy, estimate_equity
from poker.equity_pricing import (FALLBACKS, GuardedEquityPolicyV2, ledger, make_fallback,
                                  pot_layers, strength_table)
from poker.fast_game import FastGame
from tests.test_equity_policy import observation

BOARD = ('Kh', '8d', '2c', 'Qc', '4s')
LIMP_TO_RIVER = ['call', 'call', 'call', 'fold', 'fold', 'call'] + ['call'] * 8


def drive(stacks, script, seed=1):
    """Replay a scripted hand on the reference engine; 'max'/'min'/'pot'/'half' pick legal sizes."""
    game = FastGame(seed, 6, button=5, stacks=stacks)
    for action in script:
        o = game.observe()
        sizes = [a for a in o.legal if isinstance(a, int)]
        action = {'max': sizes[-1:], 'min': sizes[:1], 'pot': sizes[-2:-1], 'half': sizes[1:2]}.get(action, [action])[0]
        game.act(action)
    return game.observe()


def test_pot_layers_multiway_all_ins_with_different_stacks():
    # Seat 0 folded after 5; seats 1 and 2 are all in for 10 and 50; seat 3 (hero,
    # capped by its call) and seat 4 (still has chips) put in 100.
    paid = [5, 10, 50, 100, 100]
    in_hand = [False, True, True, True, True]
    capped = [False, True, True, True, False]
    layers = pot_layers(paid, in_hand, capped)
    assert layers == [(45, frozenset({1, 2, 3, 4})), (120, frozenset({2, 3, 4})), (100, frozenset({3, 4}))]
    assert sum(amount for amount, _ in layers) == sum(paid)
    # An uncapped player below a level stays eligible; equal sets merge.
    assert pot_layers([10, 40, 40], [True] * 3, [False, True, False]) == [(90, frozenset({0, 1, 2}))]
    # Dead money is a main-pot layer open to everyone still in the hand.
    assert pot_layers([0, 8, 8], [False, True, True], [True] * 3, dead=30) == [(46, frozenset({1, 2}))]
    assert pot_layers([0, 8, 4], [False, True, True], [True] * 3, dead=30) == [
        (38, frozenset({1, 2})), (4, frozenset({1}))]


def test_pot_layers_match_engine_showdown_payoffs():
    # Three all-ins of different depths against a covering caller: the hero
    # payout implied by the layers equals the reference engine's settlement.
    stacks = [400, 400, 30, 90, 400, 400]
    game = FastGame(7, 6, button=5, stacks=stacks)
    for action in ['max', 'max', 'call', 'fold', 'fold', 'fold']:
        o = game.observe()
        game.act(max(a for a in o.legal if isinstance(a, int)) if action == 'max' else action)
    assert not game.live
    from poker.fast_eval import evaluate
    score = [evaluate(game.hole[i] + tuple(game.board)) if game.active[i] else -1 for i in range(6)]
    layers = pot_layers(game.paid, game.active, [True] * 6)
    won = [0.0] * 6
    for amount, eligible in layers:
        best = max(score[i] for i in eligible)
        winners = [i for i in eligible if score[i] == best]
        for i in winners:
            won[i] += amount / len(winners)
    for i in range(6):
        assert won[i] - game.paid[i] == pytest.approx(game.payoffs[i], abs=1)


def test_ledger_reconstructs_contributions_with_hero_call():
    o = drive([400, 400, 40, 400, 400, 400], LIMP_TO_RIVER + ['call', 'call', 'max', 'call', 'fold'])
    paid, bets = ledger(o)
    assert (o.position, o.call, o.stacks[2]) == (2, 38, 38)
    assert paid == [1, 2, 2, 400, 400, 0] and sum(paid) == o.pot
    assert bets == [0, 0, 0, 398, 398, 0]
    paid[2] += o.call
    layers = pot_layers(paid, o.active, [s == 0 for s in o.stacks[:2]] + [True] + [s == 0 for s in o.stacks[3:]])
    assert layers == [(123, frozenset({2, 3, 4})), (720, frozenset({3, 4}))]
    # Inconsistent or missing ledgers fall back to v1 pricing.
    assert ledger(replace(o, pot=o.pot + 100)) is None
    assert ledger(replace(o, history=())) is None


def test_short_stack_call_priced_on_main_pot_only():
    # Hero has 38 behind facing two 398 all-ins: v1 sees 38 into 843 (4.5%)
    # and calls; only the 123-chip main pot is winnable, so v2 needs ~31%.
    o = drive([400, 400, 40, 400, 400, 400], LIMP_TO_RIVER + ['call', 'call', 'max', 'call', 'fold'])
    o = replace(o, hole=('8s', '7s'), board=BOARD)
    v1, v2 = GuardedEquityPolicy(1024), GuardedEquityPolicyV2(1024)
    assert v1.act(o) == 'call'
    assert v2.act(o) == 'fold'
    pots = v2.last['pots']
    assert [p['amount'] for p in pots] == [123, 720] and pots[1]['hero_share'] is None
    assert v2.last['call_ev'] < 0
    # A strong main-pot favourite still calls.
    assert GuardedEquityPolicyV2(1024).act(replace(o, hole=('Ks', '5d'))) == 'call'


def test_side_pot_equity_against_fewer_players_turns_fold_into_call():
    # A short stack shoves 10 into a 9 pot, a deep player raises to 20. The main
    # pot is three-way, but the 20-chip side pot is heads up, where the hero's
    # share is far higher than the single three-way equity v1 applies to both.
    o = drive([400, 400, 400, 12, 400, 400], LIMP_TO_RIVER + ['call', 'call', 'max', 'min', 'fold'])
    o = replace(o, hole=('Ks', '6d'), board=BOARD)
    v1, v2 = GuardedEquityPolicy(1024), GuardedEquityPolicyV2(1024)
    assert v1.act(o) == 'fold'
    assert v1.last['equity'] < v1.last['pot_odds'] + .025
    assert v2.act(o) == 'call'
    main, side = v2.last['pots']
    assert main['eligible'] == [2, 3, 4] and side['eligible'] == [2, 4]
    assert side['hero_share'] > main['hero_share'] + .15
    assert v2.last['call_ev'] > v2.last['required_ev']


def test_single_pot_keeps_v1_pricing():
    # No all-in players: one pot contested by everyone, v2 defers to v1 verbatim.
    o = drive([400] * 6, LIMP_TO_RIVER + ['call', 'call', 'half', 'call', 'fold'])
    for hole in (('Ks', '6d'), ('8s', '7s'), ('Qs', 'Jd'), ('As', '2d')):
        case = replace(o, hole=hole, board=BOARD)
        v1, v2 = GuardedEquityPolicy(256), GuardedEquityPolicyV2(256)
        assert v1.act(case) == v2.act(case)
        assert 'pricing' not in (v2.last or {})
    # Without a ledger (unverified live history) v2 also keeps v1's decision.
    for call in (5, 20, 60):
        bare = observation(('Ah', '9h'), ('Kh', '2h', '7c'), call=call)
        assert GuardedEquityPolicy(256).act(bare) == GuardedEquityPolicyV2(256).act(bare)


def value_spot():
    # Heads-up river after a raised pot and a called flop bet; checked to the hero.
    o = drive([400] * 6, ['pot', 'fold', 'fold', 'fold', 'fold', 'call', 'call', 'pot', 'call', 'call', 'call', 'call'])
    assert (o.street, o.call, o.pot, o.legal) == (3, 0, 33, ('call', 2, 16, 33, 384))
    return o


def test_value_size_smaller_against_tight_callers_larger_against_sticky():
    o = value_spot()
    two_pair = replace(o, hole=('Kd', '8s'), board=BOARD)
    top_pair = replace(o, hole=('Ks', 'Jd'), board=BOARD)
    assert GuardedEquityPolicy(1024).act(two_pair) == 33
    assert GuardedEquityPolicy(1024).act(top_pair) == 16
    tight = GuardedEquityPolicyV2(1024, profiles={None: 'tight'})
    sticky = GuardedEquityPolicyV2(1024, profiles={None: 'sticky'})
    assert tight.act(two_pair) == 16
    assert tight.last['size_ev']['16'] > tight.last['size_ev']['33']
    assert sticky.act(two_pair) == 33
    assert sticky.act(top_pair) == 33
    assert sticky.last['size_ev']['33'] > sticky.last['size_ev']['16'] > sticky.last['size_ev']['2']
    assert tight.act(top_pair) <= 16
    # Per-position profiles: the river caller (position 1) decides.
    assert GuardedEquityPolicyV2(1024, profiles={1: 'sticky'}).act(top_pair) == 33
    # Fold equity alone is worth the pot: every size is at least break-even.
    assert min(sticky.last['size_ev'].values()) > 0


def test_value_size_without_calling_information_keeps_v1_size():
    o = value_spot()
    for hole in (('Ks', 'Jd'), ('As', 'Kd'), ('Kd', '8s'), ('Ks', 'Kd'), ('Ks', '7s')):
        case = replace(o, hole=hole, board=BOARD)
        assert GuardedEquityPolicy(512).act(case) == GuardedEquityPolicyV2(512).act(case)
    # A profile for a different seat is not information about the river caller.
    case = replace(o, hole=('Ks', 'Jd'), board=BOARD)
    assert GuardedEquityPolicyV2(512, profiles={4: 'sticky'}).act(case) == 16
    # Never a size above v1's own bound, even against sticky callers.
    sticky = GuardedEquityPolicyV2(512, profiles={None: 'sticky'})
    assert sticky.act(replace(o, hole=('Ks', 'Kd'), board=BOARD)) < 384


def test_strength_table_percentiles():
    board = tuple(sorted(range(0, 20, 4)[:5]))  # 2c 3c 4c 5c 6c: a straight flush on board
    table = strength_table(board)
    assert len(table) == 47 * 46 // 2
    assert all(0 < v < 1 for v in table.values())
    from poker.fast_eval import CARD_ID
    river = tuple(sorted(CARD_ID[c] for c in BOARD))
    ranks = strength_table(river)
    assert ranks[tuple(sorted((CARD_ID['Ks'], CARD_ID['Kd'])))] > .99
    assert ranks[tuple(sorted((CARD_ID['3s'], CARD_ID['5d'])))] < .2


def test_fallback_registry_and_v1_untouched():
    assert FALLBACKS == ('equity-v1', 'equity-guarded-v1', 'equity-guarded-v2')
    assert type(make_fallback('equity-guarded-v1')) is GuardedEquityPolicy
    assert type(make_fallback('equity-v1')) is EquityPolicy
    v2 = make_fallback('equity-guarded-v2', stack_press=True)
    assert isinstance(v2, GuardedEquityPolicyV2) and v2.stack_press
    with pytest.raises(ValueError):
        make_fallback('equity-guarded-v3')
    assert not hasattr(GuardedEquityPolicy(), 'profiles')


def _committed_module():
    try:
        source = subprocess.run(['git', 'show', 'HEAD:poker/equity_policy.py'], check=True, capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1]).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git history unavailable')
    module = types.ModuleType('poker._committed_equity_policy')
    module.__package__ = 'poker'
    exec(compile(source, 'HEAD:poker/equity_policy.py', 'exec'), module.__dict__)
    return module


def test_v1_equity_and_decisions_match_committed_implementation():
    old = _committed_module()
    from poker.live_curriculum import root_game
    from poker.rule_opponents import RulePolicy
    checked = 0
    for seed in range(40):
        game = root_game(seed, 6, [100, 50, 20, 100, 200, 75])
        players = [RulePolicy(('station-v1', 'aggressive-v1', 'tight-v1')[i % 3]) for i in range(6)]
        while game.live:
            o = game.observe()
            if o.street:
                assert estimate_equity(o, 64) == old.estimate_equity(o, 64)
                assert GuardedEquityPolicy(64).act(o) == old.GuardedEquityPolicy(64).act(o)
                assert EquityPolicy(64).act(o) == old.EquityPolicy(64).act(o)
                checked += 1
            game.act(players[o.position].act(o))
    assert checked > 50
