"""Tests for Triplo v5-fast upgrades: deep_value, learned_postflop=False, and weak-seat avoidance."""
import time

from poker.equity_pricing import GuardedEquityPolicyV4, VERSION_V5, make_fallback
from poker.fast_game import FastGame
from poker.game import Observation
from poker.history_cfr import COMPACT_VERSION, HistoryPolicy
from poker.opponent_strength import OpponentField
from poker import stack_press
from scripts import openpoker_client as client
from scripts.openpoker_strong import StrongAvoider


def _row(name, balance, status='playing', **extra):
    return dict(dict(bot_name=name, hands_played=1000, chip_balance=balance,
                     chips_at_table=2000, rebuys=0, status=status), **extra)


def test_learned_postflop_false_keeps_preflop_and_routes_postflop_to_fallback():
    g = FastGame(42, players=6, stacks=[200] * 6, lean=True)
    o_pre = g.observe()
    saved = dict(schema=1, algorithm='history-parallel-es-v1', abstraction=COMPACT_VERSION,
                 menu='lean-menu-v1', players=6, stack_bb=100, sums={}, visits={})
    # Populate a learned row for the preflop UTG state
    p_tmp = HistoryPolicy(saved, min_visits=1)
    pre_key = p_tmp.key(o_pre)
    saved['sums'][pre_key] = [0.0, 1.0] + [0.0] * (len(o_pre.legal) - 2)
    saved['visits'][pre_key] = 100

    # Also build a postflop observation and put a learned row for it
    o_post = Observation(0, 0, ('As', 'Ah'), ('Ad', '7c', '2s'), (200,) * 6,
                         (True, True, False, False, False, False), 10, 0, 1,
                         ('call', 5, 10), ((2, 0, 0, 5, 3, 2), (3, 1, 0, 'call', 8, 3)))
    post_key = p_tmp.key(o_post)
    saved['sums'][post_key] = [1.0, 0.0, 0.0]  # learned would check top set
    saved['visits'][post_key] = 100

    policy = HistoryPolicy(saved, min_visits=64, learned_postflop=False)
    policy.fallback = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=False)
    act_pre = policy.act(o_pre, history_complete=True)
    assert policy.covered == 1 and act_pre == 'fold'
    act_post = policy.act(o_post, history_complete=True)
    assert policy.covered == 1  # postflop bypassed learned row
    assert isinstance(act_post, int)  # fallback bet top set


def test_deep_sizes_offers_richer_menu_at_all_depths_and_off_preflop():
    assert stack_press.deep_sizes(0, 2, 3, 4, 500, 0, 500, [(2, 500)], ('call', 'fold', 4, 5)) == ()
    # Even at 60 BB effective stack (below Stack Press's 100 BB threshold), deep_sizes offers 0.75x/1.5x/2x/shove
    extra = stack_press.deep_sizes(1, 0, 20, 2, 120, 0, 120, [(0, 120)], ('call', 10, 20))
    assert 15 in extra and 30 in extra and 40 in extra and 120 in extra
    # Respects opponent cover cap
    capped = stack_press.deep_sizes(1, 0, 20, 2, 500, 0, 500, [(0, 60)], ('call', 10, 20))
    assert max(capped) <= 60


def test_deep_value_raises_top_pair_vs_sticky_bettor_and_uses_overbet_sizes():
    fb = GuardedEquityPolicyV4(stack_press=True, deep_value=True, profiles={1: 'sticky'})
    # Hero has AA overpair on low dry flop facing a small bet from sticky opponent
    # Ledger: SB(0) raises to 5, BB(1) calls 3 -> pot 10; flop BB(1) bets 10 -> pot 20
    o = Observation(0, 0, ('As', 'Ah'), ('7d', '4c', '2s'), (495, 485, 500, 500, 500, 500),
                    (True, True, False, False, False, False), 20, 10, 1,
                    ('call', 'fold', 25, 40, 55, 70, 250, 485),
                    ((2, 0, 0, 5, 3, 2), (3, 1, 0, 'call', 7, 3), (3, 1, 1, 10, 10, 0)))
    act = fb.act(o)
    assert isinstance(act, int) and act >= 25
    assert fb.last is not None and fb.last.get('deep_value') is True


def test_strong_check_min_weak_and_max_strong():
    f = OpponentField([
        _row('hero', 90000, account_agent_id='ours'),
        _row('shark1', 90000),   # +290 BB/100
        _row('shark2', 90000),   # +290 BB/100
        _row('reg1', 18000),     # +50 BB/100
        _row('reg2', 18000),
        _row('reg3', 18000),
        _row('preset1', 18000, bot_kind='no_code'),
        _row('preset2', 18000, bot_kind='no_code'),
    ])
    # Two sharks offset by two presets: default stays, max_strong=1 leaves
    assert not f.strong_check('hero', ['shark1', 'shark2', 'preset1', 'preset2']).leave
    res_max = f.strong_check('hero', ['shark1', 'shark2', 'preset1', 'preset2'], max_strong=1)
    assert res_max.leave and res_max.reason == 'too many strong opponents'

    # Full table (4+ opponents) with only 1 weak seat: min_weak=2 leaves
    res_min = f.strong_check('hero', ['reg1', 'reg2', 'reg3', 'preset1'], min_weak=2)
    assert res_min.leave and res_min.reason == 'too few weak seats'
    # Half-empty table (only 3 opponents) does not leave early for min_weak
    assert not f.strong_check('hero', ['reg1', 'reg2', 'preset1'], min_weak=2).leave
    # Full table with 2 weak seats and 1 shark stays
    assert not f.strong_check('hero', ['shark1', 'reg1', 'preset1', 'preset2'], min_weak=2, max_strong=1).leave

    # StrongAvoider wires min_weak through check() even when strong=()
    av = StrongAvoider('leave', 220.0, min_weak=2, max_strong=1)
    av.hero, av.field, av.field_time = 'hero', f, time.monotonic()
    av.seated('T')
    av.hero_seat, av.hero_stack = 0, 5000
    av.live = {0: 'hero', 1: 'reg1', 2: 'reg2', 3: 'reg3', 4: 'preset1'}
    chk = av.check('hand_result', 5000)
    assert chk is not None and chk['leave'] is True and chk['reason'] == 'too few weak seats'


def test_openpoker_client_decision_with_deep_value_wires_extra_sizes():
    saved = dict(schema=1, algorithm='history-parallel-es-v1', abstraction=COMPACT_VERSION,
                 menu='lean-menu-v1', players=6, stack_bb=100, sums={}, visits={})
    policy = HistoryPolicy(saved, min_visits=64, learned_postflop=False)
    policy.fallback = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=True)
    snap = {
        'table_id': 'T', 'hand_id': 'H1', 'dealer_seat': 5, 'actor_seat': 0,
        'big_blind': 20, 'pot': 200, 'board': ['As', 'Ad', '7c'],
        'hero': {'seat': 0, 'turn_token': 'tok1', 'hole_cards': ['Ac', 'Kd'],
                 'valid_actions': [{'action': 'check'}, {'action': 'raise', 'min': 20, 'max': 4800}]},
        'seats': [
            {'seat': 0, 'name': 'Triplo', 'stack': 4800, 'bet': 0, 'in_hand': True, 'folded': False},
            {'seat': 1, 'name': 'p1', 'stack': 4800, 'bet': 0, 'in_hand': True, 'folded': False},
            {'seat': 2, 'name': 'p2', 'stack': 4800, 'bet': 0, 'in_hand': True, 'folded': True},
        ],
    }
    payload = client.decision(snap, policy, public_history=((0, 0, 0, 5, 3, 2), (1, 1, 0, 'call', 8, 3)),
                              history_complete=True)
    assert payload['action'] == 'raise'
    assert 20 <= payload['amount'] <= 4800
    assert client.decision.last['decision_source'] == 'equity_fallback'
    assert len(client.decision.last.get('stack_press_sizes', ())) >= 3
