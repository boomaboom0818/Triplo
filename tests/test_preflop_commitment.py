from dataclasses import replace

from poker.fast_game import FastGame
from poker.preflop_commitment import committed_preflop, guard_preflop


def test_small_repeated_calls_cannot_evade_total_commitment_limit():
    o = replace(FastGame(1, lean=True).observe(), position=0, hole=('9d', 'Td'),
                call=4, pot=26, legal=('call', 'fold', 18),
                history=((0, 0, 0, 'call', 11, 3),))
    assert committed_preflop(o) == 4  # small blind plus first call
    assert guard_preflop(o, 'call') == 'fold'  # four more would exceed 3 BB
    assert guard_preflop(replace(o, call=2), 'call') == 'call'
    assert guard_preflop(replace(o, hole=('As', 'Ah')), 'call') == 'call'


def test_raise_to_amounts_and_checks_preserve_accounting():
    o = replace(FastGame(1, lean=True).observe(), position=1, hole=('As', 'Qd'),
                call=4, pot=40, legal=('call', 'fold', 30),
                history=((1, 1, 0, 10, 10, 3), (1, 1, 0, 'call', 30, 6)))
    assert committed_preflop(o) == 16
    assert guard_preflop(o, 30) == 'call'  # call fits 10 BB, raise does not
    assert guard_preflop(replace(o, call=5), 30) == 'fold'
    assert guard_preflop(replace(o, call=0), 30) == 'call'
    assert guard_preflop(o, 'fold') == 'fold'


def test_weak_learned_continuations_are_not_capped():
    from poker.history_cfr import HistoryPolicy, observation_key
    o = replace(FastGame(1, lean=True).observe(), position=2, hole=('3h', '4h'),
                call=3, pot=8, legal=('call', 'fold', 12), history=())
    key = observation_key(o)
    learned = HistoryPolicy(dict(sums={key: [100., 0., 0.]}, visits={key: 80}, stack_bb=100),
                            preflop_commitment=True)
    assert learned.act(o, history_complete=True) == 'call'
    assert learned.covered == 1 and learned.commitment_overrides == 0
    fallback = HistoryPolicy(dict(sums={}, visits={}, stack_bb=100), learned=False,
                             preflop_commitment=True)
    assert fallback.act(o, history_complete=True) == 'fold'
    assert fallback.commitment_overrides == 0


def test_policy_applies_the_limit_only_to_a_verified_ledger():
    from poker.history_cfr import HistoryPolicy, observation_key
    saved = dict(sums={}, visits={}, stack_bb=100)
    o = replace(FastGame(1, lean=True).observe(), position=0, hole=('9d', 'Td'),
                call=4, pot=26, legal=('call', 'fold', 18),
                history=((0, 0, 0, 'call', 11, 3),))
    limited = HistoryPolicy(saved, learned=False, preflop_commitment=True)
    assert limited.act(o, history_complete=True) == 'fold'
    assert limited.commitment_overrides == 1 and limited.covered == 0
    assert HistoryPolicy(saved, learned=False).act(o, history_complete=True) == 'call'
    untouched = HistoryPolicy(saved, learned=False, preflop_commitment=True)
    assert untouched.act(o, history_complete=False) == 'call'
    assert untouched.commitment_overrides == 0
    key = observation_key(o)
    learned = HistoryPolicy(dict(sums={key: [100., 0., 0.]}, visits={key: 100}, stack_bb=100),
                            preflop_commitment=True)
    assert learned.act(o, history_complete=True) == 'fold'
    assert learned.covered == 0 and learned.commitment_overrides == 1


def test_repeated_min_reraises_stop_at_the_chart_cap():
    from poker.fast_eval import CARD_ID
    from poker.history_cfr import HistoryPolicy
    game = FastGame(1, lean=True)
    hero = 2
    game.hole[hero] = [CARD_ID['9d'], CARD_ID['Td']]
    policy = HistoryPolicy(dict(sums={}, visits={}, stack_bb=100), learned=False,
                           preflop_commitment=True)
    raises = 0
    while game.live and game.street == 0 and raises < 12:
        o = game.observe()
        if o.position == hero:
            game.act(policy.act(o, history_complete=True))
            continue
        sizes = [a for a in o.legal if isinstance(a, int)]
        if game.paid[hero] >= 6 or not sizes:
            game.act('call' if 'call' in o.legal else 'fold')
        else:
            game.act(sizes[0])
            raises += 1
    assert not game.active[hero] or game.street
    assert game.paid[hero] <= 6


def test_accounting_matches_engine_on_live_preflop_paths():
    from random import Random
    rng = Random(35)
    for seed in range(40):
        game = FastGame(seed, lean=True)
        while game.live and game.street == 0:
            o = game.observe()
            assert committed_preflop(o) == game.paid[o.position]
            game.act(rng.choice(o.legal))
