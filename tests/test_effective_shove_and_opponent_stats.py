import json

from poker.fast_game import FastGame
from poker.opponent_stats import OpponentStats
from scripts.openpoker_client import decision, opponent_profiles


# -- effective all-in on the lean menu -------------------------------------

def deep_game(sb_behind, bb_behind, hero_behind=2998):
    """3-handed preflop; internal index 2 (first to act) holds ``hero_behind``."""
    game = FastGame(1, 3, stacks=[50, 50, 50], lean=True)
    assert game.actors[0] == 2 and game.bets == [1, 2, 0]
    game.stacks = [sb_behind, bb_behind, hero_behind]
    return game


def test_deep_actor_gets_effective_all_in():
    # Hero is 3000 deep; the blinds can match only 8 in total.
    game = deep_game(7, 6)
    _, _, legal = game.menu()
    assert 8 in legal and 2998 not in legal


def test_effective_all_in_still_needs_two_pot_bound():
    game = deep_game(40, 40)                    # cover 42 > 2 + 2*3
    _, _, legal = game.menu()
    assert 42 not in legal and 2998 not in legal


def test_equal_stacks_menu_unchanged():
    for stacks in ([8, 8, 8], [200] * 6):
        game = FastGame(3, len(stacks), stacks=stacks, lean=True)
        _, _, legal = game.menu()
        i = game.actors[0]
        hi = game.stacks[i] + game.bets[i]
        top, total = max(game.bets), sum(game.paid)
        assert (hi in legal) == (hi <= top + 2 * total)


def test_cover_is_deepest_live_opponent():
    game = deep_game(7, 18)
    assert game.cover(2) == 20
    game.active[1] = False
    assert game.cover(2) == 8


def snapshot(stacks, bets, pot, hero_seat=2, board=('Ah', '7d', '2c'), call=0, names=None):
    seats = [dict(seat=i, in_hand=True, folded=False, bet=bets[i], stack=stacks[i],
                  name=(names or {}).get(i, 'p%d' % i)) for i in range(len(stacks))]
    actions = [dict(action='check')] if not call else [dict(action='call', amount=call), dict(action='fold')]
    hero_total = stacks[hero_seat] + bets[hero_seat]
    actions.append(dict(action='raise', min=max(20, 2 * max(bets)), max=hero_total))
    return dict(hand_id='h', actor_seat=hero_seat, dealer_seat=len(stacks) - 1, big_blind=20, pot=pot,
                board=list(board), seats=seats,
                hero=dict(seat=hero_seat, turn_token='t', hole_cards=['As', 'Ad'], valid_actions=actions))


class Lean:
    """Minimal lean-menu policy that records its observation and checks."""
    from poker.lean_cfr import LEAN_MENU as menu
    sums = {}
    abstraction = 'test'

    def key(self, o):
        self.seen = o
        return 'k'

    def act(self, o):
        return 'call'


def test_live_adapter_offers_effective_all_in():
    # Flop, pot 400 chips; hero 5000 deep vs opponents with 300 behind.
    policy = Lean()
    policy.sums = {'k': [1.0]}
    message = decision(snapshot([300, 300, 5000], [0, 0, 0], 400), policy)
    legal = policy.seen.legal
    assert 30 in legal and 500 not in legal          # 300 chips = 30 units, not 5000 = 500
    assert message['action'] == 'check'
    # The wire amount for the effective shove is the opponent's total, in chips.
    assert decision.last['action'] == 'check'
    policy.act = lambda o: 30
    message = decision(snapshot([300, 300, 5000], [0, 0, 0], 400), policy)
    assert message == dict(message, action='raise', amount=300)


# -- per-opponent calling stats --------------------------------------------

def action(seq, name, label, kind, to_call, pot_before, hand='h1', reason=None):
    return dict(type='player_action', table_id='t', table_seq=seq, hand_id=hand, name=name, street=label,
                action=kind, to_call_before=to_call, pot_before=pot_before, reason=reason)


def test_street_comes_from_previous_label_and_duplicates_drop():
    s = OpponentStats(min_faced=1)
    s.ingest(dict(type='hand_start', table_id='t', table_seq=1, hand_id='h1'))
    # Preflop call that closes the round is labelled 'flop': must not count.
    s.ingest(action(2, 'a', 'flop', 'call', 20, 30))
    # Flop fold facing a bet (labelled after the fact as 'showdown').
    s.ingest(action(3, 'a', 'showdown', 'fold', 40, 80))
    s.ingest(action(3, 'a', 'showdown', 'fold', 40, 80))       # replayed duplicate
    assert s.faced('a') == 1
    assert sum(r[1] for r in s.players['a']) == 1


def test_timeouts_checks_and_unknown_hands_ignored():
    s = OpponentStats(min_faced=1)
    s.ingest(action(5, 'a', 'flop', 'fold', 40, 80, hand='unseen'))   # no hand_start seen
    s.ingest(dict(type='hand_start', table_id='t', table_seq=6, hand_id='h2'))
    s.ingest(action(7, 'a', 'flop', 'call', 20, 30, hand='h2'))
    s.ingest(action(8, 'a', 'flop', 'fold', 40, 80, hand='h2', reason='timeout'))
    s.ingest(action(9, 'a', 'flop', 'check', 0, 80, hand='h2'))
    assert s.faced('a') == 0


def feed(stats, name, fold_rate_at, n=60):
    for k in range(n):
        price = (.15, .25, .4)[k % 3]
        folded = (k // 3) % 10 < round(10 * fold_rate_at(price))
        stats.observe(name, 1, 'fold' if folded else 'call', round(1000 * price), round(1000 * (1 - price)))


def test_fit_separates_sticky_from_tight_and_gates_on_sample_size():
    s = OpponentStats(min_faced=20)
    feed(s, 'sticky', lambda p: .1 + .5 * p)
    feed(s, 'tight', lambda p: .45 + 1.2 * p)
    feed(s, 'new', lambda p: .5, n=5)
    sticky, tight = s.threshold('sticky'), s.threshold('tight')
    fold = lambda t, p: t[0] + t[1] * p
    assert fold(sticky, .25) < .35 < .6 < fold(tight, .25)
    assert s.threshold('new') is None
    assert s.profiles({1: 'sticky', 3: 'new'}) == {1: sticky}


def test_save_load_roundtrip(tmp_path):
    s = OpponentStats(min_faced=1)
    feed(s, 'x', lambda p: .3)
    path = tmp_path / 'stats.json'
    s.save(path)
    loaded = OpponentStats.load(path, min_faced=1)
    assert loaded.threshold('x') == s.threshold('x')
    assert json.loads(path.read_text())['schema'] == 1
    assert OpponentStats.load(tmp_path / 'missing.json').players == {}


def test_profiles_use_padded_positions_and_skip_folded():
    class Fallback:
        profiles = {}
    stats = OpponentStats(min_faced=1)
    for name in ('a', 'b', 'c'):
        feed(stats, name, lambda p: .3)
    fb = Fallback()
    fb.opponent_stats = stats
    seats = [dict(name='a', folded=False), dict(name='hero', folded=False),
             dict(name='b', folded=True), dict(name='c', folded=False)]
    # Four players padded to six: index 2, 3 -> positions 4, 5; hero at 1.
    out = opponent_profiles(fb, seats, 2, 1)
    assert set(out) == {0, 5} and fb.profiles is out
    assert opponent_profiles(Fallback(), seats, 2, 1) is None
