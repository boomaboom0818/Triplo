import json
from random import Random

import pytest

from poker.fast_eval import CARD_ID
from poker.fast_game import FastGame
from poker.lean_abstraction import card_features, lean_key, observation_key
from poker.lean_cfr import LeanCFR, LeanPolicy, load, resume
from scripts.rule_gate import play_style


def ids(text):
    return tuple(CARD_ID[text[i:i + 2]] for i in range(0, len(text), 2))


def test_lean_menu_has_no_open_shove_or_bare_min_raise():
    game = FastGame(1, lean=True)
    assert game.menu() == (3, 2, ('call', 'fold', 4, 5))
    assert FastGame(1).menu()[2] == ('call', 'fold', 4, 5, 200)
    # The all-in returns once it is at most a twice-pot raise.
    short = FastGame(1, stacks=[200, 200, 200, 8, 200, 200], lean=True)
    assert short.menu()[2] == ('call', 'fold', 4, 5, 8)
    # Raises grow the pot geometrically, so raise wars are short.
    game, raises = FastGame(1, lean=True), 0
    while game.live and isinstance(game.menu()[2][-1], int) and game.street == 0:
        game.act(game.menu()[2][2]); raises += 1
    assert raises < 15


def test_training_and_observation_keys_agree():
    rng = Random(4)
    seen = set()
    for seed in range(150):
        game = FastGame(seed, lean=True)
        while game.live:
            actor = game.actors[0]
            pot, call, legal = game.menu()
            key = lean_key(actor, game.street, card_features(game.hole[actor], tuple(game.board)),
                           game.stacks, game.active, pot, call, legal)
            assert key == observation_key(game.observe())
            seen.add(key)
            game.act('call' if rng.random() < .6 else rng.choice(legal))
    assert len(seen) > 400


def test_card_features_separate_hands_the_old_abstraction_merged():
    board = ids('Kh7c2d')
    top, bottom, air = (card_features(ids(h), board) for h in ('AsKs', '3s2s', '9s8s'))
    assert int(top[0]) > int(bottom[0]) > int(air[0])
    assert card_features(ids('AsKs'), ()) == 'cbs' and card_features(ids('2h2d'), ()) == '00o'
    assert card_features(ids('AsQs'), ids('Ks7s2d'))[1] == '2'          # flush draw
    assert card_features(ids('9h8d'), ids('7s6c2d'))[1] == '1'          # straight draw
    assert card_features(ids('AsQs'), ids('Ks7s2d3c4h'))[1] == '0'      # no draws on the river


def test_solver_is_deterministic_and_resumes_exactly(tmp_path):
    a = LeanCFR(3)
    for _ in range(30): a.iterate()
    path = tmp_path / 'c.json'
    a.save(path)
    b = resume(path)
    for _ in range(20): a.iterate(); b.iterate()
    assert a.regrets == b.regrets and a.sums == b.sums and a.nodes == b.nodes and a.iterations == 50
    fresh = LeanCFR(3)
    for _ in range(50): fresh.iterate()
    assert fresh.regrets == a.regrets and fresh.sums == a.sums
    saved = json.loads(path.read_text())
    saved['menu'] = 'other'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load(path)


def test_updates_are_bounded_by_the_linear_weight():
    solver = LeanCFR(5)
    for _ in range(40): solver.iterate()
    # Each visit adds at most its iteration weight in total; no importance weights.
    assert max(sum(row) for row in solver.sums.values()) <= 6 * 40 * sum(range(1, 41))
    assert all(x >= 0 for row in solver.sums.values() for x in row)


def test_lean_policy_plays_legal_hands_on_the_lean_engine():
    solver = LeanCFR(2)
    for _ in range(60): solver.iterate()
    hero = LeanPolicy(solver.sums, 1)
    values, shoves = play_style(hero, 'tight-v1', 20, lean=True)
    assert len(values) == 20 and hero.decisions > 100 and 0 < hero.covered <= hero.decisions


def test_only_masked_positions_get_the_lean_menu():
    game = FastGame(1, lean=1 << 3)
    assert game.menu()[2] == ('call', 'fold', 4, 5, 200)       # position 2 keeps the full menu
    game.act('fold')
    assert game.menu()[2] == ('call', 'fold', 4, 5)            # position 3 is lean


def test_unseen_deep_all_in_is_answered_by_the_rule_fallback():
    import dataclasses
    game = FastGame(1, lean=1 << 3)
    game.act(200)                                              # a 100bb open shove: not in the trained game
    o = game.observe()
    assert o.legal == ('call', 'fold')
    policy = LeanPolicy({}, 1)
    assert policy.act(dataclasses.replace(o, hole=('As', 'Ah'))) == 'call'
    assert policy.act(dataclasses.replace(o, hole=('9s', '2h'))) == 'fold'
    assert policy.covered == 0 and policy.decisions == 2
