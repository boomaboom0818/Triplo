"""The fast engine must be indistinguishable from the pokerkit reference engine."""
from random import Random

from pokerkit import Deck, StandardHighHand

from poker.fast_eval import LABELS, NAMES, STRAIGHT_DRAW, evaluate
from poker.fast_game import FastGame
from poker.game import Game


def test_card_ids_follow_reference_deck_order():
    assert list(NAMES) == [repr(c) for c in Deck.STANDARD]


def test_evaluator_matches_reference_ordering_and_labels():
    rng = Random(3)
    previous = None
    for trial in range(1500):
        cards = rng.sample(range(52), 9)
        board = cards[2:2 + rng.choice((3, 4, 5))]
        hole = cards[:2]
        ours = evaluate(hole + board)
        theirs = StandardHighHand.from_game(''.join(NAMES[c] for c in hole), ''.join(NAMES[c] for c in board))
        assert LABELS[ours >> 20] == str(theirs.entry.label)
        if previous is not None:
            a, b = previous
            assert (ours > a) == (theirs > b) and (ours == a) == (theirs == b)
        previous = ours, theirs


def test_straight_draw_table_counts_best_window():
    mask = lambda text: sum(1 << '23456789TJQKA'.index(r) for r in text)
    assert STRAIGHT_DRAW[mask('A2345')] == 5 and STRAIGHT_DRAW[mask('TJQKA')] == 5
    assert STRAIGHT_DRAW[mask('A234')] == 4 and STRAIGHT_DRAW[mask('2789K')] == 3 and STRAIGHT_DRAW[mask('2')] == 1


def replay(seed, players, stacks, rng, passive):
    slow, fast = Game(seed, players, stacks=stacks), FastGame(seed, players, stacks=stacks)
    decisions = 0
    while slow.state.status:
        assert fast.live
        o = slow.observe()
        assert fast.observe() == o
        if rng.random() < passive:
            action = 'call'
        else:
            action = rng.choice(o.legal)
        # A clone must be independent of its parent.
        probe = fast.clone(); probe.act(action); assert fast.observe() == o
        slow.act(action); fast.act(action)
        decisions += 1
    assert not fast.live and fast.payoff() == slow.payoff()
    return decisions, len(slow.observe.__self__.state.board_cards)


def test_matches_reference_on_training_shape():
    rng = Random(11)
    decisions = showdowns = 0
    for seed in range(700):
        d, board = replay(seed, 6, None, rng, rng.choice((.2, .6, .9)))
        decisions += d; showdowns += board == 5
    assert decisions > 5000 and showdowns > 200


def test_matches_reference_with_short_tables_and_unequal_stacks():
    rng = Random(12)
    showdowns = 0
    for seed in range(1500):
        players = rng.randint(3, 6)
        stacks = [rng.choice((rng.randint(1, 12), rng.randint(2, 60), rng.randint(40, 400))) for _ in range(players)]
        _, board = replay(seed, players, stacks, rng, rng.choice((.3, .7, .95)))
        showdowns += board == 5
    assert showdowns > 400


def test_button_rotation_and_validation():
    import pytest
    assert FastGame(5, 6, button=3).order == Game(5, 6, button=3).order
    with pytest.raises(ValueError):
        FastGame(1, 2)
    game = FastGame(1)
    with pytest.raises(ValueError):
        game.act(3)
