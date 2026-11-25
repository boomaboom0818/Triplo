"""Triplo v9 big-open tooling: benchmark arms and opener models, live open report helpers."""
from collections import Counter

from poker.fast_game import FastGame
from scripts.v9_open_benchmark import BigOpener, StickyFish, parse_arm
from scripts.v9_open_report import bucket, first_open


def test_arm_specs():
    assert parse_arm('v8') is None and parse_arm('off') is None
    assert parse_arm('tr') == dict(max_open=None, fish_only=False, keep=None)
    assert parse_arm('tr3') == dict(max_open=6, fish_only=False, keep=None)
    assert parse_arm('tr3f') == dict(max_open=6, fish_only=True, keep=None)
    assert parse_arm('tr3r')['keep'] == ('raise',) and parse_arm('trc')['keep'] == ('call',)


def deals(n=400):
    for seed in range(n):
        g = FastGame(seed, button=5, lean=True)
        yield g, g.observe()


def test_big_openers_open_wide_and_big():
    for style, sizes in (('fixed', {7}), ('svan', {5, 8}), ('three', {6})):
        bot = BigOpener(None, style)
        opens = Counter()
        for g, o in deals():
            a = bot.preflop(o, g)
            opens[a if isinstance(a, int) else 'no'] += 1
        assert set(opens) - {'no'} == sizes
        share = 1 - opens['no'] / 400
        assert (.12 < share < .22) if style == 'three' else (.52 < share < .68)
    svan = BigOpener(None, 'svan')
    big = Counter()
    for g, o in deals(2000):
        a = svan.preflop(o, g)
        if isinstance(a, int):
            big[a] += 1
    assert .25 < big[8] / sum(big.values()) < .42             # live: a third of Svan opens are 4 BB


def test_big_opener_answers_a_three_bet_mostly_by_calling():
    bot = BigOpener(None, 'fixed')
    answers = Counter()
    for seed in range(600):
        g = FastGame(seed, button=5, lean=True)
        o = g.observe()
        a = bot.preflop(o, g)
        if not isinstance(a, int):
            continue
        g.history.append((o.player, o.position, 0, a, o.pot, o.call)); g.act(a, checked=False)
        o = g.observe()
        g.history.append((o.player, o.position, 0, 21, o.pot, o.call)); g.act(21, checked=False)  # HJ 3-bets
        while g.actors[0] != 2:
            o = g.observe()
            g.history.append((o.player, o.position, 0, 'fold', o.pot, o.call)); g.act('fold', checked=False)
        r = bot.preflop(g.observe(), g)
        answers['raise' if isinstance(r, int) else r] += 1
    total = sum(answers.values())
    assert .5 < answers['call'] / total < .75 and .15 < answers['fold'] / total < .35


def test_sticky_fish_rarely_folds_to_a_three_bet():
    fish = StickyFish(seed=7)
    answers = Counter()
    for seed in range(3000):
        g = FastGame(seed, button=5)
        o = g.observe()
        a = fish.act(o, g)
        if not isinstance(a, int):
            continue
        g.history.append((o.player, o.position, 0, a, o.pot, o.call)); g.act(a, checked=False)
        o = g.observe()
        top = max(g.bets)
        g.history.append((o.player, o.position, 0, 3 * top, o.pot, o.call)); g.act(3 * top, checked=False)
        while g.actors[0] != 2:
            o = g.observe()
            g.history.append((o.player, o.position, 0, 'fold', o.pot, o.call)); g.act('fold', checked=False)
        r = fish.act(g.observe(), g)
        answers['raise' if isinstance(r, int) else r] += 1
    total = sum(answers.values())
    assert total > 300 and answers['fold'] / total < .05 and answers['call'] / total > .85


def test_report_finds_the_first_open_and_buckets_sizes():
    actions = [dict(seat=3, action='fold', street='preflop'), dict(seat=4, action='call', street='preflop'),
               dict(seat=5, action='raise', amount=70, street='preflop'), dict(seat=0, action='call', street='preflop')]
    assert first_open(actions) == (2, 5, 70, 1)
    assert first_open(actions[:2] + [dict(seat=5, action='check', street='flop')]) is None
    assert [bucket(x) for x in (2, 2.5, 3, 3.5, 4, 4.5, 30)] == ['<=2.25', '2.5', '3', '3.5-4', '3.5-4', '>4', '>4']
