import dataclasses
import pytest
from poker.game import Game
from poker.rule_opponents import RulePolicy, STYLES, chen
from scripts.rule_gate import play_style, summarize


def first(hole, actions=()):
    game = Game(1)
    for action in actions:
        game.act(action)
    return dataclasses.replace(game.observe(), hole=hole)


def test_chen_orders_obvious_hands():
    assert chen(('As', 'Ah')) > chen(('Ks', 'Kh')) > chen(('As', 'Ks')) > chen(('As', 'Kh')) > chen(('7s', '2h'))
    assert chen(('2s', '2h')) == 5


def test_rule_opponents_fold_junk_and_never_fold_aces():
    for style in STYLES:
        policy = RulePolicy(style)
        assert policy.act(first(('7s', '2h'))) == 'fold'
        assert policy.act(first(('As', 'Ah'), [200])) == 'call'       # facing an all-in
        assert policy.act(first(('9s', '2h'), [200])) == 'fold'
    assert RulePolicy('tight-v1').act(first(('As', 'Ah'))) == 5        # pot-size open
    with pytest.raises(ValueError):
        RulePolicy('other')


def test_rule_opponents_always_return_legal_actions():
    for style in STYLES:
        policy = RulePolicy(style)
        for seed in range(40):
            game = Game(seed)
            while game.state.status:
                game.act(policy.act(game.observe()))
            assert abs(sum(game.payoff())) < 1e-9


class Shover:
    def act(self, o):
        return o.legal[-1] if isinstance(o.legal[-1], int) else 'call'


class Folder:
    def act(self, o):
        return 'call' if not o.call else 'fold'


def test_gate_fails_always_shove_and_scores_always_fold_at_blind_cost():
    values, shoves = play_style(Shover(), 'tight-v1', 60)
    report = summarize(values, shoves)
    assert report['bb_per_100'] < 0 and not report['passed'] and report['preflop_allins_per_100_hands'] > 50
    values, shoves = play_style(Folder(), 'tight-v1', 30)
    assert shoves == 0 and -30 <= summarize(values, shoves)['bb_per_100'] <= 0


def test_gate_requires_interval_above_zero():
    assert summarize([10., 12., 11., 9.], 0)['passed']
    assert not summarize([10., -12., 11., -9.], 0)['passed']
