from itertools import permutations
from random import Random
from poker.kuhn import Kuhn, audit
from poker.sampled_cfr import SampledCFR


def test_kernel_learns_kuhn_against_exhaustive_best_responses():
    solver = SampledCFR(2, 81)
    deals = list(permutations(range(3), 2))
    rng = Random(82)
    uniform = audit(solver.average)
    for _ in range(30000):
        solver.iterate(Kuhn(rng.choice(deals)))
    result = audit(solver.average)
    assert uniform['nash_conv'] > .8
    assert result['nash_conv'] < .06
    assert abs(result['value_player0'] + 1/18) < .015


def test_kuhn_payoffs_and_information_privacy():
    assert Kuhn((2,0),'bc').utility(0) == 2
    assert Kuhn((2,0),'cbf').utility(0) == -1
    assert Kuhn((1,2)).key() == Kuhn((1,0)).key()
    assert Kuhn((1,2),'c').key() != Kuhn((1,2),'b').key()


def test_exhaustive_audit_recognizes_analytic_kuhn_equilibrium():
    # Independent analytic policy fixes the audit's expected answer, preventing
    # a broken evaluator and a broken learner from agreeing with one another.
    def equilibrium(key, count):
        player, card, history = key.split(':')
        card = int(card)
        if player == '0':
            if not history:
                return ([2/3,1/3], [1.,0.], [0.,1.])[card]
            return ([1.,0.], [1/3,2/3], [0.,1.])[card]
        if history == 'c':
            return ([2/3,1/3], [1.,0.], [0.,1.])[card]
        return ([1.,0.], [2/3,1/3], [0.,1.])[card]
    result = audit(equilibrium)
    assert abs(result['nash_conv']) < 1e-12
    assert abs(result['value_player0']+1/18) < 1e-12


def test_revisited_coordinate_does_not_overwrite_descendant_regret():
    class State:
        actor = 0
        actions = ('a','b')
        def __init__(self, depth=0, payoff=None):
            self.depth, self.payoff = depth, payoff
        @property
        def terminal(self): return self.payoff is not None
        def key(self): return 'merged'
        def utility(self, player): return self.payoff
        def child(self, action):
            if self.depth == 0:
                return State(1) if action == 'a' else State(1,0.)
            return State(2,3. if action == 'a' else 1.)
    solver = SampledCFR(2)
    solver.walk(State(),0,1)
    assert solver.regrets['merged'] == [2.,-2.]
