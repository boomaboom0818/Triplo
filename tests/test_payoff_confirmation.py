from scripts import payoff_confirmation as p


def case(parent_low, uniform_low, worst_high=5.):
    pair = {'k': dict(interval=[parent_low, parent_low + 50]), 'tight': dict(interval=[-20., worst_high])}
    return p.judge(pair, {'k': dict(ci95=[uniform_low, uniform_low + 50])}, 'k')


def test_all_cases_must_pass_every_gate():
    good = case(.1, .1)
    assert p.passed([good] * 4)
    assert not p.passed([])
    assert not p.passed([good] * 3 + [case(0., 1.)])       # parent bound must be strictly positive
    assert not p.passed([good] * 3 + [case(1., 0.)])       # uniform bound must be strictly positive
    assert not p.passed([good] * 3 + [case(1., 1., -10.1)])  # material regression on any population
    assert p.passed([good] * 3 + [case(1., 1., -10.)])


def test_frozen_design():
    assert (p.SEED, p.BLOCKS, p.PARENT_CONFIDENCE) == (15000000, 20000, .9875)
    assert all('coverage-reuse-training-001' in str(x) for x in p.CANDIDATES)
    assert all('future-linear-training-001' in str(x) for x in p.PARENTS)
