from collections import Counter
import math

from poker.cfr import information_key
from poker.game import Game, play
from scripts.coverage_gap_audit import CoverageTrace, empty_cells, policies


def test_natural_trace_instrumentation_preserves_all_six_roles_and_rng():
    state = dict(sums={}, regrets={}, abstraction='texture-buckets-v2')
    for population in ('tight','calling','mixed'):
        for owner in range(6):
            cells, gaps = empty_cells(), Counter()
            traced = policies(state,state,population,11700000,0,owner,cells,gaps)
            plain = policies(state,state,population,11700000,0,owner)
            actual = play(traced,11700000,button=0)
            expected = play(plain,11700000,button=0)
            assert actual[:2] == expected[:2]
            assert traced[owner].rng.getstate() == plain[owner].rng.getstate()
            assert len(cells) == 120
            assert sum(c['decisions'] for c in cells.values()) == len(actual[1][owner])
            assert sum(gaps.values()) == len(actual[1][owner])


def test_gap_zero_current_reach_is_not_underflow_and_uses_prior_actions():
    o = Game(1).observe();key = information_key(o,'texture-buckets-v2')
    average = [float(a=='call') for a in o.legal]
    regret = [float(a=='fold') for a in o.legal]
    state = dict(sums={key:average},regrets={key:regret},abstraction='texture-buckets-v2')
    reference = dict(sums={key:average},regrets={key:regret})
    cells, gaps = empty_cells(), Counter()
    trace = CoverageTrace(state,reference,1,cells,gaps)
    assert trace.act(o) == 'call'
    assert trace.zero_current_reach
    state['sums'].clear()
    trace.act(o)
    c = cells[f'owner={o.player}:street={o.street}:active={sum(o.active)}']
    assert c['gap_zero_current_own_reach'] == 1 and c['gap_reference_average_positive'] == 1
    assert c['gap_regret_positive'] == 1
    trace.zero_current_reach = False; trace.log_current_reach = -10000.
    trace.act(o)
    assert c['gap_zero_current_own_reach'] == 1
    assert c['gap_tiny_nonzero_current_own_reach'] == 1
    assert c['gap_positive_current_own_reach'] == 1
    assert math.isfinite(trace.log_current_reach)
