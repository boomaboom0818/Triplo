"""Tests for the A/B metric harness (poker/ab_metric.py, scripts/ab_report.py)."""
import json

import pytest

from poker import ab_metric as M


def test_metric_caps_pairs_and_window_shares():
    nets = [100., -3., 2., -1., 0.5, -60.]
    assert M.estimate(nets, 25)['bb100'] == pytest.approx(100 * (25 - 3 + 2 - 1 + .5 - 25) / 6, abs=.01)
    assert M.estimate(nets)['n'] == 6
    d = M.paired([1., 2., 3.], [2., 3., 4.])
    assert d['bb100'] == 100 and d['ci95'] == [100, 100]
    with pytest.raises(ValueError):
        M.paired([1.], [1., 2.])
    assert M.welch([1., 2., 3.], [2., 3., 4.])['bb100'] == 100
    # Windows: two 10-minute windows, one positive.
    w = M.window_share([(0, 5.), (100, -1.), (700, -2.)])
    assert w['windows'] == 2 and w['positive_share'] == .5
    assert M.block_share([1, 1, -5, 1], 2) == dict(blocks=2, positive_share=.5)
    assert M.paired_block_share([0, 0, 0, 0], [1, -2, 0, 0], 2)['arm_ahead_share'] == .25
    # Hands needed scale with variance; capping cuts them.
    assert M.hands_for(25, 35) > M.hands_for(12.4, 35) > M.hands_for(7.4, 35)
    assert 1500 < M.hands_for(7.4, 35) < 2000


def test_metric_big_pot_breakdown_sums_to_total():
    nets, pots = [1., -2., 60., -40., 3.], [3., 12., 130., 90., 5.]
    by = M.by_pot(nets, pots)
    assert sum(v['contribution_bb100'] for v in by.values()) == pytest.approx(M.estimate(nets)['bb100'], abs=.05)
    big = M.big_pot_summary(nets, pots)
    assert big['hands'] == 2 and big['contribution_bb100'] == pytest.approx(400)
    paired = M.paired_by_pot(nets, [n + 1 for n in nets], pots, pots)
    assert sum(v['contribution_bb100'] for v in paired.values()) == pytest.approx(100, abs=.05)


def test_ab_report_reads_session_log(tmp_path):
    from scripts.ab_report import load, report
    log = tmp_path / 'session.jsonl'
    rows = []
    stacks = 5000
    for k in range(6):
        new = stacks + (100 if k % 2 else -40)
        rows.append(dict(time=1000 + 120 * k, type='hand_result', seat=2,
                         result=dict(table_id='t', hand_id=f'h{k}', total_pot=200, final_stacks={'2': new},
                                     payouts=[dict(seat=2, amount=200)] if k % 2 else [])))
        stacks = new
    log.write_text('\n'.join(json.dumps(r) for r in rows))
    rep = report(load([log], 20, 5000), [('a', 1000, 1400), ('b', 1400, None)])
    assert rep['arms']['a']['uncapped']['n'] == 3 and rep['arms']['b']['uncapped']['n'] == 2
    assert 'b-a' in rep['differences']
