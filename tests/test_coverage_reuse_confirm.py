import copy
import json
from scripts.coverage_reuse_confirm import gate_report, ROOT, SEED_BASE


def passing():
    totals = [dict(nodes=1024, seconds=10., variance_nodes=4., variance_seconds=2.),
              dict(nodes=1024, seconds=11., variance_nodes=4., variance_seconds=2.2)]
    summaries = [dict(active23_new_positive_keys=100, active23_postflop_updates=500),
                 dict(active23_new_positive_keys=300, active23_postflop_updates=500)]
    cells = {f'owner={o}:street={s}': 20 for o in range(6) for s in (1, 2, 3)}
    return dict(totals=totals, summaries=summaries, new_cells=cells, peaks=[1., 2.], max_ratio=1.,
                role_batches=[dict(nodes=10000, seconds=2.)])


def test_amended_gate_boundaries_pass_exactly():
    report = gate_report(**passing())
    assert all(report['gates'].values())
    assert report['new_unique_keys_per_node_gain'] == 3 and report['raw_update_rate_gain'] == 1


def test_each_amended_gate_fails_independently():
    def failed(change):
        case = copy.deepcopy(passing())
        change(case)
        return {k for k, v in gate_report(**case)['gates'].items() if not v}
    assert failed(lambda c: c['summaries'][1].update(active23_new_positive_keys=299)) == {'new_unique_key_rate'}
    # Many redundant updates cannot substitute for new keys, and fewer updates regress.
    assert failed(lambda c: c['summaries'][1].update(active23_new_positive_keys=299, active23_postflop_updates=5000)) == {'new_unique_key_rate'}
    assert failed(lambda c: c['summaries'][1].update(active23_postflop_updates=499)) == {'total_update_rate_not_regressed'}
    assert failed(lambda c: c['new_cells'].update({'owner=3:street=2': 19})) == {'all_role_street_new_keys'}
    assert failed(lambda c: c['new_cells'].pop('owner=3:street=2')) == {'all_role_street_new_keys'}
    assert failed(lambda c: c['totals'][1].update(variance_nodes=6.01)) == {'variance_work'}
    assert failed(lambda c: c['totals'][1].update(variance_seconds=4.01)) == {'variance_time'}
    assert failed(lambda c: c['totals'][1].update(seconds=20.1)) == {'total_time'}
    assert failed(lambda c: c.update(max_ratio=1.001)) == {'defensive_bound'}
    assert failed(lambda c: c.update(peaks=[1., float('inf')])) == {'finite_actual_weights'}
    assert failed(lambda c: c.update(role_batches=[dict(nodes=10001, seconds=1.)])) == {'bounded_batches'}
    assert failed(lambda c: c['summaries'][0].update(active23_new_positive_keys=0)) == {'new_unique_key_rate'}


def test_confirmation_is_separate_from_failed_probe():
    assert ROOT.name == 'coverage-reuse-confirm-001' and SEED_BASE == 12700000
    old = json.loads((ROOT.parent / 'coverage-reuse-probe-001' / 'independent-review.json').read_text()) \
        if (ROOT.parent / 'coverage-reuse-probe-001' / 'independent-review.json').exists() else None
    assert old is None or old['eligible_for_registration'] is False
