import json
import pytest
from scripts import coverage_reuse_training as pilot
from scripts.sustained_train import digest
from scripts.trajectory_pilot import fixed_reference_tight_coverage

NAMES = ('ROOT', 'PARENTS', 'EXPECTED_BASES', 'ADDITIONS', 'ALGORITHM', 'AVERAGING', 'TRAVERSAL', 'PREFIX_DEPTH',
         'POPULATION', 'FORK_FLAG', 'REQUIRE_TIGHT_IMPROVEMENT', 'COVERAGE_REFERENCES', 'fork_allowed', 'CRITERIA',
         'SCREEN_SEED', 'CONFIRM_SEED', 'ADDITIONAL_SOURCES', 'stage_config')


def tight(fraction, decisions=200):
    return {'tight': dict(decisions=decisions, fraction=fraction)}


@pytest.mark.parametrize('candidate,reference,parent,decisions,passed', [
    (.52, .50, .52, 200, True),      # exact +2pp over fixed reference, no parent decline
    (.5199, .50, .51, 200, False),   # below +2pp
    (.53, .50, .5301, 200, False),   # declined against immediate parent
    (.53, .50, .52, 199, False),     # too few decisions
    (.04, .01, .03, 200, False),     # below absolute 5%
    (.99, .96, .981022, 200, True),  # feasible near the ceiling, unlike a rebased +2pp
])
def test_fixed_reference_tight_gate(candidate, reference, parent, decisions, passed):
    result = fixed_reference_tight_coverage(tight(candidate, decisions), tight(reference), tight(parent))
    assert result['passed'] is passed and result['required_percentage_point_gain'] == 2.


def test_pilot_configuration(monkeypatch):
    from scripts import trajectory_pilot as d
    for name in NAMES:
        monkeypatch.setattr(d, name, getattr(d, name))
    pilot.configure()
    assert d.EXPECTED_BASES == [8906, 9150] and d.ADDITIONS == [50, 200, 500, 1000, 2000]
    assert (d.SCREEN_SEED, d.CONFIRM_SEED) == (13100000, 13200000) and d.REQUIRE_TIGHT_IMPROVEMENT
    assert d.TRAVERSAL == 'coverage-reuse-average-v10' and d.POPULATION == 'guarded-tables-v2'
    assert [p.name for p in d.COVERAGE_REFERENCES] == ['candidate-0-plus-2000.json', 'candidate-1-plus-2000.json']
    assert all('joint-average-training-001' in str(p) for p in d.COVERAGE_REFERENCES)
    assert all('future-linear-training-001' in str(p) for p in d.PARENTS)
    assert pilot.stage_config(dict(iterations=1, config={}, limits=dict(max_information_sets=5, max_iteration_nodes=7)), 2)['max_seconds'] == 7200


def test_engineering_gate_requires_reviewed_unchanged_evidence(tmp_path):
    math = tmp_path / 'math.py'
    math.write_text('x=1')
    parents = [tmp_path / 'a.json', tmp_path / 'b.json']
    for p in parents:
        p.write_text('{}')
    manifest = dict(sources={str(math): digest(math)}, inputs=[dict(path=str(p), sha256=digest(p)) for p in parents])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    for c in '01':
        (tmp_path / f'candidate-{c}.json').write_text(c)

    def write(**changes):
        rows = {c: dict(gates={g: True for g in pilot.GATES}, resources_ok=True,
                        report_sha256=digest(tmp_path / f'candidate-{c}.json')) for c in '01'}
        review = dict(eligible_for_integration=True, manifest_sha256=digest(tmp_path / 'manifest.json'), candidates=rows)
        for key, value in changes.items():
            if key == 'row':
                rows['1'].update(value)
            else:
                review[key] = value
        (tmp_path / 'independent-review.json').write_text(json.dumps(review))

    check = lambda: pilot.verify_engineering(tmp_path, parents, [math])
    write()
    check()
    for bad in (dict(eligible_for_integration=False), dict(manifest_sha256='0'), dict(row=dict(resources_ok=False)),
                dict(row=dict(report_sha256='0')), dict(row=dict(gates={g: g != 'new_unique_key_rate' for g in pilot.GATES})),
                dict(row=dict(gates={g: True for g in pilot.GATES[1:]}))):
        write(**bad)
        with pytest.raises(ValueError):
            check()
    write()
    with pytest.raises(ValueError):
        pilot.verify_engineering(tmp_path, parents[::-1], [math])
    math.write_text('x=2')
    with pytest.raises(ValueError, match='math'):
        check()
    math.write_text('x=1')
    parents[0].write_text('{"changed":1}')
    with pytest.raises(ValueError, match='parent bytes'):
        check()


def test_real_confirmation_evidence_currently_qualifies():
    if not (pilot.ENGINEERING / 'independent-review.json').exists():
        pytest.skip('local run artifacts absent')
    pilot.verify_engineering()
