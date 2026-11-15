"""Real disk v8->v10 fork, exact resume, loader and evaluation tests."""
import json
import pytest
from poker.config import load_checkpoint, CFR_ALGORITHMS
from poker.coverage_reuse_average import ALGORITHM
from poker.coverage_reuse_contract import AVERAGING, TRAVERSAL, ESTIMATOR, FORK_KIND, fork_allowed, validate
from poker.future_linear_average import AVERAGING as V8_AVERAGING
from poker.train_cfr import train

V7 = dict(game='holdem', seed=11, traversal='joint-average-v7', averaging_sampler='regret-path-mis-average-v7',
          trajectory_samples=1, opponent_population='guarded-tables-v2', population_probability=.25,
          archive_every=2, archive_size=2, regret_prefix_depth=2)
V8 = {**V7, 'traversal': 'future-linear-average-v8', 'averaging_sampler': V8_AVERAGING}
V10 = {**V8, 'traversal': TRAVERSAL, 'averaging_sampler': AVERAGING}
LEARNING = ('regrets', 'sums', 'rng', 'archive', 'iterations', 'nodes', 'diagnostics', 'average_weight_moments',
            'weight_diagnostics_started_at', 'diagnostics_started_at')


@pytest.fixture(scope='module')
def lineage(tmp_path_factory):
    root = tmp_path_factory.mktemp('v10')
    train(1, root / 'v7', **V7)
    v8 = train(3, root / 'v8', resume=root / 'v7', fork_traversal=True, **V8)
    return root, v8


def test_fork_contract_only_changes_traversal_and_sampler():
    cfg = dict(game='holdem', players=6, **{k: v for k, v in V8.items() if k != 'game'})
    new = {**cfg, 'traversal': TRAVERSAL, 'averaging_sampler': AVERAGING}
    assert fork_allowed(cfg, new)
    assert not fork_allowed(cfg, {**new, 'trajectory_samples': 2})
    assert not fork_allowed({**cfg, 'traversal': 'joint-average-v7'}, new)
    assert not fork_allowed(cfg, cfg) and not fork_allowed(new, new)
    assert ALGORITHM in CFR_ALGORITHMS


def test_failed_fork_preserves_all_learning_and_old_clock(lineage):
    root, v8 = lineage
    original = (root / 'v8').read_bytes()
    with pytest.raises(ValueError, match='incompatible'):
        train(4, root / 'implicit', resume=root / 'v8', **V10)
    with pytest.raises(ValueError, match='incompatible'):
        train(4, root / 'skip', resume=root / 'v7', fork_traversal=True, **V10)
    with pytest.raises(ValueError):
        train(4, root / 'wrong', resume=root / 'v8', fork_traversal=True, **{**V10, 'trajectory_samples': 2})
    failed = train(4, root / 'failed', resume=root / 'v8', fork_traversal=True, **{**V10, 'max_iteration_nodes': 1})
    for key in LEARNING:
        assert failed[key] == v8[key], key
    assert failed['algorithm'] == ALGORITHM and failed['average_estimator_schema'] == ESTIMATOR
    assert failed['future_average_origin'] == v8['future_average_origin'] == 1
    assert failed['future_average_completed_iterations'] == 2 and failed['coverage_reuse_started_at'] == 3
    assert failed['joint_average_diagnostics_started_at'] == v8['joint_average_diagnostics_started_at']
    assert failed['traversal_stats_started_at'] == 3 and failed['traversal_stats']['iterations'] == 0
    assert failed['recipe_fork']['kind'] == FORK_KIND and failed['recipe_fork']['previous_recipe_fork'] == v8['recipe_fork']
    assert load_checkpoint(root / 'failed')['iterations'] == 3
    assert (root / 'v8').read_bytes() == original


def test_exact_disk_resume_evaluation_and_strict_loader(lineage):
    root, v8 = lineage
    original = (root / 'v8').read_bytes()
    full = train(5, root / 'full', resume=root / 'v8', fork_traversal=True, **V10)
    split = root / 'split'
    train(4, split, resume=root / 'v8', fork_traversal=True, **V10)
    after = train(5, split, resume=split, **V10)
    for key in (*LEARNING, 'traversal_stats', 'epoch_weight_moments', 'recipe_fork', 'future_average_origin',
                'future_average_completed_iterations', 'coverage_reuse_started_at', 'traversal_stats_started_at'):
        assert full[key] == after[key], key
    assert (root / 'v8').read_bytes() == original
    assert after['future_average_origin'] == 1 and after['future_average_completed_iterations'] == 4
    assert after['coverage_reuse_started_at'] == 3 and after['traversal_stats']['iterations'] == 2
    assert all(after['sums'][key][j] >= v for key, row in v8['sums'].items() for j, v in enumerate(row))
    assert any(k.startswith('coverage-reuse-v10:') for k in after['diagnostics'])
    assert validate(load_checkpoint(split)) == 1
    from poker.evaluate import evaluate
    report = evaluate(split, 2, 19, 1)
    assert report['algorithm'] == ALGORITHM and report['populations']['mixed']['candidate_decisions'] > 0
    fork = after['recipe_fork']
    for changes in ({'future_average_origin': True}, {'future_average_origin': 3}, {'future_average_origin': 0},
                    {'future_average_completed_iterations': 2}, {'coverage_reuse_started_at': 4},
                    {'coverage_reuse_started_at': None}, {'future_average_weighting_schema': 'uniform'},
                    {'average_estimator_schema': 'joint-inclusion-future-linear-v8'},
                    {'joint_average_diagnostics_started_at': True}, {'traversal_stats_started_at': 1},
                    {'recipe_fork': None}, {'recipe_fork': fork['previous_recipe_fork']},
                    {'recipe_fork': {**fork, 'previous_recipe_fork': None}},
                    {'recipe_fork': {**fork, 'parent_algorithm': 'joint-traversal-average-mccfr-v7'}}):
        bad = root / 'bad'
        bad.write_text(json.dumps({**after, **changes}))
        with pytest.raises(ValueError):
            load_checkpoint(bad)
        with pytest.raises(ValueError):
            train(6, root / 'bad-resume', resume=bad, **V10)
    # A v10 checkpoint can never be relabelled back to v8.
    with pytest.raises(ValueError):
        train(6, root / 'back', resume=split, **V8)


def test_first_v10_iteration_uses_the_preserved_weighting_clock(lineage):
    root, v8 = lineage
    from poker.coverage_reuse_average import CoverageReuseMCCFR
    solver = CoverageReuseMCCFR(average_origin=v8['future_average_origin'])
    solver.iterations = v8['iterations']
    assert solver.average_multiplier == v8['future_average_completed_iterations'] + 1 == 3
