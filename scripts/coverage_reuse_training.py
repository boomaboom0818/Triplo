"""Engineering-gated v8->v10 coverage-only reuse continuation; one bounded staged pilot.

Protocol approved by the user on 2026-11-15 (docs/COVERAGE-REUSE-PILOT.md). Tight
coverage gain is measured against the fixed v7 reference; payoff gates stay on
the immediate v8 parents.
"""
import argparse
import json
from pathlib import Path

from poker.coverage_reuse_average import ALGORITHM
from poker.coverage_reuse_contract import AVERAGING, TRAVERSAL, fork_allowed
from scripts import trajectory_pilot as driver
from scripts.sustained_train import digest, stage_config as original_stage_config

ROOT = Path('runs/bot/coverage-reuse-training-001')
PARENTS = [Path(f'runs/bot/future-linear-training-001/candidate-{i}-plus-2000.json') for i in range(2)]
REFERENCES = [Path(f'runs/bot/joint-average-training-001/candidate-{i}-plus-2000.json') for i in range(2)]
ENGINEERING = Path('runs/bot/coverage-reuse-confirm-001')
MATH = [Path('poker') / name for name in (
    'coverage_reuse_average.py', 'cross_role_average.py', 'future_linear_average.py', 'joint_average.py',
    'prefix.py', 'trajectory.py', 'cfr.py', 'population.py')]
GATES = ('exact_regret_rng_nodes', 'exact_covered_updates', 'variance_work', 'variance_time', 'total_time',
         'new_unique_key_rate', 'total_update_rate_not_regressed', 'all_role_street_new_keys',
         'defensive_bound', 'finite_actual_weights', 'bounded_batches')


def verify_engineering(root=ENGINEERING, parents=PARENTS, math=MATH):
    root = Path(root)
    review = json.loads((root / 'independent-review.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = review.get('candidates', {})
    if (review.get('eligible_for_integration') is not True or set(rows) != {'0', '1'}
            or review.get('manifest_sha256') != digest(root / 'manifest.json')):
        raise ValueError('both independently reviewed coverage-reuse confirmations required')
    for name, row in rows.items():
        gates = row.get('gates', {})
        if (set(gates) != set(GATES) or not all(v is True for v in gates.values())
                or row.get('resources_ok') is not True
                or row.get('report_sha256') != digest(root / f'candidate-{name}.json')):
            raise ValueError('coverage-reuse confirmation gate/report mismatch')
    if any(manifest['sources'].get(str(p)) != digest(p) for p in math):
        raise ValueError('qualified math source changed')
    inputs = manifest.get('inputs', [])
    if [str(Path(v['path']).resolve()) for v in inputs] != [str(Path(p).resolve()) for p in parents]:
        raise ValueError('qualified parent set changed')
    if any(digest(Path(v['path'])) != v['sha256'] for v in inputs):
        raise ValueError('qualified parent bytes changed')


def stage_config(saved, target):
    cfg = original_stage_config(saved, target)
    cfg['max_seconds'] = 7200
    return cfg


def configure():
    driver.ROOT = ROOT
    driver.PARENTS = PARENTS
    driver.EXPECTED_BASES = [8906, 9150]
    driver.ADDITIONS = [50, 200, 500, 1000, 2000]
    driver.ALGORITHM = ALGORITHM
    driver.AVERAGING = AVERAGING
    driver.TRAVERSAL = TRAVERSAL
    driver.PREFIX_DEPTH = 2
    driver.POPULATION = 'guarded-tables-v2'
    driver.FORK_FLAG = '--fork-traversal'
    driver.REQUIRE_TIGHT_IMPROVEMENT = True
    driver.COVERAGE_REFERENCES = REFERENCES
    driver.fork_allowed = fork_allowed
    driver.CRITERIA = Path('docs/COVERAGE-REUSE-PILOT.md')
    driver.SCREEN_SEED = 13100000
    driver.CONFIRM_SEED = 13200000
    driver.stage_config = stage_config
    driver.ADDITIONAL_SOURCES = [Path(__file__), ENGINEERING / 'independent-review.json', ENGINEERING / 'manifest.json',
                                 Path('docs/COVERAGE-REUSE-CONFIRMATION.md'), Path('configs/bot.json')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    verify_engineering()
    configure()
    driver.main(args.deadline, args.resume)
