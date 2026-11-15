"""Strict dispatch metadata for coverage-only cross-role reuse (v10).

A v10 checkpoint keeps the v8 future-linear origin and weighting clock. Only a
new diagnostic epoch starts at the explicit v8->v10 fork.
"""
import math

from .coverage_reuse_average import ALGORITHM
from .future_linear_contract import ALGORITHM as PARENT_ALGORITHM, AVERAGING as PARENT_AVERAGING, WEIGHTING

TRAVERSAL = 'coverage-reuse-average-v10'
PARENT_TRAVERSAL = 'future-linear-average-v8'
AVERAGING = 'coverage-only-cross-role-average-v10'
ESTIMATOR = 'coverage-only-cross-role-v10'
FORK_KIND = f'explicit {TRAVERSAL} fork'
PARENT_FORK_KIND = f'explicit {PARENT_TRAVERSAL} fork'


def fork_allowed(before, after):
    old = dict(before)
    new = dict(after)
    if old.pop('traversal', None) != PARENT_TRAVERSAL or new.pop('traversal', None) != TRAVERSAL:
        return False
    if old.pop('averaging_sampler', None) != PARENT_AVERAGING or new.pop('averaging_sampler', None) != AVERAGING:
        return False
    return old == new


def _integer(value):
    return type(value) is int


def validate(checkpoint):
    """Return the preserved future-averaging origin or raise ValueError."""
    origin = checkpoint.get('future_average_origin')
    started = checkpoint.get('coverage_reuse_started_at')
    iterations = checkpoint.get('iterations')
    joint_start = checkpoint.get('joint_average_diagnostics_started_at')
    completed = checkpoint.get('future_average_completed_iterations')
    stats = checkpoint.get('traversal_stats', {})
    if not isinstance(stats, dict):
        raise ValueError('invalid coverage-reuse statistics')
    ratio = stats.get('max_joint_average_to_original_weight_ratio')
    if (checkpoint.get('average_estimator_schema') != ESTIMATOR
            or checkpoint.get('future_average_weighting_schema') != WEIGHTING
            or not all(map(_integer, (origin, started, iterations, joint_start, completed)))
            or not 0 <= joint_start <= origin <= started <= iterations
            or completed != iterations - origin
            or checkpoint.get('traversal_stats_started_at') != started
            or type(checkpoint.get('traversal_stats_started_at')) is not int
            or not _integer(stats.get('iterations')) or stats['iterations'] != iterations - started
            or type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1 + 1e-12):
        raise ValueError('invalid coverage-reuse checkpoint metadata')
    fork = checkpoint.get('recipe_fork')
    if fork is None:
        if origin != 0 or started != 0:
            raise ValueError('coverage-reuse clocks require a preserved fork chain')
        return origin
    if (not isinstance(fork, dict) or fork.get('parent_algorithm') != PARENT_ALGORITHM
            or fork.get('kind') != FORK_KIND or fork.get('iterations') != started):
        raise ValueError('coverage-reuse fork mismatch')
    previous = fork.get('previous_recipe_fork')
    if previous is None:
        if origin != 0:
            raise ValueError('future averaging origin requires preserved fork')
    elif (not isinstance(previous, dict) or previous.get('kind') != PARENT_FORK_KIND
            or previous.get('iterations') != origin):
        raise ValueError('future averaging clock/fork mismatch')
    return origin
