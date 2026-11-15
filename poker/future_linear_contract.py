"""Strict dispatch metadata for preserved-history future-linear averaging."""
import math

ALGORITHM='future-linear-average-mccfr-v8'
AVERAGING='future-linear-joint-average-v8'
ESTIMATOR='joint-inclusion-future-linear-v8'
WEIGHTING='inherited-uniform-future-linear-v1'


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='joint-average-v7' or new.pop('traversal',None)!='future-linear-average-v8':return False
    if old.pop('averaging_sampler',None)!='regret-path-mis-average-v7' or new.pop('averaging_sampler',None)!=AVERAGING:return False
    return old==new


def validate(checkpoint):
    origin=checkpoint.get('future_average_origin');iterations=checkpoint.get('iterations')
    joint_start=checkpoint.get('joint_average_diagnostics_started_at')
    completed=checkpoint.get('future_average_completed_iterations')
    stats=checkpoint.get('traversal_stats',{})
    if not isinstance(stats,dict):raise ValueError('invalid future averaging statistics')
    ratio=stats.get('max_joint_average_to_original_weight_ratio')
    if (checkpoint.get('average_estimator_schema')!=ESTIMATOR or checkpoint.get('future_average_weighting_schema')!=WEIGHTING
            or type(origin) is not int or type(iterations) is not int or not 0<=origin<=iterations
            or type(joint_start) is not int or not 0<=joint_start<=origin
            or type(completed) is not int or completed!=iterations-origin
            or type(checkpoint.get('traversal_stats_started_at')) is not int or checkpoint.get('traversal_stats_started_at')!=origin
            or type(stats.get('iterations')) is not int or stats.get('iterations')!=completed
            or type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0<=ratio<=1+1e-12):
        raise ValueError('invalid future averaging checkpoint metadata')
    fork=checkpoint.get('recipe_fork')
    if fork is None:
        if origin!=0:raise ValueError('future averaging origin requires preserved fork')
    elif (not isinstance(fork,dict) or fork.get('parent_algorithm')!='joint-traversal-average-mccfr-v7' or fork.get('iterations')!=origin
            or fork.get('kind')!='explicit future-linear-average-v8 fork'):
        raise ValueError('future averaging clock/fork mismatch')
    return origin
