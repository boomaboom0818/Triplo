"""Strict loaders for the currently supported policy and checkpoint contracts."""
import json
import math
from .future_linear_contract import validate as validate_future
from pathlib import Path

CFR_ALGORITHMS=('external-sampling-mccfr-v1','outcome-sampling-mccfr-v2','prefix-outcome-mccfr-v3','joint-traversal-average-mccfr-v7','future-linear-average-mccfr-v8')


def load_config(path):
    config = json.loads(Path(path).read_text())
    if (config.get('schema') != 1 or config.get('mode') not in ('bot', 'human')
            or config.get('policy') != 'blueprint'
            or config.get('model_id') != config.get('mode') + '-opponent-v1'
            or config.get('status') != ('active' if config.get('mode') == 'bot' else 'deferred')
            or config.get('objective') != 'maximize_expected_winnings'
            or config.get('adaptation') != 'disabled_pending_validation'):
        raise ValueError('unsupported policy configuration')
    return config


def load_checkpoint(path, *, model_id=None):
    checkpoint = json.loads(Path(path).read_text())
    if model_id is not None and checkpoint.get('model_id') != model_id:
        raise ValueError('checkpoint model identity mismatch; legacy artifacts are unassigned')
    if checkpoint.get('algorithm') in CFR_ALGORITHMS:
        if (checkpoint.get('schema') != 1 or checkpoint.get('engine') != 'pokerkit-0.7.7'
                or checkpoint.get('abstraction') not in ('exact-observation-v1','texture-buckets-v1','texture-buckets-v2')
                or checkpoint.get('config', {}).get('game') != 'holdem'
                or checkpoint.get('config', {}).get('players') != 6):
            raise ValueError('evaluation requires a six-player holdem CFR checkpoint')
        cfg=checkpoint['config']
        if checkpoint['algorithm'] in ('outcome-sampling-mccfr-v2','prefix-outcome-mccfr-v3','joint-traversal-average-mccfr-v7','future-linear-average-mccfr-v8'):
            samples=cfg.get('trajectory_samples')
            future=checkpoint['algorithm']=='future-linear-average-mccfr-v8'
            joint=checkpoint['algorithm']=='joint-traversal-average-mccfr-v7'
            expected='future-linear-average-v8' if future else 'joint-average-v7' if joint else 'prefix-v3' if checkpoint['algorithm']=='prefix-outcome-mccfr-v3' else 'trajectory-v2'
            sampler='future-linear-joint-average-v8' if future else 'regret-path-mis-average-v7' if joint else 'trajectory-call-mixture-v2'
            if (cfg.get('traversal')!=expected or cfg.get('averaging_sampler')!=sampler
                    or type(samples) is not int or not 1<=samples<=64):
                raise ValueError('invalid trajectory checkpoint metadata')
            if expected in ('prefix-v3','joint-average-v7','future-linear-average-v8') and (type(cfg.get('regret_prefix_depth')) is not int or cfg['regret_prefix_depth'] not in (1,2)):
                raise ValueError('invalid prefix checkpoint metadata')
            if future:validate_future(checkpoint)
            if joint:
                start=checkpoint.get('joint_average_diagnostics_started_at');iterations=checkpoint.get('iterations')
                stats=checkpoint.get('traversal_stats',{})
                ratio=stats.get('max_joint_average_to_original_weight_ratio') if isinstance(stats,dict) else None
                if (checkpoint.get('average_estimator_schema')!='joint-inclusion-v7' or type(start) is not int or type(iterations) is not int or not 0<=start<=iterations
                        or type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0<=ratio<=1+1e-12):
                    raise ValueError('invalid joint average checkpoint metadata')
        elif cfg.get('traversal','external-v1')!='external-v1' or cfg.get('averaging_sampler','uniform') not in ('uniform','call-mixture-v1'):
            raise ValueError('CFR algorithm/traversal mismatch')
        for name in ('regrets','sums'):
            rows=checkpoint.get(name)
            if not isinstance(rows,dict): raise ValueError('invalid CFR table')
            for row in rows.values():
                if (not isinstance(row,list) or not 1 <= len(row) <= 6
                        or any(type(x) not in (int,float) or not math.isfinite(x) for x in row)
                        or (name=='sums' and any(x<0 for x in row))):
                    raise ValueError('invalid CFR values')
        return checkpoint
    expected = dict(schema=1, algorithm='tabular-reinforce-v1',
                    engine='pokerkit-0.7.7', abstraction='rank-position-board-active-pot-v1',
                    stacks=200, players=6)
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError('incompatible checkpoint metadata')
    table = checkpoint.get('table')
    if not isinstance(table, dict):
        raise ValueError('checkpoint table must be an object')
    for row in table.values():
        if (not isinstance(row, list) or not 1 <= len(row) <= 6
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
            raise ValueError('invalid checkpoint logits')
    return checkpoint
