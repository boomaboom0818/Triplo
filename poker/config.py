"""Strict loaders for the currently supported policy and checkpoint contracts."""
import json
import math
from pathlib import Path


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
    if checkpoint.get('algorithm') == 'external-sampling-mccfr-v1':
        if (checkpoint.get('schema') != 1 or checkpoint.get('engine') != 'pokerkit-0.7.7'
                or checkpoint.get('abstraction') not in ('exact-observation-v1','texture-buckets-v1')
                or checkpoint.get('config', {}).get('game') != 'holdem'
                or checkpoint.get('config', {}).get('players') != 6):
            raise ValueError('evaluation requires a six-player holdem CFR checkpoint')
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
