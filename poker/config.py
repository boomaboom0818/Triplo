"""Strict loaders for the currently supported policy and checkpoint contracts."""
import json
import math
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    if (config.get('schema') != 1 or config.get('mode') not in ('bot', 'human')
            or config.get('policy') != 'blueprint'
            or config.get('adaptation') != 'disabled_pending_validation'):
        raise ValueError('unsupported policy configuration')
    return config


def load_checkpoint(path):
    checkpoint = json.loads(Path(path).read_text())
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
