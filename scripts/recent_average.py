"""Extract guarded continuation averages without changing training state.

Rows with >=64 new averaging visits use a fixed mixture of the completed and
recent normalized policies. Sparse rows retain the completed policy. Counts are
sampled averaging visits, not independent samples or confidence bounds. Inputs
must be raw exports from the same optimizer lineage; monotonicity checks cannot
prove ancestry. The output is for inference, never continuation training.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from poker.history_cfr import load
from scripts.live_gate import digest


def derive(completed, parent, mix=1., min_visits=64):
    if not 0 < mix <= 1 or type(min_visits) is not int or min_visits < 64:
        raise ValueError('mix in (0,1] and >=64 new visits required')
    if completed.get('inference_transform') or parent.get('inference_transform'):
        raise ValueError('raw accumulated averages required')
    for field in ('algorithm', 'abstraction', 'menu', 'players', 'stack_bb'):
        if completed.get(field) != parent.get(field):
            raise ValueError('incompatible ' + field)
    if completed['iterations'] <= parent['iterations']:
        raise ValueError('continuation must be newer')
    if not parent['sums'].keys() <= completed['sums'].keys():
        raise ValueError('parent rows missing')
    stats = dict(replaced=0, inherited=0, new_rows=0,
                 min_new_visits=min_visits, mix=mix)
    rows, visits = {}, {}
    for key, row in completed['sums'].items():
        old = parent['sums'].get(key, [0.] * len(row))
        if len(row) != len(old) or not row:
            raise ValueError('action coordinate mismatch')
        if any(not math.isfinite(v) or v < 0 for v in row + old):
            raise ValueError('invalid average mass')
        delta = [a - b for a, b in zip(row, old)]
        # Exported sums are rounded to .001. Reject decreases beyond rounding noise.
        if any(v < -.0011 for v in delta):
            raise ValueError('average mass decreased')
        delta = [max(0., v) for v in delta]
        end_count = completed['visits'].get(key, 0)
        old_count = parent['visits'].get(key, 0)
        if any(type(v) is not int or v < 0 for v in (end_count, old_count)):
            raise ValueError('invalid averaging visit count')
        count = end_count - old_count
        if count < 0:
            raise ValueError('visit count decreased')
        mass, recent_mass = sum(row), sum(delta)
        if count >= min_visits and recent_mass > 0 and mass > 0:
            rows[key] = [(1 - mix) * a / mass + mix * b / recent_mass
                         for a, b in zip(row, delta)]
            visits[key] = count
            stats['replaced'] += 1
            stats['new_rows'] += key not in parent['sums']
        else:
            rows[key] = row
            visits[key] = end_count
            stats['inherited'] += 1
    result = {**completed, 'sums': rows, 'visits': visits, 'regrets': {},
              'policy_only': True,
              'inference_transform': dict(version='recent-average-v1',
                                          parent_iterations=parent['iterations'], **stats)}
    return result, stats


def run(completed, parent, out, mix=1.):
    out = Path(out)
    if out.exists():
        raise ValueError('preserve existing policy')
    completed_hash, parent_hash = digest(completed), digest(parent)
    result, stats = derive(load(completed), load(parent), mix)
    result['inference_transform'].update(completed_sha256=completed_hash,
                                         parent_sha256=parent_hash)
    if digest(completed) != completed_hash or digest(parent) != parent_hash:
        raise RuntimeError('input changed')
    # Publish a complete file exclusively; failed writes never masquerade as policies.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=out.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(result, f, allow_nan=False)
        os.link(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink()
    print(json.dumps(stats), flush=True)
    return stats


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--completed', required=True, type=Path)
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mix', type=float, default=1.)
    a = p.parse_args()
    run(a.completed, a.parent, a.out, a.mix)
