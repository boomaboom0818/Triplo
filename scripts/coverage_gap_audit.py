"""Read-only natural-trajectory coverage diagnosis; never a payoff experiment."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time

from poker.cfr import CFRPolicy, information_key, normalize
from poker.config import load_checkpoint
from poker.game import play
from poker.policy import Policy
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def positive(row):
    return row is not None and any(x > 0 for x in row)


class CoverageTrace(CFRPolicy):
    def __init__(self, state, reference, seed, cells, gaps):
        super().__init__(state['sums'], seed, state['abstraction'])
        self.state, self.reference = state, reference
        self.cells, self.gaps = cells, gaps
        self.log_current_reach = 0.
        self.zero_current_reach = False

    def act(self, observation):
        key = information_key(observation, self.abstraction)
        row = self.sums.get(key)
        source = 'positive' if positive(row) else 'missing' if row is None else 'zero'
        label = f'owner={observation.player}:street={observation.street}:active={sum(observation.active)}'
        counts = self.cells[label]
        counts['decisions'] += 1
        counts[source] += 1
        regrets = self.state['regrets'].get(key)
        counts['current_regret_present'] += regrets is not None
        counts['current_regret_positive'] += positive(regrets)
        counts['reference_average_positive'] += positive(self.reference['sums'].get(key))
        counts['reference_regret_positive'] += positive(self.reference['regrets'].get(key))
        if source != 'positive':
            counts['gap_zero_current_own_reach'] += self.zero_current_reach
            counts['gap_tiny_nonzero_current_own_reach'] += not self.zero_current_reach and self.log_current_reach < math.log(1e-6)
            counts['gap_positive_current_own_reach'] += not self.zero_current_reach
            counts['gap_regret_missing'] += regrets is None
            counts['gap_regret_positive'] += positive(regrets)
            counts['gap_reference_average_positive'] += positive(self.reference['sums'].get(key))
            self.gaps[(label, hashlib.sha256(key.encode()).hexdigest())] += 1
        action = super().act(observation)
        current = normalize(regrets if regrets is not None else [0.] * len(observation.legal))
        probability = current[observation.legal.index(action)]
        if probability == 0:
            self.zero_current_reach = True
        else:
            self.log_current_reach += math.log(probability)
        return action


def empty_cells():
    return {f'owner={o}:street={s}:active={a}': Counter()
            for o in range(6) for s in range(4) for a in range(2, 7)}


def policies(state, reference, population, seed, block, owner, cells=None, gaps=None):
    ps = [Policy(seed + block*100 + owner*6 + i, {},
                 ['calling', 'tight', 'aggressive'][i % 3] if population == 'mixed' else population)
          for i in range(6)]
    rng_seed = seed + block*100 + owner*6 + owner
    ps[owner] = (CFRPolicy(state['sums'], rng_seed, state['abstraction']) if cells is None
                 else CoverageTrace(state, reference, rng_seed, cells, gaps))
    return ps


def audit(root, candidate, deadline):
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:
        raise ValueError('deadline must include timezone')
    end = parsed.timestamp()
    if not time.time() < end <= time.time() + 3600:
        raise ValueError('finite future invocation deadline within one hour required')
    resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
    os.nice(10)
    def expired(signum, frame):
        raise TimeoutError('coverage diagnosis deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, math.ceil(end-time.time())))
    started = time.monotonic()
    minimum = available_memory()
    status = 'failed'
    with run_lock(root/f'candidate-{candidate}'):
        output = root/f'candidate-{candidate}.json'
        if output.exists() or (root/f'candidate-{candidate}.metrics.json').exists():
            raise ValueError('preserve previous attempts; investigate before recovery')
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['deadline'] != deadline:
            raise ValueError('invocation differs from frozen deadline')
        for path, sha in manifest['sources'].items():
            assert digest(path) == sha, path
        for row in manifest['inputs']:
            assert digest(row['path']) == row['sha256']
        try:
            inputs = manifest['inputs'][candidate*2:candidate*2+2]
            states = [load_checkpoint(row['path']) for row in inputs]
            assert states[0]['abstraction'] == states[1]['abstraction']
            assert states[1]['recipe_fork']['parent_sha256'] == inputs[0]['sha256']
            result = {}
            for variant, state in enumerate(states):
                other = states[1-variant]
                populations = {}
                for population, blocks in manifest['blocks'].items():
                    cells, gaps = empty_cells(), Counter()
                    exact_pairs = 0
                    for block in range(blocks):
                        if time.time() >= end:
                            raise TimeoutError('coverage diagnosis deadline reached')
                        minimum = min(minimum, available_memory())
                        if minimum < 2*1024**3 or shutil.disk_usage(root).free < 10*1024**3:
                            raise RuntimeError('host reserve breached')
                        for owner in range(6):
                            ps = policies(state, other, population, manifest['seed'], block, owner, cells, gaps)
                            payoff, traces, _ = play(ps, manifest['seed']+block, button=0)
                            if block < manifest['exact_replay_blocks']:
                                baseline = policies(state, other, population, manifest['seed'], block, owner)
                                expected_payoff, expected_traces, _ = play(baseline, manifest['seed']+block, button=0)
                                assert payoff == expected_payoff and traces == expected_traces
                                assert ps[owner].rng.getstate() == baseline[owner].rng.getstate()
                                exact_pairs += 1
                        if block % 50 == 0:
                            atomic_save(root/f'candidate-{candidate}.progress.json', dict(
                                variant=variant, population=population, blocks_completed=block+1,
                                total_blocks=blocks, deadline=deadline))
                    for counts in cells.values():
                        assert counts['decisions'] == sum(counts[k] for k in ('positive','missing','zero'))
                        assert counts['missing']+counts['zero'] == counts['gap_zero_current_own_reach']+counts['gap_positive_current_own_reach']
                    populations[population] = dict(blocks=blocks, rotations=6, cells={k:dict(v) for k,v in cells.items()},
                        gap_keys=[dict(cell=k[0],sha256=k[1],visits=v) for k,v in sorted(gaps.items())],
                        exact_action_trace_rng_pairs=exact_pairs)
                result['parent' if variant == 0 else 'candidate'] = populations
            for row in manifest['inputs']:
                assert digest(row['path']) == row['sha256']
            for path, sha in manifest['sources'].items():
                assert digest(path) == sha
            atomic_save(output, dict(candidate=candidate, manifest_sha256=digest(root/'manifest.json'),
                inputs=inputs, variants=result, read_only=True,
                caveat='Fresh natural scripted trajectories only, all six roles. No payoffs reported or training updates. Reference rows queried on the same observations. Zero CURRENT own reach is distinct from sparse sampling and tiny positive reach. Unvisited cells remain explicit zeros; no forced contexts or heldout checkpoint inputs.'))
            status = 'completed'
        finally:
            peaks = {l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines()
                     if l.startswith(('VmPeak:', 'VmHWM:'))}
            atomic_save(root/f'candidate-{candidate}.metrics.json',dict(status=status,pid=os.getpid(),
                wall_seconds=time.monotonic()-started,deadline=deadline,
                address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                minimum_host_available_bytes=min(minimum,available_memory()),**peaks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', type=int, choices=(0,1), required=True)
    parser.add_argument('--deadline', required=True)
    args = parser.parse_args()
    audit(args.root, args.candidate, args.deadline)
