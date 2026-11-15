"""One preregistered high-power payoff confirmation of the existing v10 checkpoints.

Approved by the user on 2026-11-15 after a read-only power analysis showed the
2000-block pilot could only resolve about 130 bb/100. Evaluation only: no
training, no checkpoint changes. The 2000-block pilot failure stays a failure.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from poker.compare import paired_difference
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts import scaling_validation as workers
from scripts.sustained_train import digest

ROOT = Path('runs/bot/payoff-confirmation-001')
CANDIDATES = [Path(f'runs/bot/coverage-reuse-training-001/candidate-{i}-plus-2000.json') for i in range(2)]
PARENTS = [Path(f'runs/bot/future-linear-training-001/candidate-{i}-plus-2000.json') for i in range(2)]
CRITERIA = Path('docs/PAYOFF-CONFIRMATION.md')
SEED = 15000000
BLOCKS = 20000
PARENT_CONFIDENCE = .9875


def judge(pair, uniform, key):
    """Frozen gates for one learner/opponent case."""
    return dict(parent_gain_pass=pair[key]['interval'][0] > 0,
                uniform_gain_pass=uniform[key]['ci95'][0] > 0,
                material_regression=any(v['interval'][1] < -10 for v in pair.values()))


def passed(cases):
    return bool(cases) and all(c['parent_gain_pass'] and c['uniform_gain_pass'] and not c['material_regression']
                               for c in cases)


def main(deadline, resume=False):
    deadline = datetime.fromisoformat(deadline)
    if deadline.tzinfo is None or not time.time() < deadline.timestamp() <= time.time() + 12 * 3600:
        raise ValueError('finite invocation deadline within 12h required')
    ROOT.mkdir(parents=True, exist_ok=True)
    workers.ROOT = ROOT
    opponents = [workers.VALIDATION, *workers.ROBUSTNESS]
    with run_lock(ROOT / 'supervisor'):
        manifest = dict(seed=SEED, blocks=BLOCKS, parent_confidence=PARENT_CONFIDENCE, deadline=deadline.isoformat(),
                        candidates=[str(p) for p in CANDIDATES], parents=[str(p) for p in PARENTS],
                        inputs=[dict(path=str(p.resolve()), sha256=digest(p)) for p in [*CANDIDATES, *PARENTS, *opponents]],
                        criteria=str(CRITERIA), criteria_sha256=digest(CRITERIA),
                        sources={str(p): digest(p) for p in [*sorted(Path('poker').glob('*.py')), Path(__file__),
                                                              Path(workers.__file__), Path('scripts/limited_worker.py'),
                                                              Path('scripts/sustained_train.py'), Path('configs/bot.json')]})
        path = ROOT / 'manifest.json'
        if path.exists():
            if not resume or json.loads(path.read_text()) != manifest:
                raise ValueError('existing/mismatched confirmation manifest')
        elif resume:
            raise ValueError('missing confirmation manifest')
        else:
            atomic_save(path, manifest)

        def verify():
            for row in manifest['inputs']:
                if digest(Path(row['path'])) != row['sha256']:
                    raise ValueError('frozen confirmation input changed')
            for source, sha in manifest['sources'].items():
                if digest(Path(source)) != sha:
                    raise ValueError('frozen confirmation code changed')
            if digest(CRITERIA) != manifest['criteria_sha256']:
                raise ValueError('confirmation criteria changed')
        atomic_save(ROOT / 'invocation.json', dict(pid=os.getpid(), resume=resume, deadline=deadline.isoformat()))
        try:
            verify()
            requests = []
            for i in range(2):
                for population, opponent in [('validation', workers.VALIDATION), ('independent', workers.ROBUSTNESS[i])]:
                    for kind, checkpoint in [('candidate', CANDIDATES[i]), ('baseline', PARENTS[i]), ('uniform', CANDIDATES[i])]:
                        requests.append(dict(checkpoint=checkpoint, opponent=opponent, uniform=kind == 'uniform',
                                             output=ROOT / f'candidate-{i}-{population}-{kind}.json'))
            atomic_save(ROOT / 'status.json', dict(status='evaluating', requests=len(requests)))
            for index in range(0, len(requests), 2):
                workers.evaluate_parallel(requests[index:index + 2], deadline.timestamp(), seed=SEED, blocks=BLOCKS)
            verify()
            evidence = {}
            cases = []
            for i in range(2):
                evidence[str(i)] = {}
                for population in ('validation', 'independent'):
                    c, b, u = [json.loads((ROOT / f'candidate-{i}-{population}-{kind}.json').read_text())
                               for kind in ('candidate', 'baseline', 'uniform')]
                    key = c['opponent_checkpoint_hashes'][0]
                    pair = paired_difference(c, b, confidence=PARENT_CONFIDENCE)
                    uniform = paired_difference(c, u)
                    case = judge(pair, uniform, key)
                    cases.append(case)
                    evidence[str(i)][population] = dict(paired=pair, uniform=uniform, **case)
            atomic_save(ROOT / 'payoff-evidence.json', evidence)
            atomic_save(ROOT / 'status.json', dict(status='completed_pending_review', payoff_pass=passed(cases),
                                                   competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT / 'status.json', dict(status='stopped', reason=str(exc) or type(exc).__name__))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.deadline, args.resume)
