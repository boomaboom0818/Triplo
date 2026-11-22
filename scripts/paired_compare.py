"""Paired comparison of ``live_gate evaluate`` reports against a baseline report.

All reports must come from the same ``--seed`` and ``--deals`` so each block is the
same deal for every arm. Per-setup block differences are averaged with equal
weight over the setups; the standard error combines the per-setup errors.

    python -m scripts.paired_compare --baseline eval-base.json eval-a.json eval-b.json
"""
import argparse
import json
import math
from pathlib import Path


def paired(candidate, baseline):
    means, variances = [], []
    for name, case in candidate['cases'].items():
        d = [a - b for a, b in zip(case['blocks'], baseline['cases'][name]['blocks'], strict=True)]
        m = sum(d) / len(d)
        means.append(m)
        variances.append(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return sum(means) / len(means), math.sqrt(sum(variances)) / len(means), dict(zip(candidate['cases'], means))


def postflop_learned(report):
    cases = report['cases'].values()
    return sum(c['postflop_learned'] for c in cases) / sum(c['postflop_decisions'] for c in cases)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('candidates', type=Path, nargs='+')
    a = p.parse_args()
    base = json.loads(a.baseline.read_text())
    print(f"{'candidate':>24} {'diff bb/100':>12} {'SE':>6} {'95% CI':>18} {'postflop learned':>17}")
    for path in a.candidates:
        report = json.loads(path.read_text())
        if (report['seed'], report['deals']) != (base['seed'], base['deals']):
            raise SystemExit(f'{path}: seed/deals differ from the baseline; blocks are not paired')
        mean, se, _ = paired(report, base)
        print(f'{path.stem:>24} {mean:>+12.2f} {se:>6.2f} [{mean - 1.96 * se:>+7.2f},{mean + 1.96 * se:>+7.2f}] '
              f'{postflop_learned(report):>16.1%}')
    print(f"{'baseline':>24} {a.baseline.stem}, postflop learned {postflop_learned(base):.1%}")


if __name__ == '__main__':
    main()
