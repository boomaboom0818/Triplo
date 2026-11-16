"""Compare sparse-row fallback, raw learning and fallback alone on paired deals.

This is a screening experiment, not a training job or a competitive-strength gate.
Choose the seed and threshold before running; do not select them from the results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from poker.lean_cfr import LeanPolicy, load
from poker.rule_opponents import STYLES
from scripts.rule_gate import play_style, summarize


def run(checkpoint, deals, seed, min_support, out):
    if deals < 2:
        raise ValueError('at least two independent deals required')
    if not math.isfinite(min_support) or min_support <= 0:
        raise ValueError('positive finite minimum support required')
    if out.exists():
        raise ValueError('preserve existing report')
    saved = load(checkpoint)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  iterations=saved['iterations'], seed=seed, deals=deals, min_support=min_support,
                  support_definition='average row mass / sum(1..iterations); not independent visits or ESS',
                  note='Screening only; pointwise 95% normal intervals over whole deal blocks. No multiple-test correction.',
                  styles={})
    for style in STYLES:
        policies = dict(raw=LeanPolicy.from_checkpoint(saved, 5),
                        supported=LeanPolicy.from_checkpoint(saved, 5, min_support),
                        fallback_only=LeanPolicy({}, 5))
        results, samples = {}, {}
        for name, policy in policies.items():
            values, shoves = play_style(policy, style, deals, seed=seed, lean=True)
            samples[name] = values
            results[name] = dict(summarize(values, shoves),
                                 learned_decision_fraction=policy.covered / policy.decisions,
                                 thin_decisions=policy.thin, decisions=policy.decisions,
                                 deal_block_bb_per_100=values)
        differences = {}
        for control in ('raw', 'fallback_only'):
            values = [a - b for a, b in zip(samples['supported'], samples[control])]
            summary = summarize(values, 0)
            differences[control] = {k: summary[k] for k in ('bb_per_100', 'standard_error', 'interval_95')}
        report['styles'][style] = dict(policies=results, supported_minus=differences)
    # Exclusive create also prevents overwriting a report created during evaluation.
    with out.open('x') as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--deals', type=int, default=1500)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--min-support', type=float, required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.checkpoint, args.deals, args.seed, args.min_support, args.out)
    print(json.dumps({style: row['supported_minus'] for style, row in report['styles'].items()}, indent=2))
