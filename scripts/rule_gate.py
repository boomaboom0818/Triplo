"""Sanity gate: one checkpoint seat (live fallback) against five fixed rule opponents.

Every deal is replayed with the checkpoint in each of the six seats. The gate
passes only if the lower end of the 95% interval is above zero against every
rule style. Passing is necessary, never sufficient, evidence of strength.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from poker.cfr import CFRPolicy
from poker.config import load_checkpoint
from poker.evaluate import LiveFallbackPolicy
from poker.fast_game import FastGame
from poker.game import Game
from poker import lean_cfr
from poker.rule_opponents import RulePolicy, STYLES

SEED = 880000
ALWAYS_FOLD_BB_PER_100 = -25.0


def play_style(hero, style, deals, seed=SEED, lean=False):
    """Per-deal mean bb/100 of the hero over the six seat rotations.

    lean=True plays the same deals on the fast engine with the lean raise menu,
    which is the game lean checkpoints are trained on.
    """
    values, shoves = [], 0
    opponent = RulePolicy(style)
    for deal in range(deals):
        block = []
        for seat in range(6):
            game = FastGame(seed + deal, lean=True) if lean else Game(seed + deal)
            while game.live if lean else game.state.status:
                o = game.observe()
                action = (hero if o.player == seat else opponent).act(o)
                shoves += o.player == seat and not o.street and isinstance(action, int) and action >= 100
                game.act(action)
            block.append(game.payoff()[seat])
        values.append(statistics.mean(block) * 100)
    return values, shoves


def summarize(values, shoves):
    mean = statistics.mean(values)
    error = statistics.stdev(values) / len(values) ** .5
    return dict(deals=len(values), hands=6 * len(values), bb_per_100=mean, standard_error=error,
                interval_95=[mean - 1.96 * error, mean + 1.96 * error],
                preflop_allins_per_100_hands=100 * shoves / (6 * len(values)),
                passed=mean - 1.96 * error > 0)


def run(checkpoint, deals, out=None):
    if deals < 2:
        raise ValueError('at least two deals required')
    if out is not None and Path(out).exists():
        raise ValueError('preserve existing gate report')
    lean = json.loads(Path(checkpoint).read_text()).get('algorithm') == lean_cfr.ALGORITHM
    saved = lean_cfr.load(checkpoint) if lean else load_checkpoint(checkpoint, model_id='bot-opponent-v1')
    styles = {}
    for style in STYLES:
        if lean:
            hero = lean_cfr.LeanPolicy(saved['sums'], 5)
        else:
            hero = LiveFallbackPolicy(CFRPolicy(saved['sums'], 5, saved['abstraction']))
        styles[style] = summarize(*play_style(hero, style, deals, lean=lean))
        if lean:
            styles[style]['coverage'] = hero.covered / hero.decisions
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  iterations=saved['iterations'], algorithm=saved['algorithm'], seed=SEED,
                  always_fold_bb_per_100=ALWAYS_FOLD_BB_PER_100, styles=styles,
                  passed=all(row['passed'] for row in styles.values()),
                  note='Rule opponents are weak fixed sanity players. Passing is necessary, not sufficient.')
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--deals', type=int, default=1500)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.deals, args.out), indent=2))
