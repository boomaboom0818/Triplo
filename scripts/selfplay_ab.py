"""Duplicate self-play A/B for GuardedEquityPolicy variants.

Each deal is replayed with the test policy in every seat (6 rotations), and
every candidate sees exactly the same cards, seats and opponents. Because all
policies here are deterministic, the paired difference between candidates is
caused only by the candidate's own decisions.

    python -m scripts.selfplay_ab --deals 400 --stacks 400 --field guarded-old

Candidates:
  old        GuardedEquityPolicy at --baseline-ref (default HEAD)
  new        current GuardedEquityPolicy, Stack Press off
  new-press  current GuardedEquityPolicy, Stack Press on

Fields (the five opponents):
  guarded-old  five copies of the old guarded policy (comparable strength)
  rule-mix     tight/aggressive/station/shover/station rule bots (weaker field)
"""
import argparse
import json
import math
import subprocess
import sys
import types
from multiprocessing import Pool
from random import Random
from statistics import mean

from poker.game import Game
from poker.rule_opponents import RulePolicy
import poker.equity_policy as current

PLAYERS = 6
CANDIDATES = ('old', 'new', 'new-press')
FIELDS = ('guarded-old', 'rule-mix')
_old = None


def old_module(ref):
    source = subprocess.run(['git', 'show', f'{ref}:poker/equity_policy.py'],
                            check=True, capture_output=True, text=True).stdout
    module = types.ModuleType('poker._baseline_equity_policy')
    module.__package__ = 'poker'
    exec(compile(source, f'{ref}:poker/equity_policy.py', 'exec'), module.__dict__)
    return module


def make(name):
    if name == 'old':
        return _old.GuardedEquityPolicy()
    if name == 'new':
        return current.GuardedEquityPolicy()
    if name == 'new-press':
        return current.GuardedEquityPolicy(stack_press=True)
    raise ValueError(name)


def field(name):
    if name == 'guarded-old':
        return [make('old') for _ in range(PLAYERS - 1)]
    styles = ('tight-v1', 'aggressive-v1', 'station-v1', 'shover-v1', 'station-v1')
    return [RulePolicy(s) for s in styles]


class PressGame(Game):
    """Adds the client's Stack Press sizes (2x pot, half stack) for press seats."""
    def __init__(self, seed, stacks, press_seats):
        super().__init__(seed, PLAYERS, 0, stacks)
        self.press_seats = press_seats

    def observe(self):
        o = super().observe()
        s = self.state
        if o.player not in self.press_seats or not s.can_complete_bet_or_raise_to():
            return o
        stack = s.stacks[s.actor_index]
        if stack <= 200:
            return o
        lo = s.min_completion_betting_or_raising_to_amount
        hi = s.max_completion_betting_or_raising_to_amount
        bet, pot = max(s.bets), s.total_pot_amount
        extra = {min(hi, max(lo, n)) for n in (bet + 2 * pot, bet + max(2 * pot, stack // 2))}
        sizes = sorted({a for a in o.legal if isinstance(a, int)} |
                       {a for a in extra if s.can_complete_bet_or_raise_to(a)})
        return type(o)(**{**o.__dict__, 'legal': tuple(a for a in o.legal if isinstance(a, str)) + tuple(sizes)})


def hero_result(deal, seat, candidate, field_name, stack):
    others = field(field_name)
    hero = make(candidate)
    policies = others[:seat] + [hero] + others[seat:]
    press = {seat} if getattr(hero, 'stack_press', False) else set()
    g = PressGame(deal, [stack] * PLAYERS, press)
    while g.state.status:
        o = g.observe()
        g.act(policies[o.player].act(o))
    return g.payoff()[seat]


def run_deal(args):
    deal, field_name, stack, ref = args
    global _old
    if _old is None:
        _old = old_module(ref)
    # One block = this deal with the hero in all six seats; result in BB.
    return {c: sum(hero_result(deal, s, c, field_name, stack) for s in range(PLAYERS))
            for c in CANDIDATES}


def paired(a, b, seed):
    diff = [x - y for x, y in zip(a, b)]
    rng = Random(seed)
    boot = sorted(mean(rng.choices(diff, k=len(diff))) for _ in range(2000))
    to100 = 100 / PLAYERS  # each block is six hero hands
    return dict(bb_per_100=mean(diff) * to100,
                ci95=[boot[50] * to100, boot[1949] * to100],
                stderr=(math.sqrt(sum((d - mean(diff)) ** 2 for d in diff) / (len(diff) - 1) / len(diff)) * to100))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deals', type=int, default=400)
    p.add_argument('--seed', type=int, default=7_100_000)
    p.add_argument('--stacks', type=int, nargs='+', default=[200, 400], help='starting stack in units (2 = 1 BB)')
    p.add_argument('--field', choices=FIELDS, nargs='+', default=list(FIELDS))
    p.add_argument('--baseline-ref', default='HEAD')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--output', help='write JSON results here')
    a = p.parse_args()
    report = dict(deals=a.deals, seed=a.seed, baseline_ref=a.baseline_ref, results={})
    with Pool(a.workers) as pool:
        for field_name in a.field:
            for stack in a.stacks:
                jobs = [(a.seed + i, field_name, stack, a.baseline_ref) for i in range(a.deals)]
                rows = pool.map(run_deal, jobs, chunksize=4)
                series = {c: [r[c] for r in rows] for c in CANDIDATES}
                key = f'{field_name} @ {stack // 2}BB'
                to100 = 100 / PLAYERS
                report['results'][key] = dict(
                    hero_bb_per_100={c: mean(v) * to100 for c, v in series.items()},
                    new_minus_old=paired(series['new'], series['old'], a.seed),
                    press_minus_old=paired(series['new-press'], series['old'], a.seed),
                    press_minus_new=paired(series['new-press'], series['new'], a.seed),
                    blocks_changed={'new': sum(x != y for x, y in zip(series['new'], series['old'])),
                                    'new-press': sum(x != y for x, y in zip(series['new-press'], series['new']))})
                r = report['results'][key]
                print(f'\n{key}  ({a.deals} deals x 6 seats = {a.deals * 6} hero hands)', flush=True)
                for c, v in r['hero_bb_per_100'].items():
                    print(f'  {c:10s} {v:+8.1f} bb/100')
                for k in ('new_minus_old', 'press_minus_old', 'press_minus_new'):
                    d = r[k]
                    print(f'  {k:16s} {d["bb_per_100"]:+8.1f} bb/100  95% CI [{d["ci95"][0]:+.1f}, {d["ci95"][1]:+.1f}]')
                print(f'  deals where outcome differed: {r["blocks_changed"]}', flush=True)
    if a.output:
        with open(a.output, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    sys.exit(main())
