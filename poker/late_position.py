"""Widen first-in opens only for validated six-max lineups."""
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from .rule_opponents import chen

SCHEMA = 1
BEFORE = {4: (2, 3), 5: (2, 3, 4), 0: (2, 3, 4, 5)}
BEHIND = {4: (5, 0, 1), 5: (0, 1), 0: (1,)}
MIN_CHEN = {4: 5.5, 5: 4.5, 0: 6}
MIN_STACK_BB = 20
MIN_TRAIN, MIN_HOLDOUT = 30, 15
MARGIN = .03


def valid_lineup(position, names):
    return (type(position) is int and position in BEHIND and len(names) == len(BEHIND[position])
            and all(isinstance(name, str) and name.strip() for name in names) and len(set(names)) == len(names))


def lower_bound(folds, total):
    """One-sided 95% Wilson lower bound for a joint fold frequency."""
    if not total:
        return 0.
    z = 1.64485362695147
    p = folds / total
    return (p + z*z/(2*total) - z*sqrt((p*(1-p) + z*z/(4*total))/total)) / (1 + z*z/total)


def required_fold_rate(risk, pot):
    return risk / (pot + risk) + MARGIN


@dataclass(frozen=True)
class FoldEvidence:
    train: tuple
    holdout: tuple

    def lower(self):
        if self.train[1] < MIN_TRAIN or self.holdout[1] < MIN_HOLDOUT:
            return None
        return min(lower_bound(*self.train), lower_bound(*self.holdout))


class LatePositionSteals:
    def __init__(self, hero, evidence):
        self.hero = hero
        self.evidence = evidence

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if (not isinstance(data, dict) or data.get('schema') != SCHEMA or not isinstance(data.get('hero'), str)
                or not data['hero'].strip() or not isinstance(data.get('lineups'), list)):
            raise ValueError('incompatible late-position evidence file')
        evidence = {}
        for row in data['lineups']:
            if (not isinstance(row, dict) or not isinstance(row.get('behind'), list)
                    or not valid_lineup(row.get('position'), row['behind']) or data['hero'] in row['behind']):
                raise ValueError('invalid late-position opponent lineup')
            key = row['position'], tuple(row['behind'])
            if key in evidence:
                raise ValueError('duplicate late-position opponent lineup')
            counts = [row.get(k) for k in ('train', 'holdout')]
            if any(not isinstance(c, list) or len(c) != 2 or any(type(n) is not int for n in c)
                   or not 0 <= c[0] <= c[1] for c in counts):
                raise ValueError('invalid late-position fold counts')
            evidence[key] = FoldEvidence(*(tuple(c) for c in counts))
        return cls(data['hero'], evidence)

    def propose(self, o, action, seats, bb, pot, wire):
        """Return (normalized action, diagnostics), or None to keep v3.

        Seats are in small-blind-first order. The client calls this only after
        verifying the full action prefix; wire sizes and bets are in raw chips.
        """
        if action != 'fold' or o.street != 0 or len(seats) != 6 or o.position not in BEFORE or not o.call:
            return None
        pos = o.position
        if (tuple(h[1] for h in o.history) != BEFORE[pos]
                or any(h[2:4] != (0, 'fold') for h in o.history)
                or tuple(i for i, on in enumerate(o.active) if on) != tuple(sorted((pos, *BEHIND[pos])))
                or seats[pos].get('name') != self.hero or chen(o.hole) < MIN_CHEN[pos]):
            return None
        # Match the same ordinary 1/2 blind structure and exact 2.5 BB size
        # used in fitting, not a lookalike produced by chip rounding.
        if (bb <= 0 or 2*pot != 3*bb or 2*seats[0]['bet'] != bb or seats[1]['bet'] != bb
                or any(s['bet'] for s in seats[2:])
                or any(seats[i]['stack'] + seats[i]['bet'] < MIN_STACK_BB*bb for i in (pos, *BEHIND[pos]))):
            return None
        target = next((a for a in o.legal if type(a) is int and wire[a]['action'] == 'raise'
                       and 2*wire[a]['amount'] == 5*bb), None)
        if target is None:
            return None
        names = tuple(seats[i].get('name') for i in BEHIND[pos])
        if not valid_lineup(pos, names) or self.hero in names:
            return None
        evidence = self.evidence.get((pos, names))
        lower = evidence.lower() if evidence else None
        risk = wire[target]['amount'] - seats[pos]['bet']
        required = required_fold_rate(risk, pot)
        if lower is None or lower <= required:
            return None
        return target, dict(position=pos, train=evidence.train, holdout=evidence.holdout,
                            fold_lower=round(lower, 4), required=round(required, 4),
                            proposed_amount=wire[target]['amount'])
