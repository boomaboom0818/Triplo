"""Public-history commitment limit for the existing tight preflop thresholds.

Hands the tight chart continues with may call up to 3 BB (score 7–8) or 10 BB
(score 9–11) in total, not that amount on every new price. Weaker learned
continuations are left alone: the chart already folds them, and replacing
those learned actions is a different policy. Premiums (score 12+) are uncapped.
This is a conservative candidate heuristic, not an optimal preflop strategy.
Apply it only with a verified complete ledger.
"""
from .rule_opponents import chen


def committed_preflop(o):
    committed = 1 if o.position == 0 else 2 if o.position == 1 else 0
    for _, position, street, action, _, owed in o.history:
        if position != o.position or street != 0:
            continue
        if isinstance(action, int):
            # Raises are total bets on the street, not additional chips.
            committed = action
        elif action == 'call':
            committed += owed
    return committed


def guard_preflop(o, proposed):
    if o.street or proposed == 'fold':
        return proposed
    score = chen(o.hole)
    if score >= 12 or score < 7:
        return proposed
    limit = 20 if score >= 9 else 6
    committed = committed_preflop(o)
    cost = proposed if isinstance(proposed, int) else committed + o.call
    if cost <= limit:
        return proposed
    if not o.call or committed + o.call <= limit:
        return 'call'
    return 'fold' if 'fold' in o.legal else 'call'
