"""Reconstruct exact public contributions in normalized chips."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BettingContext:
    paid: tuple
    bets: tuple


def betting_context(o):
    """Replay ordinary 1/2 blinds and reconcile the pot and call price."""
    n = len(o.stacks)
    if n < 3 or len(o.active) != n or o.street not in range(4):
        raise ValueError('unsupported public state')
    if len(o.board) != (0, 3, 4, 5)[o.street]:
        raise ValueError('street disagrees with board')
    paid = [1, 2] + [0] * (n - 2)
    bets = paid.copy()
    street = 0
    for _, pos, st, action, pot, call in o.history:
        if not street <= st <= o.street or not 0 <= pos < n or sum(paid) != pot:
            raise ValueError('incomplete public ledger')
        if st != street:
            bets = [0] * n
            street = st
        if action == 'fold':
            continue
        if type(action) is int:
            if action <= max(bets):
                raise ValueError('raise-to does not increase price')
            delta = action - bets[pos]
        elif action == 'call':
            if not 0 <= call <= max(bets) - bets[pos]:
                raise ValueError('invalid recorded call')
            delta = call
        else:
            raise ValueError('unknown ledger action')
        bets[pos] += delta
        paid[pos] += delta
    if street != o.street:
        bets = [0] * n
    if sum(paid) != o.pot:
        raise ValueError('pot disagrees with complete ledger')
    if min(o.stacks) < 0:
        raise ValueError('negative stack')
    if min(o.stacks[o.position], max(bets) - bets[o.position]) != o.call:
        raise ValueError('call disagrees with public ledger')
    return BettingContext(tuple(paid), tuple(bets))
