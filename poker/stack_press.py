"""Stack Press: opt-in extra raise sizes that only the fallback may use.

The learned policy is keyed on the trained action menu, so the extra sizes must
never reach its lookup. They are offered to the fallback only, after the
learned policy has declined the state. Shared by the live adapter
(scripts/openpoker_client.py) and the offline gate (scripts/live_gate.py) so
both exercise the same path.

All amounts are normalized chips (1 = small blind); sizes are raise-to totals.
"""
from dataclasses import replace

# Effective stack (normalized chips behind) above which the sizes are offered: 100 BB.
DEPTH = 200


def enabled(policy):
    return bool(getattr(getattr(policy, 'fallback', None), 'stack_press', False))


def targets(bet, pot, hero_bet, hero_stack, opponents):
    """Raise-to targets (2x pot and half the effective stack), or () if not meaningful.

    ``opponents`` are (street bet, stack behind) pairs of every other seat
    still in the hand. Sizing uses the effective stack -- what the deepest
    opponent can still match -- not hero's own stack, and never exceeds it.
    """
    cover = min(hero_bet + hero_stack, max((b + s for b, s in opponents), default=0))
    effective = cover - hero_bet
    if effective <= DEPTH:
        return ()
    return tuple(sorted({min(cover, bet + 2 * pot), min(cover, bet + max(2 * pot, effective // 2))}))


def sizes(street, bet, pot, lo, hi, hero_bet, hero_stack, opponents, existing):
    """Extra raise-to sizes within [lo, hi] that are not already on the menu.

    Postflop only: the fallback's Stack Press logic acts only postflop, and
    preflop rules choose among raises by position in the menu.
    """
    if not street:
        return ()
    extra = {min(hi, max(lo, t)) for t in targets(bet, pot, hero_bet, hero_stack, opponents)}
    return tuple(sorted(extra - set(existing)))


class _PressFallback:
    """Proxy that widens the menu for the wrapped fallback only."""
    def __init__(self, inner, extra):
        self.inner, self.extra = inner, tuple(extra)

    def act(self, o):
        raises = sorted({a for a in o.legal if isinstance(a, int)} | set(self.extra))
        legal = tuple(a for a in o.legal if not isinstance(a, int)) + tuple(raises)
        return self.inner.act(replace(o, legal=legal))

    def __getattr__(self, name):
        return getattr(self.inner, name)


def act(policy, o, extra, **kwargs):
    """policy.act(o, **kwargs) with ``extra`` raise sizes visible to its fallback only."""
    if not extra:
        return policy.act(o, **kwargs)
    inner = policy.fallback
    policy.fallback = _PressFallback(inner, extra)
    try:
        return policy.act(o, **kwargs)
    finally:
        policy.fallback = inner
