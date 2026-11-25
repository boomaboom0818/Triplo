"""Opt-in preflop exploit layer for tables of measured calling stations (no_code fish).

Wraps a policy (normally ``HistoryPolicy``). Preflop, when every opponent who can still
act in the hand or who raised is a measured fish (postflop fold base <= FISH_BASE),
it replaces the learned decision with a range rule built on what live no_code bots do:
they fold ~70% to an open, open 22-46% and call 3-bets with dominated hands, and
re-raise wide once a pot is 3-bet. Every other decision goes to the wrapped policy.

Triplo v9 adds two opt-in unopened-pot rules that post-process the wrapped policy's
decision (both default off, so the default wrapper is decision-identical to v8):

* ``open_bb``: when every opponent who can still act behind us and every limper is a
  leaderboard fish, an open (or iso-raise over limpers) is resized to
  ``open_bb + open_limper_bb * limpers`` big blinds. Live fish fold 65-74% to opens
  of every size and call ~25% at any size, so a bigger open collects more dead money.
  Only the amount changes; the learned policy still chooses whether to raise.
* ``limp_raise``: an open-limp (or over-limp) with a hand in the top ``limp_raise``
  share (Chen rank) becomes a raise. ``limp_raise_scope='fish'`` restricts it to the
  same all-fish condition as ``open_bb``.
"""
from functools import lru_cache
from hashlib import blake2b
from itertools import combinations

from .fast_eval import NAMES
from .rule_opponents import chen
from .river_solver_fallback import FISH_BASE


@lru_cache(maxsize=1)
def _chen_table():
    scores = sorted((chen((NAMES[a], NAMES[b])) for a, b in combinations(range(52), 2)), reverse=True)
    out, total = {}, len(scores)
    for i, s in enumerate(scores):
        out.setdefault(s, i / total)
    return out


def hand_rank(hole):
    """Share of holdings with a strictly better Chen score (0 = best)."""
    return _chen_table()[chen(tuple(hole))]


class FishPreflop:
    """Preflop range rules against measured fish; delegates everything else."""

    def __init__(self, inner, open_width=.55, steal_width=.70, iso_width=.12, call_width=.30,
                 vs3bet_width=.22, jam_width=.06, steal=True, iso=True,
                 open_bb=None, open_limper_bb=1.0, open_scope='all', limp_raise=None, limp_raise_scope='all'):
        object.__setattr__(self, 'inner', inner)
        object.__setattr__(self, 'steal', steal)
        object.__setattr__(self, 'iso', iso)
        if open_scope not in ('all', 'majority') or limp_raise_scope not in ('all', 'fish'):
            raise ValueError('unknown open or limp-raise scope')
        self.open_width = open_width
        self.steal_width = steal_width
        self.iso_width = iso_width
        self.call_width = call_width
        self.vs3bet_width = vs3bet_width
        self.jam_width = jam_width
        self.overrides = 0
        # Triplo v9 unopened-pot rules (opt-in; see the module docstring).
        self.open_bb = open_bb
        self.open_limper_bb = open_limper_bb
        self.open_scope = open_scope
        self.limp_raise = limp_raise
        self.limp_raise_scope = limp_raise_scope
        self.resized = 0
        self.limp_raises = 0
        self.last_open = None

    # Attribute access (fallback, rng, counters) goes to the wrapped policy, so the
    # client and Stack Press can swap ``fallback`` exactly as before.
    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in ('open_width', 'steal_width', 'iso_width', 'call_width', 'vs3bet_width', 'jam_width',
                    'overrides', 'open_bb', 'open_limper_bb', 'open_scope', 'limp_raise', 'limp_raise_scope',
                    'resized', 'limp_raises', 'last_open'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def _fish(self, pos):
        fallback = self.inner.fallback
        if hasattr(fallback, 'is_fish'):
            return fallback.is_fish(pos)
        p = fallback.profiles.get(pos)
        return isinstance(p, (tuple, list)) and p[0] <= FISH_BASE

    def act(self, o, **kwargs):
        action = self._rule(o) if not o.street else None
        if action is None:
            action = self.inner.act(o, **kwargs)
            if not o.street and (self.open_bb or self.limp_raise):
                action = self._unopened(o, action)
            return action
        self.overrides += 1
        return action

    def _unopened(self, o, action):
        """Triplo v9: raise instead of limping (``limp_raise``) and bigger opens (``open_bb``)."""
        object.__setattr__(self, 'last_open', None)
        if any(h[2] == 0 and isinstance(h[3], int) for h in o.history):
            return action
        legal_raises = [a for a in o.legal if isinstance(a, int)]
        if not legal_raises:
            return action
        n = len(o.active)
        acted = {h[1] for h in o.history if h[2] == 0}
        behind = [i for i in range(n) if i != o.position and o.active[i] and o.stacks[i] > 0 and i not in acted]
        limpers = [h[1] for h in o.history if h[2] == 0 and h[3] == 'call' and h[5] > 0]
        others = behind + [i for i in limpers if o.active[i]]
        fish = [i for i in others if self._fish(i)]
        all_fish = bool(others) and len(fish) == len(others)
        sized_ok = all_fish if self.open_scope == 'all' else bool(others) and 2 * len(fish) >= len(others)
        baseline = action
        if (self.limp_raise and action == 'call' and o.call > 0 and hand_rank(o.hole) < self.limp_raise
                and (self.limp_raise_scope == 'all' or all_fish)):
            # The trained menu's pot-size open (2.5 BB) plus a big blind per limper.
            target = int(round(2 * (2.5 + len(limpers))))
            action = min(legal_raises, key=lambda a: (abs(a - target), a))
            self.limp_raises += 1
        if self.open_bb and isinstance(action, int) and sized_ok:
            hero_bet = {0: 1, 1: 2}.get(o.position, 0)
            cap = o.stacks[o.position] + hero_bet
            target = min(cap, int(round(2 * (self.open_bb + self.open_limper_bb * len(limpers)))))
            if target > action:
                action = target
                self.resized += 1
        if action != baseline:
            object.__setattr__(self, 'last_open', dict(baseline=baseline, action=action, limpers=len(limpers),
                                                       fish_behind=len(fish), others=len(others)))
        return action

    def _rule(self, o):
        n = len(o.active)
        raises = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        legal_raises = [a for a in o.legal if isinstance(a, int)]
        acted = {h[1] for h in o.history if h[2] == 0}
        # Players still to act this round who have not folded, excluding us.
        behind = [i for i in range(n) if i != o.position and o.active[i] and o.stacks[i] > 0 and i not in acted]
        live_others = [i for i in range(n) if i != o.position and o.active[i]]
        r = hand_rank(o.hole)
        u = int.from_bytes(blake2b(repr((o.hole, o.position, len(o.history))).encode(), digest_size=8).digest(),
                           'big') / (1 << 64)
        pot_bb = o.pot / 2

        def size(bb):
            """Nearest legal raise-to (chips are half-BB) to ``bb`` big blinds."""
            if not legal_raises:
                return None
            target = int(round(bb * 2))
            return min(legal_raises, key=lambda a: (abs(a - target), a))

        if not raises:
            if not self.steal or not behind or not all(self._fish(i) for i in behind):
                return None
            limpers = [h for h in o.history if h[2] == 0 and h[3] == 'call' and h[5] > 0]
            if limpers and not all(self._fish(h[1]) for h in limpers):
                return None
            late = len(behind) <= 2
            width = self.steal_width if late else self.open_width
            if r < width and legal_raises:
                return size(2.5 + len(limpers))
            if o.call and r >= width:
                return 'fold' if 'fold' in o.legal else 'call'
            return None
        if not self.iso:
            return None
        raisers = [h[1] for h in raises]
        if not all(self._fish(i) for i in raisers if i != o.position):
            return None
        if not all(self._fish(i) for i in live_others):
            return None
        if len(raises) == 1:
            if r < self.iso_width and legal_raises:
                in_pos = o.position not in (0, 1)
                return size(max(o.call, 2) / 2 * (3 if in_pos else 4) + 1)
            closing = o.position == 1 or not behind
            if r < self.call_width or (closing and r < self.call_width * 1.5):
                return 'call'
            return 'fold' if 'fold' in o.legal else 'call'
        if raisers[-1] == o.position:
            return None
        if len(raises) == 2 and o.position in raisers:
            if r < self.jam_width and legal_raises:
                return legal_raises[-1]
            if r < self.vs3bet_width:
                return 'call'
            return 'fold' if 'fold' in o.legal else 'call'
        if r < self.jam_width:
            return legal_raises[-1] if legal_raises else 'call'
        return None
