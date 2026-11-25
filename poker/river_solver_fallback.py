"""Triplo v5 fallback with a calibrated heads-up river subgame, plus opt-in fish exploits."""
from dataclasses import replace

from .equity_pricing import GuardedEquityPolicyV4, VERSION_V5, draw_outs, ledger
from .equity_policy import private_strength
from .fast_eval import CARD_ID

# A measured postflop fold base at or below this marks a calling station (no_code
# bots measure ~0.05-0.07 live; regulars ~0.3-0.4).
FISH_BASE = .18


class RiverSolverFallback(GuardedEquityPolicyV4):
    VERSION = VERSION_V5

    def __init__(self, *args, solve_river=True, fish_size=None, fish_thin=None, fish_nobluff=False,
                 fish_cbet=None, fish_raise=None, fish_stab=None, fish_value_cap=None, **kwargs):
        super().__init__(*args, **kwargs)
        if solve_river:
            import numpy  # Check the optional dependency before play begins.
        self.solve_river = solve_river
        self.fish_size = fish_size      # {street: pot fraction} for value bets into calling stations
        self.fish_thin = fish_thin      # (min equity, pot fraction) for thin turn/river value bets
        self.fish_nobluff = fish_nobluff
        self.fish_cbet = fish_cbet      # pot fraction for heads-up flop c-bets into a calling station
        self.fish_raise = fish_raise    # ('top'|'two', pot fraction): raise fish bets/raises on flop/turn for value
        # Triplo v9 (opt-in): (pot fraction, 'all'|'nopair') heads-up flop stab when a fish checks to us,
        # at the exact fraction (off the trained menu, like Stack Press sizes).
        self.fish_stab = fish_stab
        # Triplo v9 (opt-in): pot fraction that caps Deep Value bets and raises while any non-fish
        # opponent remains; overbets stay available only when every remaining opponent is a fish.
        self.fish_value_cap = fish_value_cap
        # Positions of opponents the leaderboard marks weak (no_code or losing), set per decision
        # by the client. When set it alone decides who is a fish: measured fold profiles also
        # flag most winning regulars, because nearly everyone calls river bets.
        self.fish_seats = None

    def act(self, o):
        action = self._act_v5(o)
        if o.street and (self.fish_size or self.fish_thin or self.fish_nobluff or self.fish_cbet or self.fish_raise
                         or self.fish_stab):
            action = self._fish_adjust(o, action)
        return action

    def _deep_value_size(self, o, proposed, press, all_sticky, strong, usable):
        """Deep Value sizing; with ``fish_value_cap``, no overbets while a regular remains."""
        if not self.fish_value_cap or self.fish_seats is None:
            return super()._deep_value_size(o, proposed, press, all_sticky, strong, usable)
        acting = [i for i, a in enumerate(o.active) if a and i != o.position and o.stacks[i] > 0]
        if acting and all(self.is_fish(i) for i in acting):
            return super()._deep_value_size(o, proposed, press, all_sticky, strong, usable)
        if o.call:
            book = ledger(o)
            if book is None:
                return super()._deep_value_size(o, proposed, press, all_sticky, strong, usable)
            top = book[1][o.position] + o.call
        else:
            top = 0
        cap = top + self.fish_value_cap * o.pot
        raises = [a for a in o.legal if isinstance(a, int)]
        # A jam that is at most pot-sized is not an overbet: keep it available.
        allowed = [a for a in raises if a <= cap or (a == raises[-1] and a <= top + o.pot)]
        if not allowed:
            allowed = raises[:1]
        clamped = max([a for a in allowed if a <= proposed], default=allowed[0])
        legal = tuple(a for a in o.legal if not isinstance(a, int)) + tuple(allowed)
        chosen = super()._deep_value_size(replace(o, legal=legal), clamped, press, False, strong, usable)
        if len(allowed) < len(raises):
            # The cap removed sizes the uncapped EV sizing could have chosen.
            self.last = dict(self.last or {}, fish_value_cap=dict(proposed=proposed, chosen=chosen, cap=cap,
                                                                  removed=len(raises) - len(allowed)))
        return chosen

    def _act_v5(self, o):
        baseline = super().act(o)
        if o.street != 3 or not self.solve_river:
            return baseline
        from .calibrated_river_solver import propose
        proposal = propose(o, baseline)
        if proposal is None:
            return baseline
        action, details = proposal
        self.last = dict(self.last or {}, river_solver=details)
        return action

    def is_fish(self, i):
        if self.fish_seats is not None:
            return i in self.fish_seats
        return (i in self.profiles or None in self.profiles) and self.threshold(i)[0] <= FISH_BASE

    def _all_fish(self, o):
        acting = [i for i, a in enumerate(o.active) if a and i != o.position and o.stacks[i] > 0]
        return bool(acting) and all(self.is_fish(i) for i in acting)

    @staticmethod
    def _nearest(o, target):
        raises = [a for a in o.legal if isinstance(a, int)]
        return min(raises, key=lambda a: (abs(a - target), a)) if raises else None

    def _fish_adjust(self, o, action):
        """Exploit measured calling stations: bigger value bets, thin value, no bluffs (opt-in)."""
        if not self._all_fish(o):
            return action
        hole = tuple(CARD_ID[c] for c in o.hole)
        board = tuple(CARD_ID[c] for c in o.board)
        strong, top_pair, pair = private_strength(hole, board)
        if o.call:
            # Fish raise flop/turn bets at ~25-35% regardless of strength and almost never
            # fold once re-raised: raise them back for value with made hands.
            if not self.fish_raise or o.street not in (1, 2) or isinstance(action, int):
                return action
            level, frac = self.fish_raise
            raises_now = [h for h in o.history if h[2] == o.street and isinstance(h[3], int)]
            if (strong or (level == 'top' and top_pair)) and len(raises_now) <= 3:
                legal = [a for a in o.legal if isinstance(a, int)]
                if legal:
                    target = (o.pot + o.call) * frac + o.call + max(0, max(h[3] for h in raises_now))
                    action = min(legal, key=lambda a: (abs(a - target), a))
                    self.last = dict(self.last or {}, fish_raise=True)
            return action
        outs = draw_outs(hole, board)
        tag = None
        if isinstance(action, int):
            if self.fish_nobluff and o.street >= 2 and not pair and outs < 8:
                action, tag = 'call', 'fish_nobluff'
            elif self.fish_size and (strong or top_pair) and o.street in self.fish_size:
                sized = self._nearest(o, o.pot * self.fish_size[o.street])
                if sized is not None and sized > action:
                    action, tag = sized, 'fish_size'
        elif (self.fish_stab and o.street == 1 and sum(o.active) == 2
              and not any(h[2] == 1 and isinstance(h[3], int) for h in o.history)
              and any(h[2] == 1 and h[1] != o.position and h[3] == 'call' for h in o.history)
              and (self.fish_stab[1] == 'all' or not pair)):
            # Checked to us heads-up on the flop by a fish: a small stab at the exact fraction.
            bet = max(2, int(round(o.pot * self.fish_stab[0])))
            raises = [a for a in o.legal if isinstance(a, int)]
            if raises and bet <= raises[-1]:
                action, tag = bet, 'fish_stab'
        elif (self.fish_cbet and o.street == 1 and sum(o.active) == 2
              and not any(h[2] == 1 and isinstance(h[3], int) for h in o.history)
              and [h[1] for h in o.history if h[2] == 0 and isinstance(h[3], int)][-1:] == [o.position]):
            bet = self._nearest(o, o.pot * self.fish_cbet)
            if bet is not None:
                action, tag = bet, 'fish_cbet'
        elif self.fish_thin and o.street >= 2 and sum(o.active) == 2 and pair:
            equity = (self.last or {}).get('equity')
            if equity is not None and equity >= self.fish_thin[0]:
                frac = self.fish_size.get(o.street, self.fish_thin[1]) if (self.fish_size and (strong or top_pair)) \
                    else self.fish_thin[1]
                bet = self._nearest(o, o.pot * frac)
                if bet is not None:
                    action, tag = bet, 'fish_thin'
        if tag:
            self.last = dict(self.last or {}, **{tag: True})
        return action
