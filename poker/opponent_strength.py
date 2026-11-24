"""Season-result opponent strength for the opt-in --avoid-strong table rule.

Ratings describe an opponent's public chip results against the whole OpenPoker
field, not the hero's EV against them. Ported from Triplo Turbo's "apex shark"
avoidance; constants are conservative public-league heuristics.
"""
from dataclasses import dataclass


STARTING_CHIPS = 5000
REBUY_CHIPS = 1500
BIG_BLIND = 20
PRIOR_HANDS = 500       # shrink short records toward zero
MIN_HANDS = 200
STRONG_BB100 = 220.0    # Turbo's apex bar
WEAK_BB100 = -15.0      # Turbo's losing self-hosted bar


@dataclass(frozen=True)
class OpponentRating:
    name: str
    hands: int | None
    net_chips: int | None
    results_bb100: float | None  # prior-shrunk season rate, uncapped
    kind: str | None             # 'no_code', 'self_hosted' or 'unknown'
    owner: str | None
    balance: int | None = None   # off-table chip_balance from leaderboard

    @classmethod
    def from_row(cls, row):
        name = row['bot_name']
        if not isinstance(name, str) or not name:
            raise ValueError('missing leaderboard name')
        hands = row.get('hands_played')
        hands = hands if type(hands) is int and hands >= 0 else None
        values = [row.get(k) for k in ('chip_balance', 'chips_at_table', 'rebuys')]
        net = rate = balance = None
        if all(type(v) is int and v >= 0 for v in values):
            balance, at_table, rebuys = values
            net = balance + at_table - STARTING_CHIPS - rebuys * REBUY_CHIPS
            if hands is not None and hands >= MIN_HANDS:
                rate = 100 * net / (BIG_BLIND * (hands + PRIOR_HANDS))
        return cls(name, hands, net, rate, row.get('bot_kind'), row.get('account_agent_id'), balance)


@dataclass(frozen=True)
class StrongCheck:
    strong: tuple   # opponents at or above the threshold
    weak: tuple     # no_code presets or clearly losing bots, which can justify staying
    leave: bool
    reason: str


class OpponentField:
    def __init__(self, rows):
        self.players = {}
        for row in rows:
            player = OpponentRating.from_row(row)
            self.players.setdefault(player.name, player)  # moving leaderboard pages can overlap

    def strong_check(self, hero, opponents, threshold=STRONG_BB100, min_weak=0, max_strong=None):
        """Leave when strong opponents outnumber weak seats, exceed max_strong, or a full table has too few weak seats.

        One strong bot is tolerated while at least as many weak bots sit with it.
        When ``max_strong`` is set, more than that many strong opponents always
        triggers a leave. When ``min_weak`` is positive and at least 4 opponents
        are seated, fewer than ``min_weak`` weak seats also triggers a leave.
        Unrated seats count as neither, and bots on the hero's own account are ignored.
        """
        me = self.players.get(hero)
        others = sorted(set(opponents) - {hero})
        strong, weak = [], []
        for name in others:
            p = self.players.get(name)
            if p is None or (me and me.owner and p.owner == me.owner):
                continue
            if p.results_bb100 is not None and p.results_bb100 >= threshold:
                strong.append(name)
            elif p.kind == 'no_code' or (p.results_bb100 is not None and p.results_bb100 <= WEAK_BB100):
                weak.append(name)
        if max_strong is not None and len(strong) > max_strong:
            return StrongCheck(tuple(strong), tuple(weak), True, 'too many strong opponents')
        if strong and len(weak) < len(strong):
            return StrongCheck(tuple(strong), tuple(weak), True, 'strong opponent')
        if min_weak > 0 and len(others) >= 4 and len(weak) < min_weak:
            return StrongCheck(tuple(strong), tuple(weak), True, 'too few weak seats')
        if not strong:
            return StrongCheck((), tuple(weak), False, 'no strong opponent')
        return StrongCheck(tuple(strong), tuple(weak), False, 'strong opponent offset by weak seats')
