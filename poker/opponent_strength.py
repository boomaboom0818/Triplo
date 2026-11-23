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

    @classmethod
    def from_row(cls, row):
        name = row['bot_name']
        if not isinstance(name, str) or not name:
            raise ValueError('missing leaderboard name')
        hands = row.get('hands_played')
        hands = hands if type(hands) is int and hands >= 0 else None
        values = [row.get(k) for k in ('chip_balance', 'chips_at_table', 'rebuys')]
        net = rate = None
        if all(type(v) is int and v >= 0 for v in values):
            balance, at_table, rebuys = values
            net = balance + at_table - STARTING_CHIPS - rebuys * REBUY_CHIPS
            if hands is not None and hands >= MIN_HANDS:
                rate = 100 * net / (BIG_BLIND * (hands + PRIOR_HANDS))
        return cls(name, hands, net, rate, row.get('bot_kind'), row.get('account_agent_id'))


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

    def strong_check(self, hero, opponents, threshold=STRONG_BB100):
        """Leave when strong opponents outnumber the weak seats that pay for them.

        One strong bot is tolerated while at least as many weak bots sit with it.
        Unrated seats count as neither, and bots on the hero's own account are ignored.
        """
        me = self.players.get(hero)
        strong, weak = [], []
        for name in sorted(set(opponents) - {hero}):
            p = self.players.get(name)
            if p is None or (me and me.owner and p.owner == me.owner):
                continue
            if p.results_bb100 is not None and p.results_bb100 >= threshold:
                strong.append(name)
            elif p.kind == 'no_code' or (p.results_bb100 is not None and p.results_bb100 <= WEAK_BB100):
                weak.append(name)
        if not strong:
            return StrongCheck((), tuple(weak), False, 'no strong opponent')
        if len(weak) >= len(strong):
            return StrongCheck(tuple(strong), tuple(weak), False, 'strong opponent offset by weak seats')
        return StrongCheck(tuple(strong), tuple(weak), True, 'strong opponent')
