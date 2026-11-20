"""Per-opponent postflop calling statistics for ``equity-guarded-v2`` value sizing.

Every postflop OpenPoker ``player_action`` that faces a bet is one observation:
the price ``to_call / (pot_before + to_call)`` and whether the player continued
(call / raise / all-in) or folded. Folding frequency is fit as a line in price,

    P(fold | price) ~= base + slope * price,

which is exactly v2's calling threshold ``base + slope * pot-odds`` on a
hand-strength percentile. The fit is regularized toward the ``balanced``
profile with a few pseudo-observations, and a player only gets a profile once
``min_faced`` real observations exist; below that v2 keeps v1's size.

Event ``street`` labels report the street *after* the action, so the street an
action happened on is the label of the previous action in the same hand
(preflop for the first one). Timeout folds are ignored. Replayed duplicates are
dropped by (table_id, table_seq).
"""
from collections import OrderedDict
import json
from pathlib import Path

STREETS = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
# Price bins (upper edges); the fit uses each bin's mean observed price.
EDGES = (.15, .22, .28, .34, .40, 1.0)
PRIOR = (.30, 1.0)          # PROFILES['balanced'] in poker.equity_pricing
PRIOR_WEIGHT = 2.0          # pseudo-observations per bin
BASE_RANGE = (0.0, .9)
SLOPE_RANGE = (0.0, 2.5)
SCHEMA = 1


def _bin(price):
    for j, edge in enumerate(EDGES):
        if price <= edge:
            return j
    return len(EDGES) - 1


class OpponentStats:
    def __init__(self, min_faced=20, seen_capacity=20000):
        self.min_faced = min_faced
        # name -> per bin [faced, folded, price_sum]
        self.players = {}
        self._hands = OrderedDict()      # hand_id -> street of the next action
        self._seen = OrderedDict()
        self._capacity = seen_capacity
        self.dirty = False

    # -- ingestion -------------------------------------------------------
    def ingest(self, event):
        """Feed any OpenPoker message; only hand_start / player_action are used."""
        kind = event.get('type')
        if kind == 'resync_response':
            for replayed in sorted(event.get('replayed_events', []),
                                   key=lambda e: e.get('table_seq') if type(e.get('table_seq')) is int else -1):
                self.ingest(replayed)
            return
        if kind not in ('hand_start', 'player_action'):
            return
        identity = (event.get('table_id'), event.get('table_seq'))
        if type(identity[1]) is int:
            if identity in self._seen:
                return
            self._seen[identity] = None
            if len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
        hand = event.get('hand_id')
        if not hand:
            return
        if kind == 'hand_start':
            self._remember(hand, 0)
            return
        street = self._hands.get(hand)
        reported = STREETS.get(event.get('street'))
        if reported is not None:
            self._remember(hand, reported)
        if street is None:
            return          # joined mid-hand: the action's street is unknown
        self.observe(event.get('name'), street, event.get('action'), event.get('to_call_before'),
                     event.get('pot_before'), event.get('reason'))

    def _remember(self, hand, street):
        self._hands[hand] = street
        self._hands.move_to_end(hand)
        if len(self._hands) > 512:
            self._hands.popitem(last=False)

    def observe(self, name, street, action, to_call, pot_before, reason=None):
        """Record one decision; returns True when it counted as a faced postflop bet."""
        if (not name or not 1 <= street <= 3 or reason is not None or action not in ('fold', 'call', 'raise', 'all_in')
                or type(to_call) is not int or type(pot_before) is not int or to_call <= 0 or pot_before < 0):
            return False
        price = to_call / (pot_before + to_call)
        row = self.players.setdefault(name, [[0, 0, 0.0] for _ in EDGES])[_bin(price)]
        row[0] += 1
        row[1] += action == 'fold'
        row[2] += price
        self.dirty = True
        return True

    # -- estimates -------------------------------------------------------
    def faced(self, name):
        return sum(r[0] for r in self.players.get(name, ()))

    def fit(self, name):
        """(base, slope, faced) regularized toward the balanced profile."""
        rows = self.players.get(name)
        points = []          # (weight, price, fold_rate)
        for j, edge in enumerate(EDGES):
            low = EDGES[j - 1] if j else 0.0
            center = (low + min(edge, .6)) / 2
            points.append((PRIOR_WEIGHT, center, PRIOR[0] + PRIOR[1] * center))
            if rows and rows[j][0]:
                n, folded, price_sum = rows[j]
                points.append((n, price_sum / n, folded / n))
        w = sum(p[0] for p in points)
        mx = sum(p[0] * p[1] for p in points) / w
        my = sum(p[0] * p[2] for p in points) / w
        sxx = sum(p[0] * (p[1] - mx) ** 2 for p in points)
        sxy = sum(p[0] * (p[1] - mx) * (p[2] - my) for p in points)
        slope = min(SLOPE_RANGE[1], max(SLOPE_RANGE[0], sxy / sxx if sxx else PRIOR[1]))
        base = min(BASE_RANGE[1], max(BASE_RANGE[0], my - slope * mx))
        return base, slope, self.faced(name)

    def threshold(self, name):
        """(base, slope) once ``min_faced`` observations exist, else None."""
        if self.faced(name) < self.min_faced:
            return None
        base, slope, _ = self.fit(name)
        return base, slope

    def profiles(self, positions):
        """{position: (base, slope)} for the given {position: name} with enough data."""
        out = {}
        for position, name in positions.items():
            threshold = self.threshold(name)
            if threshold is not None:
                out[position] = threshold
        return out

    # -- persistence -----------------------------------------------------
    def to_json(self):
        return dict(schema=SCHEMA, edges=list(EDGES), players=self.players)

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(self.to_json()))
        tmp.replace(path)
        self.dirty = False

    @classmethod
    def load(cls, path, **options):
        stats = cls(**options)
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text())
            if data.get('schema') != SCHEMA or tuple(data.get('edges', ())) != EDGES:
                raise ValueError('incompatible opponent stats file: %s' % path)
            stats.players = {k: [list(r) for r in v] for k, v in data['players'].items()}
        return stats

    @classmethod
    def from_traces(cls, paths, **options):
        """Rebuild from OpenPoker client ``--trace`` logs (wire messages)."""
        stats = cls(**options)
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    try:
                        message = json.loads(line).get('message')
                    except ValueError:
                        continue
                    if isinstance(message, dict):
                        stats.ingest(message)
        return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build/inspect per-opponent calling stats from client traces')
    parser.add_argument('traces', nargs='*', type=Path)
    parser.add_argument('--out', type=Path, help='write the stats JSON here')
    parser.add_argument('--load', type=Path, help='start from an existing stats JSON')
    parser.add_argument('--min-faced', type=int, default=20)
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args()
    stats = OpponentStats.load(args.load, min_faced=args.min_faced) if args.load else OpponentStats(args.min_faced)
    for path in args.traces:
        with open(path) as handle:
            for line in handle:
                try:
                    message = json.loads(line).get('message')
                except ValueError:
                    continue
                if isinstance(message, dict):
                    stats.ingest(message)
    ranked = sorted(stats.players, key=stats.faced, reverse=True)
    ready = [n for n in ranked if stats.faced(n) >= stats.min_faced]
    print('%d players, %d with >= %d faced postflop bets' % (len(ranked), len(ready), stats.min_faced))
    print('%-22s %6s %6s %6s %8s %8s' % ('player', 'faced', 'fold%', 'base', 'slope', 'fold@.25'))
    for name in ranked[:args.top]:
        base, slope, faced = stats.fit(name)
        folded = sum(r[1] for r in stats.players[name])
        print('%-22s %6d %5.0f%% %6.2f %8.2f %7.0f%%' % (name[:22], faced, 100 * folded / faced, base, slope,
                                                          100 * min(1, base + slope * .25)))
    if args.out:
        stats.save(args.out)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
