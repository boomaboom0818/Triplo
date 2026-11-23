"""Live state for the opt-in --avoid-strong rule: leave tables where strong bots sit down."""
import asyncio
from http.client import HTTPException
import json
import logging
import time
from urllib.request import Request, urlopen

from poker.opponent_strength import OpponentField, STRONG_BB100

LOG = logging.getLogger(__name__)
REFRESH_SECONDS = 30
MAX_AGE_SECONDS = 60        # never act on an older leaderboard read
COOLDOWN = 30               # seconds after the first leave; doubles per consecutive leave
COOLDOWN_CAP = 600
RESET_HANDS = 4             # settled hands at one table that end a run of leaves
REQUEUE_DELAY = 10          # seconds before rejoining, so the lobby does not reseat us at once


def fetch_field():
    rows = []
    for offset in range(0, 10000, 1000):
        url = ('https://api.openpoker.ai/api/season/leaderboard?sort_by=hands_played'
               f'&limit=1000&min_hands=0&offset={offset}')
        with urlopen(Request(url, headers={'User-Agent': 'TriploPoker/0.1'}), timeout=5) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise ValueError('unexpected leaderboard response')
        rows.extend(page)
        if len(page) < 1000:
            return OpponentField(rows)
    raise ValueError('incomplete leaderboard')


class StrongAvoider:
    def __init__(self, mode='report', threshold=STRONG_BB100, min_weak=0, max_strong=None):
        self.mode, self.threshold = mode, threshold
        self.min_weak, self.max_strong = min_weak, max_strong
        self.leaves = 0
        self.last_leave = float('-inf')
        self.task = self.field = None
        self.started = self.field_time = float('-inf')
        self.hero = self.leaving = None
        self.seated(None)

    def clear_field(self):
        if self.task:
            if self.task.done() and not self.task.cancelled():
                self.task.exception()  # discard even a failed read on disconnect/rollover
            else:
                self.task.cancel()
        self.task = self.field = None
        self.started = float('-inf')

    def refresh(self):
        """Start a bounded public read off the action path; never wait on it here."""
        now = time.monotonic()
        if self.task and self.task.done():
            try:
                self.field = self.task.result()
                self.field_time = self.started
            except (OSError, HTTPException, ValueError, KeyError, TypeError) as exc:
                self.field = None
                LOG.warning('Avoid strong: leaderboard unavailable: %s', exc)
            self.task = None
        if self.task is None and now - self.started >= REFRESH_SECONDS:
            self.started = now
            self.task = asyncio.create_task(asyncio.to_thread(fetch_field))

    def seated(self, table_id):
        self.table_id = table_id
        self.hero_seat = self.hero_stack = None
        self.live = {}  # seat -> name for the next hand, including new arrivals
        self.hands = self.watermark = self.snapshot_seq = 0

    def snapshot(self, msg):
        """Take the lineup from an authoritative snapshot of this table."""
        if not self.table_id or msg.get('table_id') != self.table_id:
            return
        seq = msg.get('table_seq')
        if type(seq) is not int or seq < max(self.watermark, self.snapshot_seq):
            return
        self.snapshot_seq = seq
        self.hero_seat = (msg.get('hero') or {}).get('seat', self.hero_seat)
        seats = msg.get('seats', ())
        # Away or disconnected players keep their seat and may return next hand.
        self.live = {s['seat']: s['name'] for s in seats
                     if s.get('name') and s.get('status') != 'empty' and type(s.get('seat')) is int}
        for s in seats:
            if s.get('seat') == self.hero_seat and type(s.get('stack')) is int:
                self.hero_stack = s['stack']

    def arrived(self, msg):
        """Seed the lineup from a table_joined message."""
        for p in msg.get('players') or ():
            if type(p.get('seat')) is int and p.get('name'):
                self.live[p['seat']] = p['name']
                if p['seat'] == self.hero_seat and type(p.get('stack')) is int:
                    self.hero_stack = p['stack']

    def joined(self, msg):
        if type(msg.get('seat')) is int and msg.get('name'):
            self.live[msg['seat']] = msg['name']

    def left(self, msg):
        self.live.pop(msg.get('seat'), None)

    def counted(self, msg):
        """Count a new settled hand at this table; False for stale or duplicate results."""
        seq = msg.get('table_seq')
        if (not self.table_id or msg.get('table_id') != self.table_id or self.leaving
                or type(seq) is not int or seq <= self.watermark or not msg.get('hand_id')):
            return False
        self.watermark = seq
        self.hands += 1
        stack = msg.get('final_stacks', {}).get(str(self.hero_seat))
        if type(stack) is int:
            self.hero_stack = stack
        if self.hands >= RESET_HANDS:
            self.leaves = 0
        return True

    def check(self, trigger, buy_in):
        """Check the lineup for strong opponents; None when there are none."""
        if not self.table_id or self.leaving or not self.hero:
            return None
        now = time.monotonic()
        if not self.field or now - self.field_time > MAX_AGE_SECONDS:
            return None
        opponents = [n for s, n in self.live.items() if s != self.hero_seat and n != self.hero]
        found = self.field.strong_check(self.hero, opponents, self.threshold,
                                        min_weak=self.min_weak, max_strong=self.max_strong)
        if not found.strong and not found.leave:
            return None
        detail = dict(table_id=self.table_id, trigger=trigger, threshold_bb100=self.threshold,
                      strong=list(found.strong), weak=list(found.weak), opponents=opponents,
                      field_age_s=round(now - self.field_time, 1), leave=False, reason=found.reason)
        if not found.leave:
            return detail
        cooldown = 0 if not self.leaves else min(COOLDOWN_CAP, COOLDOWN * 2 ** (self.leaves - 1))
        if now - self.last_leave < cooldown:
            return dict(detail, reason='cooldown', cooldown_s=cooldown)
        # Leaving must return enough chips to fund the next join_lobby.
        if type(self.hero_stack) is not int or self.hero_stack < buy_in:
            return dict(detail, reason='buy-in not covered by stack')
        return dict(detail, leave=True)

    def left_table(self):
        """Record a leave the client has just sent."""
        self.leaves += 1
        self.last_leave = time.monotonic()
        self.leaving = self.table_id
