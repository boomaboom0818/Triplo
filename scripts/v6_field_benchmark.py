"""Paired 5-field benchmark and HUD metrics for Triplo v5-fast and the v6 ablations (docs/TRIPLO-V6.md).

Needs the cached playable checkpoint (CHECKPOINT_CACHE) and, for the turbo field,
the offline Triplo Turbo bridge binary (BRIDGE_PATH).

Fields (2048+ paired deals/field, unequal stacks, 250 BB hero buy-in):
  1. sticky        : 5x LooseCaller(2)
  2. real_mix      : tight + station-v1 + LooseCaller(1) + pressure + turbo
  3. pressure      : 5x ScriptedOpponent('pressure')
  4. turbo         : 5x Ramsay (offline-turbo-bridge)
  5. hud_exploiter : 5x HudExploiter (Doppio-style 3-exploit adversarial bot)
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import blake2b
import json
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import mean, stdev
import subprocess
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker.equity_policy import private_strength
from poker.equity_pricing import (
    VERSION_V5,
    make_fallback,
    strength_table,
)
from poker.fast_eval import CARD_ID, evaluate
from poker.fast_game import BIG_BLIND, FastGame
from poker.history_cfr import HistoryPolicy
from poker.lean_abstraction import card_features
from poker.loose_opponents import LooseCaller
from poker.opponent_stats import OpponentStats
from poker.rule_opponents import RANKS, RulePolicy, chen
from poker import stack_press

CHECKPOINT_CACHE = Path('/tmp/triplo-factorial-playable.json')
BRIDGE_PATH = Path('/tmp/opencode/triplo-turbo-study-001/target/release/offline-turbo-bridge')
STACK_PROFILES = (
    (30, 50, 90, 125, 250),
    (50, 75, 125, 200, 350),
    (35, 100, 150, 250, 500),
)
HERO_BUYIN_BB = 250
ALL_FIELDS = ('sticky', 'real_mix', 'pressure', 'turbo', 'hud_exploiter')


class ScriptedOpponent:
    """Matches scripts/river_research_screen.py ScriptedOpponent."""
    def __init__(self, name):
        self.name = name
        self.early = (
            LooseCaller(2) if name == 'sticky' else
            RulePolicy('aggressive-v1' if name == 'pressure' else 'tight-v1')
        )

    def act(self, o):
        if o.street != 3:
            return self.early.act(o)
        board = tuple(sorted(CARD_ID[c] for c in o.board))
        hole = tuple(sorted(CARD_ID[c] for c in o.hole))
        strength = strength_table(board)[hole]
        sizes = [s for s in o.legal if type(s) is int]
        if o.call:
            price = o.call / (o.pot + o.call)
            cutoff = dict(sticky=.26 + .63 * price, tight=.54 + 1.04 * price,
                          pressure=.33 + 1.02 * price)[self.name]
            if strength >= .95 and sizes:
                return sizes[0]
            return 'call' if strength >= cutoff else 'fold'
        if sizes and (strength >= .82 or self.name == 'pressure' and (strength >= .72 or .10 < strength < .23)):
            return sizes[-1]
        return 'call'


def _hud_preflop_score(hole):
    """Playability rank for HudExploiter's 52% 3-betting range vs hero."""
    r1, r2 = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    suited = hole[0][1] == hole[1][1]
    s = chen(hole)
    # Boost suited Aces, suited connectors, and pocket pairs (Doppio's 3-bet bluff core)
    if r1 == r2:
        return max(s, 7.0 + r1 * 0.2)
    if suited and r1 == 12:
        return max(s, 7.5)
    if suited and (r1 - r2) <= 2 and r2 >= 3:
        return max(s, 6.5)
    return s


class HudExploiter:
    """Adversarial HUD shark modeled directly on Doppio's 3 exploits against Triplo:
    1. ~52% 3-bet frequency vs hero opens + 4-bet bluffs vs hero 3-bets.
    2. 100% small (0.35-0.45x pot) flop c-bet when checked to as aggressor / short-handed,
       followed by turn/river bluff shutdown.
    3. Tight folding vs hero's multi-street barrels (refusing to hero-call light).
    """
    def __init__(self, target_seat=None):
        self.target_seat = target_seat

    @staticmethod
    def _u(o, tag):
        h = blake2b(repr((tag, o.position, o.hole, o.board, o.pot, len(o.history))).encode(), digest_size=8).digest()
        return int.from_bytes(h, 'big') / (1 << 64)

    @staticmethod
    def _clamp(lo, hi, target):
        return min(hi, max(lo, int(round(target))))

    def act(self, o, game=None):
        raises = [a for a in o.legal if isinstance(a, int)]
        lo = raises[0] if raises else None
        hi = raises[-1] if raises else None
        passive = 'call' if not o.call else ('fold' if 'fold' in o.legal else 'call')

        if o.street == 0:
            return self._preflop(o, game, lo, hi, passive)
        return self._postflop(o, game, lo, hi, passive)

    def _preflop(self, o, game, lo, hi, passive):
        s = chen(o.hole)
        r1, r2 = sorted((RANKS.index(c[0]) for c in o.hole), reverse=True)
        suited = o.hole[0][1] == o.hole[1][1]
        pre_raises = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        n_raises = len(pre_raises)
        last_raiser = pre_raises[-1][1] if pre_raises else None
        facing_hero = (self.target_seat is not None and last_raiser == self.target_seat)
        top = max(game.bets) if game is not None else o.call

        if n_raises == 0:
            # Solid 6-max opening range (~28-32% average)
            thresh = {2: 8.0, 3: 7.5, 4: 6.5, 5: 5.5, 0: 5.5, 1: 8.0}[o.position]
            if lo is not None and s >= thresh:
                open_to = 5 if o.position in (4, 5) else 6
                return self._clamp(lo, hi, open_to)
            if o.call == 0:
                return 'call'
            if o.position == 0 and o.call <= 2 and s >= 5.5:
                return 'call'
            return passive

        if n_raises == 1:
            if facing_hero:
                # Exploit #1: ~52.0% 3-bet frequency vs hero opens (690/1326 combos)
                if lo is not None and o.call <= 14:
                    should_3b = (
                        r1 == r2
                        or (suited and (r1 >= 6 or r1 - r2 <= 2))
                        or r1 == 12
                        or (r1 == 11 and r2 >= 5)
                        or (r1 == 10 and r2 >= 6)
                        or (r1 == 9 and r2 >= 6)
                        or (r1 == 8 and r2 >= 6)
                    )
                    if should_3b:
                        target = max(top * 3.0, top + o.pot * 0.9)
                        return self._clamp(lo, hi, target)
                if s >= 9.0 and o.call <= 24:
                    return 'call'
                return passive
            else:
                # Normal ~14% 3-bet vs non-hero opens
                if lo is not None and o.call <= 12:
                    if s >= 10.0 or (suited and r1 == 12 and r2 <= 3 and self._u(o, '3b') < 0.6):
                        return self._clamp(lo, hi, max(top * 3.0, top + o.pot * 0.9))
                if (s >= 7.5 or r1 == r2) and o.call <= 8:
                    return 'call'
                return passive

        if n_raises == 2:
            # Facing a 3-bet
            if lo is not None and o.call <= 45:
                if s >= 12.0 or (r1 == 12 and r2 == 11):  # QQ+, JJ, AK
                    return self._clamp(lo, hi, max(top * 2.4, top + o.pot * 0.8))
                # 4-bet bluffs vs hero's wide 22.5% 3-bet
                if facing_hero and o.call <= 32:
                    if (suited and r1 == 12 and r2 <= 4) or (suited and r1 == 11 and r2 in (8, 9) and self._u(o, '4b') < 0.5):
                        return self._clamp(lo, hi, max(top * 2.3, top + o.pot * 0.75))
            if (s >= 8.5 or (r1 == r2 and r1 >= 5)) and o.call <= 26:
                return 'call'
            return passive

        # Facing a 4-bet+ (n_raises >= 3): keep value, fold bluffs cleanly
        if s >= 14.0:  # QQ+, KK, AA
            return hi if (lo is not None and s >= 16.0) else 'call'
        if (s >= 12.0 or (r1 == 12 and r2 == 11)) and o.call <= 80:
            return 'call'
        return passive

    def _postflop(self, o, game, lo, hi, passive):
        hole = tuple(CARD_ID[c] for c in o.hole)
        board = tuple(CARD_ID[c] for c in o.board)
        strong, top_pair, pair = private_strength(hole, board)
        strength = strength_table(tuple(sorted(board)))[tuple(sorted(hole))]
        feats = card_features(hole, board)
        draw = int(feats[1])  # 2=flush draw, 1=straight draw, 0=none
        top = max(game.bets) if game is not None else o.call
        price = o.call / max(1, o.pot + o.call)

        # Who was preflop aggressor?
        pre_raises = [h[1] for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        pf_agg = pre_raises[-1] if pre_raises else None

        # Did the current bettor also bet on an earlier postflop street?
        street_bettors = [h[1] for h in o.history if h[2] == o.street and isinstance(h[3], int)]
        cur_bettor = street_bettors[-1] if street_bettors else None
        prev_barrel = (
            cur_bettor is not None
            and any(h[2] in range(1, o.street) and h[1] == cur_bettor and isinstance(h[3], int) for h in o.history)
        )
        reraised = any(h[2] == o.street and isinstance(h[3], int) and h[5] > 0 for h in o.history)

        if o.street == 1:
            if not o.call:
                # Exploit #2: 100% small (0.38x pot) flop c-bet when PFR or short-handed
                if lo is not None and (pf_agg == o.position or sum(o.active) <= 3):
                    frac = 0.50 if strong else 0.38
                    return self._clamp(lo, hi, top + max(2, o.pot * frac))
                if lo is not None and (strong or top_pair):
                    return self._clamp(lo, hi, top + max(2, o.pot * 0.50))
                return 'call'
            # Facing a flop bet
            if strong:
                if lo is not None and not reraised and o.call <= o.pot:
                    return self._clamp(lo, hi, top + o.pot * 0.75)
                return 'call'
            if top_pair and o.call <= 1.25 * o.pot and not reraised:
                return 'call'
            if draw == 2 and price <= 0.34:
                return 'call'
            if draw == 1 and price <= 0.28:
                return 'call'
            if pair and price <= 0.27 and strength >= 0.52 and not reraised:
                return 'call'
            return passive

        if o.street == 2:
            if not o.call:
                # Exploit #2b: Turn bluff shutdown — only bet strong made hands / very strong TP
                if lo is not None and (strong or (top_pair and strength >= 0.82)):
                    return self._clamp(lo, hi, top + max(2, o.pot * 0.65))
                return 'call'
            # Facing a turn bet — Exploit #3: tight folding vs multi-street barrels
            if strong:
                if lo is not None and not reraised and strength >= 0.92 and o.call <= o.pot:
                    return self._clamp(lo, hi, top + o.pot * 0.75)
                return 'call'
            if draw == 2 and price <= 0.24 and not prev_barrel:
                return 'call'
            cutoff = (0.72 if prev_barrel else 0.62) + 0.65 * price
            if strength >= cutoff and not reraised:
                return 'call'
            return passive

        # River (street == 3)
        if not o.call:
            # Bluff shutdown on river: pure value
            if lo is not None and (strong or (top_pair and strength >= 0.85)):
                return self._clamp(lo, hi, top + max(2, o.pot * 0.65))
            return 'call'
        # Facing river bet — Exploit #3: respect multi-street barrels, fold one-pair light
        if strong:
            if lo is not None and not reraised and strength >= 0.96:
                return self._clamp(lo, hi, top + o.pot * 0.75)
            if o.call <= 2.5 * o.pot or strength >= 0.90:
                return 'call'
            return passive
        cutoff = (0.76 if prev_barrel else 0.66) + 0.55 * price
        if strength >= cutoff and o.call <= 1.2 * o.pot and not reraised:
            return 'call'
        return passive


def event(o, bets, stacks, action):
    i, top = o.position, max(bets)
    if action == 'fold':
        return i, o.street, 'f', 0.
    if action == 'call':
        return i, o.street, 'c' if o.call else 'x', o.call / (2 if not o.street else max(o.pot, 2))
    kind = 'j' if action - bets[i] == stacks[i] else 'r' if top or not o.street else 'b'
    return i, o.street, kind, action / 2 if not o.street else (action - bets[i]) / max(o.pot, 2)


def turbo_action(proc, g, o, events, seed):
    i, top = o.position, max(g.bets)
    hi = g.stacks[i] + g.bets[i]
    lo = min(hi, top + max(2, g.raise_size)) if g.can_raise() else None
    request = dict(
        hero=i, hole=o.hole, board=o.board,
        stacks=[s * 10 for s in g.stacks],
        bets=[s * 10 for s in g.bets],
        active=o.active, profiles=['unknown'] * 6,
        pot=o.pot * 10, call=o.call * 10,
        raises=None if lo is None else [lo * 10, hi * 10],
        events=events, mode='full',
        seed=seed * 1_000_003 + len(events) * 31 + i,
    )
    proc.stdin.write(json.dumps(request, separators=(',', ':')) + '\n')
    out = proc.stdout.readline()
    if not out:
        raise RuntimeError('Turbo bridge exited before replying')
    dec = json.loads(out)
    if 'error' in dec:
        raise RuntimeError(dec['error'])
    a = dec['action']
    if a == 'jam':
        a = hi
    elif type(a) is int:
        a = (a + 5) // 10
        a = min(hi, max(lo, a))
    return a


def hero_act_v5fast(policy, game, o):
    """Exact OpenPoker client execution path for v5-fast (with stack_press.deep_sizes)."""
    hero = o.position
    press = ()
    if (stack_press.deep_enabled(policy) or stack_press.enabled(policy)) and game.can_raise():
        top = max(game.bets)
        hi = game.stacks[hero] + game.bets[hero]
        lo = min(hi, max(game.raise_size, 2) + top)
        opponents = [(game.bets[j], game.stacks[j]) for j in range(6) if j != hero and game.active[j]]
        sizer = stack_press.deep_sizes if stack_press.deep_enabled(policy) else stack_press.sizes
        press = sizer(o.street, top, o.pot, lo, hi, game.bets[hero], game.stacks[hero], opponents, o.legal)

    cov_before = policy.covered
    sg_before = policy.safeguards
    po_before = policy.pressure_overrides
    policy.fallback.last = None
    action = stack_press.act(policy, o, press, history_complete=True)
    if policy.safeguards > sg_before:
        src = 'poker_safeguard'
    elif policy.pressure_overrides > po_before:
        src = 'pressure_guard'
    elif policy.covered > cov_before:
        src = 'learned_policy'
    else:
        src = 'preflop_rules' if not o.street else 'equity_fallback'
    return action, src


def make_lineup(field, seed, hero):
    """Return list of 6 opponent tags (None at hero seat)."""
    if field == 'sticky':
        tags = ['sticky'] * 5
    elif field == 'pressure':
        tags = ['pressure'] * 5
    elif field == 'turbo':
        tags = ['turbo'] * 5
    elif field == 'hud_exploiter':
        tags = ['hud_exploiter'] * 5
    elif field == 'real_mix':
        tags = ['tight', 'station-v1', 'loose1', 'pressure', 'turbo']
        Random(f'realmix-seats:{seed}').shuffle(tags)
    else:
        raise ValueError(field)
    tags.insert(hero, None)
    return tags


def make_bot_instances(hero):
    return {
        'sticky': LooseCaller(2),
        'loose1': LooseCaller(1),
        'station-v1': RulePolicy('station-v1'),
        'tight': ScriptedOpponent('tight'),
        'pressure': ScriptedOpponent('pressure'),
        'hud_exploiter': HudExploiter(target_seat=hero),
    }


def play_one_hand(policy, seed, hero, stacks, tags, proc, profiles_map, track_hud=False, stats_collector=None):
    """Play one hand and return (hero_bb, hud_dict)."""
    game = FastGame(seed, button=5, stacks=[s * BIG_BLIND for s in stacks], lean=1 << hero)
    bots = make_bot_instances(hero)
    need_events = 'turbo' in tags
    events = [] if need_events else None

    if profiles_map is None:
        policy.fallback.profiles = {}
    else:
        policy.fallback.profiles = {
            pos: profiles_map[tag]
            for pos, tag in enumerate(tags)
            if tag is not None and tag in profiles_map
        }

    policy.rng.seed(f'hud-ablate:{seed}:{hero}')

    hud = Counter() if track_hud else None
    hero_vpip = False
    hero_pfr = False
    hero_opened = False
    hero_bet_postflop = False
    showdown_strength = None

    while game.live:
        o = game.observe()
        pos = o.position
        if pos == hero:
            action, src = hero_act_v5fast(policy, game, o)
            if track_hud:
                if o.street == 0:
                    pre_raises = sum(1 for h in o.history if h[2] == 0 and isinstance(h[3], int))
                    if isinstance(action, int):
                        hero_vpip = True
                        hero_pfr = True
                        if pre_raises == 0:
                            hero_opened = True
                    elif action == 'call' and o.call > 0:
                        hero_vpip = True
                    if pre_raises == 1 and o.call > 0 and any(isinstance(a, int) for a in o.legal):
                        hud['3bet_opp'] += 1
                        if isinstance(action, int):
                            hud['3bet_yes'] += 1
                    if pre_raises == 2 and o.call > 0 and 'fold' in o.legal:
                        hud['face_any_3bet'] += 1
                        if action == 'fold':
                            hud['fold_any_3bet'] += 1
                        if hero_opened:
                            hud['face_3bet_after_open'] += 1
                            hud[f'f3b_src_{src}'] += 1
                            if action == 'fold':
                                hud['fold_3bet_after_open'] += 1
                                hud[f'f3b_fold_src_{src}'] += 1
                else:
                    # Postflop
                    if isinstance(action, int):
                        hero_bet_postflop = True
                        hud['pf_agg_actions'] += 1
                        hole_i = tuple(CARD_ID[c] for c in o.hole)
                        board_i = tuple(CARD_ID[c] for c in o.board)
                        strong, top_pair, pair = private_strength(hole_i, board_i)
                        if not strong and not top_pair:
                            hud['pf_bluff_non_tp'] += 1
                            if not pair:
                                hud['pf_bluff_air_draw'] += 1
                    if o.call > 0 and 'fold' in o.legal:
                        if o.street == 1:
                            # First flop bet faced
                            flop_raises = [h for h in o.history if h[2] == 1 and isinstance(h[3], int)]
                            if len(flop_raises) == 1:
                                hud['flop_face_bet'] += 1
                                if action == 'fold':
                                    hud['flop_fold_bet'] += 1
                                pre_r = [h[1] for h in o.history if h[2] == 0 and isinstance(h[3], int)]
                                if pre_r and flop_raises[0][1] == pre_r[-1]:
                                    hud['flop_face_cbet'] += 1
                                    if action == 'fold':
                                        hud['flop_fold_cbet'] += 1
                        elif o.street == 2:
                            hud['turn_face_bet'] += 1
                            if action == 'fold':
                                hud['turn_fold_bet'] += 1
                        elif o.street == 3:
                            hud['river_face_bet'] += 1
                            if action == 'fold':
                                hud['river_fold_bet'] += 1
        else:
            tag = tags[pos]
            if tag == 'turbo':
                action = turbo_action(proc, game, o, events, seed)
            elif tag == 'hud_exploiter':
                action = bots[tag].act(o, game)
            else:
                action = bots[tag].act(o)
            if stats_collector is not None and o.street >= 1 and o.call > 0:
                act_str = 'raise' if isinstance(action, int) else action
                stats_collector.observe(tag, o.street, act_str, o.call, o.pot - o.call)

        if events is not None:
            events.append(event(o, game.bets, game.stacks, action))
        game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
        game.act(action, checked=False)

    assert sum(game.payoffs) == 0
    if track_hud:
        hud['hands'] += 1
        hud['vpip'] += int(hero_vpip)
        hud['pfr'] += int(hero_pfr)
        if game.active[hero] and len(game.board) == 5 and sum(game.active) >= 2:
            hole_i = tuple(sorted(game.hole[hero]))
            board_i = tuple(sorted(game.board))
            sd = strength_table(board_i)[hole_i]
            hud['sd_count'] += 1
            hud['sd_sum_x1000'] += round(sd * 1000)
            if hero_bet_postflop:
                hud['sd_agg_count'] += 1
                hud['sd_agg_sum_x1000'] += round(sd * 1000)
    return game.payoffs[hero] / BIG_BLIND, hud


def build_policy(saved, arm_cfg):
    policy = HistoryPolicy(saved, seed=0, min_visits=64, learned_postflop=False)
    fb_kwargs = dict(stack_press=True, deep_value=True, solve_river=True)
    fb_kwargs.update(arm_cfg.get('fallback_kwargs', {}))
    policy.fallback = make_fallback(VERSION_V5, **fb_kwargs)
    return policy


# Worker globals
_SAVED = None
_ARMS = None
_PROFILES = None


def _run_batch(task):
    field, first, count, arm_names, cold_start = task
    need_bridge = field in ('turbo', 'real_mix')
    proc = (
        subprocess.Popen([str(BRIDGE_PATH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, env={'PATH': '/usr/bin:/bin'})
        if need_bridge else None
    )
    policies = {name: build_policy(_SAVED, _ARMS[name]) for name in arm_names}
    results = {name: [] for name in arm_names}
    huds = {name: Counter() for name in arm_names}
    prof_map = None if cold_start else _PROFILES
    try:
        for seed in range(first, first + count):
            rng = Random(f'hud-lineup:{seed}')
            hero = rng.randrange(6)
            lineup = list(rng.choice(STACK_PROFILES))
            rng.shuffle(lineup)
            stacks = lineup.copy()
            stacks.insert(hero, HERO_BUYIN_BB)
            tags = make_lineup(field, seed, hero)
            for name in arm_names:
                bb, hud = play_one_hand(policies[name], seed, hero, stacks, tags, proc, prof_map, track_hud=True)
                results[name].append(bb)
                huds[name].update(hud)
    finally:
        if proc is not None:
            proc.stdin.close()
            proc.stdout.close()
            if proc.wait(timeout=10):
                raise RuntimeError('Turbo bridge exited unsuccessfully')
    return results, huds


def calibrate_profiles(saved, warmup_deals=300):
    """Measure real OpponentStats (base, slope) for each opponent type offline."""
    policy = build_policy(saved, {})
    stats = OpponentStats(min_faced=10)
    proc = subprocess.Popen([str(BRIDGE_PATH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, env={'PATH': '/usr/bin:/bin'})
    try:
        # Run warmup hands across all opponent types
        for field in ('sticky', 'real_mix', 'pressure', 'hud_exploiter', 'turbo'):
            n = warmup_deals if field != 'turbo' else min(200, warmup_deals)
            for i in range(n):
                seed = 900_000_000 + hash(field) % 100000 + i
                rng = Random(f'warmup:{seed}')
                hero = rng.randrange(6)
                lineup = list(rng.choice(STACK_PROFILES))
                rng.shuffle(lineup)
                stacks = lineup.copy()
                stacks.insert(hero, HERO_BUYIN_BB)
                tags = make_lineup(field, seed, hero)
                play_one_hand(policy, seed, hero, stacks, tags, proc if field in ('turbo', 'real_mix') else None,
                              None, track_hud=False, stats_collector=stats)
    finally:
        proc.stdin.close()
        proc.stdout.close()
        proc.wait(timeout=10)
    profiles = {}
    for tag in ('sticky', 'loose1', 'station-v1', 'tight', 'pressure', 'turbo', 'hud_exploiter'):
        base, slope, faced = stats.fit(tag)
        profiles[tag] = (round(base, 4), round(slope, 4))
    return profiles, {tag: stats.fit(tag) for tag in profiles}


def est_bb100(vals):
    m = mean(vals) * 100.0
    se = stdev(vals) * 100.0 / sqrt(len(vals))
    return dict(bb100=round(m, 1), ci95=[round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)])


def summarize_hud(h):
    pct = lambda num, den: round(100.0 * num / den, 1) if den else None
    out = dict(
        hands=h['hands'],
        vpip_pct=pct(h['vpip'], h['hands']),
        pfr_pct=pct(h['pfr'], h['hands']),
        threebet_pct=pct(h['3bet_yes'], h['3bet_opp']),
        fold_to_3bet_after_open_pct=pct(h['fold_3bet_after_open'], h['face_3bet_after_open']),
        face_3bet_after_open_n=h['face_3bet_after_open'],
        fold_to_any_3bet_pct=pct(h['fold_any_3bet'], h['face_any_3bet']),
        flop_fold_to_cbet_pct=pct(h['flop_fold_cbet'], h['flop_face_cbet']),
        flop_face_cbet_n=h['flop_face_cbet'],
        flop_fold_to_bet_pct=pct(h['flop_fold_bet'], h['flop_face_bet']),
        turn_fold_pct=pct(h['turn_fold_bet'], h['turn_face_bet']),
        river_fold_pct=pct(h['river_fold_bet'], h['river_face_bet']),
        postflop_bluff_non_tp_pct=pct(h['pf_bluff_non_tp'], h['pf_agg_actions']),
        postflop_bluff_air_draw_pct=pct(h['pf_bluff_air_draw'], h['pf_agg_actions']),
        postflop_agg_n=h['pf_agg_actions'],
        avg_sd_strength=round(h['sd_sum_x1000'] / (1000.0 * h['sd_count']), 3) if h['sd_count'] else None,
        avg_sd_agg_strength=round(h['sd_agg_sum_x1000'] / (1000.0 * h['sd_agg_count']), 3) if h['sd_agg_count'] else None,
        f3b_sources={k[8:]: h[k] for k in h if k.startswith('f3b_src_')},
        f3b_fold_sources={k[13:]: h[k] for k in h if k.startswith('f3b_fold_src_')},
    )
    return out


def run_eval(arms, fields, deals, workers, seed_base, profiles, cold_start=False):
    global _SAVED, _ARMS, _PROFILES
    _ARMS = arms
    _PROFILES = profiles
    arm_names = list(arms.keys())
    base_name = arm_names[0]
    out = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork')) as pool:
        for f_idx, field in enumerate(fields):
            offset = seed_base + f_idx * 100_000
            chunk = max(32, deals // (workers * 2))
            tasks = [
                (field, offset + i, min(chunk, deals - i), arm_names, cold_start)
                for i in range(0, deals, chunk)
            ]
            all_res = {name: [] for name in arm_names}
            all_hud = {name: Counter() for name in arm_names}
            for res, hud in pool.map(_run_batch, tasks):
                for name in arm_names:
                    all_res[name].extend(res[name])
                    all_hud[name].update(hud[name])
            field_rep = {}
            base_vals = all_res[base_name]
            for name in arm_names:
                vals = all_res[name]
                delta = None if name == base_name else est_bb100([a - b for a, b in zip(vals, base_vals)])
                field_rep[name] = dict(
                    abs=est_bb100(vals),
                    delta=delta,
                    hud=summarize_hud(all_hud[name]),
                )
            out[field] = field_rep
    return out


ARM_PRESETS = {
    'v5_fast_base': {},
    'fix1_3bet': {'fallback_kwargs': {'defend_3bet': True}},
    'fix2_peel': {'fallback_kwargs': {'flop_peel': True}},
    'fix3_draws': {'fallback_kwargs': {'semi_bluff': 'draws'}},
    'fix3_bluff': {'fallback_kwargs': {'semi_bluff': True}},
    'fix12_3bet_peel': {'fallback_kwargs': {'defend_3bet': True, 'flop_peel': True}},
    'fix13_3bet_draws': {'fallback_kwargs': {'defend_3bet': True, 'semi_bluff': 'draws'}},
    'fix13_3bet_bluff': {'fallback_kwargs': {'defend_3bet': True, 'semi_bluff': True}},
    'fix23_peel_draws': {'fallback_kwargs': {'flop_peel': True, 'semi_bluff': 'draws'}},
    'fix23_peel_bluff': {'fallback_kwargs': {'flop_peel': True, 'semi_bluff': True}},
    'fix123_draws': {'fallback_kwargs': {'defend_3bet': True, 'flop_peel': True, 'semi_bluff': 'draws'}},
    'fix123_all': {'fallback_kwargs': {'defend_3bet': True, 'flop_peel': True, 'semi_bluff': True}},
}


def main():
    global _SAVED
    parser = argparse.ArgumentParser()
    parser.add_argument('--deals', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=770_000_000)
    parser.add_argument('--fields', nargs='+', default=list(ALL_FIELDS))
    parser.add_argument('--arms', nargs='+', default=['v5_fast_base', 'fix1_3bet', 'fix2_peel', 'fix3_bluff', 'fix12_3bet_peel', 'fix13_3bet_bluff', 'fix123_all'])
    parser.add_argument('--cold-arms', nargs='+', default=['v5_fast_base', 'fix1_3bet', 'fix2_peel', 'fix3_bluff', 'fix123_all'])
    parser.add_argument('--profiles-cache', type=Path, default=Path('/tmp/triplo-hud-measured-profiles-v2.json'))
    parser.add_argument('--cold-start', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    t0 = perf_counter()
    wrapper = json.loads(CHECKPOINT_CACHE.read_text())
    _SAVED = wrapper['policy']

    if args.profiles_cache.exists():
        pdata = json.loads(args.profiles_cache.read_text())
        profiles = {k: tuple(v) for k, v in pdata['profiles'].items()}
    else:
        profiles, raw_fit = calibrate_profiles(_SAVED)
        args.profiles_cache.write_text(json.dumps(dict(profiles=profiles, raw_fit=raw_fit), indent=2))
        print('Calibrated OpponentStats profiles:', json.dumps(raw_fit, indent=2), flush=True)

    arms = {k: ARM_PRESETS[k] for k in args.arms}
    res_measured = run_eval(arms, args.fields, args.deals, args.workers, args.seed, profiles, cold_start=False)
    res_cold = None
    if args.cold_start:
        cold_arms = {k: ARM_PRESETS[k] for k in args.cold_arms}
        res_cold = run_eval(cold_arms, args.fields, args.deals, args.workers, args.seed, profiles, cold_start=True)
    report = dict(
        deals_per_field=args.deals,
        seed=args.seed,
        profiles=profiles,
        measured=res_measured,
        cold_start=res_cold,
        elapsed_s=round(perf_counter() - t0, 1),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f'Done in {report["elapsed_s"]}s -> {args.out}', flush=True)
    for mode, data in (('MEASURED', res_measured), ('COLD-START', res_cold)):
        if data is None:
            continue
        print(f'\n=== {mode} ({args.deals} paired deals/field) ===', flush=True)
        for field in args.fields:
            print(f'  [{field}]', flush=True)
            for name, r in data[field].items():
                h = r['hud']
                d = r['delta']
                dstr = f'delta={d["bb100"]:+6.1f} {d["ci95"]}' if d else 'BASELINE'
                print(f'    {name:18s}: abs={r["abs"]["bb100"]:+7.1f} | {dstr:28s} | '
                      f'F3B={str(h["fold_to_3bet_after_open_pct"]):>5s} '
                      f'FCBet={str(h["flop_fold_to_cbet_pct"]):>5s} '
                      f'TF={str(h["turn_fold_pct"]):>5s} RF={str(h["river_fold_pct"]):>5s} '
                      f'BluffNT={str(h["postflop_bluff_non_tp_pct"]):>5s} '
                      f'BluffAir={str(h["postflop_bluff_air_draw_pct"]):>5s} '
                      f'SD={str(h["avg_sd_agg_strength"]):>5s}', flush=True)


if __name__ == '__main__':
    main()
