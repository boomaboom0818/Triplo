"""Snapshot-driven OpenPoker V2 adapter. Run with python -m scripts.openpoker_client."""
import hashlib
import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4
from urllib.request import Request, urlopen

from poker.cfr import CFRPolicy, information_key
from poker.config import load_checkpoint
from poker.equity_pricing import FALLBACKS, VERSION_V3, make_fallback
from poker.fast_game import LEAN_MENU
from poker.game import Observation
from poker import lean_cfr
from poker import history_cfr
from poker import stack_press
from poker.openpoker_history import PublicHistory

LOG = logging.getLogger(__name__)


def decision(snapshot, policy, public_history=(), history_complete=None):
    """Translate a complete acting-player snapshot; return an idempotent action."""
    hero = snapshot['hero']
    token, hand = hero['turn_token'], snapshot['hand_id']
    if snapshot['actor_seat'] != hero['seat']:
        raise ValueError('snapshot actor differs from hero')
    seats = sorted((s for s in snapshot['seats'] if s.get('in_hand')), key=lambda s: s['seat'])
    if not 3 <= len(seats) <= 6 or any('folded' not in s for s in seats):
        raise ValueError('need complete 3–6 player hand mapping and fold flags')
    # Triplo engine order starts left of the dealer (small blind).
    dealer = snapshot['dealer_seat']
    seats = [s for s in seats if s['seat'] > dealer] + [s for s in seats if s['seat'] <= dealer]
    position = next(i for i, s in enumerate(seats) if s['seat'] == hero['seat'])
    scale = snapshot['big_blind'] / 2
    if scale <= 0:
        raise ValueError('invalid big blind')
    unit = lambda amount: round(amount / scale)
    offered = {a['action']: a for a in hero['valid_actions']}
    passive = 'check' if 'check' in offered else 'call' if 'call' in offered else None
    legal, wire, press = [], {}, []
    if passive:
        legal.append('call'); wire['call'] = {'action': passive}
    if 'fold' in offered and passive != 'check':
        legal.append('fold'); wire['fold'] = {'action': 'fold'}
    pot = unit(snapshot['pot'])
    if 'raise' in offered:
        lo, hi = offered['raise']['min'], offered['raise']['max']
        bet = max(unit(s['bet']) for s in seats)
        # Retain exact advertised wire bounds even when normalization rounds.
        lean = getattr(policy, 'menu', None) == LEAN_MENU
        sizes = {} if lean else {unit(lo): lo, unit(hi): hi}
        for n in (bet + pot // 2, bet + pot):
            amount = min(hi, max(lo, round(n * scale)))
            sizes.setdefault(unit(amount), amount)
        # Lean checkpoints are trained without a bare minimum raise or deep all-in.
        # Their all-in is the effective one (FastGame.menu): the deepest live
        # opponent's total, so a deep hero can still put a short stack all in.
        if lean:
            me = seats[position]
            cover = max((s['stack'] + s['bet'] for s in seats if s is not me and not s['folded']), default=0)
            shove = min(hi, max(lo, cover))
            if unit(shove) <= bet + 2 * pot:
                sizes[unit(shove)] = shove
        for n, amount in sorted(sizes.items()):
            legal.append(n); wire[n] = {'action': 'raise', 'amount': amount}
        # Stack Press sizes stay off the trained menu (and so off the learned
        # key); poker.stack_press offers them to the fallback only.
        if stack_press.enabled(policy):
            me = seats[position]
            opponents = [(unit(s['bet']), unit(s['stack'])) for s in seats if s is not me and not s['folded']]
            for n in stack_press.sizes(len(snapshot['board']), bet, pot, unit(lo), unit(hi), unit(me['bet']),
                                       unit(me['stack']), opponents, legal):
                amount = min(hi, max(lo, round(n * scale)))
                if unit(amount) not in wire:
                    press.append(unit(amount)); wire[unit(amount)] = {'action': 'raise', 'amount': amount}
    elif 'all_in' in offered:
        n = unit(offered['all_in']['amount'])
        legal.append(n); wire[n] = {'action': 'all_in'}
    if not legal:
        raise ValueError('empty action menu')
    board = tuple(snapshot['board'])
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    hole = tuple(hero['hole_cards'])
    if len(hole) != 2:
        raise ValueError('missing private cards')
    stacks = tuple(unit(s['stack']) for s in seats)
    active = tuple(not s['folded'] for s in seats)
    # Training is 6-max only. A short-handed table is the same game as 6-max after
    # the earliest non-blind seats fold, so present it that way to the policy.
    pad = 6 - len(seats)
    if pad:
        stacks = stacks[:2] + (200,) * pad + stacks[2:]
        active = active[:2] + (False,) * pad + active[2:]
        if position >= 2:
            position += pad
    observation = Observation(position, position, hole, board, stacks, active,
        pot, unit(offered.get('call', {}).get('amount', 0)), street, tuple(legal), tuple(public_history))
    # Lean policies carry their own key function; poker/cfr.py stays frozen for the old line.
    key = policy.key(observation) if hasattr(policy, 'key') else information_key(observation, policy.abstraction)
    row = policy.sums.get(key)
    covered = bool(row and any(row))
    source = 'learned_policy'
    # Unseen states should not silently become uniform random raises.
    if hasattr(policy, 'fallback'):
        # Lean policies answer unseen states themselves with a tight rule player.
        before = getattr(policy, 'covered', 0)
        safeguards_before = getattr(policy, 'safeguards', 0)
        overrides_before = getattr(policy, 'pressure_overrides', 0)
        if hasattr(policy.fallback, 'last'):
            policy.fallback.last = None
        from poker.equity_policy import GuardedEquityPolicy
        profiles = opponent_profiles(policy.fallback, seats, pad, position)
        if isinstance(policy.fallback, GuardedEquityPolicy) and history_complete is not True:
            observation = replace(observation, history=())
        if isinstance(policy, history_cfr.HistoryPolicy):
            action = stack_press.act(policy, observation, press, history_complete=history_complete is True)
        else:
            action = stack_press.act(policy, observation, press)
        covered = getattr(policy, 'covered', 0) > before
        if getattr(policy, 'safeguards', 0) > safeguards_before:
            source = 'poker_safeguard'
        elif getattr(policy, 'pressure_overrides', 0) > overrides_before:
            source = 'pressure_guard'
        elif not covered:
            from poker.equity_policy import EquityPolicy
            source = ('preflop_rules' if not street else 'equity_fallback') if isinstance(policy.fallback, EquityPolicy) else 'rule_fallback'
    elif row is None or not any(row):
        source = 'safety_fallback'
        LOG.warning('Uncovered state: using check/fold/call fallback')
        action = 'call' if passive == 'check' else 'fold' if 'fold' in offered else 'call'
    else:
        action = policy.act(observation)
    steal_detail = None
    steals = getattr(policy, 'late_position_steals', None)
    if steals is not None and history_complete is True and isinstance(policy, history_cfr.HistoryPolicy):
        proposed = steals.propose(observation, action, seats, snapshot['big_blind'], snapshot['pot'], wire)
        if proposed is not None:
            target, steal_detail = proposed
            apply = not policy.late_position_report
            steal_detail.update(applied=apply, baseline_source=source, baseline_covered=covered)
            if apply:
                action, source, covered = target, 'late_position_steal', False
    LOG.info('Decision street=%s active=%s covered=%s action=%s',street,sum(observation.active),covered,wire[action]['action'])
    # Diagnostics for the optional session log; never sent on the wire.
    decision.last = dict(hand_id=hand, seat=hero['seat'], street=street, active=sum(observation.active),
                         covered=covered, decision_source=source, action=wire[action]['action'], amount=wire[action].get('amount'))
    if isinstance(policy, history_cfr.HistoryPolicy):
        decision.last['history_verified'] = history_complete is True
        decision.last['history_length'] = len(public_history)
        decision.last['observation'] = asdict(observation)
    if press:
        decision.last['stack_press_sizes'] = list(press)
    if steal_detail is not None:
        decision.last['late_position_steal'] = steal_detail
    if hasattr(policy, 'fallback') and profiles is not None:
        decision.last['opponent_profiles'] = {str(p): [round(b, 3), round(k, 3)] for p, (b, k) in profiles.items()}
    equity_detail = getattr(getattr(policy, 'fallback', None), 'last', None)
    if not covered and equity_detail is not None:
        decision.last['equity_fallback'] = equity_detail
    return dict(type='action', hand_id=hand, turn_token=token,
                client_action_id=str(uuid4()), **wire[action])


def opponent_profiles(fallback, seats, pad, position):
    """Install measured calling thresholds on an equity-guarded-v2 fallback.

    Maps each live opponent's padded 6-max position to its fitted (base, slope)
    from ``fallback.opponent_stats``; opponents without enough observations are
    left out, which makes v2 keep v1's bet size. Returns None when not enabled.
    """
    stats = getattr(fallback, 'opponent_stats', None)
    if stats is None:
        return None
    names = {}
    for i, s in enumerate(seats):
        p = i + pad if i >= 2 else i
        if p != position and not s['folded'] and s.get('name'):
            names[p] = s['name']
    fallback.profiles = stats.profiles(names)
    return fallback.profiles


def active_game(key):
    request = Request('https://api.openpoker.ai/api/me/active-game',
                      headers={'Authorization': 'Bearer ' + key,
                               'User-Agent': 'TriploPoker/0.1'})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def record(path, **row):
    """Append one JSON line to the optional read-only session log."""
    if path is None:
        return
    with Path(path).open('a') as handle:
        handle.write(json.dumps(dict(time=time.time(), **row)) + '\n')


def redact(value):
    """Never persist authentication or live action-authority tokens in diagnostics."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if k not in ('turn_token', 'api_key', 'token', 'Authorization')}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


async def run(args, policy, key):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    delay = 1
    # Retain exact payloads for token duplicates, including across reconnects.
    pending = {}
    sent_at = {}
    rejected = set()
    last_action_id = None
    log = getattr(args, 'log', None)
    trace = getattr(args, 'trace', None)
    minutes = getattr(args, 'max_minutes', None)
    stop_at = None if minutes is None else time.monotonic() + 60 * minutes
    seat = None
    history = PublicHistory()
    stats = getattr(getattr(policy, 'fallback', None), 'opponent_stats', None)
    stats_path = getattr(args, 'opponent_stats', None)
    def resync_request(table_id):
        return dict(type='resync_request', table_id=table_id,
                    last_table_seq=history.watermark if history.table_id == table_id else 0)
    while True:
        try:
            async with connect('wss://openpoker.ai/ws', additional_headers={
                    'Authorization': 'Bearer ' + key}, open_timeout=20) as ws:
                async def send(message):
                    await ws.send(json.dumps(message))
                async def join():
                    await send(dict(type='join_lobby', buy_in=args.buy_in))
                async for raw in ws:
                    msg = json.loads(raw)
                    kind = msg.get('type')
                    if trace and kind in ('hand_start', 'player_action', 'resync_response'):
                        record(trace, type='wire', message=redact(msg))
                    if kind == 'action_ack':
                        action_id = msg.get('client_action_id')
                        started = sent_at.pop(action_id, None)
                        record(log, type='action_ack', client_action_id=action_id, status=msg.get('status'),
                               latency_ms=None if started is None else 1000 * (time.monotonic() - started))
                        if msg.get('status') == 'accepted':
                            delay = 1
                    if kind != 'resync_response':
                        history.ingest(msg)
                    if stats is not None:
                        stats.ingest(msg)
                    if kind == 'connected':
                        game = await asyncio.to_thread(active_game, key)
                        if game.get('playing'):
                            await send(resync_request(game['table_id']))
                        else:
                            await join()
                    elif kind == 'lobby_joined':
                        await send(dict(type='set_auto_rebuy', enabled=args.auto_rebuy))
                    elif kind == 'your_turn':
                        # A resync supplies authoritative folds, bets, mapping and token together.
                        await send(resync_request(msg['table_id']))
                    elif kind == 'resync_response':
                        snapshot = msg['snapshot']
                        public_history, complete = history.resync(msg)
                        history.ingest(msg)
                        hero = snapshot.get('hero', {})
                        if hero.get('turn_token'):
                            identity = (snapshot['hand_id'], hero['turn_token'])
                            if identity not in pending:
                                try:
                                    pending[identity] = decision(snapshot, policy, public_history, complete)
                                    if isinstance(policy, history_cfr.HistoryPolicy):
                                        decision.last['history_reason'] = history.last_reason
                                except (KeyError, ValueError, StopIteration) as exc:
                                    LOG.error('Cannot translate snapshot: %s', exc)
                                    continue
                                seat = hero.get('seat', seat)
                                record(log, type='decision', **decision.last)
                            payload = pending[identity]
                            if payload['client_action_id'] not in rejected:
                                last_action_id = payload['client_action_id']
                                sent_at.setdefault(last_action_id, time.monotonic())
                                await send(payload)
                    elif kind == 'season_ended' and getattr(args, 'stop_on_season_end', False):
                        LOG.info('Season %s ended; stopping', msg.get('season_number'))
                        record(log, type='season_ended', season_number=msg.get('season_number'),
                               next_season_number=msg.get('next_season_number'))
                        return
                    elif kind in ('table_closed', 'season_ended', 'rebuy_confirmed'):
                        pending.clear()
                        sent_at.clear()
                        rejected.clear()
                        await join()
                    elif kind == 'table_joined':
                        seat = msg.get('seat', seat)
                        record(log, type='table_joined', table_id=msg.get('table_id'), seat=seat, players=len(msg.get('players', [])))
                    elif kind == 'hand_result':
                        pending.clear()
                        sent_at.clear()
                        rejected.clear()
                        record(log, type='hand_result', seat=seat, result={k: v for k, v in msg.items() if k != 'type'})
                        if stats is not None and stats_path is not None and stats.dirty:
                            stats.save(stats_path)
                        # Stop only between hands so no live decision is abandoned.
                        if stop_at is not None and time.monotonic() >= stop_at:
                            LOG.info('Session time limit reached between hands')
                            await send(dict(type='leave_table'))
                            return
                    elif kind == 'error' and msg.get('code') == 'already_seated':
                        await send(resync_request(msg['table_id']))
                    elif kind in ('error', 'action_rejected'):
                        code = msg.get('code', 'unknown')
                        LOG.error('Server rejection: %s', code)
                        record(log, type='action_rejected' if kind == 'action_rejected' else 'server_error', code=code)
                        if kind == 'action_rejected':
                            action_id = msg.get('client_action_id') or last_action_id
                            if action_id:
                                rejected.add(action_id)
                            if code in ('stale_turn_token', 'stale_hand_action', 'not_your_turn') and history.table_id:
                                await send(resync_request(history.table_id))
                        if code == 'auth_failed':
                            raise RuntimeError('OpenPoker authentication failed')
                    elif kind == 'busted' and not args.auto_rebuy:
                        LOG.info('Busted; automatic rebuy disabled')
                        return
        except (OSError, ConnectionClosed, TimeoutError) as exc:
            details = dict(error=type(exc).__name__, reason=str(exc), retry_seconds=delay)
            record(log, type='connection_lost', **details)
            LOG.warning('Connection lost (%s: %s); retrying in %ss', type(exc).__name__, exc, delay)
            await asyncio.sleep(delay)
            delay = min(30, delay * 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--buy-in', type=int, choices=range(1000, 5001), default=2000, metavar='1000..5000')
    parser.add_argument('--auto-rebuy', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-support', type=float, default=0.,
                        help='lean only: use tight fallback below this normalized average mass (default: disabled)')
    parser.add_argument('--min-visits', type=int, default=64,
                        help='history model: minimum sampled average visits before using a learned row')
    parser.add_argument('--fallback', choices=('tight-v1', *FALLBACKS), default='tight-v1',
                         help='rule fallback for history/lean policies; equity-v1 uses bounded range sampling')
    parser.add_argument('--late-position-steals', type=Path, metavar='FOLD_EVIDENCE_JSON',
                        help='opt in to opponent-aware late-position opens on top of equity-guarded-v3')
    parser.add_argument('--late-position-report', action='store_true',
                        help='log qualified late-position opens without changing the action')
    parser.add_argument('--trace', type=Path, help='optional redacted public events and replay snapshots for diagnostics')
    parser.add_argument('--log', type=Path, help='append decisions and hand results as JSON lines')
    parser.add_argument('--stack-press', action=argparse.BooleanOptionalAction, default=False,
                        help='Stack Press: with equity-guarded-v1, near-nut hands may raise up to the '
                             'whole stack (good vs weaker fields, risky vs strong tight ones)')
    parser.add_argument('--opponent-stats', type=Path,
                        help='equity-guarded-v2: per-opponent postflop calling stats JSON (loaded, updated live, '
                             'saved after each hand); enables value sizing for opponents with enough data')
    parser.add_argument('--opponent-min-faced', type=int, default=20,
                        help='faced postflop bets needed before an opponent gets a measured calling profile')
    parser.add_argument('--max-minutes', type=float, help='stop after the first completed hand past this many minutes')
    parser.add_argument('--stop-on-season-end', action='store_true',
                        help='exit when the current season ends instead of joining the next one')
    args = parser.parse_args()
    if args.late_position_steals is not None and args.fallback != VERSION_V3:
        parser.error(f'--late-position-steals requires --fallback {VERSION_V3}')
    if args.late_position_report and args.late_position_steals is None:
        parser.error('--late-position-report requires --late-position-steals')
    if args.late_position_report and args.log is None:
        parser.error('--late-position-report requires --log')
    if json.loads(args.checkpoint.read_text()).get('algorithm') in history_cfr.ALGORITHMS:
        saved = history_cfr.load(args.checkpoint)
    else:
        try:
            saved = lean_cfr.load(args.checkpoint)
        except ValueError:
            saved = load_checkpoint(args.checkpoint, model_id='bot-opponent-v1')
            if saved.get('abstraction') != 'texture-buckets-v2':
                parser.error('client requires a history-public-v1, history-compact-v1, lean-v3 or texture-buckets-v2 checkpoint')
    key = os.environ.get('OPEN_POKER_API_KEY')
    if not key:
        parser.error('set OPEN_POKER_API_KEY in the environment')
    logging.basicConfig(level=logging.INFO)
    LOG.info('Checkpoint sha256=%s abstraction=%s iterations=%s',hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),saved['abstraction'],saved['iterations'])
    if saved['abstraction'] in history_cfr.VERSIONS:
        if args.min_support != 0:
            parser.error('--min-support requires a lean checkpoint')
        policy = history_cfr.HistoryPolicy(saved, args.seed, args.min_visits)
        LOG.info('History model: learned rows require verified replay and min_visits=%s', args.min_visits)
        if policy.pressure_guard is not None:
            LOG.info('Decision guard: %s', policy.pressure_guard.version)
    elif saved['abstraction'] == lean_cfr.VERSION:
        policy = lean_cfr.LeanPolicy.from_checkpoint(saved, args.seed, args.min_support)
        LOG.info('Sparse-row fallback min_support=%s', args.min_support)
    else:
        if args.min_support != 0:
            parser.error('--min-support requires a lean checkpoint')
        policy = CFRPolicy(saved['sums'], args.seed, saved['abstraction'])
    if args.fallback in FALLBACKS:
        if not hasattr(policy, 'fallback'):
            parser.error('equity fallbacks require a history or lean checkpoint')
        if args.stack_press and not args.fallback.startswith('equity-guarded'):
            parser.error('--stack-press requires an equity-guarded fallback')
        policy.fallback = make_fallback(args.fallback, stack_press=args.stack_press)
        LOG.info('Fallback: %s (256 deterministic range samples), Stack Press %s',
                 args.fallback, 'ON' if args.stack_press else 'OFF')
    if args.late_position_steals is not None:
        if not isinstance(policy, history_cfr.HistoryPolicy):
            parser.error('--late-position-steals requires a history checkpoint')
        from poker.late_position import LatePositionSteals
        try:
            policy.late_position_steals = LatePositionSteals.load(args.late_position_steals)
        except (OSError, ValueError) as exc:
            parser.error(f'cannot load late-position evidence: {exc}')
        policy.late_position_report = args.late_position_report
        LOG.info('Late-Position Steals: %s (%d saved lineups)', 'report' if args.late_position_report else 'apply',
                 len(policy.late_position_steals.evidence))
    if args.opponent_stats is not None:
        if args.fallback not in ('equity-guarded-v2', VERSION_V3):
            parser.error('--opponent-stats requires --fallback equity-guarded-v2 or -v3')
        from poker.opponent_stats import OpponentStats
        policy.fallback.opponent_stats = OpponentStats.load(args.opponent_stats, min_faced=args.opponent_min_faced)
        stats = policy.fallback.opponent_stats
        LOG.info('Opponent stats: %d players, %d with >= %d faced postflop bets', len(stats.players),
                 sum(stats.faced(n) >= stats.min_faced for n in stats.players), stats.min_faced)
    try:
        asyncio.run(run(args, policy, key))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
