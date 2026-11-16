"""Snapshot-driven OpenPoker V2 adapter. Run with python -m scripts.openpoker_client."""
import hashlib
import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from urllib.request import Request, urlopen

from poker.cfr import CFRPolicy, information_key
from poker.config import load_checkpoint
from poker.fast_game import LEAN_MENU
from poker.game import Observation
from poker import lean_cfr
from poker import history_cfr
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
    legal, wire = [], {}
    if passive:
        legal.append('call'); wire['call'] = {'action': passive}
    if 'fold' in offered:
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
        if lean and unit(hi) <= bet + 2 * pot:
            sizes[unit(hi)] = hi
        for n, amount in sorted(sizes.items()):
            legal.append(n); wire[n] = {'action': 'raise', 'amount': amount}
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
    # Unseen states should not silently become uniform random raises.
    if hasattr(policy, 'fallback'):
        # Lean policies answer unseen states themselves with a tight rule player.
        before = getattr(policy, 'covered', 0)
        if isinstance(policy, history_cfr.HistoryPolicy):
            action = policy.act(observation, history_complete=history_complete is True)
        else:
            action = policy.act(observation)
        covered = getattr(policy, 'covered', 0) > before
    elif row is None or not any(row):
        LOG.warning('Uncovered state: using check/fold/call fallback')
        action = 'call' if passive == 'check' else 'fold' if 'fold' in offered else 'call'
    else:
        action = policy.act(observation)
    LOG.info('Decision street=%s active=%s covered=%s action=%s',street,sum(observation.active),covered,wire[action]['action'])
    # Diagnostics for the optional session log; never sent on the wire.
    decision.last = dict(hand_id=hand, seat=hero['seat'], street=street, active=sum(observation.active),
                         covered=covered, action=wire[action]['action'], amount=wire[action].get('amount'))
    if isinstance(policy, history_cfr.HistoryPolicy):
        decision.last['history_verified'] = history_complete is True
        decision.last['history_length'] = len(public_history)
        decision.last['observation'] = asdict(observation)
    return dict(type='action', hand_id=hand, turn_token=token,
                client_action_id=str(uuid4()), **wire[action])


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


async def run(args, policy, key):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    delay = 1
    # Retain exact payloads for token duplicates, including across reconnects.
    pending = {}
    log = getattr(args, 'log', None)
    minutes = getattr(args, 'max_minutes', None)
    stop_at = None if minutes is None else time.monotonic() + 60 * minutes
    seat = None
    history = PublicHistory()
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
                    if kind != 'resync_response':
                        history.ingest(msg)
                    if kind == 'connected':
                        game = await asyncio.to_thread(active_game, key)
                        if game.get('playing'):
                            await send(resync_request(game['table_id']))
                        else:
                            await join()
                        delay = 1
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
                            await send(pending[identity])
                    elif kind in ('table_closed', 'season_ended', 'rebuy_confirmed'):
                        pending.clear()
                        await join()
                    elif kind == 'table_joined':
                        seat = msg.get('seat', seat)
                        record(log, type='table_joined', table_id=msg.get('table_id'), seat=seat, players=len(msg.get('players', [])))
                    elif kind == 'hand_result':
                        pending.clear()
                        record(log, type='hand_result', seat=seat, result={k: v for k, v in msg.items() if k != 'type'})
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
                        if code == 'auth_failed':
                            raise RuntimeError('OpenPoker authentication failed')
                    elif kind == 'busted' and not args.auto_rebuy:
                        LOG.info('Busted; automatic rebuy disabled')
                        return
        except (OSError, ConnectionClosed, TimeoutError):
            LOG.warning('Connection lost; retrying in %ss', delay)
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
    parser.add_argument('--log', type=Path, help='append decisions and hand results as JSON lines')
    parser.add_argument('--max-minutes', type=float, help='stop after the first completed hand past this many minutes')
    args = parser.parse_args()
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
    elif saved['abstraction'] == lean_cfr.VERSION:
        policy = lean_cfr.LeanPolicy.from_checkpoint(saved, args.seed, args.min_support)
        LOG.info('Sparse-row fallback min_support=%s', args.min_support)
    else:
        if args.min_support != 0:
            parser.error('--min-support requires a lean checkpoint')
        policy = CFRPolicy(saved['sums'], args.seed, saved['abstraction'])
    try:
        asyncio.run(run(args, policy, key))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
