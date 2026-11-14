"""Snapshot-driven OpenPoker V2 adapter. Run with python -m scripts.openpoker_client."""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from uuid import uuid4
from urllib.request import Request, urlopen

from poker.cfr import CFRPolicy, information_key
from poker.config import load_checkpoint
from poker.game import Observation

LOG = logging.getLogger(__name__)


def decision(snapshot, policy):
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
        sizes = {unit(lo): lo, unit(hi): hi}
        for n in (bet + pot // 2, bet + pot):
            amount = min(hi, max(lo, round(n * scale)))
            sizes.setdefault(unit(amount), amount)
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
    observation = Observation(position, position, hole, board,
        tuple(unit(s['stack']) for s in seats), tuple(not s['folded'] for s in seats),
        pot, unit(offered.get('call', {}).get('amount', 0)), street, tuple(legal), ())
    key = information_key(observation, policy.abstraction)
    row = policy.sums.get(key)
    # Unseen states should not silently become uniform random raises.
    if row is None or not any(row):
        LOG.warning('Uncovered state: using check/fold/call fallback')
        action = 'call' if passive == 'check' else 'fold' if 'fold' in offered else 'call'
    else:
        action = policy.act(observation)
    return dict(type='action', hand_id=hand, turn_token=token,
                client_action_id=str(uuid4()), **wire[action])


def active_game(key):
    request = Request('https://api.openpoker.ai/api/me/active-game',
                      headers={'Authorization': 'Bearer ' + key,
                               'User-Agent': 'TriploPoker/0.1'})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


async def run(args, policy, key):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    delay = 1
    # Retain exact payloads for token duplicates, including across reconnects.
    pending = {}
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
                    if kind == 'connected':
                        game = await asyncio.to_thread(active_game, key)
                        if game.get('playing'):
                            await send(dict(type='resync_request', table_id=game['table_id']))
                        else:
                            await join()
                        delay = 1
                    elif kind == 'lobby_joined':
                        await send(dict(type='set_auto_rebuy', enabled=args.auto_rebuy))
                    elif kind == 'your_turn':
                        # A resync supplies authoritative folds, bets, mapping and token together.
                        await send(dict(type='resync_request', table_id=msg['table_id']))
                    elif kind == 'resync_response':
                        snapshot = msg['snapshot']
                        hero = snapshot.get('hero', {})
                        if hero.get('turn_token'):
                            identity = (snapshot['hand_id'], hero['turn_token'])
                            if identity not in pending:
                                try:
                                    pending[identity] = decision(snapshot, policy)
                                except (KeyError, ValueError, StopIteration) as exc:
                                    LOG.error('Cannot translate snapshot: %s', exc)
                                    continue
                            await send(pending[identity])
                    elif kind in ('table_closed', 'season_ended', 'rebuy_confirmed'):
                        pending.clear()
                        await join()
                    elif kind == 'hand_result':
                        pending.clear()
                    elif kind == 'error' and msg.get('code') == 'already_seated':
                        await send(dict(type='resync_request', table_id=msg['table_id']))
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
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint, model_id='bot-opponent-v1')
    if saved.get('abstraction') != 'texture-buckets-v2':
        parser.error('client requires a texture-buckets-v2 CFR checkpoint')
    key = os.environ.get('OPEN_POKER_API_KEY')
    if not key:
        parser.error('set OPEN_POKER_API_KEY in the environment')
    logging.basicConfig(level=logging.INFO)
    policy = CFRPolicy(saved['sums'], args.seed, saved['abstraction'])
    try:
        asyncio.run(run(args, policy, key))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
