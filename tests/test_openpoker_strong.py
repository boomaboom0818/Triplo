import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from poker.opponent_strength import OpponentField
from scripts import openpoker_client as client
from scripts import openpoker_strong
from scripts.openpoker_strong import StrongAvoider


def row(name, balance, **extra):
    return dict(dict(bot_name=name, hands_played=1000, chip_balance=balance,
                     chips_at_table=2000, rebuys=0, status='playing'), **extra)


# 1000 hands + 500 prior: a 90000 balance is +290 BB/100 (strong), two rebuys at zero is -20 (weak).
def field():
    return OpponentField([row('hero', 90000, account_agent_id='ours'),
                          row('sibling', 90000, account_agent_id='ours'),
                          row('shark', 90000), row('shark2', 90000),
                          row('reg', 18000), row('loser', 0, rebuys=2),
                          row('preset', 18000, bot_kind='no_code')])


def test_rating_and_strong_rule():
    f = field()
    assert f.players['shark'].results_bb100 == pytest.approx(290)
    assert f.players['reg'].results_bb100 == pytest.approx(50)
    assert f.players['loser'].results_bb100 == pytest.approx(-20)
    assert f.players['hero'].results_bb100 is not None
    short = OpponentField([row('new', 90000, hands_played=199)])
    assert short.players['new'].results_bb100 is None  # too few hands to rate
    calm = f.strong_check('hero', ['hero', 'reg', 'sibling', 'stranger'])
    assert not calm.leave and calm.strong == () and calm.reason == 'no strong opponent'
    alone = f.strong_check('hero', ['shark', 'reg'])
    assert alone.leave and alone.strong == ('shark',)
    paid = f.strong_check('hero', ['shark', 'preset', 'reg'])
    assert not paid.leave and paid.weak == ('preset',)
    assert f.strong_check('hero', ['shark', 'shark2', 'loser']).leave
    assert not f.strong_check('hero', ['shark', 'shark2', 'loser', 'preset']).leave
    assert not f.strong_check('hero', ['shark', 'reg'], threshold=300).leave


def avoider(mode='leave'):
    a = StrongAvoider(mode)
    a.hero, a.field, a.field_time = 'hero', field(), time.monotonic()
    a.seated('T')
    a.hero_seat, a.hero_stack = 3, 2000
    a.live = {3: 'hero', 0: 'reg'}
    return a


def test_gates_and_backoff():
    a = avoider()
    assert a.check('hand_result', 2000) is None  # nobody strong
    a.joined(dict(seat=1, name='shark'))
    found = a.check('player_joined', 2000)
    assert found['leave'] and found['strong'] == ['shark'] and found['trigger'] == 'player_joined'
    a.hero_stack = 1999
    assert a.check('hand_result', 2000)['reason'] == 'buy-in not covered by stack'
    a.hero_stack = 2000
    a.field_time -= 61
    assert a.check('hand_result', 2000) is None  # stale leaderboard: never act
    a.field_time = time.monotonic()
    a.left_table()
    assert a.leaving == 'T' and a.check('hand_result', 2000) is None  # already leaving
    def reseat(table):
        a.seated(table); a.leaving = None
        a.hero_seat, a.hero_stack, a.live = 3, 2000, {3: 'hero', 1: 'shark'}
    reseat('T2')
    assert a.check('table_joined', 2000)['cooldown_s'] == 30
    a.last_leave -= 31
    assert a.check('table_joined', 2000)['leave']
    a.left_table()
    reseat('T3')
    a.last_leave -= 31
    assert a.check('table_joined', 2000)['cooldown_s'] == 60  # doubles per consecutive leave
    a.left(dict(seat=1))
    assert a.check('hand_result', 2000) is None
    for n in range(1, 5):
        assert a.counted(dict(type='hand_result', table_id='T3', hand_id=f'h{n}', table_seq=n,
                              final_stacks={'3': 2100}))
    assert not a.counted(dict(type='hand_result', table_id='T3', hand_id='h4', table_seq=4))  # duplicate
    assert a.leaves == 0 and a.hero_stack == 2100  # a table that sticks resets the backoff


def test_snapshot_keeps_occupied_seats_and_hero_stack():
    a = StrongAvoider()
    a.seated('T')
    a.snapshot(dict(table_id='T', table_seq=5, hero={'seat': 3}, seats=[
        dict(seat=0, name='shark', status='away', stack=9000),
        dict(seat=1, name='', status='empty'),
        dict(seat=3, name='hero', status='active', stack=2500)]))
    assert a.live == {0: 'shark', 3: 'hero'} and a.hero_seat == 3 and a.hero_stack == 2500
    a.snapshot(dict(table_id='other', table_seq=9, seats=[]))
    a.snapshot(dict(table_id='T', table_seq=4, seats=[]))  # older than the last one
    assert a.live == {0: 'shark', 3: 'hero'}


def test_failed_background_read_is_retried_later_not_each_hand(monkeypatch):
    def fail(): raise OSError('offline')
    monkeypatch.setattr(openpoker_strong, 'fetch_field', fail)
    async def check():
        a = StrongAvoider()
        a.refresh()
        await asyncio.gather(a.task, return_exceptions=True)
        a.refresh()
        assert a.field is None and a.task is None
    asyncio.run(check())


def snapshot(n):
    return dict(type='table_state', table_id='T', hand_id=f'h{n}', table_seq=n*10, hero={'seat': 3},
                seats=[dict(seat=0, name='reg', status='active', stack=2000),
                       dict(seat=3, name='hero', status='active', stack=2000)])


def play(monkeypatch, tmp_path, mode, messages):
    websockets = pytest.importorskip('websockets.asyncio.client')
    reads = []
    def refresh(self):
        reads.append(1)
        self.field, self.field_time = field(), time.monotonic()
    monkeypatch.setattr(StrongAvoider, 'refresh', refresh)
    monkeypatch.setattr(openpoker_strong, 'REQUEUE_DELAY', .05)
    monkeypatch.setattr(client, 'active_game', lambda _: dict(playing=True, table_id='T', seat=3))
    sent = []
    class Socket:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def send(self, message): sent.append(json.loads(message))
        def __aiter__(self): return messages(sent)
    monkeypatch.setattr(websockets, 'connect', lambda *a, **k: Socket())
    log = tmp_path / 'session.jsonl'
    args = SimpleNamespace(stop_on_season_end=True, auto_rebuy=False, buy_in=2000, log=log)
    if mode is not None:
        args.avoid_strong, args.strong_bb100 = mode, None
    asyncio.run(client.run(args, object(), 'test-key'))
    return sent, [json.loads(line) for line in log.read_text().splitlines()], reads


@pytest.mark.parametrize('mode', ['leave', 'report'])
def test_strong_arrival_leaves_and_requeues_after_delay(monkeypatch, tmp_path, mode):
    async def messages(sent):
        yield json.dumps(dict(type='connected', name='hero'))
        yield json.dumps(snapshot(1))
        yield json.dumps(dict(type='hand_result', table_id='T', hand_id='h1', table_seq=15,
                              final_stacks={'0': 2000, '3': 2000}))
        assert not any(m['type'] == 'leave_table' for m in sent)
        yield json.dumps(snapshot(2))
        yield json.dumps(dict(type='player_joined', table_id='T', seat=1, name='shark', stack=5000))
        leaves = [m for m in sent if m['type'] == 'leave_table']
        assert leaves == ([dict(type='leave_table')] if mode == 'leave' else [])
        if mode == 'leave':
            yield json.dumps(dict(type='error', code='leave_pending'))
            # The server may start the next hand before it processes our leave.
            yield json.dumps(dict(type='your_turn', table_id='T'))
            assert sent[-1]['type'] == 'resync_request'
            yield json.dumps(dict(type='player_left', seat=3, name='hero'))
            assert not any(m['type'] == 'join_lobby' for m in sent)  # waits for the seat to refill
            await asyncio.sleep(.2)
            assert [m for m in sent if m['type'] == 'join_lobby'] == [dict(type='join_lobby', buy_in=2000)]
            yield json.dumps(dict(type='table_closed', table_id='T'))  # already requeued: no second join
            yield json.dumps(dict(type='table_joined', table_id='T9', seat=2,
                                  players=[dict(seat=2, name='hero', stack=2000), dict(seat=4, name='reg')]))
        yield json.dumps(dict(type='season_ended'))
    sent, rows, _ = play(monkeypatch, tmp_path, mode, messages)
    checks = [r for r in rows if r['type'] == 'strong_opponent_check']
    assert len(checks) == 1 and checks[0]['strong'] == ['shark'] and checks[0]['leave']
    assert checks[0]['leave_requested'] == (mode == 'leave')
    assert not any(r['type'] == 'server_error' for r in rows)
    assert len([m for m in sent if m['type'] == 'join_lobby']) == (1 if mode == 'leave' else 0)


def test_off_is_unchanged(monkeypatch, tmp_path):
    async def messages(sent):
        yield json.dumps(dict(type='connected', name='hero'))
        yield json.dumps(snapshot(1))
        yield json.dumps(dict(type='player_joined', table_id='T', seat=1, name='shark', stack=5000))
        yield json.dumps(dict(type='hand_result', table_id='T', hand_id='h1', table_seq=15,
                              final_stacks={'0': 2000, '3': 2000}))
        yield json.dumps(dict(type='season_ended'))
    for mode in (None, 'off'):
        sent, rows, reads = play(monkeypatch, tmp_path, mode, messages)
        assert not reads and not any(m['type'] == 'leave_table' for m in sent)
        assert not any(r['type'] == 'strong_opponent_check' for r in rows)


def test_silent_socket_reconnects_when_idle_timeout_is_set(monkeypatch, tmp_path):
    websockets = pytest.importorskip('websockets.asyncio.client')
    monkeypatch.setattr(client, 'active_game', lambda _: dict(playing=True, table_id='T', seat=3))
    real_sleep = asyncio.sleep
    monkeypatch.setattr(client.asyncio, 'sleep', lambda *_: real_sleep(0))
    opened, sent = [], []
    class Socket:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def send(self, message): sent.append(json.loads(message))
        def __aiter__(self): return self.messages()
        async def messages(self):
            opened.append(1)
            yield json.dumps(dict(type='connected', name='hero'))
            if len(opened) == 1:
                await asyncio.Event().wait()  # the server goes silent
            yield json.dumps(dict(type='season_ended'))
    monkeypatch.setattr(websockets, 'connect', lambda *a, **k: Socket())
    log = tmp_path / 'session.jsonl'
    args = SimpleNamespace(stop_on_season_end=True, auto_rebuy=False, buy_in=2000, log=log, idle_timeout=.05)
    asyncio.run(client.run(args, object(), 'test-key'))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    lost = [r for r in rows if r['type'] == 'connection_lost']
    assert len(opened) == 2 and len(lost) == 1 and lost[0]['idle'] and lost[0]['error'] == 'IdleTimeout'
    assert [m['type'] for m in sent].count('resync_request') == 2  # resumes the seated table


def test_idle_timeout_defaults_to_never(monkeypatch, tmp_path):
    async def messages(sent):
        yield json.dumps(dict(type='connected', name='hero'))
        await asyncio.sleep(.2)  # long silence is fine when the watchdog is off
        yield json.dumps(dict(type='season_ended'))
    sent, rows, _ = play(monkeypatch, tmp_path, None, messages)
    assert not any(r['type'] == 'connection_lost' for r in rows)
