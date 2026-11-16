import dataclasses
from copy import deepcopy
from random import Random

from poker.fast_game import FastGame
from poker.history_cfr import HistoryPolicy, observation_key
from poker.openpoker_history import PublicHistory
from scripts.openpoker_client import decision

STREET=('preflop','flop','turn','river')


def fixture(seed=1, steps=3, players=6):
    game=FastGame(seed,players,button=0,lean=True)
    events=[dict(type='hand_start',table_id='table',hand_id='hand',table_seq=1,
                 dealer_seat=0,blinds=dict(small_blind=10,big_blind=20))]
    rng=Random(seed); seq=1
    for _ in range(steps):
        if not game.live: break
        o=game.observe()
        # Fold/check/call and pot raises, with enough nonterminal examples.
        action=rng.choice(o.legal)
        after=game.clone(); after.act(action)
        if not after.live: break
        game=after; seq+=rng.randrange(2,8)
        delta=sum(game.paid)-o.pot
        events.append(dict(type='player_action',table_id='table',hand_id='hand',table_seq=seq,
            seat=o.player,street=STREET[o.street],
            action=('check' if not o.call else 'call') if action=='call' else 'fold' if action=='fold' else 'raise',
            # Deliberately vary amount encoding: ledger uses pot differences.
            amount=delta*10 if action in ('call','fold') else action*10,
            pot=sum(game.paid)*10,stack=game.stacks[o.position]*10,
            contribution_delta=delta*10,pot_before=o.pot*10,to_call_before=o.call*10))
    o=game.observe(); lo=min((a for a in o.legal if isinstance(a,int)),default=None)
    legal=[dict(action='check') if not o.call else dict(action='call',amount=o.call*10)]
    if 'fold' in o.legal: legal.append(dict(action='fold'))
    if lo is not None:
        # Match the server's true min/max; adapter reconstructs lean sizes.
        top=max(game.bets); hi=game.stacks[o.position]+game.bets[o.position]
        legal.append(dict(action='raise',min=min(hi,top+max(2,game.raise_size))*10,max=hi*10))
    snap=dict(table_id='table',hand_id='hand',table_seq=seq+10,dealer_seat=0,big_blind=20,
        street=STREET[o.street],pot=o.pot*10,board=list(o.board),actor_seat=o.player,
        seats=[dict(seat=game.order[i],in_hand=True,folded=not game.active[i],
                    stack=game.stacks[i]*10,bet=game.bets[i]*10) for i in range(players)],
        hero=dict(seat=o.player,hole_cards=list(o.hole),turn_token='token',valid_actions=legal))
    return game,events,snap


def test_sequenced_replay_matches_training_keys_across_legal_hands():
    for seed in range(1,45):
        game,events,snapshot=fixture(seed,12)
        tracker=PublicHistory()
        # Replay order is sorted; duplicate resync must not append actions twice.
        response=dict(snapshot=snapshot,to_table_seq=snapshot['table_seq'],replayed_events=events[::-1])
        history,complete=tracker.resync(response)
        assert complete,tracker.last_reason
        assert history==tuple(game.history)
        again,complete=tracker.resync(response)
        assert complete and again==history
        o=game.observe()
        key=observation_key(o)
        rows={key:[1.]+[0.]*(len(o.legal)-1)}
        policy=HistoryPolicy(dict(sums=rows,visits={key:64},stack_bb=100))
        decision(snapshot,policy,history,complete)
        # A tactical safeguard can supersede a learned row, but the key is exact.
        assert decision.last['history_verified']
        assert policy.key(dataclasses.replace(o,history=history))==key
        if not policy.safeguards: assert policy.covered==1


def test_missing_replay_and_conflicting_snapshots_disable_learning():
    _,events,snap=fixture(1,4)
    tracker=PublicHistory()
    history,complete=tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=events[1:]))
    assert not complete and not history
    tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=events))
    wrong=deepcopy(snap); wrong['pot']+=10
    assert not tracker.reconstruct(wrong)[1]
    wrong=deepcopy(snap)
    actor=events[-1]['seat']
    next(s for s in wrong['seats'] if s['seat']==actor)['stack']+=10
    assert not tracker.reconstruct(wrong)[1]
    # Never use a trained root row without verification, even when it exists.
    game,events,snap=fixture(9,0)
    key=observation_key(game.observe())
    policy=HistoryPolicy(dict(sums={key:[1,0,0,0]},visits={key:100},stack_bb=100))
    decision(snap,policy)
    assert not decision.last['history_verified'] and policy.covered==0


def test_bad_duplicate_table_change_and_truncation_are_conservative():
    _,events,snap=fixture(4,4)
    tracker=PublicHistory(capacity=2)
    for event in events: tracker.ingest(event)
    if len(events)>2: assert not tracker.reconstruct(snap)[1]
    tracker=PublicHistory()
    for event in events: tracker.ingest(event)
    conflict=deepcopy(events[-1]); conflict['pot']+=20
    tracker.ingest(conflict)
    assert not tracker.reconstruct(snap)[1]
    tracker.ingest(dict(type='hand_start',table_id='another',hand_id='new',table_seq=1))
    assert not tracker.reconstruct(snap)[1]


def test_short_handed_history_matches_existing_six_seat_embedding():
    for players in (3,4,5):
        game,events,snap=fixture(3,4,players)
        tracker=PublicHistory()
        history,complete=tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=events))
        assert complete,tracker.last_reason
        pad=6-players
        assert [h[1] for h in history[:pad]]==list(range(2,2+pad))
        for old,new in zip(game.history,history[pad:]):
            assert old[3:]==new[3:]
            assert new[1]==(old[1]+pad if old[1]>=2 else old[1])


def test_async_client_replays_history_and_reuses_payload_for_duplicate_turn(tmp_path,monkeypatch):
    import asyncio
    import json
    import sys
    from types import ModuleType, SimpleNamespace
    import scripts.openpoker_client as client
    game,events,snap=fixture(3,4)
    key=observation_key(game.observe())
    policy=HistoryPolicy(dict(sums={key:[1.]+[0.]*(len(game.menu()[2])-1)},
                              visits={key:100},stack_bb=100))
    response=dict(type='resync_response',table_id='table',table_seq=snap['table_seq'],
                  to_table_seq=snap['table_seq'],snapshot=snap,replayed_events=events[::-1])
    messages=[dict(type='connected'),*events,response,response,dict(type='hand_result')]
    class Socket:
        def __init__(self): self.sent=[]
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def __aiter__(self): return self
        async def __anext__(self):
            if not messages: raise StopAsyncIteration
            return json.dumps(messages.pop(0))
        async def send(self,message): self.sent.append(json.loads(message))
    socket=Socket()
    try:
        import websockets.asyncio.client as websocket_client
        monkeypatch.setattr(websocket_client,'connect',lambda *a,**k:socket)
    except ModuleNotFoundError:
        ws_mod=ModuleType('websockets')
        ws_async=ModuleType('websockets.asyncio')
        ws_client=ModuleType('websockets.asyncio.client')
        ws_exc=ModuleType('websockets.exceptions')
        ws_client.connect=lambda *a,**k:socket
        ws_exc.ConnectionClosed=Exception
        monkeypatch.setitem(sys.modules,'websockets',ws_mod)
        monkeypatch.setitem(sys.modules,'websockets.asyncio',ws_async)
        monkeypatch.setitem(sys.modules,'websockets.asyncio.client',ws_client)
        monkeypatch.setitem(sys.modules,'websockets.exceptions',ws_exc)
    monkeypatch.setattr(client,'active_game',lambda key:dict(playing=True,table_id='table'))
    log=tmp_path/'session.jsonl'
    args=SimpleNamespace(log=log,max_minutes=0,auto_rebuy=False,buy_in=2000)
    asyncio.run(client.run(args,policy,'test-placeholder'))
    requests=[m for m in socket.sent if m['type']=='resync_request']
    assert requests[0]['last_table_seq']==0
    actions=[m for m in socket.sent if m['type']=='action']
    assert len(actions)==2 and actions[0]==actions[1]
    rows=[json.loads(line) for line in log.read_text().splitlines()]
    decisions=[r for r in rows if r['type']=='decision']
    assert len(decisions)==1
    assert decisions[0]['history_verified'] and decisions[0]['history_reason']=='verified'


def test_history_loader_in_client_selects_guarded_policy(tmp_path,monkeypatch):
    import sys
    import scripts.openpoker_client as client
    from poker.history_cfr import save
    from poker.sampled_cfr import SampledCFR
    checkpoint=tmp_path/'model.json'
    save(checkpoint,SampledCFR(6,1),1,20)
    monkeypatch.setattr(sys,'argv',['client','--checkpoint',str(checkpoint),'--max-minutes','1'])
    monkeypatch.setenv('OPEN_POKER_API_KEY','test-placeholder')
    selected=[]
    async def fake_run(args,policy,key): selected.append(policy)
    monkeypatch.setattr(client,'run',fake_run)
    client.main()
    assert isinstance(selected[0],HistoryPolicy) and selected[0].min_visits==64


def test_missing_actions_unsequenced_replay_and_new_hand_are_unavailable():
    _,events,snap=fixture(4,6)
    tracker=PublicHistory()
    partial=[events[0],*events[2:]]
    assert not tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=partial))[1]
    tracker=PublicHistory()
    unknown=deepcopy(events); unknown[0]['table_seq']=None
    assert not tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=unknown))[1]
    tracker=PublicHistory()
    tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=events))
    new=deepcopy(snap); new['hand_id']='new-hand'
    assert not tracker.reconstruct(new)[1]
    off_grid=deepcopy(snap); off_grid['pot']+=1
    assert not tracker.reconstruct(off_grid)[1]


def test_all_in_event_amount_is_not_mistaken_for_raise_to():
    # Find an actual nonterminal all-in in the engine-generated replay fixtures.
    found=False
    for seed in range(1,100):
        game,events,snap=fixture(seed,25)
        allins=[e for e in events[1:] if e['action']=='raise' and e['stack']==0]
        if not allins: continue
        found=True
        for e in allins:
            e['action']='all_in'
            e['amount']=999999  # deliberately ambiguous; authoritative delta wins
        tracker=PublicHistory()
        history,complete=tracker.resync(dict(snapshot=snap,to_table_seq=snap['table_seq'],replayed_events=events))
        assert complete,tracker.last_reason
        assert history==tuple(game.history)
        break
    assert found


def test_off_grid_live_chip_stacks_and_bets_reconstruct_cleanly():
    # Live OpenPoker stacks and bets (e.g., 4784 chips, 612-chip turn raise on a 10/20 table)
    # are not multiples of small_blind=10. Replay validates exact chips and normalizes output.
    events=[
        dict(type='hand_start',table_id='live',hand_id='h1',table_seq=10,dealer_seat=0,blinds=dict(small_blind=10,big_blind=20)),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=11,seat=3,street='preflop',action='raise',amount=50,pot=80,stack=5362),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=12,seat=4,street='preflop',action='call',amount=50,pot=130,stack=1950),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=13,seat=0,street='preflop',action='fold',amount=None,pot=130,stack=4784),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=14,seat=1,street='preflop',action='fold',amount=None,pot=130,stack=724),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=15,seat=2,street='preflop',action='call',amount=30,pot=160,stack=7066),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=16,seat=2,street='flop',action='check',amount=None,pot=160,stack=7066),
        dict(type='player_action',table_id='live',hand_id='h1',table_seq=17,seat=3,street='flop',action='raise',amount=112,pot=272,stack=5250),
    ]
    snap=dict(table_id='live',hand_id='h1',table_seq=18,dealer_seat=0,big_blind=20,street='flop',pot=272,
              board=['8s','7c','Jh'],actor_seat=4,
              seats=[
                  dict(seat=0,in_hand=True,folded=True,stack=4784,bet=0),
                  dict(seat=1,in_hand=True,folded=True,stack=724,bet=0),
                  dict(seat=2,in_hand=True,folded=False,stack=7066,bet=0),
                  dict(seat=3,in_hand=True,folded=False,stack=5250,bet=112),
                  dict(seat=4,in_hand=True,folded=False,stack=1950,bet=0),
              ],
              hero=dict(seat=4,hole_cards=['Qc','Js'],turn_token='tok',
                        valid_actions=[dict(action='call',amount=112),dict(action='fold'),dict(action='raise',min=224,max=1950)]))
    tracker=PublicHistory()
    history,complete=tracker.resync(dict(snapshot=snap,to_table_seq=18,replayed_events=events))
    assert complete,tracker.last_reason
    assert len(history)==8  # 1 pad fold + 7 real actions
    assert history[-1][3]==round(112/10)

