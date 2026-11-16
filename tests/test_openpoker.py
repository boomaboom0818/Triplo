from copy import deepcopy
from poker.cfr import CFRPolicy
from scripts.openpoker_client import decision


def snapshot():
    return dict(hand_id='h1', actor_seat=2, dealer_seat=5, big_blind=20,
                pot=30, board=[], seats=[dict(seat=i, in_hand=True, folded=False,
                stack=1990 if i == 0 else 1980 if i == 1 else 2000,
                bet=10 if i == 0 else 20 if i == 1 else 0) for i in range(6)],
                hero=dict(seat=2, hole_cards=['Ah', 'Kd'], turn_token='t1',
                valid_actions=[dict(action='fold'), dict(action='call', amount=20),
                               dict(action='raise', min=40, max=2000)]))


def test_unseen_folds_and_check_is_free():
    s = snapshot()
    policy = CFRPolicy({}, abstraction='texture-buckets-v2')
    a = decision(s, policy)
    assert a['action'] == 'fold' and a['turn_token'] == 't1' and a['hand_id'] == 'h1'
    s['hero']['valid_actions'] = [dict(action='check')]
    assert decision(s, policy)['action'] == 'check'


def test_normalized_menu_maps_to_wire_bounds():
    class Policy:
        abstraction = 'texture-buckets-v2'
        class Rows:
            def get(self, key): return [1]
        sums = Rows()
        def act(self, o):
            assert o.position == 2 and o.pot == 3 and o.call == 2
            assert o.legal == ('call', 'fold', 4, 5, 200)
            return 200
    s = snapshot()
    before = deepcopy(s)
    a = decision(s, Policy())
    assert a['action'] == 'raise' and a['amount'] == 2000
    assert s == before


def test_rejects_incomplete_fold_state():
    import pytest
    s = snapshot()
    del s['seats'][0]['folded']
    with pytest.raises(ValueError):
        decision(s, CFRPolicy({}, abstraction='texture-buckets-v2'))


def test_active_game_sets_user_agent_and_bearer(monkeypatch):
    import io
    import scripts.openpoker_client as client
    def respond(request, timeout):
        assert request.get_header('Authorization') == 'Bearer test-placeholder'
        assert request.get_header('User-agent') == 'TriploPoker/0.1'
        assert timeout == 15
        return io.BytesIO(b'{"playing": false}')
    monkeypatch.setattr(client, 'urlopen', respond)
    assert client.active_game('test-placeholder') == {'playing': False}


def test_session_log_records_diagnostics_without_changing_wire_payload(tmp_path):
    import json
    from scripts.openpoker_client import record
    action = decision(snapshot(), CFRPolicy({}, abstraction='texture-buckets-v2'))
    assert set(action) == {'type', 'hand_id', 'turn_token', 'client_action_id', 'action'}
    assert decision.last == dict(hand_id='h1', seat=2, street=0, active=6, covered=False, action='fold', amount=None)
    record(None, type='ignored')
    log = tmp_path / 'session.jsonl'
    record(log, type='decision', **decision.last)
    record(log, type='hand_result', seat=2, result=dict(payouts=[dict(seat=2, amount=30)]))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r['type'] for r in rows] == ['decision', 'hand_result'] and all(r['time'] > 0 for r in rows)
    assert rows[1]['result']['payouts'][0]['amount'] == 30


def test_short_handed_table_maps_to_six_max_with_early_seats_folded():
    from poker.cfr import information_key
    from poker.game import Game
    seen = []
    class Policy:
        abstraction = 'texture-buckets-v2'
        class Rows:
            def get(self, key): seen.append(key); return [1]
        sums = Rows()
        def act(self, o): seen.append(o); return 'call'
    s = snapshot()
    # Five-handed: seat 2 is absent, dealer 5, hero seat 3 is first to act.
    s['seats'] = [x for x in s['seats'] if x['seat'] != 2]
    s['actor_seat'] = s['hero']['seat'] = 3
    decision(s, Policy())
    key, o = seen
    assert o.position == 3 and o.active == (True, True, False, True, True, True) and len(o.stacks) == 6
    # Identical information set to the trained 6-max state after one early fold.
    game = Game(1); game.act('fold')
    trained = game.observe()
    assert trained.position == 3
    import dataclasses
    assert key == information_key(dataclasses.replace(trained, hole=('Ah', 'Kd')), 'texture-buckets-v2')
    # Blinds keep their positions.
    s['actor_seat'] = s['hero']['seat'] = 0
    seen.clear(); decision(s, Policy())
    assert seen[1].position == 0


def test_lean_policy_gets_lean_menu_and_rule_fallback():
    from poker.fast_game import FastGame
    from poker.lean_abstraction import observation_key
    from poker.lean_cfr import LeanPolicy
    seen = []
    class Spy(LeanPolicy):
        def act(self, o): seen.append(o); return super().act(o)
    # Trained game state for the same spot: first to act, 100bb, lean menu.
    trained = FastGame(1, lean=True).observe()
    key = observation_key(__import__('dataclasses').replace(trained, hole=('Ah', 'Kd')))
    policy = Spy({key: [0, 0, 0, 1]}, 1)
    a = decision(snapshot(), policy)
    assert seen[0].legal == ('call', 'fold', 4, 5) and a['action'] == 'raise' and a['amount'] == 50
    # Facing a 100bb shove the lean menu has no answer; the tight rule player decides.
    s = snapshot()
    s['pot'] = 2030; s['seats'][5]['bet'] = 2000; s['seats'][5]['stack'] = 0
    s['hero']['valid_actions'] = [dict(action='fold'), dict(action='call', amount=2000)]
    assert decision(s, policy)['action'] == 'fold'             # AKo is not a premium for the tight player
    s['hero']['hole_cards'] = ['Ah', 'As']
    assert decision(s, policy)['action'] == 'call'
    # A short stack brings the all-in back into the lean menu at the exact wire amount.
    s = snapshot()
    s['seats'][2]['stack'] = 75; s['hero']['valid_actions'][2] = dict(action='raise', min=40, max=75)
    decision(s, policy)
    assert seen[-1].legal == ('call', 'fold', 4, 5, 8)
