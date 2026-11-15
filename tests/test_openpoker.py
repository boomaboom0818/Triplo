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
