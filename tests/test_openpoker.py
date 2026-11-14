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
