import math
from itertools import permutations
from random import Random

import pytest
from poker.fast_eval import CARD_ID
from poker.river import RiverGame, RiverState, audit, best_response, expected_value
from poker.sampled_cfr import SampledCFR


def ids(*names): return tuple(CARD_ID[n] for n in names)


def kuhn_river():
    board=ids('2c','4d','7h','9s','Jc')
    hands=[ids('As','Ad'),ids('Ks','Kd'),ids('Qs','Qd')]
    return RiverGame(board,[hands,hands])


def test_exact_river_game_agrees_with_kuhn_and_removes_blocked_deals():
    g=kuhn_river()
    assert len(g.joint)==6
    assert sum(g.joint.values())==pytest.approx(1.)
    assert all(i!=j for i,j in g.joint)
    low_high=(0,2)
    assert RiverState(g,low_high,'bc').utility(0)==-2
    assert RiverState(g,low_high,'bf').utility(0)==1
    assert RiverState(g,low_high,'cbf').utility(0)==-1
    result=audit(g,lambda s:[.5,.5])
    assert result['nash_conv']==pytest.approx(11/12)
    solver=SampledCFR(2,8)
    for _ in range(20000): solver.iterate(g.sample(solver.rng))
    policy=lambda s:solver.average(s.key(),2)
    learned=audit(g,policy)
    assert learned['nash_conv']<.05
    assert learned['value_player0']==pytest.approx(-1/18,abs=.01)


def test_key_does_not_depend_on_other_players_actual_cards():
    g=kuhn_river()
    assert RiverState(g,(0,1)).key()==RiverState(g,(0,2)).key()
    assert RiverState(g,(0,1),'c').key()!=RiverState(g,(0,2),'c').key()


def test_weighted_ranges_blockers_and_ties():
    board=ids('As','Ks','Qs','Js','Ts')
    a=ids('2c','3d'); b=ids('4h','5h'); c=ids('2c','6h')
    g=RiverGame(board,[[a,b],[b,c]],[[1,3],[2,4]])
    pair_ab=(g.ranges[0].index(tuple(sorted(a))),g.ranges[1].index(tuple(sorted(b))))
    pair_bc=(g.ranges[0].index(tuple(sorted(b))),g.ranges[1].index(tuple(sorted(c))))
    assert len(g.joint)==2
    assert g.joint[pair_ab]==pytest.approx(2/14)
    assert g.joint[pair_bc]==pytest.approx(12/14)
    assert RiverState(g,pair_ab,'bc').utility(0)==0
    assert expected_value(g,[lambda s:[1.,0.]]*2)==0
    with pytest.raises(ValueError): RiverGame(board,[[a],[a]])
    with pytest.raises(ValueError): RiverGame(board,[[a],[b]],[[math.nan],[1]])


def test_heldout_games_are_new_cards_with_same_public_structure():
    from scripts.river_experiment import cases
    train,heldout=cases()
    assert len(train)==len(heldout)==8
    assert {g.board for g in train}.isdisjoint(g.board for g in heldout)
    for a,b in zip(train,heldout): assert a.signature==b.signature


def test_policy_learns_value_betting_bluffing_calls_and_folds():
    g=kuhn_river(); solver=SampledCFR(2,22)
    for _ in range(30000): solver.iterate(g.sample(solver.rng))
    # Player1 has a unique equilibrium policy in this Kuhn-equivalent river.
    probability=lambda index,h:solver.average(RiverState(g,(1 if index!=1 else 0,index),h).key(),2)
    assert probability(2,'c')[1]>.95  # value-bet highest holding after a check
    assert .20<probability(0,'c')[1]<.45  # mix bluffs with the lowest holding
    assert probability(0,'b')[0]>.95  # fold a holding that cannot win
    assert .20<probability(1,'b')[1]<.45  # bluff-catch sometimes with the middle
    assert probability(2,'b')[1]>.95  # call the highest holding


def test_saved_river_policy_transfers_only_matching_public_structure(tmp_path):
    import json
    from poker.river import RiverPolicy
    from scripts.river_experiment import cases
    train,heldout=cases()
    game=train[0]; state=RiverState(game,next(iter(game.joint)))
    checkpoint=dict(schema=1,algorithm='bounded-river-es-v1',sums={state.key():[1.,3.]})
    policy=RiverPolicy(checkpoint)
    other=RiverState(heldout[0],state.pair)
    assert policy.distribution(other)==(.25,.75)
    checkpoint['sums'][state.key()][0]=99
    assert policy.distribution(state)==(.25,.75)
    with pytest.raises(ValueError,match='support'):
        policy.probabilities(game,1,game.ranges[1][0],'c')
    with pytest.raises(ValueError,match='range'):
        policy.probabilities(game,0,(50,51))
    path=tmp_path/'p.json'; path.write_text(json.dumps(dict(schema=1,algorithm='wrong',sums={})))
    with pytest.raises(ValueError): RiverPolicy.load(path)


def test_solve_unseen_river_game_with_unreached_branch_has_complete_policy_support():
    from poker.river_solver import solve
    board=ids('2c','4d','7h','9s','Jc')
    # Every P0 holding beats every P1 holding, so P0 quickly converges to betting
    # 100% of the time and P0's 'cb' node can become unreached during opponent passes.
    g=RiverGame(board,[[ids('As','Ad'),ids('Ks','Kd')],[ids('Qc','Qd'),ids('Tc','Td')]],[[1,2],[2,1]],2.,1.5)
    policy,ckpt=solve(g,iterations=500,seed=5,max_seconds=10)
    stats=audit(g,policy.distribution)
    assert stats['exploitability']<.02
    assert ckpt['signature']==g.signature
    with pytest.raises(ValueError): solve(g,iterations=0)

