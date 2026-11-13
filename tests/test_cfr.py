from poker.cfr import MCCFR,Kuhn,Holdem,normalize,holdem_key
from poker.train_cfr import train
from poker.game import Game


def test_regret_matching():
    assert normalize([-1,3,1])==[0,.75,.25]
    assert normalize([0,0])==[.5,.5]


def test_resume(tmp_path):
    full=train(20,tmp_path/'full.json')
    path=tmp_path/'split.json';train(7,path)
    split=train(20,path,resume=path)
    for name in ('regrets','sums','rng','iterations','nodes'):assert full[name]==split[name]


def test_information_boundaries_and_branch_isolation():
    state=Holdem(1);actor,key,actions=state.info()
    child=state.child('call')
    assert state.game.history==[] and child.game.history
    # Changing another player's private cards cannot change the actor's key.
    other=(state.game.state.actor_index+1)%6
    state.game.state.hole_cards[other]=[]
    assert state.info()[1]==key
    assert holdem_key(Game(1).observe())==key


def test_kuhn_payoffs():
    for h,expected in [('pp',[1,-1]),('bp',[1,-1]),('pbp',[-1,1]),('bb',[2,-2])]:
        s=Kuhn(0,(2,0),h);assert s.terminal() and s.returns()==expected


def test_kuhn_value():
    from itertools import permutations
    solver=MCCFR(7)
    for _ in range(8000):solver.step(Kuhn,2)
    def value(s):
        if s.terminal():return s.returns()[0]
        _,key,actions=s.info()
        return sum(p*value(s.child(a)) for p,a in zip(normalize(solver.sums[key]),actions))
    result=sum(value(Kuhn(0,c)) for c in permutations(range(3),2))/6
    assert abs(result+1/18)<.025  # Known two-player Kuhn value, not hold'em evidence.


def test_three_player_regret_and_average_weights():
    class Toy:
        def __init__(self,h=()):self.h=h
        def terminal(self):return len(self.h)==3
        def info(self):return len(self.h),str(len(self.h)),(0,1)
        def child(self,a):return Toy(self.h+(a,))
        def returns(self):return [self.h[i]-self.h[(i+1)%3] for i in range(3)]
    solver=MCCFR(1);delta={}
    solver.regret(Toy(),2,delta)
    assert delta['2']==[-.5,.5]
    solver.average(Toy(),2)
    # Two uniformly sampled external decisions: inverse sampling reach = 4.
    assert solver.sums['2']==[2,2]


def test_budget_rollback_and_resume(tmp_path):
    import pytest
    from poker.cfr import TraversalLimit
    solver=MCCFR(1);before=solver.rng.getstate()
    with pytest.raises(TraversalLimit):solver.step(Kuhn,2,max_nodes=2)
    assert solver.regrets=={} and solver.sums=={} and solver.nodes==0
    assert solver.rng.getstate()==before and solver.iterations==0
    solver.step(Kuhn,2)
    reference=MCCFR(1);reference.step(Kuhn,2)
    assert solver.regrets==reference.regrets and solver.sums==reference.sums
    path=tmp_path/'bounded.json'
    stopped=train(5,path,max_iteration_nodes=2)
    assert stopped['iterations']==0
    resumed=train(5,path,resume=path)
    full=train(5,tmp_path/'full.json')
    assert resumed['regrets']==full['regrets'] and resumed['sums']==full['sums']
    assert path.with_name(path.name+'.previous').exists()


def test_output_lock(tmp_path):
    import pytest
    from poker.train_cfr import run_lock
    path=tmp_path/'locked.json'
    with run_lock(path):
        with pytest.raises(ValueError,match='another trainer'):train(1,path)
