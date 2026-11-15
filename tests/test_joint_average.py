import copy,math
import pytest
from poker.cfr import TraversalLimit,Holdem
from poker.prefix import PrefixMCCFR
from poker.joint_average import JointAverageMCCFR,joint_weight
from tests.test_adaptive_average import Replay,NeedChoice
from tests.test_stratified_average import Toy


def exact(s,state,out,owner=0,reach=1.):
    if state.terminal():return
    actor,key,actions=state.info();strategy=s.strategy(key,actions)
    if actor==owner:
        row=out.setdefault(key,[0.]*len(actions))
        for j,p in enumerate(strategy):row[j]+=reach*p
    for a,p in zip(actions,strategy):exact(s,state.child(a),out,owner,reach*(p if actor==owner else 1.))


class AliasToy(Toy):
    def info(self):return self.actors[len(self.h)],'depth:'+str(len(self.h)),('call','fold')
    def child(self,a):return AliasToy(self.actors,self.paid,self.h+(('call','fold').index(a),))


@pytest.mark.parametrize('depth',[1,2])
@pytest.mark.parametrize('root',[[1.,3.],[0.,3.]])
@pytest.mark.parametrize('opponents',[{}, {1:'guarded-v2',2:'guarded-v2'}])
@pytest.mark.parametrize('kind',[Toy,AliasToy])
def test_pair_of_inclusion_estimators_exactly_matches_original_target(depth,root,opponents,kind):
    state=kind((0,1,0,2,0),0);ref=JointAverageMCCFR(depth=depth);ref.regrets={state.info()[1]:root};expected={};exact(ref,state,expected)
    actual={}
    for source in ('regret','average'):
        pending=[()];total=0.
        while pending:
            path=pending.pop();s=JointAverageMCCFR(depth=depth);s.regrets=ref.regrets;s.opponents=opponents;s.rng=Replay(path)
            try:
                if source=='regret':s.outcome(state,0,{})
                else:s.trajectory_average(state,0)
            except NeedChoice as choice:
                pending.extend(path+(j,) for j,w in enumerate(choice.weights) if w>0);continue
            q=s.rng.p;total+=q
            for key,row in s.sums.items():
                dest=actual.setdefault(key,[0.]*len(row))
                for j,x in enumerate(row):dest[j]+=q*x
        assert total==pytest.approx(1.)
    for key,row in expected.items():assert actual.get(key,[0.,0.])==pytest.approx(row)


def test_full_step_preserves_exact_regret_rng_and_nodes_for_every_owner():
    for opponents in ({},{i:'guarded-v2' for i in range(6)}):
        a=PrefixMCCFR(samples=1,depth=2);b=JointAverageMCCFR(samples=1,depth=2)
        factory=lambda seed:Holdem(seed,6,'texture-buckets-v2')
        a.step(factory,6,max_nodes=100000,opponents=opponents);b.step(factory,6,max_nodes=100000,opponents=opponents)
        assert a.regrets==b.regrets and a.rng.getstate()==b.rng.getstate() and a.nodes==b.nodes
        assert b.traversal_stats['max_joint_average_to_original_weight_ratio']<=1+1e-12


def test_defensive_bound_and_rollback():
    for own,qa,pb in [(0.,0.,0.),(-2.,-10.,-math.inf),(-4.,-20.,-5.)]:
        w,ratio=joint_weight(own,qa,pb,.125);assert ratio<=1+1e-12 and w<=.125*math.exp(-qa)*(1+1e-12)
    assert joint_weight(-math.inf,-1.,-1.,1.)==(0.,0.)
    with pytest.raises(TraversalLimit,match='underflowed positive'):joint_weight(-1000.,0.,0.,1.)
    s=JointAverageMCCFR();before=copy.deepcopy((s.regrets,s.sums,s.rng.getstate(),s.nodes,s.iterations,s.diagnostics,s.traversal_stats,s.epoch_weight_moments))
    with pytest.raises(TraversalLimit):s.step(lambda seed:Toy(),3,max_nodes=3)
    assert before==(s.regrets,s.sums,s.rng.getstate(),s.nodes,s.iterations,s.diagnostics,s.traversal_stats,s.epoch_weight_moments)
