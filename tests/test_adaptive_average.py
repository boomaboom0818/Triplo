import copy
import pytest
from poker.adaptive_average import AdaptiveAverageMCCFR
from poker.cfr import TraversalLimit
from tests.test_stratified_average import Toy,exact


class NeedChoice(Exception):
    def __init__(self,weights):self.weights=weights


class Replay:
    def __init__(self,path):self.path=iter(path);self.p=1.
    def choices(self,actions,weights):
        try:j=next(self.path)
        except StopIteration:raise NeedChoice(weights)
        self.p*=weights[j]/sum(weights);return [actions[j]]


@pytest.mark.parametrize('cap',[1.,2.,4.])
@pytest.mark.parametrize('root',[[1.,3.],[0.,3.]])
@pytest.mark.parametrize('actors',[(0,1,2,0),(1,0,2,0)])
def test_complete_variable_sampling_tree_has_exact_original_target(cap,root,actors):
    ref=AdaptiveAverageMCCFR(cap=cap);ref.regrets={str(()):root};expected={};exact(ref,Toy(actors),expected)
    actual={};total=0.;pending=[()]
    while pending:
        path=pending.pop();s=AdaptiveAverageMCCFR(cap=cap);s.regrets=ref.regrets;s.rng=Replay(path)
        try:s.trajectory_average(Toy(actors),0)
        except NeedChoice as choice:
            pending.extend(path+(j,) for j,w in enumerate(choice.weights) if w>0);continue
        q=s.rng.p;total+=q
        for key,row in s.sums.items():
            dest=actual.setdefault(key,[0.,0.])
            for j,x in enumerate(row):dest[j]+=q*x
        assert all(v['maximum']<=cap*(1+1e-12) for v in s.average_weight_moments.values())
    assert total==pytest.approx(1.)
    for key,row in expected.items():assert actual.get(key,[0.,0.])==pytest.approx(row)


def test_adaptive_average_budget_failure_rolls_back_all_learning():
    s=AdaptiveAverageMCCFR(cap=1);before=copy.deepcopy((s.regrets,s.sums,s.rng.getstate(),s.nodes,s.iterations,s.diagnostics,s.traversal_stats,s.epoch_weight_moments))
    with pytest.raises(TraversalLimit):s.step(lambda seed:Toy(),3,max_nodes=2)
    assert before==(s.regrets,s.sums,s.rng.getstate(),s.nodes,s.iterations,s.diagnostics,s.traversal_stats,s.epoch_weight_moments)
