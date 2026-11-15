from random import Random
import pytest
from poker.cfr import Holdem
from poker.stratified_average import StratifiedMCCFR
from poker.live_owner_average import LiveOwnerMCCFR


@pytest.mark.parametrize('mode',[0,1])
def test_exact_per_path_updates_for_every_owner_and_fewer_nodes(mode):
    pruned=0
    for owner in range(6):
        for deal in (10500000,10500001,10500002,10500003):
            state=Holdem(deal,6,'texture-buckets-v2');a=StratifiedMCCFR();b=LiveOwnerMCCFR()
            a.rng=Random(deal+owner*100);b.rng=Random(deal+owner*100)
            a.average_path(state,owner,mode,.125);b.average_path(state,owner,mode,.125)
            assert a.sums==b.sums and a.average_weight_moments==b.average_weight_moments
            assert b.nodes<=a.nodes;pruned+=b.nodes<a.nodes
    assert pruned>0


@pytest.mark.parametrize('all_in',[False,True])
def test_dead_owner_has_no_future_average_contribution(all_in):
    state=Holdem(10500000,6,'texture-buckets-v2');owner=state.info()[0]
    action=max(a for a in state.info()[2] if isinstance(a,int)) if all_in else 'fold'
    state=state.child(action);assert not state.terminal()
    a=StratifiedMCCFR();b=LiveOwnerMCCFR();a.average_path(state,owner,0);b.average_path(state,owner,0)
    assert a.sums==b.sums=={} and b.nodes==1 and b.nodes<a.nodes
