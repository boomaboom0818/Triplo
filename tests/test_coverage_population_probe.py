from types import SimpleNamespace
from copy import deepcopy
from poker.cfr import Holdem
from poker.prefix import PrefixMCCFR
from scripts.coverage_population_probe import guarded_strategy,ReachProbe


def test_guarded_profile_is_legal_and_depends_only_on_public_price_menu():
    for legal in [('call',),('call',2,4),('call','fold'),('call','fold',2,4)]:
        for price in (0,2):
            o=SimpleNamespace(call=price,legal=legal,hole=('As','Ah'))
            weights=guarded_strategy(o);o.hole=('2s','3h')
            assert weights==guarded_strategy(o) and abs(sum(weights)-1)<1e-12
            assert all(x>=0 for x in weights)
    assert guarded_strategy(SimpleNamespace(call=2,legal=('call','fold',4)))[1]>.8


def test_probe_discards_all_learning_updates():
    s=ReachProbe(True);s.reference_sums={};before=deepcopy(s.regrets);delta={}
    s.outcome(Holdem(1),3,delta)
    assert s.regrets==before and s.sums=={} and s.iterations==0 and delta


def test_historical_probe_matches_uninstrumented_frozen_estimator():
    probe=ReachProbe(False);probe.reference_sums={};reference=PrefixMCCFR(1,depth=2)
    left={};right={};probe.outcome(Holdem(9),3,left);reference.outcome(Holdem(9),3,right)
    assert left==right and probe.nodes==reference.nodes and probe.rng.getstate()==reference.rng.getstate()
