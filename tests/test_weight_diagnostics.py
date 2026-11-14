from types import SimpleNamespace
from poker.cfr import MCCFR,Holdem,TraversalLimit
from poker.abstraction import VERSION_V2
from poker.train_cfr import train
import pytest


def test_weight_moments_and_rollback():
    solver=MCCFR(11)
    state=SimpleNamespace(game=SimpleNamespace(observe=lambda:SimpleNamespace(street=1,active=(True,True,False))))
    for weight in (2,3):solver.record_average_weight(state,weight)
    assert solver.average_weight_moments['street=1:active=2']==dict(updates=2,total=5.,squared_total=13.,maximum=3.)
    previous=dict(solver.average_weight_moments)
    with pytest.raises(TraversalLimit):solver.step(lambda seed:Holdem(seed,abstraction=VERSION_V2),6,max_nodes=2)
    assert solver.average_weight_moments==previous


def test_weight_diagnostics_exact_holdem_resume(tmp_path,monkeypatch):
    opts=dict(game='holdem',abstraction=VERSION_V2,averaging_sampler='call-mixture-v1',max_iteration_nodes=500000)
    a=train(2,tmp_path/'full',**opts)
    p=tmp_path/'split';train(1,p,**opts);b=train(2,p,resume=p,**opts)
    assert a['average_weight_moments'] and a['average_weight_moments']==b['average_weight_moments']
    for field in ('sums','regrets','rng'):assert a[field]==b[field]
    assert a['weight_diagnostics_started_at']==b['weight_diagnostics_started_at']==0
    monkeypatch.setattr(MCCFR,'record_average_weight',lambda *args:None)
    without=train(2,tmp_path/'without-diagnostics',**opts)
    for field in ('sums','regrets','rng','nodes'):assert a[field]==without[field]
