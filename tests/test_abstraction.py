from dataclasses import replace
from poker.abstraction import bucket_key,cards_bucket,VERSION
from poker.game import Game
from poker.train_cfr import train


def test_suit_relabeling_and_strength():
    o=Game(1).observe()
    o=replace(o,hole=('As','Ks'),board=('Qs','Js','2d'),street=1)
    transformed=replace(o,hole=('Ah','Kh'),board=('Qh','Jh','2c'))
    assert bucket_key(o)==bucket_key(transformed)
    assert cards_bucket(('As','Ad'),('Ac','Ah','2d'))!=cards_bucket(('7c','8d'),('Ac','Ah','2d'))


def test_buckets_ignore_absolute_card_suits_preflop():
    o=Game(1).observe()
    assert bucket_key(replace(o,hole=('As','Kd')))==bucket_key(replace(o,hole=('Ah','Kc')))
    assert bucket_key(replace(o,hole=('As','Ks')))!=bucket_key(replace(o,hole=('As','Kd')))


def test_population_resume(tmp_path):
    options=dict(abstraction=VERSION,population_probability=.5,archive_every=3)
    full=train(10,tmp_path/'full.json',**options)
    path=tmp_path/'split.json';train(4,path,**options)
    split=train(10,path,resume=path,**options)
    for field in ('sums','regrets','archive','rng'):assert split[field]==full[field]


def test_v2_suit_invariance_and_version_separation():
    from poker.abstraction import bucket_key_v2
    o=Game(1).observe()
    a=replace(o,hole=('As','Kd'))
    b=replace(o,hole=('Ah','Kc'))
    assert bucket_key_v2(a)==bucket_key_v2(b)
    assert bucket_key_v2(a)!=bucket_key(a)
    assert bucket_key_v2(a)!=bucket_key_v2(replace(a,legal=('call',)))


def test_diagnostics_resume_and_rollback(tmp_path):
    from poker.cfr import MCCFR,Holdem,TraversalLimit
    import pytest
    solver=MCCFR(11)
    with pytest.raises(TraversalLimit):
        solver.step(lambda seed:Holdem(seed,abstraction=VERSION),6,max_nodes=2)
    assert not solver.diagnostics
    options=dict(game='holdem',abstraction=VERSION,max_iteration_nodes=500000)
    full=train(2,tmp_path/'full-diag.json',**options)
    path=tmp_path/'split-diag.json'
    train(1,path,**options)
    split=train(2,path,resume=path,**options)
    assert full['diagnostics']==split['diagnostics']
    assert full['sums']==split['sums']
    assert any(k.startswith('regret:') for k in full['diagnostics'])
    assert any(k.startswith('average:') for k in full['diagnostics'])


def test_v2_checkpoint_load_and_v1_resume_rejected(tmp_path):
    from poker.abstraction import VERSION_V2
    from poker.config import load_checkpoint
    import pytest
    path=tmp_path/'v2.json'
    train(1,path,game='holdem',abstraction=VERSION_V2,max_iteration_nodes=500000)
    assert load_checkpoint(path)['abstraction']==VERSION_V2
    with pytest.raises(ValueError,match='incompatible'):
        train(2,path,resume=path,game='holdem',abstraction=VERSION)
