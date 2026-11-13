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
