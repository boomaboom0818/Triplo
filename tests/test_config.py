import json
import pytest
from poker.config import load_config, load_checkpoint
from poker.policy import Policy
from poker.game import Game

def test_same_strength_modes():
    a=load_config('configs/bot.json');b=load_config('configs/human.json')
    assert {k:v for k,v in a.items() if k!='mode'}=={k:v for k,v in b.items() if k!='mode'}
    o=Game(1).observe()
    assert Policy(1,mode='bot').act(o)==Policy(1,mode='human').act(o)

def test_shared_training_table():
    table={}; p=Policy(table=table)
    assert p.table is table

def test_checkpoint_validation(tmp_path):
    checkpoint=load_checkpoint('artifacts/baseline.json')
    checkpoint['engine']='unknown'
    path=tmp_path/'bad.json';path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):load_checkpoint(path)
