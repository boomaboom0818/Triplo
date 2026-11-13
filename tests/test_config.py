import json
import pytest
from poker.config import load_config, load_checkpoint
from poker.policy import Policy
from poker.game import Game

def test_model_targets():
    a=load_config('configs/bot.json');b=load_config('configs/human.json')
    assert a['model_id'] != b['model_id']
    assert a['status']=='active' and b['status']=='deferred'
    assert a['objective']==b['objective']=='maximize_expected_winnings'
    from poker.evaluate import evaluate
    with pytest.raises(ValueError,match='deferred'):
        evaluate('nonexistent.json',2,1,1,'configs/human.json')
    with pytest.raises(ValueError,match='identity mismatch'):
        load_checkpoint('artifacts/baseline.json',model_id=a['model_id'])

def test_shared_training_table():
    table={}; p=Policy(table=table)
    assert p.table is table

def test_checkpoint_validation(tmp_path):
    checkpoint=load_checkpoint('artifacts/baseline.json')
    checkpoint['engine']='unknown'
    path=tmp_path/'bad.json';path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):load_checkpoint(path)
