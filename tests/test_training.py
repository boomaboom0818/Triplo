import json
import pytest
from poker.train import train, configuration

def test_exact_resume_across_archive_boundary(tmp_path):
    cfg=configuration(); cfg.update(archive_every=5,checkpoint_every=4)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    full=train(12,None,tmp_path/'full.json',config=config)
    split=tmp_path/'split.json'
    train(7,None,split,config=config)
    resumed=train(12,None,split,config=config,resume=split)
    assert full['table']==resumed['table']
    assert full['training']==resumed['training']
    with pytest.raises(ValueError,match='output exists'):train(15,None,split,config=config)
    with pytest.raises(ValueError,match='differs'):train(15,42,split,config=config,resume=split)

@pytest.mark.parametrize('field,value',[('players',2),('archive_size',0),('learning_rate',float('nan')),('population',['tight'])])
def test_invalid_config(tmp_path,field,value):
    cfg=configuration();cfg[field]=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):configuration(path)
