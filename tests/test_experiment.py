import json
import pytest
from poker.experiment import experiment


def test_budget_stop_has_no_false_completion(tmp_path):
    cfg=json.load(open('configs/cfr-bot.json'));cfg['max_iteration_nodes']=1
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    output=tmp_path/'experiment'
    experiment(path,output,[1,2],2)
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='budget_stopped' and status['completed']==0
    assert not list(output.glob('*copies*'))
    with pytest.raises(ValueError,match='new experiment directory'):
        experiment(path,output,[1,2],2)
