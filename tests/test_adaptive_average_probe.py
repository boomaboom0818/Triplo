import json,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from poker.cfr import TraversalLimit
from poker.train_cfr import train
from scripts import adaptive_average_probe as a
from scripts import stratified_average_probe as d


def setup(tmp_path,monkeypatch):
    parent=tmp_path/'parent';train(1,parent,game='holdem',seed=11)
    monkeypatch.setattr(d,'PARENTS',[parent,parent]);monkeypatch.setattr(d,'DEALS',[10500000]);monkeypatch.setattr(d,'REPLICATES',2)
    monkeypatch.setattr(d,'available_memory',lambda:10*1024**3);monkeypatch.setattr(d.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    return parent


def test_custom_candidate_uses_actual_cap_gate_and_preserves_parent(tmp_path,monkeypatch):
    parent=setup(tmp_path,monkeypatch);before=parent.read_bytes();root=tmp_path/'root'
    d.main(0,time.time()+60,implementation=a.ObservedAdaptive,root=root,criteria=a.CRITERIA,additional_sources=[Path(a.__file__)],candidate_label='adaptive',batch_fn=a.batch,summary_fn=a.summarize,gate_fn=a.gates,batch_max_seconds=2.,batch_node_limit=10000)
    r=json.loads((root/'candidate-0.json').read_text());v=r['variants']['adaptive']
    assert parent.read_bytes()==before and r['candidate_label']=='adaptive'
    assert v['maximum_average_weight']<=512*(1+1e-12) and v['max_weight_to_cap_ratio']<=1+1e-12
    assert 'max_pointwise_weight_ratio' not in v and r['gates']['bounds_pass']
    d.verify(json.loads((root/'candidate-0.manifest.json').read_text()))


def test_cost_failure_retains_exact_batch_and_never_publishes_partial_result(tmp_path,monkeypatch):
    parent=setup(tmp_path,monkeypatch);before=parent.read_bytes();root=tmp_path/'failed'
    def fail(*args):raise TraversalLimit('run time budget reached')
    with pytest.raises(TraversalLimit):d.main(0,time.time()+60,root=root,batch_fn=fail,batch_max_seconds=2.,batch_node_limit=10000)
    assert not (root/'candidate-0.json').exists() and parent.read_bytes()==before
    m=json.loads((root/'candidate-0.metrics.json').read_text())
    assert m['status']=='failed' and m['last_batch']==dict(owner=0,deal_seed=10500000,variant='baseline',replicate=0,node_limit=10000,max_seconds=2.)
