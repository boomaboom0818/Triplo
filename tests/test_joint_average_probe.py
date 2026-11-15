import json,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from poker.game import Game
from poker.train_cfr import train
from scripts import stratified_average_probe as d
from scripts import joint_average_probe as j


def test_all_role_shared_trace_checksums_and_parent_bytes(tmp_path,monkeypatch):
    parent=tmp_path/'parent';train(1,parent,game='holdem',seed=11);before=parent.read_bytes();root=tmp_path/'root'
    observation=Game(1).observe()
    class Tiny:
        def __init__(self,depth=0):self.depth=depth;self.game=SimpleNamespace(observe=lambda:replace(observation,player=self.depth,position=self.depth,legal=('call',),street=1))
        def terminal(self):return self.depth==6
        def info(self):return self.depth,'node:'+str(self.depth),('call',)
        def child(self,a):return Tiny(self.depth+1)
        def returns(self):return [0.]*6
    monkeypatch.setattr(d,'PARENTS',[parent,parent]);monkeypatch.setattr(d,'DEALS',[10800000]);monkeypatch.setattr(j,'DEALS',[10800000]);monkeypatch.setattr(d,'REPLICATES',2)
    monkeypatch.setattr(j,'CONTROLS',{});monkeypatch.setattr(j,'VERIFIED',0)
    monkeypatch.setattr(d,'Holdem',lambda *args:Tiny());monkeypatch.setattr(d,'available_memory',lambda:10*1024**3);monkeypatch.setattr(d.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    d.main(0,time.time()+60,implementation=j.ObservedJoint,baseline_implementation=j.Baseline,root=root,criteria=j.CRITERIA,additional_sources=[Path(j.__file__)],candidate_label='joint',batch_fn=j.batch,summary_fn=j.summarize,gate_fn=j.gates,batch_max_seconds=2.,batch_node_limit=10000)
    x=json.loads((root/'candidate-0.json').read_text());v=x['variants']['joint']
    assert parent.read_bytes()==before and v['exact_regret_rng_nodes_batches_verified']==v['expected_batches']==12
    assert x['gates']['exact_control_pass'] and x['gates']['bounds_pass']
    assert v['max_pointwise_weight_ratio']<=1+1e-12 and set(v['source_groups'])
    d.verify(json.loads((root/'candidate-0.manifest.json').read_text()))
