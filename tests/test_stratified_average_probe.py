import json,time
from random import Random
from types import SimpleNamespace
from poker.cfr import Holdem
from poker.trajectory import TrajectoryMCCFR
from poker.train_cfr import train
from scripts import stratified_average_probe as p


def test_instrumented_eight_path_baseline_exactly_matches_original():
    saved=dict(regrets={},sums={},abstraction='texture-buckets-v2')
    a=p.Baseline();a.setup(saved);b=TrajectoryMCCFR();a.rng=Random(10400000);b.rng=Random(10400000)
    state=Holdem(10500000,6,saved['abstraction']);p.batch(a,state,0,False)
    for _ in range(8):b.trajectory_average(state,0,1/8)
    assert a.sums==b.sums and a.nodes==b.nodes and a.rng.getstate()==b.rng.getstate()


def test_probe_covers_all_roles_and_preserves_checkpoint(tmp_path,monkeypatch):
    parent=tmp_path/'parent.json';train(1,parent,game='holdem',seed=11);before=parent.read_bytes()
    monkeypatch.setattr(p,'ROOT',tmp_path/'root');monkeypatch.setattr(p,'PARENTS',[parent,parent]);monkeypatch.setattr(p,'DEALS',[10500000]);monkeypatch.setattr(p,'REPLICATES',2)
    monkeypatch.setattr(p,'available_memory',lambda:10*1024**3);monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    p.main(0,time.time()+60);r=json.loads((p.ROOT/'candidate-0.json').read_text())
    assert parent.read_bytes()==before and r['read_only'] and len(r['fixtures'])==6
    assert {v['owner'] for v in r['fixtures']}==set(range(6))
    assert r['variants']['stratified']['max_pointwise_weight_ratio']<=4/3+1e-12
    assert r['eligible_for_learning_pilot']==all(r['gates'].values())
    p.verify(json.loads((p.ROOT/'candidate-0.manifest.json').read_text()))
