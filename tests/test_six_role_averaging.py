import json,time
from types import SimpleNamespace
from random import Random
import pytest
from poker.cfr import CFRPolicy,information_key,Holdem
from poker.game import play,Game
from poker.trajectory import TrajectoryMCCFR
from poker.train_cfr import train
from scripts import six_role_averaging as d


def empty():return dict(sums={},regrets={},abstraction='texture-buckets-v2')


def test_controlled_contexts_cover_every_role_street_and_active_count():
    s=empty();cells=set()
    for owner in range(6):
        for street in (1,2,3):
            for active in (2,3,6):
                state,reach=d.context(s,10010000,owner,street,active);o=state.game.observe()
                assert (o.player,o.street,sum(o.active))==(owner,street,active)
                assert reach<=0;cells.add((owner,street,active))
    assert len(cells)==54


@pytest.mark.parametrize('population',d.POPULATIONS)
def test_reach_instrumentation_preserves_actions_payoffs_and_rng_for_all_roles(population):
    s=empty()
    for owner in range(6):
        groups={};ordinary=d.policies(s,population,owner,123);traced=d.policies(s,population,owner,123,groups)
        a=play(ordinary,10000000);b=play(traced,10000000)
        assert a[:2]==b[:2] and ordinary[owner].rng.getstate()==traced[owner].rng.getstate()
        assert sum(v['decisions'] for v in groups.values())==len(b[1][owner])


def test_zero_reach_is_distinct_from_tiny_finite_log_probability():
    o=Game(1).observe();s=empty();key=information_key(o,s['abstraction'])
    weights=[0.]*len(o.legal);weights[o.legal.index('fold')]=1.;s['regrets'][key]=weights
    s['sums'][key]=[float(a=='call') for a in o.legal];groups={};p=d.ReachPolicy(s,1,groups,'test')
    assert p.act(o)=='call' and p.log_reach==-float('inf')
    d.observe_group(groups,'test',s,o,-1000.)
    row=next(iter(groups.values()));assert row['zero_current_reach']==0 and row['tiny_current_reach']==1
    d.observe_group(groups,'test',s,o,p.log_reach);assert row['zero_current_reach']==1


def test_root_averaging_diagnostics_preserve_original_estimator():
    s=empty();a=d.ObservedAverage(s,4,{});b=TrajectoryMCCFR();a.rng=Random(12);b.rng=Random(12)
    state=Holdem(10020000,6,s['abstraction']);a.trajectory_average(state,4);b.trajectory_average(state,4)
    assert a.sums==b.sums and a.nodes==b.nodes and a.rng.getstate()==b.rng.getstate()


def test_full_diagnostic_preserves_checkpoint_and_accounts_for_roles(tmp_path,monkeypatch):
    parent=tmp_path/'parent.json';train(1,parent,game='holdem',seed=11);before=parent.read_bytes()
    monkeypatch.setattr(d,'ROOT',tmp_path/'root');monkeypatch.setattr(d,'PARENTS',[parent,parent])
    monkeypatch.setattr(d,'NATURAL_DEALS',[10000000]);monkeypatch.setattr(d,'CONTEXT_DEALS',[10010000]);monkeypatch.setattr(d,'ROOT_DEALS',[10020000]);monkeypatch.setattr(d,'REPLICATES',2)
    monkeypatch.setattr(d,'available_memory',lambda:10*1024**3);monkeypatch.setattr(d.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    d.main(0,time.time()+60);r=json.loads((d.ROOT/'candidate-0.json').read_text())
    assert parent.read_bytes()==before and len(r['controlled_contexts'])==54
    assert {v['owner'] for v in r['root_averaging_fixtures']}==set(range(6))
    d.verify(json.loads((d.ROOT/'candidate-0.manifest.json').read_text()))
