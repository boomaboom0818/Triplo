import json
import pytest
from pathlib import Path
from scripts import joint_average_training as p
from scripts.sustained_train import digest


def test_launcher_requires_both_successes_exact_sources_and_both_parent_bytes(tmp_path,monkeypatch):
    parents=[tmp_path/'a',tmp_path/'b']
    for i,path in enumerate(parents):path.write_text(str(i))
    q=tmp_path/'qualification.json';monkeypatch.setattr(p,'ENGINEERING',q);monkeypatch.setattr(p,'PARENTS',parents)
    report=dict(algorithm=p.ALGORITHM,independently_verified=True,eligible_for_learning_pilot=True,candidates=[dict(candidate=i,gates=dict(passed=True)) for i in range(2)],algorithm_sources={str(x):digest(x) for x in p.ALGORITHM_SOURCES},parents=[dict(path=str(x),sha256=digest(x)) for x in parents])
    q.write_text(json.dumps(report));p.verify_engineering()
    report['candidates'][1]['gates']['passed']=False;q.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='both'):p.verify_engineering()
    report['candidates'][1]['gates']['passed']=True;report['algorithm_sources']['poker/joint_average.py']='changed';q.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='source'):p.verify_engineering()
    report['algorithm_sources']['poker/joint_average.py']=digest(Path('poker/joint_average.py'));q.write_text(json.dumps(report));parents[1].write_text('changed')
    with pytest.raises(ValueError,match='bytes'):p.verify_engineering()


def test_fixed_curve_and_bounded_child_runtime_preserve_memory_and_coverage_gates(monkeypatch):
    from scripts import trajectory_pilot as d
    for name in ('ROOT','PARENTS','EXPECTED_BASES','ADDITIONS','ALGORITHM','AVERAGING','TRAVERSAL','PREFIX_DEPTH','POPULATION','FORK_FLAG','REQUIRE_TIGHT_IMPROVEMENT','fork_allowed','CRITERIA','SCREEN_SEED','CONFIRM_SEED','ADDITIONAL_SOURCES','stage_config'):monkeypatch.setattr(d,name,getattr(d,name))
    p.configure()
    assert d.ADDITIONS==[50,200,500,1000,2000] and d.EXPECTED_BASES==[4906,5150]
    assert (d.SCREEN_SEED,d.CONFIRM_SEED)==(10200000,10300000) and d.REQUIRE_TIGHT_IMPROVEMENT
    saved=dict(iterations=4906,config={'seed':11},limits=dict(max_iteration_nodes=500000,max_information_sets=1250000))
    cfg=d.stage_config(saved,6906)
    assert cfg['max_seconds']==7200 and cfg['max_iteration_nodes']==500000 and cfg['max_information_sets']==1250000 and cfg['checkpoint_every']==50
