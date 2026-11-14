import time
import pytest
from scripts import sustained_train as runner


def test_stage_preserves_recipe_and_caps():
    saved=dict(iterations=1000,config={'seed':11,'averaging_sampler':'call-mixture-v1','archive_size':2},limits={'max_information_sets':1250000,'max_iteration_nodes':500000})
    cfg=runner.stage_config(saved,1250)
    assert cfg['averaging_sampler']=='call-mixture-v1'
    assert cfg['archive_size']==2
    assert cfg['max_information_sets']==1250000
    assert cfg['max_iteration_nodes']==500000
    with pytest.raises(ValueError):runner.stage_config(saved,1000)


def test_deadline_launches_no_child(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch'))
    with pytest.raises(TimeoutError):runner.run_child([],tmp_path/'log',time.time()-1)


def test_timeout_signals_then_waits_for_checkpoint(tmp_path,monkeypatch):
    import subprocess,signal
    events=[]
    class Child:
        returncode=None
        def wait(self,timeout):
            events.append(('wait',timeout))
            if len(events)==1:raise subprocess.TimeoutExpired('child',timeout)
            self.returncode=0;return 0
        def poll(self):return self.returncode
        def send_signal(self,sig):events.append(('signal',sig))
        def kill(self):pytest.fail('graceful save must have time to complete')
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:Child())
    with pytest.raises(TimeoutError):runner.run_child(['poker.train_cfr'],tmp_path/'log',time.time()+10)
    assert ('signal',signal.SIGTERM) in events
    assert ('wait',120) in events


def test_evaluation_uses_declared_block_count(tmp_path,monkeypatch):
    import json
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('candidate')
    validation=tmp_path/'validation';validation.write_text('opponent')
    def fake_run(args,output,deadline):
        assert args[args.index('--blocks')+1]=='2000'
        row={'independent_deal_blocks':2000,'deal_block_bb_per_100':[0]*2000}
        output.write_text(json.dumps(dict(checkpoint_sha256=runner.digest(checkpoint),seed=5500000,copies=1,uniform_control=False,live_fallback=False,opponent_checkpoint_hashes=['checkpoint:'+runner.digest(validation)],populations={str(i):row for i in range(6)})))
    monkeypatch.setattr(runner,'run_child',fake_run)
    report=runner.evaluate(checkpoint,tmp_path/'report.json',validation,5500000,False,time.time()+10,blocks=2000)
    assert report['populations']['0']['independent_deal_blocks']==2000
    with pytest.raises(ValueError,match='provenance'):
        runner.evaluate(checkpoint,tmp_path/'report.json',validation,5500000,False,time.time()+10,blocks=1000)
