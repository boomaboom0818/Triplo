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
