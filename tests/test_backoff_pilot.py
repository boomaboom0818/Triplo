import json,subprocess,sys,time
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts import backoff_pilot as pilot
from poker.train_cfr import run_lock


def setup_workers(tmp_path,monkeypatch,commands):
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    monkeypatch.setattr(pilot,'POLL_SECONDS',.01)
    monkeypatch.setattr(pilot,'available_memory',lambda:12*1024**3)
    monkeypatch.setattr(pilot.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(pilot,'command',lambda job,deadline:[sys.executable,'-c',commands[job['variant']]])
    children=[];real=subprocess.Popen
    def spawn(*a,**kw):
        p=real(*a,**kw);children.append(p);return p
    monkeypatch.setattr(pilot.subprocess,'Popen',spawn)
    return children


def test_peer_failure_stops_running_child_and_preserves_logs(tmp_path,monkeypatch):
    children=setup_workers(tmp_path,monkeypatch,{'bad':'raise RuntimeError("deliberate")','peer':'import time;time.sleep(20)'})
    jobs=[dict(output=tmp_path/name,variant=name) for name in ('bad','peer')]
    with pytest.raises(RuntimeError,match='worker failed'):pilot.run_batch(jobs,time.time()+5)
    assert len(children)==2 and all(p.poll() is not None for p in children)
    assert 'deliberate' in (tmp_path/'bad.log').read_text()


def test_child_owns_output_lock_and_publishes_atomically(tmp_path,monkeypatch):
    output=tmp_path/'report.json'
    # A real child must acquire the SAME publication lock used by component.main.
    code="from pathlib import Path;from poker.train_cfr import run_lock;from poker.train import atomic_save;p=Path("+repr(str(output))+");\nwith run_lock(p):atomic_save(p,{'completed':True})"
    children=setup_workers(tmp_path,monkeypatch,{'ok':code})
    pilot.run_batch([dict(output=output,variant='ok')],time.time()+5)
    assert json.loads(output.read_text())=={'completed':True} and children[0].returncode==0
    with run_lock(output):pass


def test_deadline_stops_child_and_low_reserve_launches_none(tmp_path,monkeypatch):
    children=setup_workers(tmp_path,monkeypatch,{'peer':'import time;time.sleep(20)'})
    jobs=[dict(output=tmp_path/'peer',variant='peer')]
    with pytest.raises(TimeoutError):pilot.run_batch(jobs,time.time()+.05)
    assert len(children)==1 and children[0].poll() is not None
    monkeypatch.setattr(pilot,'available_memory',lambda:4*1024**3)
    with pytest.raises(RuntimeError,match='reserve'):pilot.run_batch(jobs,time.time()+5)
    assert len(children)==1
