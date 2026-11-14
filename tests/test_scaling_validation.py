import json,subprocess,sys
import pytest
from scripts import scaling_validation as scaling
from poker.compare import paired_difference


def test_resource_headroom_is_a_separate_gate():
    m={'status':'completed','address_space_limit_bytes':scaling.LIMIT,'peak_rss_bytes':1024**3,'VmPeak':2*1024**3}
    assert scaling.resource_pass([m],3*1024**3)
    assert not scaling.resource_pass([m],1024**3)
    assert not scaling.resource_pass([m],3*1024**3,False)
    assert not scaling.resource_pass([{**m,'VmPeak':scaling.LIMIT}],3*1024**3)


@pytest.mark.parametrize('memory_gib',[3,4])
def test_peak_measurement_worker(tmp_path,memory_gib):
    metrics=tmp_path/'metrics.json'
    subprocess.run([sys.executable,'-m','scripts.limited_worker','--metrics',str(metrics),'--memory-gib',str(memory_gib),'poker.train_cfr','--iterations','1','--output',str(tmp_path/'checkpoint')],check=True,capture_output=True)
    m=json.loads(metrics.read_text());assert m['status']=='completed'
    assert m['address_space_limit_bytes']==memory_gib*1024**3 and m['peak_rss_bytes']>0 and m['VmPeak']>0


def test_primary_interval_confidence_is_explicit():
    def report(v):return dict(seed=1,copies=1,opponent_checkpoint_hashes=[],populations={'x':{'deal_block_bb_per_100':v}})
    result=paired_difference(report([3,4,5]),report([1,2,3]),confidence=.9875)['x']
    assert result['interval']==[2,2] and result['confidence']==.9875
    assert 'ci95' not in result
    with pytest.raises(ValueError):paired_difference(report([1,2]),report([0,0]),confidence=1)


def test_second_launch_failure_checkpoints_first_child(tmp_path,monkeypatch):
    import signal
    events=[]
    class Child:
        code=None
        def poll(self):return self.code
        def send_signal(self,sig):events.append(sig)
        def wait(self,timeout):self.code=0;events.append(timeout);return 0
    count=0
    def launch(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:raise OSError('second launch failed')
        return Child()
    monkeypatch.setattr(scaling,'available_memory',lambda:20*1024**3)
    monkeypatch.setattr(scaling.subprocess,'Popen',launch)
    jobs=[{'output':tmp_path/f'{i}.json','log':tmp_path/f'{i}.log','metrics':tmp_path/f'{i}-metrics.json','args':[]} for i in range(2)]
    with pytest.raises(OSError):scaling.run_pair(jobs,__import__('time').time()+60)
    assert events==[signal.SIGTERM,120]


def test_duplicate_output_never_launches(tmp_path,monkeypatch):
    monkeypatch.setattr(scaling,'available_memory',lambda:20*1024**3)
    monkeypatch.setattr(scaling.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='duplicate'):scaling.run_pair([{'output':tmp_path/'same'}]*2,__import__('time').time()+60)


def test_frozen_protocol_and_inputs_cannot_change(tmp_path):
    parent=tmp_path/'parent';parent.write_text('parent')
    code=tmp_path/'algorithm.py';code.write_text('code')
    criteria=tmp_path/'criteria';criteria.write_text('criteria')
    entry={'path':str(parent),'sha256':scaling.digest(parent)}
    manifest={'parents':[entry],'robustness':[entry],'validation':entry,'algorithm_sources':{str(code):scaling.digest(code)},'criteria':str(criteria),'criteria_sha256':scaling.digest(criteria)}
    scaling.verify_inputs(manifest)
    code.write_text('changed')
    with pytest.raises(ValueError,match='algorithm changed'):scaling.verify_inputs(manifest)
    code.write_text('code');criteria.write_text('relaxed')
    with pytest.raises(ValueError,match='criteria changed'):scaling.verify_inputs(manifest)


def test_parallel_confirmation_cache_and_atomic_outputs(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('model')
    opponent=tmp_path/'opponent';opponent.write_text('opponent')
    requests=[{'checkpoint':checkpoint,'opponent':opponent,'uniform':bool(i),'output':tmp_path/f'{i}.json'} for i in range(2)]
    interrupted=tmp_path/'0.json.pending';interrupted.write_text('previous interrupted attempt')
    def run(jobs,deadline):
        assert len(jobs)==2 and all(j['module']=='poker.evaluate' for j in jobs)
        for j in jobs:
            row={'independent_deal_blocks':2000,'deal_block_bb_per_100':[0]*2000}
            report={'checkpoint_sha256':scaling.digest(checkpoint),'opponent_checkpoint_hashes':[scaling.digest(opponent)],'seed':8200000,'copies':1,'uniform_control':'--uniform-control' in j['args'],'live_fallback':False,'populations':{str(i):row for i in range(6)}}
            report['opponent_checkpoint_hashes']=['checkpoint:'+report['opponent_checkpoint_hashes'][0]]
            j['log'].write_text(json.dumps(report))
    monkeypatch.setattr(scaling,'run_pair',run)
    scaling.evaluate_parallel(requests,9999999999)
    assert all(r['output'].exists() for r in requests)
    assert next(tmp_path.glob('0.json.pending.attempt-*')).read_text()=='previous interrupted attempt'
    monkeypatch.setattr(scaling,'run_pair',lambda jobs,deadline:pytest.fail('duplicate evaluation') if jobs else None)
    scaling.evaluate_parallel(requests,9999999999)


def test_corrupt_confirmation_stays_unpublished(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('model')
    output=tmp_path/'report.json';opponent=tmp_path/'opponent';opponent.write_text('opponent')
    monkeypatch.setattr(scaling,'run_pair',lambda jobs,deadline:jobs[0]['log'].write_text('interrupted JSON'))
    with pytest.raises(json.JSONDecodeError):scaling.evaluate_parallel([{'checkpoint':checkpoint,'opponent':opponent,'uniform':False,'output':output}],9999999999)
    assert not output.exists() and (tmp_path/'report.json.pending').exists()


def test_two_evaluators_keep_three_gib_caps(tmp_path,monkeypatch):
    captured=[]
    class Child:
        pid=99999999
        def poll(self):return 0
    def launch(*args,**kwargs):kwargs['preexec_fn']();return Child()
    monkeypatch.setattr(scaling,'ROOT',tmp_path)
    monkeypatch.setattr(scaling,'available_memory',lambda:9*1024**3)
    monkeypatch.setattr(scaling.shutil,'disk_usage',lambda path:__import__('types').SimpleNamespace(free=20*1024**3))
    monkeypatch.setattr(scaling,'scaling_limits',lambda limit:captured.append(limit))
    monkeypatch.setattr(scaling.subprocess,'Popen',launch)
    jobs=[{'module':'poker.evaluate','output':tmp_path/f'{i}.json','log':tmp_path/f'{i}.pending','metrics':tmp_path/f'{i}.metrics','args':[]} for i in range(2)]
    scaling.run_pair(jobs,__import__('time').time()+60)
    assert captured==[3*1024**3,3*1024**3]
