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
