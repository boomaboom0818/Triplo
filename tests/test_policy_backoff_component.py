from collections import Counter
import json,signal,time
import pytest
from poker.cfr import information_key,CFRPolicy
from poker.game import Game
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from poker import evaluate as evaluator
from scripts.policy_backoff_component import BackoffPolicy,evaluate_component
from scripts import policy_backoff_component as component


def test_average_takes_precedence_and_backoff_requires_positive_regret():
    o=Game(1).observe();key=information_key(o,'texture-buckets-v2');n=len(o.legal)
    mean=[1.]+[0.]*(n-1);reg=[0.,1.]+[0.]*(n-2)
    sink=Counter();p=BackoffPolicy({key:mean},{key:reg},1,'texture-buckets-v2',use_regrets=True,sink=sink)
    assert p.act(o)==o.legal[0]
    p.sums={};assert p.act(o)==o.legal[1]
    assert any(k.endswith('source=positive_regret') for k in sink)
    p.regrets={key:[-1.]*n};p.act(o);assert any(k.endswith('source=uniform') for k in sink)


def test_baselines_match_and_only_candidate_policy_changes(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate';opponent=tmp_path/'opponent'
    source=train(1,candidate,game='holdem',traversal='prefix-v3',averaging_sampler=AVERAGING,trajectory_samples=1)
    heldout=train(1,opponent,game='holdem',seed=29)
    original=candidate.read_bytes();op_original=opponent.read_bytes()
    for variant in ('average','uniform'):
        baseline=evaluator.evaluate(candidate,2,9700000,1,opponent_checkpoints=[opponent],uniform_control=variant=='uniform')
        report=evaluate_component(candidate,opponent,2,9700000,variant)
        for pop in baseline['populations']:assert report['populations'][pop]['deal_block_bb_per_100']==baseline['populations'][pop]['deal_block_bb_per_100']
    real=evaluator.CFRPolicy;calls=[]
    def spy(sums,*args):calls.append(sums);return real(sums,*args)
    monkeypatch.setattr(evaluator,'CFRPolicy',spy)
    report=evaluate_component(candidate,opponent,2,9700000,'backoff')
    assert len(calls)==60 and all(s==heldout['sums'] for s in calls)
    assert evaluator.CFRPolicy is spy and report['inference_component_only']
    assert candidate.read_bytes()==original and opponent.read_bytes()==op_original
    assert report['raw_coverage_scope'].startswith('average table only')


def test_injection_is_restored_after_failure(tmp_path,monkeypatch):
    c=tmp_path/'c';o=tmp_path/'o'
    train(1,c,game='holdem',traversal='prefix-v3',averaging_sampler=AVERAGING,trajectory_samples=1)
    train(1,o,game='holdem',seed=29)
    before=(evaluator.CFRPolicy,evaluator.load_checkpoint)
    def fail(*args,**kwargs):raise RuntimeError('deliberate play failure')
    monkeypatch.setattr(evaluator,'play',fail)
    with pytest.raises(RuntimeError,match='deliberate'):evaluate_component(c,o,2,9700000,'backoff')
    assert (evaluator.CFRPolicy,evaluator.load_checkpoint)==before


def test_deadline_preserves_failed_attempt_and_restores_handlers(tmp_path,monkeypatch):
    before={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    monkeypatch.setattr(component,'available_memory',lambda:10*1024**3)
    monkeypatch.setattr(component,'evaluate_component',lambda *a:time.sleep(2))
    output=tmp_path/'report.json'
    with pytest.raises(TimeoutError,match='deadline'):component.main(None,None,output,2,9700000,'backoff',time.time()+.02)
    assert not output.exists()
    metrics=json.loads((tmp_path/'report.json.metrics.json').read_text())
    assert metrics['status']=='failed' and len(metrics['attempts'])==1 and 'deadline' in metrics['error']
    assert before=={s:signal.getsignal(s) for s in before}
