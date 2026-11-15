import json,time
import pytest
from collections import Counter
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from poker.evaluate import evaluate
from scripts.coverage_attribution import audit_one


@pytest.mark.parametrize('seed',[8600000,9500000])
def test_attribution_exactly_replays_existing_evidence_without_source_changes(tmp_path,seed):
    candidate=tmp_path/'candidate';opponent=tmp_path/'opponent'
    train(1,candidate,game='holdem',traversal='prefix-v3',averaging_sampler=AVERAGING,trajectory_samples=1)
    train(1,opponent,game='holdem',seed=29)
    report=evaluate(candidate,2,seed,1,opponent_checkpoints=[opponent])
    output=tmp_path/'report';output.write_text(json.dumps(report));original=candidate.read_bytes();op_original=opponent.read_bytes()
    result=audit_one(candidate,opponent,output,2,time.time()+30,seed=seed,parent_checkpoint=candidate)
    assert candidate.read_bytes()==original and opponent.read_bytes()==op_original
    assert result['read_only'] and len(result['populations'])==3
    for row in result['populations'].values():
        assert row['exact_payoff_match'] and sum(x['hands'] for x in row['hand_associations'].values())==12
        assert all(sum(v[k] for k in ('call','fold','raise_count'))==v['decisions'] for v in row['decisions'].values())
        assert all(v['new_positive_average']==0 and v['new_positive_regret']==0 for v in row['decisions'].values())
    with pytest.raises(ValueError,match='provenance'):audit_one(candidate,opponent,output,2,time.time()+30,seed=seed+1)


def test_replay_identifies_average_actions_outside_current_strategy_support():
    from poker.game import Game
    from poker.cfr import information_key
    from scripts.coverage_attribution import TracedPolicy
    o=Game(1).observe();key=information_key(o,'texture-buckets-v2')
    mean=[0.]*len(o.legal);mean[o.legal.index('call')]=1
    current=[0.]*len(o.legal);current[o.legal.index('fold')]=1
    groups={};p=TracedPolicy({key:mean},1,'texture-buckets-v2',groups=groups,missing=Counter(),regrets={key:current})
    assert p.act(o)=='call' and p.current_own_reach==0
    p.sums={};p.act(o)
    missing=next(v for k,v in groups.items() if 'source=missing' in k)
    assert missing['current_unreachable']==1 and missing['positive_regret_row']==1
    # A floating-point underflow is not a zero-probability action.
    p.current_own_reach=0.;p.current_zero_support=False;p.act(o)
    assert missing['current_unreachable']==1 and missing['current_tiny_reach']==1
