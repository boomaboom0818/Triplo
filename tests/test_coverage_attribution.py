import json,time
from poker.train_cfr import train
from poker.trajectory import AVERAGING
from poker.evaluate import evaluate
from scripts.coverage_attribution import audit_one


def test_attribution_exactly_replays_existing_evidence_without_source_changes(tmp_path):
    candidate=tmp_path/'candidate';opponent=tmp_path/'opponent'
    train(1,candidate,game='holdem',traversal='prefix-v3',averaging_sampler=AVERAGING,trajectory_samples=1)
    train(1,opponent,game='holdem',seed=29)
    report=evaluate(candidate,2,8600000,1,opponent_checkpoints=[opponent])
    output=tmp_path/'report';output.write_text(json.dumps(report));original=candidate.read_bytes();op_original=opponent.read_bytes()
    result=audit_one(candidate,opponent,output,2,time.time()+30)
    assert candidate.read_bytes()==original and opponent.read_bytes()==op_original
    assert result['read_only'] and len(result['populations'])==3
    for row in result['populations'].values():
        assert row['exact_payoff_match'] and sum(x['hands'] for x in row['hand_associations'].values())==12
        assert all(sum(v[k] for k in ('call','fold','raise_count'))==v['decisions'] for v in row['decisions'].values())
