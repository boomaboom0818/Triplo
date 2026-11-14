import json
import pytest
from poker.evaluate import evaluate
from poker.train import atomic_save


def test_reject_candidate_as_held_out():
    path='artifacts/bot/cfr-smoke.json'
    with pytest.raises(ValueError,match='candidate checkpoint'):
        evaluate(path,2,1,1,opponent_checkpoints=[path])


def test_reject_archived_training_opponent(tmp_path):
    path='artifacts/bot/cfr-smoke.json'
    checkpoint=json.load(open(path));checkpoint['archive']=[checkpoint['sums']]
    candidate=tmp_path/'candidate.json';atomic_save(candidate,checkpoint)
    with pytest.raises(ValueError,match='overlaps'):
        evaluate(candidate,2,1,1,opponent_checkpoints=[path])


def test_coverage_distinguishes_missing_zero_and_positive_rows():
    from poker.evaluate import coverage_counts, record_coverage
    counts = coverage_counts()
    for row in (None, [0.0, 0.0], [0.0, 2.0]):
        record_coverage(counts, row)
    assert counts == dict(decisions=3, missing_rows=1, zero_weight_rows=1,
                          positive_weight_rows=1)


def test_evaluation_coverage_reconciles_by_street():
    report = evaluate('artifacts/bot/cfr-smoke.json', 2, 123456, 1)
    for population in report['populations'].values():
        counts = population['coverage_counts']
        assert counts['decisions'] == population['candidate_decisions']
        assert counts['positive_weight_rows'] + counts['zero_weight_rows'] == population['covered_candidate_decisions']
        for key in counts:
            assert sum(street[key] for street in population['coverage_by_street'].values()) == counts[key]
        assert sum(counts[key] for key in ('missing_rows', 'zero_weight_rows', 'positive_weight_rows')) == counts['decisions']
        assert population['uniform_fallback_fraction'] == (counts['missing_rows'] + counts['zero_weight_rows']) / counts['decisions']
        assert population['latency_scope'] == 'all_table_decisions'


def test_live_fallback_matches_adapter_rules():
    from poker.evaluate import LiveFallbackPolicy
    from poker.cfr import CFRPolicy
    from poker.game import Game
    from dataclasses import replace
    policy=LiveFallbackPolicy(CFRPolicy({},abstraction='texture-buckets-v2'))
    o=Game(1).observe()
    assert policy.act(replace(o,call=2,legal=('call','fold',4)))=='fold'
    assert policy.act(replace(o,call=0,legal=('call',4)))=='call'
    assert policy.act(replace(o,call=2,legal=('call',)))=='call'
