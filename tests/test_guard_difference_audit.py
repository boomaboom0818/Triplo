import json
from poker.history_cfr import ParallelHistory
from scripts.live_gate import evaluate
from scripts.guard_difference_audit import run


def test_selected_paired_differences_replay_exactly(tmp_path):
    checkpoint = tmp_path / 'empty.json'
    ParallelHistory().export_policy(checkpoint)
    report = tmp_path / 'report.json'
    evaluate(checkpoint, checkpoint, report, 9152, 2)
    output = tmp_path / 'audit.json'
    result = run(report, report, 'six-100-tight', output, count=1)
    assert len(result['traces']) == 2
    for trace in result['traces']:
        assert trace['difference_bb'] == 0
        assert trace['candidate'] == trace['control']
    assert json.loads(output.read_text())['case'] == 'six-100-tight'
