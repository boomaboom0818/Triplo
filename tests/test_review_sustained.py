import json
import pytest
from scripts import review_sustained as reviewer


def test_report_checks_mean_mode_and_samples(tmp_path):
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('policy')
    path=tmp_path/'report.json'
    manifest={'seed':5400000,'blocks':2,'validation':{'sha256':'opponent'}}
    report=dict(checkpoint_sha256=reviewer.digest(checkpoint),seed=5400000,copies=1,uniform_control=False,live_fallback=False,opponent_checkpoint_hashes=['checkpoint:opponent'],populations={str(i):dict(deal_block_bb_per_100=[1,3],independent_deal_blocks=2,bb_per_100=2) for i in range(6)})
    path.write_text(json.dumps(report));assert reviewer.validate_report(path,checkpoint,manifest)==report
    report['populations']['0']['bb_per_100']=999
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='mean mismatch'):reviewer.validate_report(path,checkpoint,manifest)
    report['live_fallback']=True;path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='provenance'):reviewer.validate_report(path,checkpoint,manifest)


def test_incomplete_stage_left_untouched(tmp_path):
    parent=tmp_path/'parent';parent.write_text('parent')
    validation=tmp_path/'validation';validation.write_text('opponent')
    manifest={'parents':{'candidate-0':{'path':str(parent),'sha256':reviewer.digest(parent)}},'validation':{'path':str(validation),'sha256':reviewer.digest(validation)},'stages':[1250],'purpose':'exploratory'}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    checkpoint=tmp_path/'candidate-0-1250.json';checkpoint.write_text('currently writing')
    result=reviewer.review(tmp_path)
    assert result['pending']==['candidate-0-1250']
    assert checkpoint.read_text()=='currently writing'
