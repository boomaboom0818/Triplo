import json
import pytest
from scripts import review_curve


def test_review_skips_active_report_without_overwriting(tmp_path):
    checkpoint=tmp_path/'candidate-a-500.json'
    checkpoint.write_text(json.dumps({'iterations':500}))
    pending=tmp_path/'candidate-a-500-standard.json';pending.write_text('')
    result=review_curve.review(tmp_path)
    assert result['stages']=={}
    assert result['pending']==['candidate-a-500']
    assert pending.read_text()==''


def test_reader_rejects_live_mode_mismatch(tmp_path,monkeypatch):
    import hashlib
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('checkpoint')
    validation=tmp_path/'validation';validation.write_text('opponent')
    monkeypatch.setattr(review_curve,'VALIDATION',validation)
    report=tmp_path/'report.json'
    report.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seed=5300000,copies=1,uniform_control=False,live_fallback=True,opponent_checkpoint_hashes=['checkpoint:'+hashlib.sha256(validation.read_bytes()).hexdigest()],populations={})))
    with pytest.raises(ValueError,match='provenance'):review_curve.read_report(report,checkpoint)
