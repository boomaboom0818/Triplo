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
