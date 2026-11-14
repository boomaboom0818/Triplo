from copy import deepcopy
import pytest
from scripts.crossplay_check import verify_lineages


def test_crossplay_keeps_other_seed_out_of_training_population():
    candidate={'config':{'seed':11},'sums':{'one':[1,2]},'archive':[{'old':[2,1]}]}
    parent=deepcopy(candidate)
    opponent={'config':{'seed':29},'sums':{'independent':[3,1]}}
    verify_lineages(candidate,parent,opponent)
    opponent['config']['seed']=11
    with pytest.raises(ValueError,match='independent'):verify_lineages(candidate,parent,opponent)
    opponent['config']['seed']=29;opponent['sums']=candidate['archive'][0]
    with pytest.raises(ValueError,match='overlaps'):verify_lineages(candidate,parent,opponent)
    parent['config']['seed']=99
    with pytest.raises(ValueError,match='recipe'):verify_lineages(candidate,parent,opponent)


@pytest.mark.parametrize('stage,seed,blocks,deadline',[(0,1,2,'2026-11-14T17:31:00+00:00'),(2000,-1,2000,'2026-11-14T17:31:00+00:00'),(2000,6000000,1,'2026-11-14T17:31:00+00:00'),(2000,6000000,2000,'2026-11-14T17:31:00')])
def test_crossplay_rejects_invalid_design(stage,seed,blocks,deadline):
    from datetime import datetime
    from scripts.crossplay_check import validate_design
    with pytest.raises(ValueError):validate_design(stage,seed,blocks,datetime.fromisoformat(deadline))
