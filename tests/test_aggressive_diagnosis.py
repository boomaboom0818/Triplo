import pytest
from poker.history_cfr import ParallelHistory, compact_observation_key
from poker.live_curriculum import root_game
from scripts.diagnose_aggressive_history import run, normalize, tv


def test_frozen_average_mass_and_regret_disagreement_are_measured(tmp_path):
    solver=ParallelHistory()
    o=root_game(19,6,[100]*6).observe()
    key=compact_observation_key(o);start=solver._slot(key,len(o.legal))
    solver.sums[start+1]=1;solver.regrets[start]=1;solver.visits[start]=64
    parent=tmp_path/'parent.json';solver.export_policy(parent)
    solver.sums[start+1]=2;solver.visits[start]=100
    candidate=tmp_path/'candidate.json';solver.export_policy(candidate)
    solver.save_state(tmp_path/'state')
    out=tmp_path/'report.json'
    report=run(candidate,parent,parent,tmp_path/'state',out,deals=2,seed=19)
    rows=report['consumed_development_decisions']
    assert rows and all(r['parent_fraction']==pytest.approx(.5) for r in rows)
    assert all(r['new_visits']==36 for r in rows)
    assert all(r['regret_tv']==pytest.approx(1.) for r in rows)
    assert report['all_learned']['mean_policy_tv_from_parent']==0
    with pytest.raises(ValueError):run(candidate,parent,parent,tmp_path/'state',out,deals=2,seed=19)
