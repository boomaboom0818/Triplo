import dataclasses
import json
from random import Random

import pytest
from poker.fast_game import FastGame
from poker.history_cfr import PokerState, HistoryPolicy, observation_key, save, resume, load
from poker.sampled_cfr import SampledCFR
from poker.lean_abstraction import observation_key as old_key
from scripts.lean_state_audit import audit


def test_history_separates_previously_colliding_legal_paths():
    for collision in audit(300)['history_collisions']:
        keys = []
        for name in ('first', 'second'):
            game = FastGame(17, lean=True)
            for event in collision[name]['history']:
                game.act(event[3])
            assert old_key(game.observe()) == collision['key']
            keys.append(observation_key(game.observe()))
        assert keys[0] != keys[1]


def test_training_keys_match_observations_and_never_see_other_hole_cards():
    game = FastGame(4, stacks=[40]*6, lean=True)
    state = PokerState(game)
    rng = Random(1)
    while not state.terminal:
        assert state.key() == observation_key(state.game.observe())
        copy = state.game.clone()
        actor = copy.actors[0]
        holes = list(copy.hole)
        holes[(actor+1)%6] = (0,1)
        copy.hole = holes
        assert PokerState(copy).key() == state.key()
        child = state.child(rng.choice(state.actions))
        assert len(child.game.history) == len(state.game.history) + 1
        state = child


def test_history_resume_is_exact_and_rejects_old_checkpoints(tmp_path):
    solver = SampledCFR(6,9)
    def step(s):
        s.iterate(PokerState(FastGame(s.rng.randrange(2**63), stacks=[8]*6, lean=True)))
    for _ in range(4): step(solver)
    path = tmp_path / 'state.json'
    save(path, solver,9,4)
    restored, saved = resume(path)
    step(solver); step(restored)
    assert solver.regrets == restored.regrets and solver.sums == restored.sums
    assert solver.visits == restored.visits and solver.rng.getstate() == restored.rng.getstate()
    saved['abstraction'] = 'lean-v3'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError): load(path)


def test_basic_play_guards_and_sparse_fallback():
    policy = HistoryPolicy(dict(sums={}, visits={}, stack_bb=100))
    first = FastGame(1,lean=True).observe()
    for hole in (('As','Ah'),('Ac','Ad')):
        assert policy.act(dataclasses.replace(first,hole=hole)) != 'fold'
    assert policy.act(dataclasses.replace(first,hole=('7s','2h'))) == 'fold'
    river = dataclasses.replace(first,street=3,board=('As','Ks','Qs','Js','Ts'),
                               hole=('2c','3d'),call=100,legal=('call','fold'))
    assert policy.act(river) == 'call'  # board nuts cannot lose
    nuts = dataclasses.replace(river,board=('As','Ks','Qs','Js','2d'),hole=('Ts','3d'))
    assert policy.act(nuts) == 'call'
    # Even a corrupted/poor learned row cannot fold the nuts.
    key = observation_key(nuts)
    policy.sums[key] = [0,100]
    policy.visits[key] = 1000
    assert policy.act(nuts) == 'call'
    assert policy.covered == 0


def test_action_coordinates_and_player_masks_remain_distinct():
    o = FastGame(1,lean=True).observe()
    assert observation_key(o) != observation_key(dataclasses.replace(o,legal=('call','fold',4,6)))
    assert observation_key(o) != observation_key(dataclasses.replace(o,active=(True,True,True,False,True,True)))


def test_guarded_policy_plays_complete_hands_with_legal_actions_and_balanced_chips():
    policy = HistoryPolicy(dict(sums={},visits={},stack_bb=20))
    for seed in range(12):
        game = FastGame(seed,stacks=[40]*6,lean=True)
        decisions = 0
        while game.live:
            o = game.observe()
            action = policy.act(o)
            assert action in o.legal
            assert action != 'fold' or o.call > 0
            game.act(action)
            decisions += 1
        assert sum(game.payoff()) == 0 and decisions > 0


def test_visit_gate_and_missing_history_use_fallback():
    o = dataclasses.replace(FastGame(1,lean=True).observe(),hole=('7s','2h'))
    key = observation_key(o)
    saved = dict(sums={key:[0.,0.,0.,10.]},visits={key:63},stack_bb=100)
    policy = HistoryPolicy(saved)
    assert policy.act(o) == 'fold'
    policy.visits[key] = 64
    assert policy.act(o) == 5
    # Nonroot observation with no history must not reuse a populated table row.
    later = dataclasses.replace(o,pot=8)
    policy.sums[observation_key(later)] = [0.,0.,0.,10.]
    policy.visits[observation_key(later)] = 1000
    assert policy.act(later) == 'fold'


def test_trainer_refuses_overwrite_and_resume_config_changes(tmp_path, monkeypatch):
    import scripts.history_train as trainer
    monkeypatch.setattr(trainer.resource,'setrlimit',lambda *args:None)
    # Small tmpfs cannot satisfy the production 10GiB reserve, so use a
    # deterministic reserve stop before any traversal in this persistence test.
    monkeypatch.setattr(trainer,'available_memory',lambda:0)
    trainer.train(tmp_path, .001,seed=5,stack_bb=20)
    original = (tmp_path/'checkpoint.json').read_bytes()
    with pytest.raises(ValueError,match='resume'):
        trainer.train(tmp_path,.001,seed=5,stack_bb=20)
    with pytest.raises(ValueError,match='configuration'):
        trainer.train(tmp_path,.001,seed=5,stack_bb=100,resuming=True)
    assert (tmp_path/'checkpoint.json').read_bytes() == original
    trainer.train(tmp_path,.001,seed=5,stack_bb=20,resuming=True)
    assert json.loads((tmp_path/'progress.json').read_text())['stop_reason']=='host reserve'


def test_trainer_signal_finishes_iteration_before_checkpoint(tmp_path, monkeypatch):
    import signal
    import shutil
    from collections import namedtuple
    import scripts.history_train as trainer
    monkeypatch.setattr(trainer.resource,'setrlimit',lambda *args:None)
    monkeypatch.setattr(trainer,'available_memory',lambda:20*1024**3)
    usage=namedtuple('Usage','total used free')(100*1024**3,0,100*1024**3)
    monkeypatch.setattr(shutil,'disk_usage',lambda path:usage)
    original=trainer.SampledCFR.iterate
    def interrupted_iteration(self, root):
        signal.raise_signal(signal.SIGTERM)
        original(self,root)
    monkeypatch.setattr(trainer.SampledCFR,'iterate',interrupted_iteration)
    solver=trainer.train(tmp_path,.1,seed=2,stack_bb=2)
    restored,saved=resume(tmp_path/'checkpoint.json')
    assert solver.iterations == restored.iterations == 1
    assert solver.regrets == restored.regrets
    assert json.loads((tmp_path/'progress.json').read_text())['stop_reason']=='interrupted'


def test_local_gate_uses_checkpoint_stacks_and_records_rule_control(tmp_path):
    from scripts.history_gate import run
    solver = SampledCFR(6,1)
    checkpoint=tmp_path/'model.json'
    save(checkpoint,solver,1,4)
    out=tmp_path/'gate.json'
    report=run(checkpoint,out,deals=2,seed=27)
    assert report['stack_bb']==4
    for style in report['styles'].values():
        assert style['candidate']['hands']==12
        assert style['candidate']['learned_fraction']==0
        assert style['paired_difference']['interval_95']==[0.,0.]
        assert 'preflop_allins_per_100_hands' not in style['candidate']
    with pytest.raises(ValueError): run(checkpoint,out,deals=2)


def test_compact_history_separates_collisions_and_parallel_history_resumes_and_exports(tmp_path, monkeypatch):
    from poker.history_cfr import COMPACT_VERSION, ParallelHistory, compact_observation_key
    import scripts.history_train as trainer
    for collision in audit(300)['history_collisions']:
        keys = []
        for name in ('first', 'second'):
            game = FastGame(17, lean=True)
            for event in collision[name]['history']:
                game.act(event[3])
            keys.append(compact_observation_key(game.observe()))
        assert keys[0] != keys[1]

    # Verify deterministic multi-core round, save_state/load_state, and export_policy
    s1 = ParallelHistory(seed=7, stack_bb=20, population_probability=0.5, abstraction=COMPACT_VERSION)
    s1.round(workers=2, batch=4)
    state_dir = tmp_path / 'state'
    s1.save_state(state_dir)
    s2 = ParallelHistory.load_state(state_dir)
    s1.round(workers=2, batch=4)
    s2.round(workers=2, batch=4)
    assert s1.iterations == s2.iterations == 16
    assert s1.regrets == s2.regrets and s1.sums == s2.sums and s1.visits == s2.visits

    policy_path = tmp_path / 'policy.json'
    s2.export_policy(policy_path)
    loaded = load(policy_path)
    policy = HistoryPolicy(loaded, min_visits=1)
    assert policy.abstraction == COMPACT_VERSION
    o = FastGame(1, stacks=[40]*6, lean=True).observe()
    assert policy.act(o) in o.legal

