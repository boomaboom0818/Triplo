from random import Random
import pytest
from poker.live_curriculum import root_game, sample_root, VERSION
from poker.history_cfr import ParallelHistory, COMPACT_VERSION, LIVE_ALGORITHM


def test_short_table_embedding_and_unequal_stacks():
    for players in (3,4,5,6):
        game=root_game(19,players,[20,50,75,100,200,500])
        assert sum(game.active)==players
        assert game.active==[True,True]+[False]*(6-players)+[True]*(players-2)
        assert game.paid==[1,2,0,0,0,0]
        assert len(game.history)==6-players
        assert game.stacks[5]==1000
    a,b=Random(91),Random(91)
    roots=[sample_root(a) for _ in range(100)]
    assert [g.stacks for g in roots]==[sample_root(b).stacks for _ in range(100)]
    assert {sum(g.active) for g in roots}=={3,4,5,6}
    assert any(len(set(g.stacks[2:]))>1 for g in roots)


def test_curriculum_resume_is_exact_and_versioned(tmp_path):
    a=ParallelHistory(901,100,.75,COMPACT_VERSION,VERSION)
    a.round(1,2)
    a.save_state(tmp_path/'state')
    b=ParallelHistory.load_state(tmp_path/'state')
    assert b.curriculum==VERSION and b.meta()['algorithm']==LIVE_ALGORITHM
    a.round(1,2);b.round(1,2)
    assert a.index==b.index and a.regrets==b.regrets and a.sums==b.sums and a.visits==b.visits
    with pytest.raises(ValueError):ParallelHistory(curriculum='unknown')


def test_deep_aggressive_population_finishes_without_tiny_raise_wars():
    from poker.live_curriculum import TrainingAggressor
    for seed in range(40):
        game=root_game(seed,6,[1000]*6,lean=False)
        policies=[TrainingAggressor(i%3) for i in range(6)]
        while game.live:
            o=game.observe(); game.act(policies[o.position].act(o))
            assert len(game.history)<150
        assert sum(game.payoffs)==0


def test_forked_training_preserves_parent_and_rejects_resume_distribution_change(tmp_path,monkeypatch):
    import json
    import scripts.history_train as trainer
    monkeypatch.setattr(trainer.resource,'setrlimit',lambda *args:None)
    monkeypatch.setattr(trainer,'available_memory',lambda:0)
    parent=tmp_path/'parent';ParallelHistory(11,100,.5).save_state(parent)
    before={p.name:p.read_bytes() for p in parent.iterdir()}
    root=tmp_path/'candidate'
    trainer.train(root,.001,seed=19,stack_bb=100,workers=2,population=.75,abstraction=COMPACT_VERSION,
                  curriculum=VERSION,initialize_from=parent)
    assert before=={p.name:p.read_bytes() for p in parent.iterdir()}
    saved=ParallelHistory.load_state(root/'state')
    assert saved.seed==19 and saved.curriculum==VERSION and saved.population_probability==.75
    with pytest.raises(ValueError,match='configuration mismatch'):
        trainer.train(root,.001,seed=19,stack_bb=100,workers=2,population=.75,abstraction=COMPACT_VERSION,
                      resuming=True,curriculum=None)


def test_aggressive_curriculum_emphasizes_target_and_preserves_diversity():
    from poker.live_curriculum import AGGRESSIVE_VERSION, TrainingAggressor, opponent
    rng=Random(203)
    games=[sample_root(rng,AGGRESSIVE_VERSION) for _ in range(1000)]
    target=sum(sum(g.active)==6 and g.stacks==[199,198,200,200,200,200] for g in games)
    assert target>550
    assert {sum(g.active) for g in games}=={3,4,5,6}
    assert any(max(g.stacks)>400 for g in games)
    assert sum(isinstance(opponent(rng,AGGRESSIVE_VERSION),TrainingAggressor) for _ in range(1000))>750
    solver=ParallelHistory(701,100,.8,COMPACT_VERSION,AGGRESSIVE_VERSION)
    solver.round(1,2)
    assert solver.iterations==2 and solver.nodes>0
