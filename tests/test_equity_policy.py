from dataclasses import replace
import pytest
from poker.game import Observation
from poker.equity_policy import EquityPolicy, estimate_equity


def observation(hole=('2c','3d'), board=('Ah','Kh','Qh','Jh','Th'), active=(True,True,False,False,False,False), call=20):
    return Observation(0,0,hole,board,(200,200,200,200,200,200),active,100,call,
                       {3:1,4:2,5:3}[len(board)],('call','fold',40,100),())


def test_board_royal_splits_with_every_opponent():
    for players in (2,3,6):
        o=observation(active=tuple(i<players for i in range(6)))
        equity,error=estimate_equity(o,64)
        assert equity==pytest.approx(1/players)
        assert error<1e-7
    assert EquityPolicy(64).act(observation(call=0))=='call'


def test_nuts_value_bet_and_drawing_dead_fold():
    p=EquityPolicy(128)
    nuts=observation(('Ah','Kh'),('Qh','Jh','Th','2d','3c'))
    assert estimate_equity(nuts,64)==(1.,0.)
    assert isinstance(p.act(nuts),int)
    dead=observation(('2c','3d'),('Ah','Ad','Ac','Ks','Qh'),call=200)
    assert p.act(dead)=='fold'


def test_flush_draw_calls_small_price_and_rejects_large_price():
    p=EquityPolicy(256)
    draw=observation(('Ah','9h'),('Kh','2h','7c'),call=5)
    assert p.act(draw)=='call'
    assert p.act(replace(draw,call=200))=='fold'


def test_equity_reproducible_and_respects_multiway_competition():
    o=observation(('As','Kd'),('Ac','7h','2s'))
    first=estimate_equity(o,256)
    assert first==estimate_equity(o,256)
    multi=replace(o,active=(True,)*6)
    assert estimate_equity(multi,256)[0]<first[0]
    assert EquityPolicy(64).act(replace(o,call=0))!='fold'


def test_preflop_preserves_baseline_and_validates_budget():
    p=EquityPolicy(64)
    o=replace(observation(),street=0,board=())
    assert p.act(o)==p.preflop.act(o)
    with pytest.raises(ValueError): EquityPolicy(0)


def test_private_strength_excludes_shared_board_categories():
    from poker.equity_policy import private_strength
    from poker.fast_eval import CARD_ID
    def strength(hole, board):
        return private_strength(tuple(map(CARD_ID.__getitem__, hole)), tuple(map(CARD_ID.__getitem__, board)))
    assert not strength(('As','Kd'), ('7c','7d','7h'))[0]
    assert not strength(('As','2d'), ('Kc','Kd','Qc','Qd'))[0]
    assert not strength(('As','Qd'), ('Kc','Qh','2c'))[1]
    assert strength(('Qs','Qd'), ('Kc','Qh','2c'))[0]
    assert strength(('As','Ad'), ('Kc','Qh','2c'))[1]
    assert strength(('7s','Ad'), ('7c','7d','2h'))[0]


def test_guarded_preflop_counts_repeated_small_calls():
    from poker.equity_policy import GuardedEquityPolicy, preflop_commitment
    # KQs scores ten: each call fits the old 20-chip price limit, but the
    # cumulative cost exceeds it after repeated small raises.
    o=replace(observation(('Ks','Qs')), street=0, board=(), call=4,
              history=((0,0,0,18,3,1),(1,1,0,22,20,16)), legal=('call','fold',26))
    assert preflop_commitment(o)==18
    assert EquityPolicy(32).act(o)=='call'
    assert GuardedEquityPolicy(32).act(o)=='fold'
    assert GuardedEquityPolicy(32).act(replace(o,call=0))=='call'
    assert GuardedEquityPolicy(32).act(replace(o,hole=('As','Ah')))!='fold'


def test_guarded_does_not_raise_board_trips_under_pressure(monkeypatch):
    import poker.equity_policy as module
    monkeypatch.setattr(module, 'estimate_equity', lambda *a: (.95, 0.))
    o=observation(('As','Kd'), ('7c','7d','7h'), call=10)
    assert module.GuardedEquityPolicy(32).act(o)=='call'
    assert isinstance(module.GuardedEquityPolicy(32).act(
        replace(o,hole=('7s','Ad'))), int)
