import pytest
from poker.game import Game,play
from poker.policy import Policy
@pytest.mark.parametrize('n',range(3,7))
def test_conservation_and_replay(n):
    for seed in range(20):
        def run():return play([Policy(i+seed) for i in range(n)],seed,seed%n)[0]
        a=run(); assert sum(a)==0; assert a==run()
def test_order_and_privacy():
    g=Game(4,6,0); o=g.observe()
    assert o.player==3 and len(o.hole)==2 and not o.board
    assert not hasattr(o,'deck') and not hasattr(o,'state')
    for _ in range(6):g.act('call')
    assert g.observe().player==1
    assert len(g.observe().board)==3
def test_illegal():
    with pytest.raises(ValueError):Game(1).act(9999)
def test_unequal_allins():
    g=Game(5,6,stacks=[10,20,30,40,50,60])
    while g.state.status:
        o=g.observe();g.act(o.legal[-1] if isinstance(o.legal[-1],int) else 'call')
    assert sum(g.payoff())==0

def test_folded_blinds_and_uncalled_raise():
    g=Game(1)
    g.act(200)
    while g.state.status:g.act('fold')
    assert g.payoff()==[0,-.5,-1,1.5,0,0]

@pytest.mark.parametrize('button',range(6))
def test_button_rotation(button):
    g=Game(1,button=button)
    assert g.observe().player==(button+3)%6
    for _ in range(6):g.act('call')
    assert g.observe().player==(button+1)%6

def test_fold_down_to_heads_up():
    g=Game(1)
    for _ in range(4):g.act('fold')
    assert sum(g.observe().active)==2
    while g.state.status:g.act('call')
    assert sum(g.payoff())==0

def test_short_allin_does_not_reopen():
    # Three handed: button raises to 10, SB calls, BB can only raise to 15.
    g=Game(1,3,button=0,stacks=[200,200,15])
    g.state.complete_bet_or_raise_to(10)
    g.state.check_or_call()
    g.state.complete_bet_or_raise_to(15)
    assert not g.state.can_complete_bet_or_raise_to()
    assert all(not isinstance(a,int) for a in g.observe().legal)
