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
