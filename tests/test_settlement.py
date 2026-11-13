"""Exact-value engine contracts: integer chip payouts, not just conservation."""
from pokerkit import Automation, NoLimitTexasHoldem


def state(stacks, holes):
    auto=tuple(a for a in Automation if a not in (Automation.HOLE_DEALING,Automation.BOARD_DEALING))
    s=NoLimitTexasHoldem.create_state(auto,True,0,(1,2),2,stacks,len(stacks))
    for hole in holes:s.deal_hole(hole)
    return s


def runout(s,board):
    for cards in board:s.deal_board(cards)
    assert not s.status
    return s.payoffs


def test_three_side_pots_and_uncalled_excess():
    s=state((10,20,30,40),('AsAd','KsKd','QsQd','JsJd'))
    # UTG 30, button 40, SB 10, BB 20: 10 unmatched chips returned.
    while s.actor_index is not None:
        if s.can_complete_bet_or_raise_to():s.complete_bet_or_raise_to(s.max_completion_betting_or_raising_to_amount)
        else:s.check_or_call()
    assert runout(s,('2c3c4h','8s','9d'))==[30,10,-10,-30]


def test_board_split_odd_chip_left_of_button():
    s=state((200,200,200),('2c3c','4c5c','6c7c'))
    s.complete_bet_or_raise_to(4);s.fold();s.check_or_call()
    for board in ('AsKsQs','Js','Ts'):
        s.deal_board(board)
        while s.actor_index is not None:s.check_or_call()
    # Folded SB contributes 1. Nine-chip pot: first eligible seat (BB) gets 5.
    assert s.payoffs==[-1,1,0]
