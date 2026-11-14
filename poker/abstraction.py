"""Deterministic coarse public/own-card abstraction; intentionally imperfect recall.

Version changes require new checkpoints. No simulator or opponent cards are inputs.
"""
from collections import Counter
from functools import lru_cache
import json
from pokerkit import StandardHighHand

VERSION='texture-buckets-v1'
RANKS='23456789TJQKA'

@lru_cache(maxsize=100000)
def cards_bucket(hole,board):
    ranks=sorted((RANKS.index(c[0]) for c in hole),reverse=True)
    if not board:return ('pre',*ranks,hole[0][1]==hole[1][1])
    cards=hole+board
    hand=StandardHighHand.from_game(''.join(hole),''.join(board))
    # Hand category, private ranks, suitedness, board texture and draw potential.
    suits=Counter(c[1] for c in cards);board_suits=Counter(c[1] for c in board)
    counts=Counter(c[0] for c in board)
    values={RANKS.index(c[0])+2 for c in cards}
    if 14 in values:values.add(1)
    straight=max(sum(v in values for v in range(low,low+5)) for low in range(1,11))
    return ('post',str(hand.entry.label),ranks[0]//3,ranks[1]//3,
            max(suits.values()),straight,max(board_suits.values()),
            tuple(sorted(counts.values(),reverse=True)),
            max(RANKS.index(c[0]) for c in board)//3,
            sum(c[0] in {x[0] for x in board} for c in hole))


def bucket_key(o):
    # Numeric raises are represented by ordered menu slots and coarse pot ratios.
    legal=tuple(a if isinstance(a,str) else min(20,round(a/max(2,o.pot)*4)) for a in o.legal)
    raises=tuple(min(3,sum(isinstance(a,int) and street==s for _,_,street,a,_,_ in o.history)) for s in range(o.street+1))
    return json.dumps((VERSION,o.position,o.street,cards_bucket(tuple(sorted(o.hole)),o.board),
        o.active,min(8,o.call*8//max(1,o.pot)),
        tuple(min(8,stack*2//max(1,o.pot)) for stack in o.stacks),legal,raises),separators=(',',':'))


VERSION_V2='texture-buckets-v2'

def bucket_key_v2(o):
    """Experimental compact state; ordered raise slots retain their pot ratios.

    Own position and relative active counts replace exact opponent seat masks.
    Effective stack buckets replace the six-seat stack vector. This loses detail
    deliberately and requires independent training; v1 tables remain unchanged.
    """
    cards=cards_bucket(tuple(sorted(o.hole)),o.board)
    if o.street:
        cards=(cards[0],cards[1],min(4,cards[4]),min(4,cards[5]),cards[6],cards[7])
    legal=tuple(a if isinstance(a,str) else min(20,round(a/max(2,o.pot)*4)) for a in o.legal)
    active=[s for i,s in enumerate(o.stacks) if o.active[i] and i!=o.position]
    effective=min(o.stacks[o.position],max(active,default=0))
    spr=min(4,effective//max(1,o.pot))
    return json.dumps((VERSION_V2,o.position,o.street,cards,sum(o.active),
        sum(o.active[:o.position]),min(4,o.call*4//max(1,o.pot)),spr,legal),separators=(',',':'))
