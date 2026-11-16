"""Versioned training distribution; held-out gate opponents live elsewhere.

Short tables use the same six-seat embedding as the live adapter: absent seats
are forced folds before action. Heads-up starts remain unsupported by the client.
"""
from hashlib import blake2b
from .fast_game import FastGame
from .fast_eval import CARD_ID
from .lean_abstraction import card_features
from .loose_opponents import LooseCaller, PROFILES

VERSION = 'live-conditions-v1'
AGGRESSIVE_VERSION = 'live-aggressive-v2'
VERSIONS = (VERSION, AGGRESSIVE_VERSION)
DEPTHS = (20, 50, 75, 100, 200, 500)
WEIGHTS = (1, 4, 2, 4, 2, 1)


def root_game(seed, players, stacks_bb, lean=True):
    if players not in (3,4,5,6) or len(stacks_bb)!=6 or any(type(s) is not int or s<2 for s in stacks_bb):
        raise ValueError('need 3-6 players and six positive stack depths >=2 BB')
    game=FastGame(seed,6,button=5,stacks=[2*s for s in stacks_bb],lean=lean)
    for _ in range(6-players):
        game.act('fold')
    return game


def sample_root(rng, version=VERSION):
    if version not in VERSIONS:
        raise ValueError('unsupported curriculum')
    if version == AGGRESSIVE_VERSION and rng.random() < .60:
        return root_game(rng.randrange(2**63), 6, [100]*6)
    players=rng.choices((3,4,5,6),(1,2,3,4))[0]
    depth=rng.choices(DEPTHS,WEIGHTS)[0]
    stacks=[depth]*6
    if rng.random()<.5:
        stacks=[rng.choices(DEPTHS,WEIGHTS)[0] for _ in range(6)]
        if rng.random()<.25:
            stacks[rng.randrange(6)]=1000
    return root_game(rng.randrange(2**63),players,stacks)


class TrainingAggressor:
    """Varied value/pressure profiles, distinct from the deterministic rule gate."""
    def __init__(self, profile):
        self.profile=profile

    def act(self,o):
        raises=[a for a in o.legal if isinstance(a,int)]
        # Exclude tiny reraises in large pots: they create thousands of recursive
        # actions at deep stacks. Still retain full-menu all-ins for pressure.
        if raises:
            meaningful=[a for a in raises if a>=o.call+max(2,o.pot//2)]
            raises=meaningful or raises[-1:]
        fold='fold' if o.call and 'fold' in o.legal else 'call'
        if not o.street:
            a,b=(CARD_ID[c]>>2 for c in o.hole)
            premium=a==b and a>=8 or min(a,b)>=10
            playable=a==b or max(a,b)>=10 or o.hole[0][1]==o.hole[1][1]
            if raises and (premium or playable and o.call<=2):
                # Mix ordinary opens, reraises and occasional oversized pressure.
                target=(3+self.profile)*max(2,o.call)
                return min(raises,key=lambda x:abs(x-target))
            return 'call' if playable and o.call<=4+4*self.profile else fold
        bucket=int(card_features(tuple(CARD_ID[c] for c in o.hole),tuple(CARD_ID[c] for c in o.board))[0])
        mix=int.from_bytes(blake2b(repr((o.hole,o.board,o.history,self.profile)).encode(),digest_size=2).digest(),'big')/65536
        if raises and (bucket>=7 or not o.call and mix<(.15+.1*self.profile)):
            return min(raises,key=lambda x:abs(x-(.5+.5*self.profile)*o.pot))
        return 'call' if bucket>=5 and o.call<=o.pot*(.4+.2*self.profile) else fold


def opponent(rng, version=VERSION):
    if version not in VERSIONS:
        raise ValueError('unsupported curriculum')
    if rng.random() < (.20 if version == AGGRESSIVE_VERSION else .5):
        return LooseCaller(rng.randrange(len(PROFILES)))
    return TrainingAggressor(rng.randrange(3))
