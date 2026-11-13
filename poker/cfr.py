"""Reference external-sampling MCCFR. No multiplayer equilibrium guarantee.

Chance is sampled by the caller. Regret traversal enumerates updater actions and
samples others on-policy. A separate averaging traversal enumerates owner actions
and samples others uniformly, correcting their sampling reach. This avoids the
incorrect multiplayer 'simple averaging' shortcut.
"""
import copy
from random import Random


def normalize(weights):
    positive=[max(0.0,x) for x in weights]
    total=sum(positive)
    return [x/total for x in positive] if total else [1/len(weights)]*len(weights)


class MCCFR:
    def __init__(self,seed=1):
        self.rng=Random(seed);self.regrets={};self.sums={};self.iterations=0
        self.nodes=0

    def strategy(self,key,actions):
        row=self.regrets.get(key,[0.0]*len(actions))
        if len(row)!=len(actions):raise ValueError('information-set action mismatch')
        return normalize(row)

    def regret(self,state,updater,delta):
        self.nodes+=1
        if state.terminal():return state.returns()[updater]
        actor,key,actions=state.info()
        strategy=self.strategy(key,actions)
        if actor!=updater:
            return self.regret(state.child(self.rng.choices(actions,strategy)[0]),updater,delta)
        values=[self.regret(state.child(a),updater,delta) for a in actions]
        value=sum(p*v for p,v in zip(strategy,values))
        row=delta.setdefault(key,[0.0]*len(actions))
        for i,v in enumerate(values):row[i]+=v-value
        return value

    def average(self,state,owner,own_reach=1.0,inverse_sample=1.0):
        self.nodes+=1
        if state.terminal():return
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            row=self.sums.setdefault(key,[0.0]*len(actions))
            for i,p in enumerate(strategy):row[i]+=own_reach*inverse_sample*p
            for a,p in zip(actions,strategy):
                if p:self.average(state.child(a),owner,own_reach*p,inverse_sample)
        else:
            self.average(state.child(self.rng.choice(actions)),owner,own_reach,inverse_sample*len(actions))

    def step(self,factory,players):
        # All updates use the frozen strategy from the start of this iteration.
        delta={}
        for player in range(players):
            self.regret(factory(self.rng.randrange(2**63)),player,delta)
            self.average(factory(self.rng.randrange(2**63)),player)
        for key,values in delta.items():
            row=self.regrets.setdefault(key,[0.0]*len(values))
            for i,v in enumerate(values):row[i]+=v
        self.iterations+=1


class Kuhn:
    """Two-player three-card Kuhn, an algorithm test only."""
    def __init__(self,seed,cards=None,history=''):
        self.cards=tuple(Random(seed).sample(range(3),2)) if cards is None else cards
        self.history=history
    def terminal(self):return self.history in ('pp','bp','bb','pbp','pbb')
    def returns(self):
        h=self.history
        if h=='bp':v=1
        elif h=='pbp':v=-1
        else:v=(1 if self.cards[0]>self.cards[1] else -1)*(2 if h.endswith('bb') else 1)
        return [v,-v]
    def info(self):
        actor=len(self.history)%2
        return actor,f'kuhn:{actor}:{self.cards[actor]}:{self.history}',('p','b')
    def child(self,action):return Kuhn(0,self.cards,self.history+action)


class Holdem:
    def __init__(self,seed,players=6):
        from .game import Game
        self.game=Game(seed,players)
    def terminal(self):return not self.game.state.status
    def returns(self):return self.game.payoff()
    def info(self):
        o=self.game.observe()
        return o.player,holdem_key(o),o.legal
    def child(self,action):
        result=copy.deepcopy(self);result.game.act(action);return result


def holdem_key(o):
    # Exact cards and full observed history preserve recall in this reference.
    # No opponent cards, deck, simulator state or future board is included.
    import json
    return json.dumps((o.player,o.position,sorted(o.hole),o.board,o.stacks,
                       o.active,o.pot,o.call,o.street,o.legal,o.history),separators=(',',':'))


class CFRPolicy:
    def __init__(self,sums,seed=0):self.sums=sums;self.rng=Random(seed)
    def act(self,o):
        row=self.sums.get(holdem_key(o),[0.0]*len(o.legal))
        if len(row)!=len(o.legal):raise ValueError('information-set action mismatch')
        return self.rng.choices(o.legal,normalize(row))[0]
