"""Reference external-sampling MCCFR. No multiplayer equilibrium guarantee.

Chance is sampled by the caller. Regret traversal enumerates updater actions and
samples others on-policy. A separate averaging traversal enumerates owner actions
and samples others with full support, correcting their sampling reach. This avoids the
incorrect multiplayer 'simple averaging' shortcut.
"""
import copy
import time
from random import Random
from .abstraction import bucket_key,bucket_key_v2,VERSION,VERSION_V2
from collections import Counter


def normalize(weights):
    positive=[max(0.0,x) for x in weights]
    total=sum(positive)
    return [x/total for x in positive] if total else [1/len(weights)]*len(weights)


class TraversalLimit(RuntimeError):
    """An iteration exceeded its configured resource budget."""


class MCCFR:
    def __init__(self,seed=1):
        self.rng=Random(seed);self.regrets={};self.sums={};self.iterations=0
        self.nodes=0;self.diagnostics=Counter();self.average_weight_moments={}

    def visit(self):
        if getattr(self,"stop_requested",lambda:False)():raise KeyboardInterrupt
        self.nodes+=1
        if getattr(self,"node_limit",None) is not None and self.nodes>self.node_limit:
            raise TraversalLimit("iteration node budget exceeded")
        if getattr(self,"deadline",None) is not None and time.monotonic()>self.deadline:
            raise TraversalLimit("run time budget reached")

    def record_visit(self,state,traversal,owner):
        if not hasattr(state,'game'):return
        o=state.game.observe()
        kind='enumerated' if o.player==owner else 'sampled'
        key=f'{traversal}:street={o.street}:active={sum(o.active)}:{kind}'
        getattr(self,'pending_diagnostics',self.diagnostics)[key]+=1

    def record_average_weight(self,state,weight):
        if not hasattr(state,'game'):return
        o=state.game.observe();key=f'street={o.street}:active={sum(o.active)}'
        table=getattr(self,'pending_weight_moments',self.average_weight_moments)
        row=table.setdefault(key,dict(updates=0,total=0.0,squared_total=0.0,maximum=0.0))
        row['updates']+=1;row['total']+=weight;row['squared_total']+=weight*weight
        row['maximum']=max(row['maximum'],weight)

    def strategy(self,key,actions):
        row=self.regrets.get(key,[0.0]*len(actions))
        if len(row)!=len(actions):raise ValueError('information-set action mismatch')
        return normalize(row)

    def regret(self,state,updater,delta):
        self.visit()
        if state.terminal():return state.returns()[updater]
        self.record_visit(state,'regret',updater)
        actor,key,actions=state.info()
        strategy=self.strategy(key,actions)
        if actor!=updater:
            frozen=getattr(self,'opponents',{}).get(actor)
            if frozen is not None:strategy=normalize(frozen.get(key,[0.0]*len(actions)))
            return self.regret(state.child(self.rng.choices(actions,strategy)[0]),updater,delta)
        values=[self.regret(state.child(a),updater,delta) for a in actions]
        value=sum(p*v for p,v in zip(strategy,values))
        row=delta.setdefault(key,[0.0]*len(actions))
        for i,v in enumerate(values):row[i]+=v-value
        return value

    def average(self,state,owner,own_reach=1.0,inverse_sample=1.0):
        self.visit()
        if state.terminal():return
        self.record_visit(state,'average',owner)
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.record_average_weight(state,own_reach*inverse_sample)
            row=getattr(self,'pending_sums',self.sums).setdefault(key,[0.0]*len(actions))
            for i,p in enumerate(strategy):row[i]+=own_reach*inverse_sample*p
            for a,p in zip(actions,strategy):
                if p:self.average(state.child(a),owner,own_reach*p,inverse_sample)
        else:
            if getattr(self,'averaging_sampler','uniform')=='call-mixture-v1' and 'call' in actions:
                # Full support: 50% uniform slots, 50% call. Correct each sampled reach.
                probabilities=[.5/len(actions)+(.5 if a=='call' else 0) for a in actions]
                index=self.rng.choices(range(len(actions)),probabilities)[0]
                self.average(state.child(actions[index]),owner,own_reach,inverse_sample/probabilities[index])
            else:
                self.average(state.child(self.rng.choice(actions)),owner,own_reach,inverse_sample*len(actions))

    def step(self,factory,players,max_nodes=None,deadline=None,opponents=None):
        # An interrupted traversal must not leak partial averages or RNG state.
        self.opponents=opponents or {}
        rng_state=self.rng.getstate(); old_nodes=self.nodes
        self.node_limit=None if max_nodes is None else self.nodes+max_nodes
        self.deadline=deadline; self.pending_sums={};self.pending_diagnostics=Counter();self.pending_weight_moments={};delta={}
        try:
            for player in range(players):
                self.regret(factory(self.rng.randrange(2**63)),player,delta)
                self.average(factory(self.rng.randrange(2**63)),player)
        except BaseException:
            self.rng.setstate(rng_state);self.nodes=old_nodes
            raise
        else:
            for destination,updates in ((self.regrets,delta),(self.sums,self.pending_sums)):
                for key,values in updates.items():
                    row=destination.setdefault(key,[0.0]*len(values))
                    for i,v in enumerate(values):row[i]+=v
            for key,values in self.pending_weight_moments.items():
                row=self.average_weight_moments.setdefault(key,dict(updates=0,total=0.0,squared_total=0.0,maximum=0.0))
                for field in ('updates','total','squared_total'):row[field]+=values[field]
                row['maximum']=max(row['maximum'],values['maximum'])
            self.diagnostics.update(self.pending_diagnostics)
            self.iterations+=1
        finally:
            del self.pending_sums
            del self.pending_diagnostics
            del self.pending_weight_moments
            self.node_limit=None;self.deadline=None


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
    def __init__(self,seed,players=6,abstraction="exact-observation-v1"):
        from .game import Game
        self.game=Game(seed,players);self.abstraction=abstraction
    def terminal(self):return not self.game.state.status
    def returns(self):return self.game.payoff()
    def info(self):
        o=self.game.observe()
        return o.player,information_key(o,self.abstraction),o.legal
    def child(self,action):
        result=copy.deepcopy(self);result.game.act(action);return result


def holdem_key(o):
    # Exact cards and full observed history preserve recall in this reference.
    # No opponent cards, deck, simulator state or future board is included.
    import json
    return json.dumps((o.player,o.position,sorted(o.hole),o.board,o.stacks,
                       o.active,o.pot,o.call,o.street,o.legal,o.history),separators=(',',':'))


def information_key(o,abstraction):
    if abstraction=='exact-observation-v1':return holdem_key(o)
    if abstraction==VERSION:return bucket_key(o)
    if abstraction==VERSION_V2:return bucket_key_v2(o)
    raise ValueError('unsupported abstraction')


class CFRPolicy:
    def __init__(self,sums,seed=0,abstraction="exact-observation-v1"):
        self.sums=sums;self.rng=Random(seed);self.abstraction=abstraction
    def act(self,o):
        row=self.sums.get(information_key(o,self.abstraction),[0.0]*len(o.legal))
        if len(row)!=len(o.legal):raise ValueError('information-set action mismatch')
        return self.rng.choices(o.legal,normalize(row))[0]
