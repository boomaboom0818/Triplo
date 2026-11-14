"""Versioned bounded prefix regret sampling; averaging estimator unchanged."""
import math
from .cfr import normalize,TraversalLimit
from .trajectory import TrajectoryMCCFR

ALGORITHM='prefix-outcome-mccfr-v3'


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='trajectory-v2' or new.pop('traversal',None)!='prefix-v3':return False
    depth=new.pop('regret_prefix_depth',None)
    return type(depth) is int and depth in (1,2) and old==new


class PrefixMCCFR(TrajectoryMCCFR):
    """Enumerate at most the first two updater decisions, then sample as v2."""
    def __init__(self,seed=1,samples=8,depth=1):
        super().__init__(seed,samples)
        if type(depth) is not int or not 0<=depth<=2:raise ValueError('prefix depth must be0..2')
        self.depth=depth
        self.epoch_weight_moments={}

    def step(self,*args,**kwargs):
        self.pending_epoch_weight_moments={}
        try:
            super().step(*args,**kwargs)
            for key,row in self.pending_epoch_weight_moments.items():
                dest=self.epoch_weight_moments.setdefault(key,dict(updates=0,total=0.0,squared_total=0.0,maximum=0.0))
                for name in ('updates','total','squared_total'):dest[name]+=row[name]
                dest['maximum']=max(dest['maximum'],row['maximum'])
        finally:del self.pending_epoch_weight_moments

    def record_weight(self,state,kind,weight):
        if not math.isfinite(weight*weight):raise TraversalLimit('nonfinite epoch weight')
        if not hasattr(state,'game') or not hasattr(self,'pending_epoch_weight_moments'):return
        o=state.game.observe();key=f'{kind}:street={o.street}:active={sum(o.active)}'
        row=self.pending_epoch_weight_moments.setdefault(key,dict(updates=0,total=0.0,squared_total=0.0,maximum=0.0))
        row['updates']+=1;row['total']+=weight;row['squared_total']+=weight*weight
        row['maximum']=max(row['maximum'],weight)

    def record_average_weight(self,state,weight):
        self.record_weight(state,'average',weight)
        super().record_average_weight(state,weight)

    def outcome(self,state,updater,delta,inverse_own_sample=1.,scale=1.,own_depth=0):
        if not math.isfinite(inverse_own_sample):raise TraversalLimit('nonfinite importance weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_inverse_own_sample']=max(self.pending_traversal_stats['max_inverse_own_sample'],inverse_own_sample)
        self.visit()
        if state.terminal():return state.returns()[updater]
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        self.record_visit(state,'regret',updater,sampling='enumerated' if actor==updater and own_depth<self.depth else 'sampled')
        if actor!=updater:
            frozen=getattr(self,'opponents',{}).get(actor)
            if isinstance(frozen,str):
                from .population import profile_strategy
                strategy=profile_strategy(state.game.observe(),frozen)
            elif frozen is not None:strategy=normalize(frozen.get(key,[0.]*len(actions)))
            action=self.rng.choices(actions,strategy)[0]
            return self.outcome(state.child(action),updater,delta,inverse_own_sample,scale,own_depth)
        self.record_weight(state,'regret_prefix',inverse_own_sample*scale)
        if own_depth<self.depth:
            values=[self.outcome(state.child(a),updater,delta,inverse_own_sample,scale,own_depth+1) for a in actions]
            value=sum(p*v for p,v in zip(strategy,values))
        else:
            q=[.5*p+.5/len(actions) for p in strategy];index=self.rng.choices(range(len(actions)),q)[0]
            action_value=self.outcome(state.child(actions[index]),updater,delta,inverse_own_sample/q[index],scale,own_depth+1)/q[index]
            values=[action_value if j==index else 0. for j in range(len(actions))]
            value=strategy[index]*action_value
        row=delta.setdefault(key,[0.]*len(actions))
        for j,v in enumerate(values):
            row[j]+=scale*inverse_own_sample*(v-value)
            if not math.isfinite(row[j]):raise TraversalLimit('nonfinite regret update')
        return value

