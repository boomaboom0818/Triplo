"""V6 exactly enumerates dangerous external branches instead of clipping weights."""
import math
from .prefix import PrefixMCCFR
from .cfr import TraversalLimit

ALGORITHM='adaptive-average-mccfr-v6'
AVERAGING='adaptive-external-enumeration-average-v6'
WEIGHT_CAP=4096


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='prefix-v3' or new.pop('traversal',None)!='adaptive-average-v6':return False
    if old.pop('averaging_sampler',None)!='trajectory-call-mixture-v2' or new.pop('averaging_sampler',None)!=AVERAGING:return False
    return new.pop('average_weight_cap',None)==WEIGHT_CAP and old==new


class AdaptiveAverageMCCFR(PrefixMCCFR):
    def __init__(self,seed=1,samples=8,depth=2,cap=WEIGHT_CAP):
        super().__init__(seed,samples,depth)
        if not math.isfinite(cap) or cap<1:raise ValueError('adaptive cap must be finite and>=1')
        self.average_cap=cap

    def trajectory_average(self,state,owner,weight=1.):
        self._average(state,owner,weight,self.average_cap*weight)

    def _action(self,state,owner,kind):
        if not hasattr(state,'game'):return
        o=state.game.observe();role='owner' if o.player==owner else 'other'
        getattr(self,'pending_diagnostics',self.diagnostics)[f'average-actions-v6:street={o.street}:active={sum(o.active)}:actor={role}:{kind}']+=1

    def _average(self,state,owner,weight,cap):
        if not math.isfinite(weight*weight) or weight<=0 or weight>cap*(1+1e-12):raise TraversalLimit('invalid adaptive average weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_average_weight']=max(self.pending_traversal_stats['max_average_weight'],weight)
        self.visit()
        if state.terminal():return
        game=getattr(state,'game',None)
        if game is not None and hasattr(game,'order'):
            position=game.order.index(owner)
            if not game.state.statuses[position] or game.state.stacks[position]==0:return
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.record_average_weight(state,weight);row=getattr(self,'pending_sums',self.sums).setdefault(key,[0.]*len(actions))
            for j,p in enumerate(strategy):
                row[j]+=weight*p
                if not math.isfinite(row[j]):raise TraversalLimit('nonfinite adaptive average update')
            self._action(state,owner,'sampled')
            self._average(state.child(self.rng.choices(actions,strategy)[0]),owner,weight,cap)
            return
        q=[.5/len(actions)+(.5 if a=='call' else 0.) for a in actions] if 'call' in actions else [1/len(actions)]*len(actions)
        high=[j for j,p in enumerate(q) if weight/p>cap]
        low=[j for j in range(len(actions)) if j not in high]
        # Guaranteed inclusion has probability1: no sampling correction. Never
        # clip/drop these branches. The remaining set has its own conditional q.
        for j in high:
            self._action(state,owner,'enumerated')
            self._average(state.child(actions[j]),owner,weight,cap)
        if low:
            mass=sum(q[j] for j in low);j=self.rng.choices(low,[q[j] for j in low])[0]
            self._action(state,owner,'sampled')
            self._average(state.child(actions[j]),owner,weight*mass/q[j],cap)
