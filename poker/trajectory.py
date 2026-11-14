"""Versioned outcome-sampled regret and on-policy-owner strategy averaging.

One path per sample avoids enumerating the updater's exponentially many action
sequences. Regret uses a full-support 50% uniform / 50% current-policy sampler,
with separate corrections for earlier updater choices and sampled action values.
Opponents remain on-policy (including the explicitly selected frozen population).
Eight independent paths per player are averaged, retaining one iteration's update
scale. Averaging samples owner actions on-policy and other actions with the existing
call mixture; inverse external sampling reach remains necessary in multiplayer.
This estimator does not imply multiplayer equilibrium or competitive strength.
"""
import math
from .cfr import MCCFR,normalize,TraversalLimit

ALGORITHM='outcome-sampling-mccfr-v2'
AVERAGING='trajectory-call-mixture-v2'


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.get('averaging_sampler')!='call-mixture-v1':return False
    if new.pop('traversal',None)!='trajectory-v2':return False
    samples=new.pop('trajectory_samples',None)
    if type(samples) is not int or not 1<=samples<=64:return False
    if new.pop('averaging_sampler',None)!=AVERAGING:return False
    old.pop('averaging_sampler')
    return old==new


class TrajectoryMCCFR(MCCFR):
    def __init__(self,seed=1,samples=8):
        super().__init__(seed)
        if type(samples) is not int or not 1<=samples<=64:raise ValueError('invalid trajectory sample count')
        self.samples=samples
        self.traversal_stats=dict(iterations=0,total_nodes=0,max_iteration_nodes=0,max_inverse_own_sample=0.0,max_average_weight=0.0)

    def step(self,*args,**kwargs):
        before=self.nodes;self.pending_traversal_stats=dict(max_inverse_own_sample=0.0,max_average_weight=0.0)
        try:
            super().step(*args,**kwargs)
            nodes=self.nodes-before;stats=self.traversal_stats
            stats['iterations']+=1;stats['total_nodes']+=nodes
            stats['max_iteration_nodes']=max(stats['max_iteration_nodes'],nodes)
            for key,value in self.pending_traversal_stats.items():stats[key]=max(stats[key],value)
        finally:del self.pending_traversal_stats

    def traverse(self,factory,players,delta):
        for player in range(players):
            for _ in range(self.samples):
                self.outcome(factory(self.rng.randrange(2**63)),player,delta,scale=1/self.samples)
                self.trajectory_average(factory(self.rng.randrange(2**63)),player,weight=1/self.samples)

    def outcome(self,state,updater,delta,inverse_own_sample=1.0,scale=1.0):
        if not math.isfinite(inverse_own_sample):raise TraversalLimit('nonfinite importance weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_inverse_own_sample']=max(self.pending_traversal_stats['max_inverse_own_sample'],inverse_own_sample)
        self.visit()
        if state.terminal():return state.returns()[updater]
        self.record_visit(state,'regret',updater)
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor!=updater:
            frozen=getattr(self,'opponents',{}).get(actor)
            if isinstance(frozen,str):
                from .population import profile_strategy
                strategy=profile_strategy(state.game.observe(),frozen)
                getattr(self,'pending_diagnostics',self.diagnostics)[f'population:{frozen}:sampled_actions']+=1
            elif frozen is not None:strategy=normalize(frozen.get(key,[0.0]*len(actions)))
            action=self.rng.choices(actions,strategy)[0]
            return self.outcome(state.child(action),updater,delta,inverse_own_sample,scale)
        q=[.5*p+.5/len(actions) for p in strategy]
        index=self.rng.choices(range(len(actions)),q)[0]
        action_value=self.outcome(state.child(actions[index]),updater,delta,inverse_own_sample/q[index],scale)/q[index]
        value=strategy[index]*action_value
        if not math.isfinite(action_value) or not math.isfinite(value):raise TraversalLimit('nonfinite regret estimate')
        row=delta.setdefault(key,[0.0]*len(actions))
        for i in range(len(actions)):
            row[i]+=scale*inverse_own_sample*((action_value if i==index else 0.0)-value)
            if not math.isfinite(row[i]):raise TraversalLimit('nonfinite regret update')
        return value

    def trajectory_average(self,state,owner,weight=1.0):
        if not math.isfinite(weight*weight):raise TraversalLimit('nonfinite averaging weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_average_weight']=max(self.pending_traversal_stats['max_average_weight'],weight)
        self.visit()
        if state.terminal():return
        self.record_visit(state,'average',owner)
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.record_average_weight(state,weight)
            row=getattr(self,'pending_sums',self.sums).setdefault(key,[0.0]*len(actions))
            for i,p in enumerate(strategy):row[i]+=weight*p
            # Sample probability supplies the owner's reach; do not divide by it.
            action=self.rng.choices(actions,strategy)[0]
            self.trajectory_average(state.child(action),owner,weight)
        else:
            q=[.5/len(actions)+(.5 if a=='call' else 0.0) for a in actions] if 'call' in actions else [1/len(actions)]*len(actions)
            index=self.rng.choices(range(len(actions)),q)[0]
            self.trajectory_average(state.child(actions[index]),owner,weight/q[index])
