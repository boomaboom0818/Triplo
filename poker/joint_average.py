"""V7: reuse regret-tree inclusion probabilities for exact defensive averaging."""
import math
from .prefix import PrefixMCCFR
from .cfr import normalize,TraversalLimit
from .population import profile_strategy

ALGORITHM='joint-traversal-average-mccfr-v7'
AVERAGING='regret-path-mis-average-v7'


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='prefix-v3' or new.pop('traversal',None)!='joint-average-v7':return False
    if old.pop('averaging_sampler',None)!='trajectory-call-mixture-v2' or new.pop('averaging_sampler',None)!=AVERAGING:return False
    return old==new


def log_probability(p):return math.log(p) if p>0 else -math.inf


def joint_weight(log_own,log_qa,log_pb,scale):
    if log_own==-math.inf:return 0.,0.
    log_pa=log_own+log_qa;high=max(log_pa,log_pb)
    log_sum=high+math.log(math.exp(log_pa-high)+math.exp(log_pb-high))
    try:weight=math.exp(math.log(scale)+log_own-log_sum)
    except OverflowError:raise TraversalLimit('nonfinite joint average weight') from None
    if weight==0:raise TraversalLimit('underflowed positive joint average weight')
    ratio=math.exp(log_pa-log_sum)
    if not math.isfinite(weight*weight) or ratio>1+1e-12:raise TraversalLimit('invalid joint average weight')
    return weight,ratio


def call_proposal(actions):
    return [.5/len(actions)+(.5 if a=='call' else 0.) for a in actions] if 'call' in actions else [1/len(actions)]*len(actions)


class JointAverageMCCFR(PrefixMCCFR):
    def __init__(self,seed=1,samples=8,depth=2):
        super().__init__(seed,samples,depth)
        self.traversal_stats['max_joint_average_to_original_weight_ratio']=0.

    def traverse(self,factory,players,delta):
        diagnostics=getattr(self,'pending_diagnostics',self.diagnostics)
        for player in range(players):
            for _ in range(self.samples):
                diagnostics['joint-traversal-v7:regret-trees']+=1
                self.outcome(factory(self.rng.randrange(2**63)),player,delta,scale=1/self.samples)
                diagnostics['joint-traversal-v7:average-paths']+=1
                self.trajectory_average(factory(self.rng.randrange(2**63)),player,weight=1/self.samples)

    def opponent_strategy(self,state,actor,key,actions,current,record=False):
        frozen=getattr(self,'opponents',{}).get(actor)
        if isinstance(frozen,str):
            if record:getattr(self,'pending_diagnostics',self.diagnostics)[f'population:{frozen}:sampled_actions']+=1
            return profile_strategy(state.game.observe(),frozen)
        return normalize(frozen.get(key,[0.]*len(actions))) if frozen is not None else current

    def average_contribution(self,state,key,actions,strategy,log_own,log_qa,log_pb,scale,source):
        weight,ratio=joint_weight(log_own,log_qa,log_pb,scale)
        if not weight:
            getattr(self,'pending_diagnostics',self.diagnostics)['joint-average-v7:zero-own-target']+=1
            return
        self.average_update_source=source
        self.average_update_ratio=ratio
        self.record_average_weight(state,weight)
        if hasattr(self,'pending_traversal_stats'):
            stats=self.pending_traversal_stats;stats['max_average_weight']=max(stats['max_average_weight'],weight)
            name='max_joint_average_to_original_weight_ratio';stats[name]=max(stats.get(name,0.),ratio)
        row=getattr(self,'pending_sums',self.sums).setdefault(key,[0.]*len(actions))
        for j,p in enumerate(strategy):
            row[j]+=weight*p
            if not math.isfinite(row[j]):raise TraversalLimit('nonfinite joint average update')
        if hasattr(state,'game'):
            o=state.game.observe()
            getattr(self,'pending_diagnostics',self.diagnostics)[f'joint-average-v7:source={source}:owner={o.player}:street={o.street}:active={sum(o.active)}']+=1

    def outcome(self,state,updater,delta,inverse_own_sample=1.,scale=1.,own_depth=0,log_own=0.,log_qa=0.,log_pb=0.):
        # Regret computation and every RNG operation/order match v3 exactly.
        if not math.isfinite(inverse_own_sample):raise TraversalLimit('nonfinite importance weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_inverse_own_sample']=max(self.pending_traversal_stats['max_inverse_own_sample'],inverse_own_sample)
        self.visit()
        if state.terminal():return state.returns()[updater]
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        self.record_visit(state,'regret',updater,sampling='enumerated' if actor==updater and own_depth<self.depth else 'sampled')
        if actor!=updater:
            strategy=self.opponent_strategy(state,actor,key,actions,strategy,True)
            action=self.rng.choices(actions,strategy)[0];j=actions.index(action);qa=call_proposal(actions)
            return self.outcome(state.child(action),updater,delta,inverse_own_sample,scale,own_depth,log_own,log_qa+log_probability(qa[j]),log_pb+log_probability(strategy[j]))
        self.average_contribution(state,key,actions,strategy,log_own,log_qa,log_pb,scale,'regret')
        self.record_weight(state,'regret_prefix',inverse_own_sample*scale)
        if own_depth<self.depth:
            values=[self.outcome(state.child(a),updater,delta,inverse_own_sample,scale,own_depth+1,log_own+log_probability(p),log_qa,log_pb) for a,p in zip(actions,strategy)]
            value=sum(p*v for p,v in zip(strategy,values))
        else:
            q=[.5*p+.5/len(actions) for p in strategy];index=self.rng.choices(range(len(actions)),q)[0]
            action_value=self.outcome(state.child(actions[index]),updater,delta,inverse_own_sample/q[index],scale,own_depth+1,log_own+log_probability(strategy[index]),log_qa,log_pb+log_probability(q[index]))/q[index]
            values=[action_value if j==index else 0. for j in range(len(actions))]
            value=strategy[index]*action_value
        row=delta.setdefault(key,[0.]*len(actions))
        for j,v in enumerate(values):
            row[j]+=scale*inverse_own_sample*(v-value)
            if not math.isfinite(row[j]):raise TraversalLimit('nonfinite regret update')
        return value

    def trajectory_average(self,state,owner,weight=1.):
        self._average_path(state,owner,weight,0,0.,0.,0.)

    def _average_path(self,state,owner,scale,own_depth,log_own,log_qa,log_pb):
        self.visit()
        if state.terminal():return
        self.record_visit(state,'average',owner,sampling='sampled')
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.average_contribution(state,key,actions,strategy,log_own,log_qa,log_pb,scale,'average')
            action=self.rng.choices(actions,strategy)[0];j=actions.index(action)
            qb=1. if own_depth<self.depth else .5*strategy[j]+.5/len(actions)
            self._average_path(state.child(action),owner,scale,own_depth+1,log_own+log_probability(strategy[j]),log_qa,log_pb+log_probability(qb))
        else:
            qa=call_proposal(actions);j=self.rng.choices(range(len(actions)),qa)[0]
            qb=self.opponent_strategy(state,actor,key,actions,strategy)[j]
            self._average_path(state.child(actions[j]),owner,scale,own_depth,log_own,log_qa+log_probability(qa[j]),log_pb+log_probability(qb))
