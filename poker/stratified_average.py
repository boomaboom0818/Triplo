"""Explicit v4: v3 regret, deterministic6A/2guardedB corrected averaging paths."""
import math
from .cfr import TraversalLimit
from .prefix import PrefixMCCFR
from .population import profile_strategy

ALGORITHM='stratified-average-mccfr-v4'
AVERAGING='stratified-call-guarded-average-v4'


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='prefix-v3' or new.pop('traversal',None)!='stratified-v4':return False
    if old.pop('averaging_sampler',None)!='trajectory-call-mixture-v2' or new.pop('averaging_sampler',None)!=AVERAGING:return False
    samples=new.get('trajectory_samples')
    return type(samples) is int and samples>=4 and samples%4==0 and old==new


def prefix_weight(log_a,log_b,scale):
    if not math.isfinite(log_a) or math.isnan(log_b) or log_b==math.inf or not math.isfinite(scale) or scale<=0:raise ValueError('invalid stratified prefix/scale')
    high=max(log_a,log_b)
    log_mix=high+math.log(.75*math.exp(log_a-high)+.25*math.exp(log_b-high))
    try:weight=math.exp(math.log(scale)-log_mix)
    except OverflowError:raise TraversalLimit('nonfinite stratified averaging weight') from None
    if not math.isfinite(weight*weight):raise TraversalLimit('nonfinite stratified averaging weight')
    ratio=math.exp(log_a-log_mix)
    if ratio>4/3+1e-12:raise TraversalLimit('stratified weight bound violated')
    return weight,ratio


class StratifiedMCCFR(PrefixMCCFR):
    def __init__(self,seed=1,samples=8,depth=2):
        super().__init__(seed,samples,depth)
        if samples%4:raise ValueError('stratified sample count must be a multiple of4')
        self.traversal_stats['max_average_to_call_weight_ratio']=0.

    def traverse(self,factory,players,delta):
        for player in range(players):
            for sample in range(self.samples):
                self.outcome(factory(self.rng.randrange(2**63)),player,delta,scale=1/self.samples)
                self.average_path(factory(self.rng.randrange(2**63)),player,0 if sample<3*self.samples//4 else 1,1/self.samples)

    def trajectory_average(self,*args,**kwargs):
        raise ValueError('stratified averaging requires the fixed component schedule')

    def record_balance_ratio(self,ratio):
        if hasattr(self,'pending_traversal_stats'):
            key='max_average_to_call_weight_ratio';self.pending_traversal_stats[key]=max(self.pending_traversal_stats.get(key,0.),ratio)

    def average_path(self,state,owner,mode,scale=1.):
        if mode not in (0,1):raise ValueError('averaging component must be0/1')
        getattr(self,'pending_diagnostics',self.diagnostics)[f'averaging-component-v4:{"call" if mode==0 else "guarded"}']+=1
        self._average(state,owner,mode,scale,0.,0.)

    def _average(self,state,owner,mode,scale,log_a,log_b):
        weight,ratio=prefix_weight(log_a,log_b,scale);self.record_balance_ratio(ratio)
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_average_weight']=max(self.pending_traversal_stats['max_average_weight'],weight)
        self.visit()
        if state.terminal():return
        self.record_visit(state,'average',owner,sampling='sampled')
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.record_average_weight(state,weight)
            row=getattr(self,'pending_sums',self.sums).setdefault(key,[0.]*len(actions))
            for i,p in enumerate(strategy):
                row[i]+=weight*p
                if not math.isfinite(row[i]):raise TraversalLimit('nonfinite stratified average update')
            self._average(state.child(self.rng.choices(actions,strategy)[0]),owner,mode,scale,log_a,log_b)
        else:
            qa=[.5/len(actions)+(.5 if a=='call' else 0.) for a in actions] if 'call' in actions else [1/len(actions)]*len(actions)
            qb=profile_strategy(state.game.observe(),'guarded-v2')
            index=self.rng.choices(range(len(actions)),qa if mode==0 else qb)[0]
            self._average(state.child(actions[index]),owner,mode,scale,
                log_a+math.log(qa[index]),log_b+math.log(qb[index]) if qb[index]>0 else -math.inf)
