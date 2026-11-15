"""V5 removes averaging tails after the owner folds or has no chips to act."""
from .stratified_average import StratifiedMCCFR

ALGORITHM='live-owner-stratified-average-mccfr-v5'
AVERAGING='live-owner-stratified-average-v5'


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='prefix-v3' or new.pop('traversal',None)!='stratified-live-v5':return False
    if old.pop('averaging_sampler',None)!='trajectory-call-mixture-v2' or new.pop('averaging_sampler',None)!=AVERAGING:return False
    samples=new.get('trajectory_samples')
    return type(samples) is int and samples>=4 and samples%4==0 and old==new


class LiveOwnerMCCFR(StratifiedMCCFR):
    def average_path(self,state,owner,mode,scale=1.):
        if mode not in (0,1):raise ValueError('averaging component must be0/1')
        getattr(self,'pending_diagnostics',self.diagnostics)[f'averaging-component-v5:{"call" if mode==0 else "guarded"}']+=1
        self._average(state,owner,mode,scale,0.,0.)

    def _average(self,state,owner,mode,scale,log_a,log_b):
        game=getattr(state,'game',None)
        if not state.terminal() and game is not None and hasattr(game,'order'):
            position=game.order.index(owner)
            if not game.state.statuses[position] or game.state.stacks[position]==0:
                # No future owner decision exists, hence this tail has zero
                # contribution to every average coordinate. Regret is untouched.
                self.visit()
                return
        super()._average(state,owner,mode,scale,log_a,log_b)
