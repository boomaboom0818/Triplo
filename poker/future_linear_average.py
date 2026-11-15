"""Unregistered v8 prototype: preserve history, linearly weight future averages."""
import math
from .joint_average import JointAverageMCCFR
from .cfr import TraversalLimit

ALGORITHM='future-linear-average-mccfr-v8'
AVERAGING='future-linear-joint-average-v8'


class FutureLinearAverageMCCFR(JointAverageMCCFR):
    def __init__(self,seed=1,samples=8,depth=2,average_origin=0):
        if type(average_origin) is not int or average_origin<0:
            raise ValueError('invalid future averaging origin')
        super().__init__(seed,samples,depth)
        self.average_origin=average_origin

    @property
    def average_multiplier(self):
        k=self.iterations-self.average_origin+1
        if type(self.iterations) is not int or k<1:
            raise ValueError('future averaging clock precedes fork')
        return k

    def step(self,*args,**kwargs):
        self.average_multiplier  # Validate before any transactional work.
        return super().step(*args,**kwargs)

    def average_contribution(self,state,key,actions,strategy,log_own,log_qa,log_pb,scale,source):
        try:weighted=scale*self.average_multiplier
        except OverflowError:raise TraversalLimit('nonfinite future averaging scale') from None
        if weighted<=0 or not math.isfinite(weighted*weighted):
            raise TraversalLimit('nonfinite future averaging scale')
        return super().average_contribution(state,key,actions,strategy,log_own,log_qa,log_pb,weighted,source)
