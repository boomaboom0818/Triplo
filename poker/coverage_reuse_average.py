"""Unregistered v10: cross-role reuse only at frozen uncovered average rows.

The prefix-measurable partition chooses two complete, unbiased estimators. It
never truncates weights or removes a target coordinate. Covered rows retain the
original v8 owner-pair estimator; uncovered rows use all original role sources.
"""
from dataclasses import dataclass
from .cross_role_average import CrossRoleAverageMCCFR, Prefix
from .future_linear_average import FutureLinearAverageMCCFR
from .joint_average import log_probability

ALGORITHM = 'coverage-only-cross-role-average-mccfr-v10'


@dataclass(frozen=True)
class CoveragePrefix(Prefix):
    external: tuple

    @classmethod
    def root(cls,players):
        p=Prefix.root(players)
        return cls(p.own,p.average,p.regret,p.depths,(0.,)*players)

    def child(self,actor,current,call,frozen,sample,depth):
        p=super().child(actor,current,call,frozen,sample,depth)
        log_call=log_probability(call)
        return CoveragePrefix(p.own,p.average,p.regret,p.depths,
            tuple(v if k==actor else v+log_call for k,v in enumerate(self.external)))


class CoverageReuseMCCFR(CrossRoleAverageMCCFR):
    def __init__(self,*args,coverage_reference=None,**kwargs):
        super().__init__(*args,**kwargs)
        # None means the actual pre-step table. Explicit empty dict is a valid
        # frozen reference for direct read-only/exhaustive source experiments.
        self.coverage_reference=coverage_reference

    def outcome(self,state,updater,delta,inverse_own_sample=1.,scale=1.):
        return self._regret(state,updater,delta,inverse_own_sample,scale,CoveragePrefix.root(self.role_count))

    def trajectory_average(self,state,owner,weight=1.):
        self._cross_average(state,owner,weight,CoveragePrefix.root(self.role_count))

    def contribute(self,state,actor,key,actions,strategy,prefix,scale,source,source_owner):
        reference=self.sums if self.coverage_reference is None else self.coverage_reference
        covered=any(v>0 for v in reference.get(key,()))
        diagnostics=getattr(self,'pending_diagnostics',self.diagnostics)
        if covered:
            if actor!=source_owner:
                diagnostics['coverage-reuse-v10:covered-foreign-omitted']+=1
                return
            diagnostics['coverage-reuse-v10:covered-own-pair']+=1
            # Preserve the original separate own/external log accumulation,
            # rather than subtracting two large logs to recover external reach.
            return FutureLinearAverageMCCFR.average_contribution(self,state,key,actions,strategy,
                prefix.own[actor],prefix.external[actor],prefix.regret[actor],scale,source)
        diagnostics['coverage-reuse-v10:uncovered-cross-role']+=1
        return super().contribute(state,actor,key,actions,strategy,prefix,scale,source,source_owner)
