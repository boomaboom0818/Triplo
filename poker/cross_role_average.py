"""Unregistered prototype: balance all existing role-specific traversal sources.

Every A_k path and B_k regret tree has eight samples in the original schedule.
At a literal prefix of actor j, sum their inclusion probabilities, then weight
by j's CURRENT own reach. No new tree, random draw, or regret update is added.
"""
import math
from dataclasses import dataclass

from .cfr import TraversalLimit
from .future_linear_average import FutureLinearAverageMCCFR
from .joint_average import call_proposal, log_probability

ALGORITHM = 'cross-role-joint-average-mccfr-v9'


@dataclass(frozen=True)
class Prefix:
    own: tuple
    average: tuple
    regret: tuple
    depths: tuple

    @classmethod
    def root(cls, players):
        return cls((0.,)*players, (0.,)*players, (0.,)*players, (0,)*players)

    def child(self, actor, current, call, frozen, sample, depth):
        lc, la, lf = map(log_probability, (current, call, frozen))
        lb = 0. if self.depths[actor] < depth else log_probability(sample)
        return Prefix(
            tuple(x+lc if k == actor else x for k,x in enumerate(self.own)),
            tuple(x+(lc if k == actor else la) for k,x in enumerate(self.average)),
            tuple(x+(lb if k == actor else lf) for k,x in enumerate(self.regret)),
            tuple(x+1 if k == actor else x for k,x in enumerate(self.depths)))


def cross_weight(prefix, actor, scale):
    own = prefix.own[actor]
    if own == -math.inf:
        return 0., 0.
    values = prefix.average + prefix.regret
    high = max(values)
    if not math.isfinite(high) or not math.isfinite(own) or scale <= 0 or not math.isfinite(scale):
        raise TraversalLimit('invalid cross-role inclusion probability')
    log_total = high + math.log(sum(math.exp(v-high) for v in values))
    try:
        weight = math.exp(math.log(scale)+own-log_total)
    except OverflowError:
        raise TraversalLimit('nonfinite cross-role average weight') from None
    if weight == 0:
        raise TraversalLimit('underflowed positive cross-role average weight')
    ratio = math.exp(prefix.average[actor]-log_total)
    if not math.isfinite(weight*weight) or ratio > 1+1e-12:
        raise TraversalLimit('invalid cross-role defensive bound')
    return weight, ratio


class CrossRoleAverageMCCFR(FutureLinearAverageMCCFR):
    def __init__(self, seed=1, samples=8, depth=2, average_origin=0, players=6):
        super().__init__(seed,samples,depth,average_origin)
        if type(players) is not int or not 2 <= players <= 6:
            raise ValueError('invalid source role count')
        self.role_count = players

    def traverse(self, factory, players, delta):
        if players != self.role_count:
            raise ValueError('cross-role source count differs from actual traversal schedule')
        super().traverse(factory,players,delta)

    def contribute(self, state, actor, key, actions, strategy, prefix, scale, source, source_owner):
        weight, ratio = cross_weight(prefix,actor,scale*self.average_multiplier)
        diagnostics = getattr(self,'pending_diagnostics',self.diagnostics)
        if not weight:
            diagnostics['cross-role-v9:zero-own-target'] += 1
            return
        self.average_update_ratio = ratio
        self.average_update_source = source
        self.record_average_weight(state,weight)
        if hasattr(self,'pending_traversal_stats'):
            stats = self.pending_traversal_stats
            stats['max_average_weight'] = max(stats['max_average_weight'],weight)
            name = 'max_joint_average_to_original_weight_ratio'
            stats[name] = max(stats.get(name,0.),ratio)
        row = getattr(self,'pending_sums',self.sums).setdefault(key,[0.]*len(actions))
        for j,p in enumerate(strategy):
            row[j] += weight*p
            if not math.isfinite(row[j]):
                raise TraversalLimit('nonfinite cross-role average update')
        if hasattr(state,'game'):
            o = state.game.observe()
            diagnostics[f'cross-role-v9:source={source}:owner={actor}:street={o.street}:active={sum(o.active)}'] += 1
            diagnostics['cross-role-v9:reused-other-role'] += actor != source_owner

    def outcome(self,state,updater,delta,inverse_own_sample=1.,scale=1.):
        return self._regret(state,updater,delta,inverse_own_sample,scale,Prefix.root(self.role_count))

    def _regret(self,state,updater,delta,inverse,scale,prefix):
        if not math.isfinite(inverse):
            raise TraversalLimit('nonfinite importance weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_inverse_own_sample'] = max(self.pending_traversal_stats['max_inverse_own_sample'],inverse)
        self.visit()
        if state.terminal():
            return state.returns()[updater]
        actor,key,actions = state.info()
        strategy = self.strategy(key,actions)
        self.record_visit(state,'regret',updater,sampling='enumerated' if actor == updater and prefix.depths[updater] < self.depth else 'sampled')
        self.contribute(state,actor,key,actions,strategy,prefix,scale,'regret',updater)
        frozen = self.opponent_strategy(state,actor,key,actions,strategy,actor != updater)
        qa = call_proposal(actions)
        qb = [.5*p+.5/len(actions) for p in strategy]
        def next_prefix(j):
            return prefix.child(actor,strategy[j],qa[j],frozen[j],qb[j],self.depth)
        if actor != updater:
            action = self.rng.choices(actions,frozen)[0]
            j = actions.index(action)
            return self._regret(state.child(action),updater,delta,inverse,scale,next_prefix(j))
        self.record_weight(state,'regret_prefix',inverse*scale)
        if prefix.depths[updater] < self.depth:
            values = [self._regret(state.child(a),updater,delta,inverse,scale,next_prefix(j)) for j,a in enumerate(actions)]
            value = sum(p*v for p,v in zip(strategy,values))
        else:
            index = self.rng.choices(range(len(actions)),qb)[0]
            action_value = self._regret(state.child(actions[index]),updater,delta,inverse/qb[index],scale,next_prefix(index))/qb[index]
            values = [action_value if j == index else 0. for j in range(len(actions))]
            value = strategy[index]*action_value
        row = delta.setdefault(key,[0.]*len(actions))
        for j,v in enumerate(values):
            row[j] += scale*inverse*(v-value)
            if not math.isfinite(row[j]):
                raise TraversalLimit('nonfinite regret update')
        return value

    def trajectory_average(self,state,owner,weight=1.):
        self._cross_average(state,owner,weight,Prefix.root(self.role_count))

    def _cross_average(self,state,owner,scale,prefix):
        self.visit()
        if state.terminal():
            return
        self.record_visit(state,'average',owner,sampling='sampled')
        actor,key,actions = state.info()
        strategy = self.strategy(key,actions)
        self.contribute(state,actor,key,actions,strategy,prefix,scale,'average',owner)
        frozen = self.opponent_strategy(state,actor,key,actions,strategy)
        qa = call_proposal(actions)
        if actor == owner:
            action = self.rng.choices(actions,strategy)[0]
            j = actions.index(action)
        else:
            j = self.rng.choices(range(len(actions)),qa)[0]
        child = prefix.child(actor,strategy[j],qa[j],frozen[j],.5*strategy[j]+.5/len(actions),self.depth)
        self._cross_average(state.child(actions[j]),owner,scale,child)
