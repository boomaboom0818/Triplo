"""Heads-up river subgames with public weighted ranges and exact showdown values.

One bet (half-pot or pot, configured per game), no raises. A canonical key retains
public joint chance weights and showdown outcomes, so only strategically identical
river games share a policy. Utilities are net of each player's half of the initial
pot; ties split. This module is a bounded laboratory, not a full-game live solver.
"""
from itertools import product
import json
import math

from .fast_eval import CARD_ID, NAMES, evaluate
from .game import Observation
from .rule_opponents import RulePolicy


class RiverGame:
    def __init__(self, board, ranges, weights=None, pot=2., bet=1.):
        self.board = tuple(board)
        self.ranges = tuple(tuple(tuple(sorted(h)) for h in r) for r in ranges)
        if (len(self.board) != 5 or len(set(self.board)) != 5 or len(self.ranges) != 2
                or any(not 1 <= len(r) <= 3 for r in self.ranges)
                or not all(math.isfinite(x) and x > 0 for x in (pot, bet))):
            raise ValueError('five-card board, two ranges of up to three hands, positive pot/bet required')
        self.pot, self.bet = pot, bet
        weights = weights or tuple([1.] * len(r) for r in self.ranges)
        if len(weights) != 2 or any(len(w) != len(r) for w,r in zip(weights,self.ranges)):
            raise ValueError('one weight per holding required')
        if any(not math.isfinite(x) or x <= 0 for w in weights for x in w):
            raise ValueError('positive finite range weights required')
        # Public sorting makes representation invariant to input range order.
        ordered=[]
        for hands, mass in zip(self.ranges, weights):
            if len(set(hands)) != len(hands): raise ValueError('duplicate holding')
            rows=[]
            for hand,w in zip(hands,mass):
                if len(hand)!=2 or len(set(hand+self.board))!=7 or any(c not in range(52) for c in hand+self.board):
                    raise ValueError('invalid or overlapping board/holding cards')
                rows.append((evaluate(hand+self.board),hand,w))
            ordered.append(sorted(rows))
        self.ranges=tuple(tuple(x[1] for x in rows) for rows in ordered)
        scores=tuple([x[0] for x in rows] for rows in ordered)
        mass=tuple([x[2] for x in rows] for rows in ordered)
        self.joint={}
        outcomes=[]
        for i,j in product(range(len(self.ranges[0])),range(len(self.ranges[1]))):
            legal=not set(self.ranges[0][i]) & set(self.ranges[1][j])
            if legal: self.joint[i,j]=mass[0][i]*mass[1][j]
            outcomes.append(None if not legal else (scores[0][i]>scores[1][j])-(scores[0][i]<scores[1][j]))
        total=sum(self.joint.values())
        if not total: raise ValueError('ranges have no compatible joint deal')
        self.joint={k:v/total for k,v in self.joint.items()}
        self.outcomes=dict(zip(product(range(len(self.ranges[0])),range(len(self.ranges[1]))),outcomes))
        # Preserve exact binary chance probabilities; do not merge nearby weights.
        self.signature=json.dumps([len(self.ranges[0]),len(self.ranges[1]),bet/pot,outcomes,
            [self.joint.get((i,j),0.).hex() for i,j in self.outcomes]],separators=(',',':'))

    def sample(self, rng):
        pair=rng.choices(list(self.joint),list(self.joint.values()))[0]
        return RiverState(self,pair)


class RiverState:
    def __init__(self, game, pair, history=''):
        self.game, self.pair, self.history=game,pair,history
        if pair not in game.joint: raise ValueError('impossible private deal')

    @property
    def terminal(self): return self.history in ('cc','bc','bf','cbc','cbf')
    @property
    def actor(self): return len(self.history)%2
    @property
    def actions(self): return ('f','c') if self.history.endswith('b') else ('c','b')
    def key(self):
        # Opponent's actual holding never enters the key.
        return json.dumps([self.game.signature,self.actor,self.pair[self.actor],self.history],separators=(',',':'))
    def child(self, action):
        if self.terminal or action not in self.actions: raise ValueError('illegal river action')
        return RiverState(self.game,self.pair,self.history+action)
    def utility(self, player):
        if not self.terminal: raise ValueError('nonterminal utility')
        if self.history in ('bf','cbf'):
            winner=0 if self.history=='bf' else 1
            value=self.game.pot/2*(1 if winner==0 else -1)
        else:
            sign=self.game.outcomes[self.pair]
            value=sign*(self.game.pot/2+(0 if self.history=='cc' else self.game.bet))
        return value if player==0 else -value


def expected_value(game, policies):
    def walk(state):
        if state.terminal: return state.utility(0)
        probs=policies[state.actor](state)
        return sum(p*walk(state.child(a)) for p,a in zip(probs,state.actions))
    return sum(w*walk(RiverState(game,pair)) for pair,w in game.joint.items())


def best_response(game, policy, player):
    histories=('', 'cb') if player==0 else ('c','b')
    coordinates=[(hand,h) for hand in range(len(game.ranges[player])) for h in histories]
    best=-math.inf; chosen=None
    # Exhaustive pure policies choose per own hand/history, never per hidden deal.
    for choices in product((0,1),repeat=len(coordinates)):
        table=dict(zip(coordinates,choices))
        def response(state):
            choice=table[state.pair[player],state.history]
            return [float(j==choice) for j in range(2)]
        policies=[policy,policy]; policies[player]=response
        value=expected_value(game,policies)*(1 if player==0 else -1)
        if value>best: best,chosen=value,dict(table)
    return best,chosen


def audit(game, policy):
    value=expected_value(game,[policy,policy])
    br0,_=best_response(game,policy,0); br1,_=best_response(game,policy,1)
    return dict(value_player0=value,nash_conv=br0+br1,exploitability=(br0+br1)/2)


def rule_policy(style='tight-v1'):
    rule=RulePolicy(style)
    def policy(state):
        g=state.game; facing=state.history.endswith('b')
        # Units are half-BB, matching the full-game rule player.
        call=round(g.bet*2) if facing else 0
        pot=round((g.pot+(g.bet if facing else 0))*2)
        legal=('call','fold') if facing else ('call',round(g.bet*2))
        o=Observation(state.actor,state.actor,
            tuple(NAMES[c] for c in g.ranges[state.actor][state.pair[state.actor]]),
            tuple(NAMES[c] for c in g.board),(200,200),(True,True),pot,call,3,legal,())
        action=rule.act(o)
        index=(0 if action=='fold' else 1) if facing else (0 if action=='call' else 1)
        return [float(j==index) for j in range(2)]
    return policy


class RiverPolicy:
    """Saved bounded-river policy. Unsupported public games fail explicitly."""
    def __init__(self, checkpoint):
        if checkpoint.get('schema')!=1 or checkpoint.get('algorithm')!='bounded-river-es-v1':
            raise ValueError('incompatible bounded river checkpoint')
        # Copy rows so continued training cannot alter a frozen evaluation policy.
        self.sums={k:tuple(v) for k,v in checkpoint['sums'].items()}
        if any(len(r)!=2 or any(not math.isfinite(x) or x<0 for x in r) or sum(r)<=0 for r in self.sums.values()):
            raise ValueError('invalid river probability mass')

    @classmethod
    def load(cls, path):
        from pathlib import Path
        return cls(json.loads(Path(path).read_text()))

    def probabilities(self, game, player, hole, history=''):
        if history not in ('','c','b','cb') or len(history)%2!=player:
            raise ValueError('invalid acting player/history')
        try: index=game.ranges[player].index(tuple(sorted(hole)))
        except ValueError: raise ValueError('own holding outside public range') from None
        key=json.dumps([game.signature,player,index,history],separators=(',',':'))
        if key not in self.sums: raise ValueError('public river game has no trained policy support')
        row=self.sums[key]; total=sum(row)
        return tuple(x/total for x in row)

    def distribution(self,state):
        return self.probabilities(state.game,state.actor,
            state.game.ranges[state.actor][state.pair[state.actor]],state.history)
