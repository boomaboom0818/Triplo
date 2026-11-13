import math
from random import Random
from .model import Opponents

def key(o):
    ranks='23456789TJQKA'
    a,b=(ranks.index(c[0]) for c in o.hole)
    return str((o.street,o.position, min(a,b)//3,max(a,b)//3,a==b,
                o.hole[0][1]==o.hole[1][1],sum(o.active),
                min(4,o.call*4//max(1,o.pot)),len(o.legal),
                tuple(c[0] for c in o.board)))

class Policy:
    def __init__(self, seed=0, table=None, style='blueprint', mode='bot'):
        self.rng=Random(seed); self.table={} if table is None else table; self.style=style
        self.mode=mode; self.model=Opponents()
    def probabilities(self,o):
        logits=self.table.get(key(o),[0.0]*len(o.legal))
        if len(logits) != len(o.legal):
            raise ValueError('checkpoint action count mismatch')
        weights=[math.exp(max(-20,min(20,x))) for x in logits]
        return [w/sum(weights) for w in weights]
    def act(self,o):
        if self.style=='calling': return 'call'
        if self.style=='tight': return 'fold' if o.call and 'fold' in o.legal else 'call'
        if self.style=='aggressive': return o.legal[-1] if self.rng.random()<.65 else 'call'
        return self.rng.choices(o.legal,self.probabilities(o))[0]
