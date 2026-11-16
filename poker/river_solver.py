"""Solve a previously unseen bounded public river game from its explicit ranges."""
import json
import time
from .river import RiverPolicy
from .sampled_cfr import SampledCFR


def solve(game, iterations=20000, seed=1, max_seconds=30):
    if type(iterations) is not int or not 1<=iterations<=100000 or not 0<max_seconds<=120:
        raise ValueError('bounded positive solve budget required')
    start=time.monotonic()
    solver=SampledCFR(2,seed)
    for _ in range(iterations):
        if time.monotonic()-start>max_seconds:
            raise TimeoutError('bounded river solve deadline')
        solver.iterate(game.sample(solver.rng))
    sums=dict(solver.sums)
    for p in (0,1):
        for i in range(len(game.ranges[p])):
            for h in (('', 'cb') if p==0 else ('c','b')):
                key=f'[{json.dumps(game.signature)},{p},{i},{json.dumps(h)}]'
                sums.setdefault(key,solver.strategy(key,2))
    checkpoint=dict(schema=1,algorithm='bounded-river-es-v1',iterations=solver.iterations,
                    seed=seed,sums=sums,regrets=solver.regrets,visits=solver.visits,
                    rng=solver.rng.getstate(),signature=game.signature,seconds=time.monotonic()-start)
    return RiverPolicy(checkpoint),checkpoint
