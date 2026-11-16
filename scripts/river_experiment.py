"""Train explicit-range river games, freeze policy, then evaluate untouched boards.

Held-out games have the same public payoff/chance structure as training games.
This tests exact structure transfer to new cards, not arbitrary-range generalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from poker.river import RiverGame, RiverState, RiverPolicy, audit, expected_value, rule_policy
from poker.sampled_cfr import SampledCFR


def generate(rng, weights, bet):
    deck=list(range(52)); rng.shuffle(deck)
    return RiverGame(deck[:5],([deck[5:7],deck[7:9],deck[9:11]],
                              [deck[11:13],deck[13:15],deck[15:17]]),weights,2.,bet)


def cases():
    rng=Random(311)
    train=[]
    # Fixed before seeing any held-out payoff. Range families and bet sizes vary.
    for weights in (([1,1,1],[1,1,1]),([2,3,5],[5,3,2])):
        for bet in (1.,2.):
            for _ in range(2): train.append(generate(rng,weights,bet))
    heldout=[]
    for target in train:
        found=None
        weights=([1,1,1],[1,1,1]) if len(set(target.joint.values()))==1 else ([2,3,5],[5,3,2])
        for _ in range(30000):
            candidate=generate(rng,weights,target.bet)
            if candidate.signature==target.signature and candidate.board not in [g.board for g in train+heldout]:
                found=candidate; break
        if found is None: raise RuntimeError('cannot construct held-out isomorphic board')
        heldout.append(found)
    return train,heldout


def describe(game):
    from poker.fast_eval import NAMES
    return dict(board=[NAMES[c] for c in game.board],ranges=[[[NAMES[c] for c in h] for h in r] for r in game.ranges],
                joint=[dict(pair=list(p),probability=w) for p,w in game.joint.items()],
                pot=game.pot,bet=game.bet,signature=game.signature)


def run(root, iterations=40000):
    if not 1<=iterations<=200000: raise ValueError('bounded training iterations required')
    root.mkdir(parents=True,exist_ok=False)
    train,heldout=cases()
    def save(name,data):
        with (root/name).open('x') as f: json.dump(data,f,indent=2)
    sources=['poker/river.py','poker/sampled_cfr.py','poker/rule_opponents.py','poker/fast_eval.py','scripts/river_experiment.py']
    manifest=dict(training=[describe(g) for g in train],heldout=[describe(g) for g in heldout],
        iterations=iterations,seeds=[121,122,123],
        sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        gate='Each seed: held-out mean exploitability below 0.03 BB and positive exact seat-balanced advantage over tight rule baseline.')
    save('manifest.json',manifest)
    for source in sources:
        p=root/'source'/source; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(Path(source).read_bytes())
    reports=[]
    for seed in manifest['seeds']:
        solver=SampledCFR(2,seed)
        for _ in range(iterations):
            game=solver.rng.choice(train); solver.iterate(game.sample(solver.rng))
        # Freeze the learned policy before evaluating untouched cards.
        checkpoint=dict(schema=1,algorithm='bounded-river-es-v1',iterations=solver.iterations,seed=seed,
                        sums=solver.sums,regrets=solver.regrets,visits=solver.visits,rng=solver.rng.getstate())
        save(f'policy-{seed}.json',checkpoint)
        learned=RiverPolicy(checkpoint).distribution
        rows=[]
        baseline=rule_policy()
        for game in heldout:
            stats=audit(game,learned)
            # Equal seats: learned as P0 and P1, exact over all chance outcomes.
            advantage=(expected_value(game,[learned,baseline])-expected_value(game,[baseline,learned]))/2
            rules=audit(game,baseline)
            rows.append(dict(learned=stats,rule=rules,advantage_bb_per_hand=advantage))
        mean_exploitability=sum(r['learned']['exploitability'] for r in rows)/len(rows)
        mean_advantage=sum(r['advantage_bb_per_hand'] for r in rows)/len(rows)
        reports.append(dict(seed=seed,mean_exploitability=mean_exploitability,mean_advantage_bb_per_hand=mean_advantage,
            passed=mean_exploitability<.03 and mean_advantage>0,cases=rows))
        print(json.dumps({k:v for k,v in reports[-1].items() if k!='cases'}),flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in manifest['sources'].items()):
        raise RuntimeError('experiment source changed')
    save('summary.json',dict(reports=reports,passed=all(r['passed'] for r in reports),
        note='Exact expectations for these bounded games; variation across training seeds is not sampling error in evaluation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--iterations',type=int,default=40000)
    a=p.parse_args(); run(a.root,a.iterations)
