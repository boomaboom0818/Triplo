"""Test on-demand solving on unseen boards, range weights AND bet/pot ratio.

No blueprint policy or test payoff is used to choose actions. Each public game is
solved from scratch with a fixed budget, then evaluated by exhaustive response.
"""
import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from poker.river import audit,expected_value,rule_policy
from poker.river_solver import solve
from scripts.river_experiment import cases,generate,describe


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    train,heldout=cases(); known={g.signature for g in train+heldout}
    rng=Random(913)
    games=[generate(rng,([1,2,4],[4,2,1]),1.5) for _ in range(8)]
    assert all(g.signature not in known for g in games)
    def save(name,data):
        with (root/name).open('x') as f:json.dump(data,f,indent=2)
    sources=['poker/river_solver.py','poker/river.py','poker/sampled_cfr.py','scripts/river_unseen.py',
             'scripts/river_experiment.py','poker/rule_opponents.py','poker/fast_eval.py']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    save('manifest.json',dict(games=[describe(g) for g in games],seeds=[131,132,133],
        iterations_per_game=20000,deadline_per_game_seconds=30,sources=hashes,
        gate='Each seed mean exploitability <0.03BB and positive mean seat-balanced gain over tight rule baseline.',
        note='Evaluates online solving of new public games, not pretrained-policy extrapolation.'))
    for path in sources:
        target=root/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes())
    reports=[]
    baseline=rule_policy()
    for seed in (131,132,133):
        rows=[]
        for i,game in enumerate(games):
            policy,checkpoint=solve(game,20000,seed+i,30)
            save(f'policy-{seed}-case-{i}.json',checkpoint)
            learned=policy.distribution
            stats=audit(game,learned)
            gain=(expected_value(game,[learned,baseline])-expected_value(game,[baseline,learned]))/2
            rows.append(dict(learned=stats,rule=audit(game,baseline),advantage_bb_per_hand=gain,
                             seconds=checkpoint['seconds']))
        exp=sum(r['learned']['exploitability'] for r in rows)/8
        gain=sum(r['advantage_bb_per_hand'] for r in rows)/8
        reports.append(dict(seed=seed,mean_exploitability=exp,mean_advantage_bb_per_hand=gain,
                            passed=exp<.03 and gain>0,cases=rows))
        print(json.dumps({k:v for k,v in reports[-1].items() if k!='cases'}),flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError('experiment source changed')
    save('summary.json',dict(reports=reports,passed=all(r['passed'] for r in reports)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
