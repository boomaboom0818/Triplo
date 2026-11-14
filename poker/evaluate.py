import argparse,json,time,statistics,hashlib
from pathlib import Path
from random import Random
from .game import play
from .policy import Policy
from .cfr import CFRPolicy,information_key
from .config import load_checkpoint, load_config

def coverage_counts():
    return dict(decisions=0, missing_rows=0, zero_weight_rows=0, positive_weight_rows=0)


def record_coverage(counts, row):
    counts['decisions'] += 1
    if row is None:
        counts['missing_rows'] += 1
    elif not any(weight > 0 for weight in row):
        counts['zero_weight_rows'] += 1
    else:
        counts['positive_weight_rows'] += 1


def evaluate(path,blocks,seed,copies,config_path="configs/bot.json",opponent_checkpoints=(),uniform_control=False):
    config=load_config(config_path)
    if config["status"] != "active":
        raise ValueError("human-opponent model is deferred; use configs/bot.json")
    if not 1 <= copies <= 6: raise ValueError("copies must be between 1 and 6")
    if blocks<2: raise ValueError('at least two independent blocks required')
    checkpoint=load_checkpoint(path, model_id=config["model_id"]); table=checkpoint.get('table',{}); reports={}
    opponents={}
    candidate_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for opponent_path in opponent_checkpoints:
        digest=hashlib.sha256(Path(opponent_path).read_bytes()).hexdigest()
        if digest==candidate_hash:raise ValueError('held-out opponent is the candidate checkpoint')
        opponent=load_checkpoint(opponent_path,model_id='bot-opponent-v1')
        if opponent['algorithm']!='external-sampling-mccfr-v1':raise ValueError('opponent must be CFR checkpoint')
        if opponent['sums']==checkpoint.get('sums') or opponent['sums'] in checkpoint.get('archive',[]):
            raise ValueError('held-out opponent overlaps candidate training population')
        opponents['checkpoint:'+digest]=opponent
    if uniform_control:
        checkpoint=dict(checkpoint,sums={});table={}
    for population in ['calling','tight','aggressive','blueprint','mixed']+list(opponents):
        start=time.perf_counter(); values=[]; latency=[];covered=0;decisions=0
        by_active={str(i):coverage_counts() for i in range(2,7)}
        coverage=coverage_counts(); by_street={str(i):coverage_counts() for i in range(4)}
        for block in range(blocks):
            results=[]
            # A block reuses one deal across all six candidate seat rotations.
            for rotation in range(6):
                candidates={(rotation+i)%6 for i in range(copies)}
                policies=[Policy(seed+block*100+rotation*6+i,table if i in candidates else {},
                                 'blueprint' if i in candidates else (['calling','tight','aggressive'][i%3] if population=='mixed' else population), mode=config['mode']) for i in range(6)]
                if checkpoint['algorithm']=='external-sampling-mccfr-v1':
                    for i in candidates:
                        policies[i]=CFRPolicy(checkpoint['sums'],seed+block*100+rotation*6+i,checkpoint['abstraction'])
                if population in opponents:
                    opponent=opponents[population]
                    for i in range(6):
                        if i not in candidates:
                            policies[i]=CFRPolicy(opponent['sums'],seed+block*100+rotation*6+i,opponent['abstraction'])
                payoff,traces,times=play(policies,seed+block,button=0)
                results.extend(payoff[i] for i in candidates); latency.extend(times)
                if checkpoint['algorithm']=='external-sampling-mccfr-v1':
                    for i in candidates:
                        for observation,_ in traces[i]:
                            decisions+=1
                            row=checkpoint['sums'].get(information_key(observation,checkpoint['abstraction']))
                            covered+=row is not None
                            record_coverage(coverage,row)
                            record_coverage(by_street[str(observation.street)],row)
                            if observation.street:record_coverage(by_active[str(sum(observation.active))],row)
            values.append(statistics.mean(results)*100)
        rng=Random(seed); boot=sorted(statistics.mean(rng.choices(values,k=blocks)) for _ in range(2000))
        reports[population]=dict(bb_per_100=statistics.mean(values),ci95_cluster_bootstrap=[boot[50],boot[1949]],
            independent_deal_blocks=blocks,table_hands=6*blocks,candidate_hands=6*blocks*copies,
            candidate_decisions=decisions,covered_candidate_decisions=covered,
            policy_coverage=covered/decisions if decisions else None,
            coverage_counts=coverage, coverage_by_street=by_street,
            postflop_coverage_by_active=by_active,deal_block_bb_per_100=values,
            uniform_fallback_fraction=(coverage['missing_rows']+coverage['zero_weight_rows'])/decisions if decisions else None,
            latency_scope="all_table_decisions",
            decision_p95_ms=1000*sorted(latency)[int(.95*(len(latency)-1))],
            decision_mean_ms=1000*statistics.mean(latency),wall_seconds=time.perf_counter()-start)
    return dict(uniform_control=uniform_control,checkpoint_sha256=candidate_hash,opponent_checkpoint_hashes=list(opponents),algorithm=checkpoint['algorithm'],config=config,seed=seed,copies=copies,populations=reports,warning='Smoke benchmark only; no strong held-out opponent integrated.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/bot/seed1.json')
    p.add_argument('--blocks',type=int,default=20);p.add_argument('--seed',type=int,default=100000)
    p.add_argument('--copies',type=int,choices=range(1,7),default=1)
    p.add_argument('--config',default='configs/bot.json')
    p.add_argument('--opponent-checkpoints',nargs='*',default=[])
    p.add_argument('--uniform-control',action='store_true')
    a=p.parse_args(); print(json.dumps(evaluate(a.checkpoint,a.blocks,a.seed,a.copies,a.config,a.opponent_checkpoints,a.uniform_control),indent=2))
