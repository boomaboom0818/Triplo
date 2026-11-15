"""Frozen full-source-set comparison of v8 and unregistered cross-role v9."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from random import Random
import resource
import shutil
import signal
import time

from poker.cfr import Holdem
from poker.config import load_checkpoint
from poker.cross_role_average import CrossRoleAverageMCCFR
from poker.future_linear_average import FutureLinearAverageMCCFR
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory
from scripts.stratified_average_probe import ObservationMetrics, summarize
from scripts.vector_variance import VectorVariance, SCHEMA

ROOT = Path('runs/bot/cross-role-probe-001')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


class Observed:
    def record_average_weight(self,state,weight):
        self.max_ratio = max(self.max_ratio,self.average_update_ratio)
        super().record_average_weight(state,weight)


class Baseline(Observed,ObservationMetrics,FutureLinearAverageMCCFR):
    pass


class Candidate(Observed,ObservationMetrics,CrossRoleAverageMCCFR):
    pass


def selected(saved,seed,kind):
    if kind == 'guarded':
        return {i:'guarded-v2' for i in range(6)}
    rng=Random(seed+1);out={}
    for i in range(6):
        if saved['archive'] and rng.random()<saved['config']['population_probability']:
            out[i]=rng.choice(saved['archive'])
    return out


def full_batch(solver,state,end):
    delta={};roles=[]
    for owner in range(6):
        before=solver.nodes;started=time.monotonic()
        solver.node_limit=before+10000
        solver.deadline=started+min(2.,end-time.time())
        for _ in range(8):
            solver.rng.randrange(2**63);solver.outcome(state,owner,delta,scale=1/8)
            solver.rng.randrange(2**63);solver.trajectory_average(state,owner,1/8)
        seconds=time.monotonic()-started
        if seconds>2.:
            raise RuntimeError('full role8+8 batch exceeded2seconds')
        roles.append(dict(owner=owner,nodes=solver.nodes-before,seconds=seconds))
    signature=hashlib.sha256(json.dumps(dict(regrets=delta,rng=solver.rng.getstate(),nodes=solver.nodes),sort_keys=True).encode()).hexdigest()
    return signature,roles


def run(candidate,deadline):
    parsed=datetime.fromisoformat(deadline)
    if parsed.tzinfo is None or not time.time()<parsed.timestamp()<=time.time()+2*3600:
        raise ValueError('finite future deadline within two hours required')
    end=parsed.timestamp()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    def expired(signum,frame):
        raise TimeoutError('cross-role engineering deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,math.ceil(end-time.time())))
    with run_lock(ROOT/f'candidate-{candidate}'):
        out=ROOT/f'candidate-{candidate}.json';metrics=ROOT/f'candidate-{candidate}.metrics.json'
        if out.exists() or metrics.exists():
            raise ValueError('preserve completed/failed attempts; no unchanged retry')
        manifest=json.loads((ROOT/'manifest.json').read_text())
        assert manifest['deadline']==deadline
        def verify():
            for p,h in manifest['sources'].items():
                assert digest(p)==h,p
            for row in manifest['inputs']:
                assert digest(row['path'])==row['sha256']
        verify()
        status='failed';error=None;current=None;start=time.monotonic();minimum=available_memory()
        try:
            if minimum<8*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:
                raise RuntimeError('cross-role preflight reserves')
            saved=load_checkpoint(manifest['inputs'][candidate]['path'])
            assert saved['iterations']==(8906,9150)[candidate]
            assert saved['future_average_completed_iterations']==2000
            solvers=[Baseline(average_origin=saved['future_average_origin']),Candidate(average_origin=saved['future_average_origin'])]
            for solver in solvers:
                solver.regrets=saved['regrets'];solver.iterations=saved['iterations'];solver.setup(saved);solver.max_ratio=0.
            fixtures=[];totals=[dict(nodes=0,seconds=0.,variance_nodes=0.,variance_seconds=0.) for _ in solvers]
            for pop_index,kind in enumerate(manifest['populations']):
                for deal_index,deal in enumerate(manifest['deals']):
                    estimators=[VectorVariance(),VectorVariance()];pairs=[]
                    nodes=[[],[]];walls=[[],[]]
                    state=Holdem(deal,6,saved['abstraction'])
                    for replicate in range(manifest['replicates']):
                        seed=11900000+candidate*100000+pop_index*10000+deal_index*100+replicate
                        signatures=[];batches=[]
                        for variant,solver in enumerate(solvers):
                            current=dict(population=kind,deal=deal,replicate=replicate,variant=variant)
                            minimum=min(minimum,available_memory())
                            if minimum<2*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:
                                raise RuntimeError('cross-role host reserve breached')
                            solver.rng=Random(seed);solver.sums={};solver.nodes=0
                            solver.opponents=selected(saved,seed,kind)
                            signature,roles=full_batch(solver,state,end)
                            signatures.append(signature);batches.append(roles)
                            estimators[variant].add(solver.sums)
                            nodes[variant].append(solver.nodes);walls[variant].append(sum(r['seconds'] for r in roles))
                        assert signatures[0]==signatures[1], 'regret/RNG/node mismatch'
                        pairs.append(dict(replicate=replicate,seed=seed,signature=signatures[0],exact_regret_rng_nodes=True,roles=batches))
                        atomic_save(ROOT/f'candidate-{candidate}.progress.json',dict(completed_fixtures=len(fixtures),current=current,completed_replicates=replicate+1,deadline=deadline))
                    variants=[]
                    for j,estimator in enumerate(estimators):
                        variance=estimator.trace();mean_nodes=sum(nodes[j])/len(nodes[j]);mean_seconds=sum(walls[j])/len(walls[j])
                        variants.append(dict(variance_schema=SCHEMA,variance_trace=variance,mean_nodes=mean_nodes,mean_seconds=mean_seconds))
                        totals[j]['nodes']+=sum(nodes[j]);totals[j]['seconds']+=sum(walls[j])
                        totals[j]['variance_nodes']+=variance*mean_nodes;totals[j]['variance_seconds']+=variance*mean_seconds
                    fixtures.append(dict(population=kind,deal=deal,variants=variants,pairs=pairs))
            verify()
            summaries=[summarize(s) for s in solvers]
            ratio=lambda key:totals[1][key]/totals[0][key] if totals[0][key]>0 else None
            rate=[summaries[i]['active23_postflop_updates']/totals[i]['nodes'] for i in range(2)]
            gain=rate[1]/rate[0] if rate[0] else None
            new_cells={}
            for owner in range(6):
                for street in (1,2,3):
                    keys=set()
                    for active in (2,3):
                        keys.update(solvers[1].new_keys.get(f'owner={owner}:street={street}:active={active}',set()))
                    new_cells[f'owner={owner}:street={street}']=len(keys)
            peaks=[max((v['maximum'] for v in s.average_weight_moments.values()),default=0.) for s in solvers]
            gates=dict(exact_regret_rng_nodes=True,variance_work=ratio('variance_nodes') is not None and ratio('variance_nodes')<=1.5,
                variance_time=ratio('variance_seconds') is not None and ratio('variance_seconds')<=2,
                total_time=ratio('seconds')<=2,coverage_rate=gain is not None and gain>=3,
                all_role_street_new_keys=all(v>=20 for v in new_cells.values()),
                defensive_bound=solvers[1].max_ratio<=1+1e-12,
                finite_actual_weights=all(p>0 and math.isfinite(p*p) for p in peaks),
                bounded_batches=all(r['nodes']<=10000 and r['seconds']<=2 for f in fixtures for p in f['pairs'] for v in p['roles'] for r in v))
            atomic_save(out,dict(candidate=candidate,manifest_sha256=digest(ROOT/'manifest.json'),status='completed',read_only=True,
                fixtures=fixtures,totals=totals,summaries=summaries,new_keys_by_role_street=new_cells,peak_actual_weights=peaks,
                variance_work_ratio=ratio('variance_nodes'),variance_time_ratio=ratio('variance_seconds'),total_time_ratio=ratio('seconds'),coverage_rate_gain=gain,
                gates=gates,eligible_for_registration=all(gates.values()),note='Full equal-count sources, unchanged future-linear target. No training/payoff/strength evidence.'))
            status='completed'
        except BaseException as exc:
            error=repr(exc)
            raise
        finally:
            peaks={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith(('VmPeak:','VmHWM:'))}
            atomic_save(metrics,dict(status=status,error=error,last_case=current,pid=os.getpid(),deadline=deadline,
                address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],minimum_host_available_bytes=minimum,
                wall_seconds=time.monotonic()-start,**peaks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=(0,1),required=True);parser.add_argument('--deadline',required=True)
    args=parser.parse_args();run(args.candidate,args.deadline)
