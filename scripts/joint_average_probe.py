"""Same-trace v3 control: exact regret/RNG/node equality while reusing averages."""
import argparse,hashlib,json,os,resource
from pathlib import Path
from datetime import datetime
from random import Random
from collections import Counter
from poker.prefix import PrefixMCCFR
from poker.joint_average import JointAverageMCCFR
from scripts import stratified_average_probe as driver

ROOT=Path('runs/bot/joint-average-probe-001')
CRITERIA=Path('docs/JOINT-AVERAGING-PILOT.md')
DEALS=list(range(10800000,10800008))
CONTROLS={}
VERIFIED=0


class Baseline(driver.ObservationMetrics,PrefixMCCFR):
    def __init__(self):super().__init__(depth=2)


class ObservedJoint(driver.ObservationMetrics,JointAverageMCCFR):
    def setup(self,saved):super().setup(saved);self.max_ratio=0.;self.sources={}
    def record_average_weight(self,state,weight):
        self.max_ratio=max(self.max_ratio,self.average_update_ratio)
        o=state.game.observe();label=f'source={self.average_update_source}:owner={o.player}:street={o.street}:active={sum(o.active)}'
        self.sources[label]=self.sources.get(label,0)+1
        super().record_average_weight(state,weight)


def batch(solver,state,owner,is_candidate):
    global VERIFIED
    start=repr(solver.rng.getstate());key=hashlib.sha256(start.encode()).hexdigest()
    selection=Random();selection.setstate(solver.rng.getstate());cfg=solver.saved['config'];archive=solver.saved['archive'];opponents={}
    if archive:
        for player in range(6):
            if selection.random()<cfg['population_probability']:opponents[player]=selection.choice(archive)
    if cfg.get('opponent_population')=='guarded-tables-v2' and selection.random()<.25:opponents={p:'guarded-v2' for p in range(6)}
    solver.opponents=opponents;delta={}
    for _ in range(8):
        solver.rng.randrange(2**63);solver.outcome(state,owner,delta,scale=1/8)
        solver.rng.randrange(2**63);solver.trajectory_average(state,owner,1/8)
    signature=hashlib.sha256(json.dumps(dict(regret=delta,rng=solver.rng.getstate(),nodes=solver.nodes),sort_keys=True).encode()).hexdigest()
    if is_candidate:
        if CONTROLS.get(key)!=signature:raise ValueError('joint averaging changed regret updates, RNG or traversal nodes')
        VERIFIED+=1
    else:CONTROLS[key]=signature


def summarize(solver,name):
    result=driver.summarize(solver)
    if name=='joint':result.update(source_groups=solver.sources,exact_regret_rng_nodes_batches_verified=VERIFIED,expected_batches=6*len(DEALS)*driver.REPLICATES)
    return result


def gates(original,variants,fixtures):
    result=dict(original);v=variants['joint']
    result['bounds_pass']=v['max_pointwise_weight_ratio']<=1+1e-12 and all(x['max_nodes']<=10000 for f in fixtures for x in f['variants'].values())
    result['exact_control_pass']=v['exact_regret_rng_nodes_batches_verified']==v['expected_batches']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,choices=(0,1),required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:raise ValueError('deadline needs timezone')
    driver.DEALS=DEALS
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    driver.main(a.candidate,deadline.timestamp(),implementation=ObservedJoint,baseline_implementation=Baseline,root=ROOT,criteria=CRITERIA,additional_sources=[Path(__file__),ROOT/'protocol.json'],candidate_label='joint',batch_fn=batch,summary_fn=summarize,gate_fn=gates,batch_max_seconds=2.,batch_node_limit=10000)
