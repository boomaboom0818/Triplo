"""Bounded engineering of exact selective enumeration; one fixed cap, no clipping."""
import argparse,os,resource
from pathlib import Path
from datetime import datetime
from poker.adaptive_average import AdaptiveAverageMCCFR,WEIGHT_CAP
from scripts import stratified_average_probe as driver

ROOT=Path('runs/bot/adaptive-average-probe-001')
CRITERIA=Path('docs/ADAPTIVE-AVERAGING-PILOT.md')


class ObservedAdaptive(driver.ObservationMetrics,AdaptiveAverageMCCFR):
    def setup(self,saved):super().setup(saved);self.max_ratio=0.
    def record_average_weight(self,state,weight):
        self.max_ratio=max(self.max_ratio,weight/(WEIGHT_CAP/8));super().record_average_weight(state,weight)


def batch(solver,state,owner,is_candidate):
    for _ in range(8):solver.trajectory_average(state,owner,1/8)


def summarize(solver,name):
    result=driver.summarize(solver)
    if name=='adaptive':
        result['max_weight_to_cap_ratio']=result.pop('max_pointwise_weight_ratio')
        result['per_path_weight_cap']=WEIGHT_CAP/8
        result['maximum_average_weight']=max((v['max_weight'] for v in solver.groups.values()),default=0.)
    return result


def gates(original,variants,fixtures):
    result=dict(original);result['bounds_pass']=variants['adaptive']['max_weight_to_cap_ratio']<=1+1e-12 and all(v['max_nodes']<=10000 for f in fixtures for v in f['variants'].values())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,choices=(0,1),required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    driver.main(a.candidate,deadline.timestamp(),implementation=ObservedAdaptive,root=ROOT,criteria=CRITERIA,additional_sources=[Path(__file__)],candidate_label='adaptive',batch_fn=batch,summary_fn=summarize,gate_fn=gates,batch_max_seconds=2.,batch_node_limit=10000)
