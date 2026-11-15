"""Same fixed engineering limits, adding only provably unused owner-tail pruning."""
import argparse,os,resource
from pathlib import Path
from datetime import datetime
from poker.live_owner_average import LiveOwnerMCCFR
from scripts import stratified_average_probe as driver

ROOT=Path('runs/bot/live-owner-average-probe-001')
CRITERIA=Path('docs/LIVE-OWNER-AVERAGING-PILOT.md')


class ObservedLiveOwner(driver.ObservationMetrics,LiveOwnerMCCFR):
    def record_balance_ratio(self,ratio):self.max_ratio=max(self.max_ratio,ratio);super().record_balance_ratio(ratio)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,choices=(0,1),required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    driver.main(a.candidate,deadline.timestamp(),implementation=ObservedLiveOwner,root=ROOT,criteria=CRITERIA,additional_sources=[Path(__file__)])
