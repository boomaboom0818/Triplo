"""New bounded production pilot; v2 failed gain gate is preserved, not revised."""
import argparse
from pathlib import Path
from poker.prefix import ALGORITHM,fork_allowed
from scripts import trajectory_pilot as driver


def configure():
    driver.ROOT=Path('runs/bot/prefix-pilot-001')
    driver.PARENTS=[Path(f'runs/bot/traversal-pilot-001/candidate-{i}-plus-1500.json') for i in range(2)]
    driver.EXPECTED_BASES=[4106,4350]
    driver.ADDITIONS=[50,200,500]
    driver.ALGORITHM=ALGORITHM;driver.TRAVERSAL='prefix-v3';driver.PREFIX_DEPTH=2
    driver.fork_allowed=fork_allowed
    driver.CRITERIA=Path('docs/PREFIX-PILOT.md')
    driver.SCREEN_SEED=8500000;driver.CONFIRM_SEED=8600000
    driver.ADDITIONAL_SOURCES=[Path(__file__)]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();configure();driver.main(a.deadline,a.resume)
