"""Prospective small guarded-table curriculum; preserve all prior failed gates."""
import argparse
from pathlib import Path
from poker.prefix import ALGORITHM
from poker.population import population_fork_allowed
from scripts import trajectory_pilot as driver


def configure():
    driver.ROOT=Path('runs/bot/coverage-curriculum-001')
    driver.PARENTS=[Path(f'runs/bot/prefix-pilot-001/candidate-{i}-plus-500.json') for i in range(2)]
    driver.EXPECTED_BASES=[4606,4850];driver.ADDITIONS=[50,150,300]
    driver.ALGORITHM=ALGORITHM;driver.TRAVERSAL='prefix-v3';driver.PREFIX_DEPTH=2
    driver.POPULATION='guarded-tables-v2';driver.FORK_FLAG='--fork-population'
    driver.REQUIRE_TIGHT_IMPROVEMENT=True
    driver.fork_allowed=population_fork_allowed
    driver.CRITERIA=Path('docs/COVERAGE-CURRICULUM.md')
    driver.SCREEN_SEED=9500000;driver.CONFIRM_SEED=9600000
    driver.ADDITIONAL_SOURCES=[Path(__file__)]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();configure();driver.main(a.deadline,a.resume)
