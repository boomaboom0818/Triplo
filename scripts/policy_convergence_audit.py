"""Frozen read-only inventory of mean inertia and current-regret disagreement."""
import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

from poker.train_cfr import run_lock
from poker.train import atomic_save
from scripts.six_role_averaging import context

ROOT = Path('runs/bot/policy-convergence-audit-001')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def strategy(row):
    total = sum(row)
    return [v/total for v in row] if total else [1/len(row)]*len(row)


def inventory(before, after, regrets):
    n = len(after or before or regrets)
    assert n
    assert all(not row or len(row) == n for row in (before, after, regrets))
    assert all(math.isfinite(v) for row in (before, after, regrets) for v in row)
    assert all(v >= 0 for row in (before, after) for v in row)
    assert all(a >= b for a, b in zip(after, before))
    b, a = sum(before), sum(after)
    assert a >= b
    positive = [max(0,v) for v in regrets] if regrets else [0.]*n
    current = strategy(positive)
    average = strategy(after or [0.]*n)
    old = strategy(before or [0.]*n)
    return dict(old_positive=b>0, final_positive=a>0, newly_positive=b==0 and a>0,
                positive_current_regret=any(positive), inherited_mass_fraction=b/a if a else None,
                mean_parent_tv=.5*sum(abs(x-y) for x,y in zip(average,old)) if b else None,
                mean_current_regret_tv=.5*sum(abs(x-y) for x,y in zip(average,current)) if a else None)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('RAM availability unavailable')


def main(candidate, deadline):
    end = datetime.fromisoformat(deadline)
    assert end.tzinfo is not None and time.time() < end.timestamp() <= time.time()+3600
    resource.setrlimit(resource.RLIMIT_AS, (3*1024**3,3*1024**3))
    os.nice(10)
    def stop(signum,frame):
        raise TimeoutError('read-only audit finite deadline reached')
    signal.signal(signal.SIGALRM,stop)
    signal.alarm(max(1,math.ceil(end.timestamp()-time.time())))
    start=time.monotonic();status='failed';minimum_available=available()
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/f'candidate-{candidate}'):
        manifest=json.loads((ROOT/'manifest.json').read_text())
        assert manifest['deadline']==deadline
        for path,sha in manifest['sources'].items():
            assert digest(path)==sha,path
        for row in manifest['inputs']:
            assert digest(row['path'])==row['sha256']
        try:
            inputs=manifest['inputs'][candidate*2:candidate*2+2]
            parent,final=[json.loads(Path(p['path']).read_text()) for p in inputs]
            assert parent['iterations']==[4906,5150][candidate]
            assert final['iterations']==parent['iterations']+2000
            assert final['recipe_fork']['parent_sha256']==inputs[0]['sha256']
            assert final['traversal_stats']['iterations']==2000
            cells={f'owner={o}:street={s}:active={a}':dict(counts=Counter(),inherited=[],drift=[],disagreement=[])
                   for o in range(6) for s in range(4) for a in range(2,7)}
            keys=set(parent['sums'])|set(final['sums'])|set(final['regrets'])
            assert set(parent['sums'])<=set(final['sums'])
            for index,key in enumerate(keys):
                if index%4096==0:
                    minimum_available=min(minimum_available,available())
                    assert minimum_available>=2*1024**3,'host reserve breached'
                    assert shutil.disk_usage(ROOT).free>=10*1024**3,'disk reserve breached'
                    atomic_save(ROOT/f'candidate-{candidate}.progress.json',dict(keys=index,total_keys=len(keys)))
                fields=json.loads(key);owner,street,active=fields[1],fields[2],fields[4]
                label=f'owner={owner}:street={street}:active={active}';cell=cells[label]
                row=inventory(parent['sums'].get(key,[]),final['sums'].get(key,[]),final['regrets'].get(key,[]))
                cell['counts']['keys']+=1
                for field in ('old_positive','final_positive','newly_positive','positive_current_regret'):
                    cell['counts'][field]+=row[field]
                cell['counts']['current_regret_without_average']+=row['positive_current_regret'] and not row['final_positive']
                for field,dest in [('inherited_mass_fraction','inherited'),('mean_parent_tv','drift'),('mean_current_regret_tv','disagreement')]:
                    if row[field] is not None:cell[dest].append(row[field])
                cell['counts']['old_positive_inherited_over_99pct']+=row['old_positive'] and row['inherited_mass_fraction']>=.99
            def stats(values):
                if not values:return dict(n=0)
                v=sorted(values)
                return dict(n=len(v),mean=sum(v)/len(v),minimum=v[0],median=v[len(v)//2],p90=v[int(.9*(len(v)-1))],maximum=v[-1])
            summary={k:dict(counts=dict(v['counts']),inherited_mass_fraction=stats(v['inherited']),mean_parent_tv=stats(v['drift']),mean_current_regret_tv=stats(v['disagreement'])) for k,v in cells.items()}
            controlled=[]
            for deal in range(11200000,11200004):
                for owner in range(6):
                    for street in (1,2,3):
                        for active in (2,3,6):
                            state,reach=context(final,deal,owner,street,active)
                            from poker.cfr import information_key
                            key=information_key(state.game.observe(),final['abstraction'])
                            row=inventory(parent['sums'].get(key,[]),final['sums'].get(key,[]),final['regrets'].get(key,[])) if key in keys else dict(old_positive=False,final_positive=False,newly_positive=False,positive_current_regret=False,inherited_mass_fraction=None,mean_parent_tv=None,mean_current_regret_tv=None)
                            controlled.append(dict(deal=deal,owner=owner,street=street,active=active,zero_current_own_reach=reach==-math.inf,**row))
            assert len(controlled)==216
            assert {(r['owner'],r['street'],r['active']) for r in controlled}=={(o,s,a) for o in range(6) for s in (1,2,3) for a in (2,3,6)}
            for row in inputs:assert digest(row['path'])==row['sha256']
            output=ROOT/f'candidate-{candidate}.json';assert not output.exists(),'preserve completed audit'
            atomic_save(output,dict(candidate=candidate,inputs=inputs,manifest_sha256=digest(ROOT/'manifest.json'),
                cells=summary,controlled_contexts=controlled,total_keys=len(keys),
                caveat='Unweighted table coordinates and conditional contexts; no visitation/ESS/payoff/strength inference.'))
            status='completed'
        finally:
            peaks={}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmPeak:','VmHWM:')):
                    key,value,*_=line.split();peaks[key.rstrip(':')]=int(value)*1024
            atomic_save(ROOT/f'candidate-{candidate}.metrics.json',dict(pid=os.getpid(),status=status,wall_seconds=time.monotonic()-start,
                address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                minimum_host_available_bytes=min(minimum_available,available()),deadline=deadline,**peaks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=[0,1],required=True);parser.add_argument('--deadline',required=True)
    args=parser.parse_args();main(args.candidate,args.deadline)
