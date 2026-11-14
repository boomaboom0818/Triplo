"""Run one allowed worker with a hard address-space cap and lifetime peak metrics."""
import argparse,os,resource,runpy,sys,time
from pathlib import Path



def run(module,arguments,metrics,memory_gib=3):
    if module not in ('poker.train_cfr','poker.evaluate'):raise ValueError('unsupported worker')
    limit=memory_gib*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority<10:os.nice(10-priority)
    start=time.monotonic();status='failed'
    try:
        sys.argv=[module,*arguments]
        runpy.run_module(module,run_name='__main__')
        status='completed'
    except SystemExit as exc:
        if exc.code in (None,0):status='completed'
        raise
    finally:
        from poker.train import atomic_save
        fields={}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmPeak:','VmHWM:')):
                key,value,*_=line.split();fields[key.rstrip(':')]=int(value)*1024
        result=dict(module=module,pid=os.getpid(),status=status,wall_seconds=time.monotonic()-start,
            cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],**fields)
        import json
        previous=json.loads(metrics.read_text()) if metrics.exists() else {}
        attempts=previous.get('attempts',[])
        attempts.append(dict(result))
        result['attempts']=attempts
        for key in ('peak_rss_bytes','VmPeak','VmHWM'):
            result[key]=max(a.get(key,0) for a in attempts)
        atomic_save(metrics,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--memory-gib',type=int,choices=range(1,9),default=3)
    p.add_argument('module');p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();run(a.module,a.arguments,a.metrics,a.memory_gib)
