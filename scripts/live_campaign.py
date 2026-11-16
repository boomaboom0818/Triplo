"""Execute a finite manifest-driven curriculum campaign with one fresh holdout.

Never edits the incumbent or the running client. A qualifying report identifies
the exact candidate hash that may be deployed; rejection preserves the champion.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .live_gate import assess, digest, SCENARIOS, FAMILY_ALPHA, MARGIN


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def run(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest['sources'].items():
        if digest(p)!=h:raise ValueError('training source changed: '+p)
    gate_path=root/'gate-plan.json'
    plan=dict(scenarios=SCENARIOS,alpha=FAMILY_ALPHA,margin_bb100=MARGIN,
              screen_seed=manifest['screen_seed'],confirmation_seed=manifest['confirmation_seed'],
              incumbent_sha256=digest(manifest['incumbent']),
              sources={p:digest(p) for p in ('scripts/live_gate.py','scripts/live_campaign.py','poker/equity_policy.py')})
    if gate_path.exists():
        if json.loads(gate_path.read_text())!=json.loads(json.dumps(plan)):raise ValueError('gate plan changed')
    else:write(gate_path,plan)
    if (root/'promotion.json').exists():raise ValueError('campaign already evaluated; never reuse its holdout')
    def command(args,log):
        with log.open('a') as f:
            subprocess.run([sys.executable,'-m',*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    def evaluation(checkpoint,tag,phase,fallback=False):
        out=root/f'{phase}-{tag}.json'
        if out.exists():
            r=json.loads(out.read_text())
            if r['seed']!=manifest[phase+'_seed'] or r['deals']!=manifest[phase+'_deals']:
                raise ValueError('cached evaluation configuration mismatch')
            if not fallback and r['checkpoint_sha256']!=digest(checkpoint):raise ValueError('cached checkpoint changed')
            return r
        args=['scripts.live_gate','evaluate','--checkpoint',str(checkpoint),'--incumbent',manifest['incumbent'],
              '--out',str(out),'--seed',str(manifest[phase+'_seed']),'--deals',str(manifest[phase+'_deals'])]
        if fallback:args.append('--fallback-only')
        command(args,root/f'{phase}-{tag}.log')
        return json.loads(out.read_text())
    parent_iterations=json.loads((Path(manifest['parent_state'])/'meta.json').read_text())['iterations']
    candidates={}
    for seed in manifest['candidate_seeds']:
        candidate=root/f'candidate-{seed}';checkpoint=candidate/'checkpoint.json'
        if not checkpoint.exists():
            command(['scripts.history_train','--root',str(candidate),'--minutes',str(manifest['minutes_per_candidate']),
                     '--seed',str(seed),'--stack-bb','100','--workers',str(manifest['workers']),
                     '--population',str(manifest['population']),'--abstraction','history-compact-v1',
                     '--curriculum',manifest['curriculum'],'--initialize-from',manifest['parent_state'],
                     '--max-rows',str(manifest['max_rows'])],root/f'candidate-{seed}.log')
        progress=json.loads((candidate/'progress.json').read_text())
        if progress['iterations']<=parent_iterations or progress['stop_reason'] not in ('deadline','row limit','host reserve'):
            raise ValueError('candidate has no completed training or stopped unexpectedly')
        candidates[str(seed)]=checkpoint
    champion=evaluation(manifest['incumbent'],'incumbent','screen')
    fallback=evaluation(manifest['incumbent'],'fallback','screen',True)
    screens={}
    for seed,path in candidates.items():
        result=assess(evaluation(path,'candidate-'+seed,'screen'),champion,fallback,confirmation=False)
        screens[seed]=result
    # Selection uses only development screens. No tuning or retries on confirmation.
    selected=max(screens,key=lambda seed:min(screens[seed]['aggregate'][label]['mean'] for label in ('vs_incumbent','vs_fallback')))
    selection=dict(seed=selected,checkpoint=str(candidates[selected]),sha256=digest(candidates[selected]),screens=screens)
    if (root/'selection.json').exists():
        if json.loads((root/'selection.json').read_text())!=selection:raise ValueError('selection changed')
    else:write(root/'selection.json',selection)
    champion=evaluation(manifest['incumbent'],'incumbent','confirmation')
    fallback=evaluation(manifest['incumbent'],'fallback','confirmation',True)
    candidate=evaluation(candidates[selected],'candidate-'+selected,'confirmation')
    result=assess(candidate,champion,fallback)
    if digest(manifest['incumbent'])!=plan['incumbent_sha256'] or digest(candidates[selected])!=selection['sha256']:
        raise ValueError('checkpoint changed during campaign')
    for p,h in {**manifest['sources'],**plan['sources']}.items():
        if digest(p)!=h:raise ValueError('source changed during campaign')
    result.update(checkpoint=str(candidates[selected]),selection=selected)
    write(root/'promotion.json',result)
    print(json.dumps(dict(promote=result['promote'],checkpoint=result['checkpoint'],checks=result['checks']),indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
