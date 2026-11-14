"""Verify completed stage reports and isolate training versus live fallback effects.

Reads only completed, parseable reports; never writes evaluation output paths.
"""
import argparse,json,hashlib
from pathlib import Path
from poker.compare import paired_difference
from poker.train import atomic_save
from scripts.sampling_curve import ROOT,PARENT,VALIDATION


def read_report(path,checkpoint,uniform=False,live=False):
    r=json.loads(path.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if (r['checkpoint_sha256']!=digest(checkpoint) or r['seed']!=5300000 or r['copies']!=1
        or r.get('uniform_control',False)!=uniform or r.get('live_fallback',False)!=live
        or r['opponent_checkpoint_hashes']!=['checkpoint:'+digest(VALIDATION)]
        or len(r['populations'])!=6
        or any(v['independent_deal_blocks']!=1000 or len(v['deal_block_bb_per_100'])!=1000 for v in r['populations'].values())):
        raise ValueError(f'evaluation provenance mismatch: {path}')
    return r


def review(root=ROOT):
    results={};pending=[]
    for checkpoint in sorted(root.glob('candidate-*-*.json')):
        pieces=checkpoint.stem.split('-')
        if len(pieces)!=3 or not pieces[2].isdigit():continue
        name='-'.join(pieces[:2]);stage=int(pieces[2]);key=f'{name}-{stage}'
        # A checkpoint file can be complete on disk before training target completes.
        meta=json.loads(checkpoint.read_text())
        if meta['iterations']!=stage:continue
        del meta
        paths=[root/f'{name}-parent.json',root/f'{name}-uniform.json',root/f'{key}-standard.json',root/f'{key}-live-fallback.json']
        if any(not p.exists() or not p.stat().st_size for p in paths):pending.append(key);continue
        try:
            parent=read_report(paths[0],PARENT/f'{name}.json')
            uniform=read_report(paths[1],PARENT/f'{name}.json',uniform=True)
            standard=read_report(paths[2],checkpoint)
            live=read_report(paths[3],checkpoint,live=True)
        except json.JSONDecodeError:
            # Legacy active evaluator writes stdout directly; leave its path alone.
            pending.append(key);continue
        coverage={}
        for pop,v in standard['populations'].items():
            post=[x for street,x in v['coverage_by_street'].items() if street!='0']
            count=sum(x['decisions'] for x in post);hits=sum(x['positive_weight_rows'] for x in post)
            coverage[pop]=dict(decisions=count,positive=hits,fraction=hits/count if count else 0)
        opponent=standard['opponent_checkpoint_hashes'][0]
        pairs={label:paired_difference(a,b) for label,a,b in [('training_vs_parent',standard,parent),('training_vs_uniform',standard,uniform),('fallback_effect_same_checkpoint',live,standard)]}
        required=('calling','mixed',opponent)
        results[key]=dict(checkpoint_sha256=standard['checkpoint_sha256'],coverage=coverage,paired=pairs,
            coverage_pass=all(coverage[x]['decisions']>=200 and coverage[x]['fraction']>=.05 for x in required),
            payoff_pass=all(pairs[k][opponent]['ci95'][0]>0 for k in ('training_vs_parent','training_vs_uniform')),
            tight_action_counts_standard=standard['populations']['tight']['action_counts'],
            tight_action_counts_live=live['populations']['tight']['action_counts'])
    summary=dict(stages=results,pending=pending,competitive_strength_established=False)
    atomic_save(root/'verified-review.json',summary)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();r=review(args.output)
    print(json.dumps({'verified_stages':list(r['stages']),'pending':r['pending']}))
