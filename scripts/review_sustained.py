"""Incremental read-only evidence verification for a staged continuation manifest."""
import argparse,json,math,statistics
from pathlib import Path
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.config import load_checkpoint
from poker.compare import paired_difference
from scripts.sustained_train import digest


def validate_report(path,checkpoint,manifest,uniform=False):
    report=json.loads(path.read_text())
    if (report['checkpoint_sha256']!=digest(checkpoint)
        or report['opponent_checkpoint_hashes']!=['checkpoint:'+manifest['validation']['sha256']]
        or report['seed']!=manifest['seed'] or report['copies']!=1
        or report.get('uniform_control',False)!=uniform or report.get('live_fallback',False)
        or len(report['populations'])!=6):raise ValueError(f'report provenance mismatch: {path}')
    for v in report['populations'].values():
        values=v['deal_block_bb_per_100']
        if len(values)!=manifest['blocks'] or v['independent_deal_blocks']!=len(values):raise ValueError('report sample count mismatch')
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):raise ValueError('invalid payoff samples')
        if not math.isclose(statistics.mean(values),v['bb_per_100'],abs_tol=1e-9):raise ValueError('report mean mismatch')
    return report


def review(root):
    with run_lock(root/'evidence-review'):
        manifest=json.loads((root/'manifest.json').read_text())
        if digest(Path(manifest['validation']['path']))!=manifest['validation']['sha256']:raise ValueError('validation checkpoint changed')
        output=root/'verified-review.json'
        old=json.loads(output.read_text()) if output.exists() else {}
        entries={};pending=[]
        for name,parent in manifest['parents'].items():
            parent_path=Path(parent['path'])
            if digest(parent_path)!=parent['sha256']:raise ValueError('initial parent changed')
            for index,stage in enumerate(manifest['stages']):
                key=f'{name}-{stage}';checkpoint=root/f'{key}.json'
                paths=[root/f'{name}-parent.json',root/f'{name}-uniform.json',root/f'{key}-evaluation.json',root/f'{key}-paired.json']
                if not checkpoint.exists() or any(not p.exists() for p in paths):pending.append(key);continue
                # Reports are atomically published by this runner; malformed final output is a failure.
                source=parent_path if index==0 else root/f'{name}-{manifest["stages"][index-1]}.json'
                lineage=json.loads((root/f'{key}-parent.json').read_text())
                if lineage!={'path':str(source.resolve()),'sha256':digest(source)}:raise ValueError('stage parent changed')
                hashes={str(p):digest(p) for p in [checkpoint,*paths]}
                hashes['manifest']=digest(root/'manifest.json')
                previous=old.get('stages',{}).get(key)
                if previous and previous['verified_hashes']==hashes:
                    entries[key]=previous;continue
                saved=load_checkpoint(checkpoint,model_id='bot-opponent-v1')
                original=load_checkpoint(parent_path,model_id='bot-opponent-v1')
                if saved['iterations']!=stage or saved['stop_reason']!='target_reached' or saved['config']!=original['config']:raise ValueError('stage checkpoint mismatch')
                del original
                compute={k:saved[k] for k in ('iterations','nodes','seconds')}
                compute['information_sets']=len(set(saved['sums'])|set(saved['regrets']));del saved
                reports={k:validate_report(p,c,manifest,u) for k,p,c,u in [('parent',paths[0],parent_path,False),('uniform',paths[1],parent_path,True),('candidate',paths[2],checkpoint,False)]}
                paired={k:paired_difference(reports['candidate'],reports[k]) for k in ('parent','uniform')}
                if paired!=json.loads(paths[3].read_text()):raise ValueError('paired statistics mismatch')
                coverage={}
                for pop,v in reports['candidate']['populations'].items():
                    rows=[r for street,r in v['coverage_by_street'].items() if street!='0']
                    n=sum(r['decisions'] for r in rows);hits=sum(r['positive_weight_rows'] for r in rows)
                    coverage[pop]={'decisions':n,'positive':hits,'fraction':hits/n if n else 0}
                opponent=reports['candidate']['opponent_checkpoint_hashes'][0]
                entries[key]=dict(verified_hashes=hashes,checkpoint=str(checkpoint.resolve()),compute=compute,paired=paired,coverage=coverage,
                    coverage_screen=all(coverage[k]['decisions']>=200 and coverage[k]['fraction']>=.05 for k in ('calling','mixed',opponent)),
                    payoff_screen=all(paired[k][opponent]['ci95'][0]>0 for k in paired))
        result=dict(purpose=manifest.get('purpose','unspecified'),stages=entries,pending=pending,competitive_strength_established=False)
        atomic_save(output,result)
        return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    result=review(parser.parse_args().output)
    print(json.dumps({'verified':list(result['stages']),'pending':result['pending'],'purpose':result['purpose']}))
