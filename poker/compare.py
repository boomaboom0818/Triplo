"""Paired whole-deal differences; screening intervals are not strength claims."""
import math
from random import Random
from statistics import mean


def paired_difference(candidate,control):
    for key in ('seed','copies','opponent_checkpoint_hashes'):
        if candidate[key]!=control[key]:raise ValueError('unmatched evaluation design')
    if candidate['populations'].keys()!=control['populations'].keys():raise ValueError('unmatched populations')
    result={}
    for name,population in candidate['populations'].items():
        a=population['deal_block_bb_per_100'];b=control['populations'][name]['deal_block_bb_per_100']
        if len(a)!=len(b) or len(a)<2:raise ValueError('unmatched deal blocks')
        for row,values in ((population,a),(control['populations'][name],b)):
            if row.get('independent_deal_blocks',len(values))!=len(values):raise ValueError('deal count mismatch')
            if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):raise ValueError('invalid payoff samples')
        differences=[x-y for x,y in zip(a,b)]
        rng=Random(candidate['seed'])
        samples=sorted(mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
        result[name]={'difference_bb_per_100':mean(differences),'ci95':[samples[50],samples[1949]],'independent_deal_blocks':len(a)}
    return result
