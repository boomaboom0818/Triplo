from itertools import product
import pytest
from poker.cfr import MCCFR
from poker.train_cfr import train
from poker.compare import paired_difference


def test_call_mixture_exact_expected_average():
    class Toy:
        def __init__(self,h=()):self.h=h
        def terminal(self):return len(self.h)==3
        def info(self):return len(self.h),'owner' if len(self.h)==2 else str(len(self.h)),('call','fold')
        def child(self,a):return Toy(self.h+(a,))
    expected=[0.,0.]
    for path in product(range(2),repeat=2):
        choices=iter(path)
        class Fixed:
            def choices(self,*args):return [next(choices)]
        solver=MCCFR();solver.averaging_sampler='call-mixture-v1';solver.rng=Fixed()
        solver.average(Toy(),2)
        probability=1
        for i in path:probability*=.75 if i==0 else .25
        for i,v in enumerate(solver.sums['owner']):expected[i]+=probability*v
    assert expected==pytest.approx([2.,2.])


def test_sampler_resume(tmp_path):
    options=dict(averaging_sampler='call-mixture-v1')
    a=train(5,tmp_path/'a.json',**options)
    p=tmp_path/'b.json';train(2,p,**options);b=train(5,p,resume=p,**options)
    for key in ('sums','regrets','rng'):assert a[key]==b[key]
    with pytest.raises(ValueError,match='incompatible'):train(6,p,resume=p)


def test_paired_constant_difference():
    def report(v):return dict(seed=1,copies=1,opponent_checkpoint_hashes=[],populations={'x':{'deal_block_bb_per_100':v}})
    result=paired_difference(report([11,21,31]),report([10,20,30]))
    assert result['x']['ci95']==[1,1]
    assert result['x']['difference_bb_per_100']==1


def test_uniform_evaluation_records_paired_samples():
    from poker.evaluate import evaluate
    r=evaluate('artifacts/bot/cfr-smoke.json',2,42,1,uniform_control=True)
    assert r['uniform_control']
    for v in r['populations'].values():
        assert len(v['deal_block_bb_per_100'])==2
        assert v['covered_candidate_decisions']==0
        assert v['uniform_fallback_fraction']==1
        assert sum(x['decisions'] for x in v['postflop_coverage_by_active'].values())==sum(x['decisions'] for k,x in v['coverage_by_street'].items() if k!='0')
