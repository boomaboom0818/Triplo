"""Exact replay attribution of existing evidence, not another efficacy test."""
import argparse,json,os,resource,statistics,time,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime
from poker.config import load_checkpoint
from poker.cfr import CFRPolicy,information_key,normalize
from poker.policy import Policy
from poker.game import play
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.sustained_train import digest,validate_evaluation
from scripts.scaling_validation import available_memory


class TracedPolicy(CFRPolicy):
    def __init__(self,*args,groups,missing,regrets,**kwargs):
        super().__init__(*args,**kwargs);self.groups=groups;self.missing=missing;self.markers=set()
        self.regrets=regrets;self.current_own_reach=1.;self.current_zero_support=False
    def act(self,o):
        key=information_key(o,self.abstraction);row=self.sums.get(key)
        source='positive' if row and any(row) else 'missing' if row is None else 'zero'
        action=super().act(o);category='raise' if isinstance(action,int) else action
        group=f'street={o.street}:active={sum(o.active)}:source={source}:price={"paid" if o.call else "free"}'
        counts=self.groups.setdefault(group,dict(decisions=0,call=0,fold=0,raise_count=0,overbet_raise=0,big_paid_call=0,current_unreachable=0,current_tiny_reach=0,regret_row=0,positive_regret_row=0))
        counts['decisions']+=1;counts['raise_count' if category=='raise' else category]+=1
        regrets=self.regrets.get(key)
        counts['current_unreachable']+=self.current_zero_support
        counts['current_tiny_reach']+=not self.current_zero_support and self.current_own_reach<1e-6
        counts['regret_row']+=regrets is not None
        counts['positive_regret_row']+=bool(regrets and any(x>0 for x in regrets))
        counts['overbet_raise']+=isinstance(action,int) and action>o.pot
        big_call=action=='call' and o.call>o.pot/2
        counts['big_paid_call']+=big_call
        if source!='positive':
            self.markers.add('fallback_postflop' if o.street else 'fallback_preflop')
            self.missing[hashlib.sha256(key.encode()).hexdigest()]+=1
        if big_call:self.markers.add('large_call_seen')
        current=normalize(regrets if regrets is not None else [0.]*len(o.legal))
        probability=current[o.legal.index(action)]
        self.current_zero_support=self.current_zero_support or probability==0
        self.current_own_reach*=probability
        return action


def audit_one(checkpoint,validation,report_path,blocks,deadline):
    before=digest(checkpoint);op_hash=digest(validation);saved=load_checkpoint(checkpoint);op=load_checkpoint(validation)
    report=json.loads(report_path.read_text());fullblocks=next(iter(report['populations'].values()))['independent_deal_blocks']
    validate_evaluation(report,checkpoint,validation,8600000,False,fullblocks)
    if not 1<=blocks<=min(50,fullblocks):raise ValueError('bounded subset of existing evidence required')
    result={};key='checkpoint:'+op_hash
    for pop in ('tight','mixed',key):
        groups={};missing=Counter();hand_groups={};values=[]
        for block in range(blocks):
            outcomes=[]
            for rotation in range(6):
                if time.time()>=deadline:raise TimeoutError('attribution deadline reached')
                if available_memory()<2*1024**3:raise RuntimeError('hostreserve2GiB reached')
                policies=[Policy(8600000+block*100+rotation*6+i,{},['calling','tight','aggressive'][i%3] if pop=='mixed' else pop) for i in range(6)]
                candidate=TracedPolicy(saved['sums'],8600000+block*100+rotation*6+rotation,saved['abstraction'],groups=groups,missing=missing,regrets=saved['regrets'])
                policies[rotation]=candidate
                if pop==key:
                    for i in range(6):
                        if i!=rotation:policies[i]=CFRPolicy(op['sums'],8600000+block*100+rotation*6+i,op['abstraction'])
                payoff,_,_=play(policies,8600000+block,button=0);value=payoff[rotation];outcomes.append(value)
                label=','.join(sorted(candidate.markers)) or 'no_fallback'
                group=hand_groups.setdefault(label,dict(hands=0,total_payoff_bb=0.))
                group['hands']+=1;group['total_payoff_bb']+=value
            values.append(statistics.mean(outcomes)*100)
        expected=report['populations'][pop]['deal_block_bb_per_100'][:blocks]
        if any(abs(a-b)>1e-10 for a,b in zip(values,expected)):raise ValueError('instrumented replay changed original payoffs')
        result[pop]=dict(decisions=groups,hand_associations=hand_groups,top_missing_key_hashes=missing.most_common(30),replayed_blocks=blocks,exact_payoff_match=True,deal_block_bb_per_100=values)
    assert digest(checkpoint)==before and digest(validation)==op_hash
    return dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=before,opponent_sha256=op_hash,read_only=True,populations=result,caveat='Action/coverage associations from existing confirmation blocks, not fresh payoff evidence or proof of causality. Current-unreachable refers only to prior candidate actions having zero probability under frozen current regret strategy, while mean strategy generated the replay. Regret-row presence does not validate its action quality. No private cards exported, policy RNG/actions/checkpoints unchanged.')


def main(output,deadline):
    if not time.time()<deadline<=time.time()+3600:raise ValueError('finite future deadline within1h required')
    with run_lock(output):
        if output.exists():raise ValueError('attribution exists; preserve instead of rerun')
        root=Path('runs/bot/prefix-pilot-001');validation=Path('runs/bot/validation-extended-001/validation-750.json')
        result={}
        for i in range(2):
            result[str(i)]=audit_one(root/f'candidate-{i}-plus-500.json',validation,root/f'candidate-{i}-validation-candidate.json',50,deadline)
            atomic_save(output.with_suffix('.progress.json'),result)
        atomic_save(output,dict(status='completed',read_only=True,source_sha256=digest(Path(__file__)),deadline=deadline,replays=result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10);main(a.output,d.timestamp())
