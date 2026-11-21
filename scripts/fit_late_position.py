"""Fit late-position fold evidence from completed OpenPoker public hand records."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

from poker.late_position import (BEFORE, BEHIND, MIN_STACK_BB, SCHEMA, FoldEvidence,
                                 required_fold_rate, valid_lineup)


def record_open(hand, hero):
    """One observed opening at the exact policy size, including failed steals."""
    try:
        bb, sb = hand['big_blind'], hand['small_blind']
        if hand['num_players'] != 6 or type(bb) is not int or bb <= 0 or 2*sb != bb:
            return None
        button = hand['dealer_seat']
        seats = sorted(hand['players'], key=lambda s: (s['seat'] <= button, s['seat']))
        if len(seats) != 6 or len({s['seat'] for s in seats}) != 6 or seats[-1]['seat'] != button:
            return None
        pos = next(i for i,s in enumerate(seats) if s.get('bot_name') == hero)
        if pos not in BEFORE or sum(s.get('bot_name') == hero for s in seats) != 1:
            return None
        responders = BEHIND[pos]
        names = tuple(seats[i].get('bot_name') for i in responders)
        if not valid_lineup(pos, names) or any(seats[i]['stack_start'] < MIN_STACK_BB*bb for i in (pos, *responders)):
            return None
        actions = sorted(hand['action_log'], key=lambda a: a['sequence_in_hand'])
        order = (*BEFORE[pos], pos, *responders)
        for j,i in enumerate(order):
            a = actions[j]
            if a['sequence_in_hand'] != j+1 or a['seat'] != seats[i]['seat']:
                return None
            if j < len(BEFORE[pos]):
                if a['action'] != 'fold' or a['street'] != 'preflop':
                    return None
            elif i == pos:
                if a['action'] != 'raise' or a['street'] != 'preflop' or 2*a['amount'] != 5*bb:
                    return None
            elif a['action'] in ('call', 'raise', 'all_in') and (a['street'] == 'preflop' or
                    i == responders[-1] and a['street'] == 'flop' and a['action'] != 'raise'):
                return (pos, names), False
            elif a['action'] != 'fold' or not (a['street'] == 'preflop' or
                                              i == responders[-1] and a['street'] == 'showdown'):
                return None
        # Do not accept a truncated success: all opponents must have acted and
        # the hand must have ended without a board or any additional actions.
        if len(actions) == len(order) and hand['board'] == []:
            return (pos, names), True
    except (KeyError, IndexError, StopIteration, TypeError):
        return None
    return None


def timepoint(value):
    t = datetime.fromisoformat(value)
    if t.utcoffset() is None:
        raise ValueError('cutoffs and hand timestamps must include a timezone')
    return t


def fit(records, hero, holdout_from):
    if not hero.strip():
        raise ValueError('hero name is required')
    cutoff = timepoint(holdout_from)
    seen = {}; counts = defaultdict(lambda: [[0, 0], [0, 0]])
    for r in records:
        opening = record_open(r, hero)
        if opening is None:
            continue
        hid = r['hand_id']; started = timepoint(r['started_at']); ended = timepoint(r['ended_at'])
        if not isinstance(hid, str) or not hid or ended < started:
            raise ValueError('invalid completed hand identity or timestamps')
        sample = (started, ended, opening)
        if hid in seen:
            if seen[hid] != sample:
                raise ValueError(f'conflicting records for {hid}')
            continue
        seen[hid] = sample
        if started < cutoff <= ended:
            continue
        lineup, folded = opening
        group = counts[lineup][int(started >= cutoff)]
        group[0] += int(folded)
        group[1] += 1
    return dict(schema=SCHEMA, hero=hero, holdout_from=holdout_from, lineups=[
        dict(position=pos, behind=list(names), train=train, holdout=holdout)
        for (pos,names),(train,holdout) in sorted(counts.items())])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path, help='JSON array of public evidence hand records, with full rosters and action logs')
    p.add_argument('--hero', required=True, help='exact OpenPoker bot name')
    p.add_argument('--holdout-from', required=True, help='ISO timestamp with timezone; newer completed hands validate the fit')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    records = json.loads(args.evidence.read_text())
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        p.error('evidence must be a JSON array of hand records')
    data = fit(records, args.hero, args.holdout_from)
    ready = 0
    for row in data['lineups']:
        lower = FoldEvidence(row['train'], row['holdout']).lower()
        ready += lower is not None and lower > required_fold_rate(5 - (row['position'] == 0), 3)
    with args.out.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    train, holdout = (sum(row[k][1] for row in data['lineups']) for k in ('train', 'holdout'))
    print(f'{train} training and {holdout} holdout opens, {ready}/{len(data["lineups"])} lineups qualify')


if __name__ == '__main__':
    main()
