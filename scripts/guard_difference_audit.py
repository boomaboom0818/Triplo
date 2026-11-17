"""Replay largest paired development differences; outcomes are not action EVs."""
import argparse
import json
from pathlib import Path
from random import Random

from poker.equity_policy import EquityPolicy
from poker.history_cfr import load
from poker.live_curriculum import root_game
from poker.rule_opponents import RulePolicy
from scripts.live_gate import SCENARIOS, digest, make_policy


def replay(saved, report, case_index, deal, position):
    name, players, stacks, style = SCENARIOS[case_index]
    if style == 'incumbent' or style == 'mixed':
        raise ValueError('this focused audit requires fixed rule opponents')
    hero = make_policy(saved, not report['fallback_only'])
    baseline = EquityPolicy()
    seed = report['seed'] + case_index * 100000 + deal
    hero.rng = Random(f'hero:{seed}:{position}')
    game = root_game(seed, players, list(stacks), lean=1 << position)
    opponent = RulePolicy(style)
    records = []
    while game.live:
        o = game.observe()
        if o.position != position:
            game.act(opponent.act(o))
            continue
        before, overrides = hero.covered, hero.pressure_overrides
        action = hero.act(o, history_complete=True)
        fallback = baseline.act(o)
        key = hero.key(o)
        row = saved['sums'].get(key, [])
        mass = sum(row)
        records.append(dict(street=o.street, hole=o.hole, board=o.board, pot=o.pot,
                            call=o.call, legal=o.legal, history=o.history, action=action,
                            learned=hero.covered > before, overridden=hero.pressure_overrides > overrides,
                            fallback=fallback, equity=baseline.last, key=key,
                            probabilities=[x / mass for x in row] if mass else [],
                            visits=saved['visits'].get(key, 0)))
        game.act(action)
    payoff = game.payoffs[position] / 2
    case = report['cases'][name]
    expected = case['hand_payoffs'][deal][case['positions'].index(position)]
    if payoff != expected:
        raise RuntimeError('replay differs from paired evaluation')
    return dict(payoff_bb=payoff, decisions=records)


def run(candidate_path, control_path, case_name, out, count=4):
    if Path(out).exists() or not 1 <= count <= 20:
        raise ValueError('new output and 1..20 selected differences required')
    candidate, control = [json.loads(Path(p).read_text()) for p in (candidate_path, control_path)]
    for field in ('seed', 'deals', 'source_sha256', 'scenarios', 'incumbent_sha256'):
        if candidate[field] != control[field]:
            raise ValueError('unpaired reports: ' + field)
    if any(digest(p) != sha for p, sha in candidate['source_sha256'].items()):
        raise ValueError('replay source changed')
    index = [s[0] for s in SCENARIOS].index(case_name)
    c, r = candidate['cases'][case_name], control['cases'][case_name]
    if c['positions'] != r['positions'] or any(
            len(case['hand_payoffs']) != candidate['deals'] or
            any(len(row) != len(case['positions']) for row in case['hand_payoffs'])
            for case in (c, r)):
        raise ValueError('truncated or misaligned hand returns')
    differences = sorted((a - b, deal, position)
                         for deal, (cs, rs) in enumerate(zip(c['hand_payoffs'], r['hand_payoffs']))
                         for position, a, b in zip(c['positions'], cs, rs))
    selected = list(dict.fromkeys(differences[:count] + differences[-count:]))
    policies = []
    for report in (candidate, control):
        if not report['fallback_only'] and digest(report['checkpoint']) != report['checkpoint_sha256']:
            raise ValueError('checkpoint changed')
        policies.append(load(report['checkpoint']))
    traces = []
    for difference, deal, position in selected:
        traces.append(dict(difference_bb=difference, deal=deal, position=position,
                           candidate=replay(policies[0], candidate, index, deal, position),
                           control=replay(policies[1], control, index, deal, position)))
    result = dict(case=case_name, candidate_report=str(candidate_path), control_report=str(control_path),
                  report_sha256={str(p): digest(p) for p in (candidate_path, control_path)},
                  note='Selected extreme paired differences on consumed development deals. Correlated outcomes, not solved action EVs.',
                  traces=traces)
    with Path(out).open('x') as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--control', required=True, type=Path)
    p.add_argument('--case', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    run(a.candidate, a.control, a.case, a.out)
