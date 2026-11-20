import json

import pytest

from poker.game import Observation
from poker.history_cfr import ParallelHistory, load
from scripts import depth_ladder as dl
from scripts import live_report_depth as lr


def test_conditions_share_one_baseline_and_hold_opponents_short_in_ladder_a():
    conds = dl.conditions((50, 200, 1500))
    assert conds[0] == ('equal-50', 'baseline', 50, 50)
    assert ('hero-200-vs-50', 'A', 200, 50) in conds and ('equal-1500', 'B', 1500, 1500) in conds
    assert len(conds) == 5
    with pytest.raises(ValueError):
        dl.conditions((100, 200))
    assert dl.table_stacks(3, 400, 50) == [50, 50, 50, 400, 50, 50]


def test_effective_depth_ignores_folded_and_hero():
    o = Observation(0, 1, ('As', 'Kd'), (), (3000, 400, 100, 2000, 100, 100),
                    (True, True, True, False, False, True), 3, 1, 0, ('call', 'fold'), ())
    assert dl.effective_bb(o) == 200  # min(hero 400, deepest live 3000) units / 2
    assert dl.depth_bucket(50) == '35-75' and dl.depth_bucket(1500) == '>750' and dl.depth_bucket(25) == '15-35'


@pytest.fixture(scope='module')
def empty_policy(tmp_path_factory):
    path = tmp_path_factory.mktemp('ladder') / 'empty.json'
    ParallelHistory().export_policy(path)
    return path


def test_ladder_is_paired_and_hero_stack_alone_cannot_matter_without_coverage(tmp_path, empty_policy):
    report = dl.run(tmp_path / 'ladder.json', empty_policy, seed=4242, deals=4, depths=(50, 400),
                    lineups=('tight-v1', 'station-v1'), workers=1, shard=3, saved=load(empty_policy))
    a = report['analysis']
    assert set(a['conditions']) == {'equal-50', 'hero-400-vs-50', 'equal-400'}
    for name in a['conditions']:
        for lineup in ('tight-v1', 'station-v1'):
            cell = report['cells'][f'{name}|{lineup}']
            assert cell['hands'] == 4 * 6 and None not in cell['blocks']
            assert set(cell['sources']) <= set(dl.SOURCES)
    # Ladder A keeps effective depth at 50 BB, so a fallback-only hero plays almost
    # every hand identically. Not all: the hero's own menu still drops the
    # all-in once its own stack exceeds call + 2 pots, which shifts size indexes.
    deep = a['conditions']['hero-400-vs-50']
    assert deep['hero_action_identical_to_baseline'] > .9
    assert a['conditions']['equal-50']['hero_action_identical_to_baseline'] == 1
    assert a['conditions']['equal-50']['vs_baseline'] is None
    assert 'hero-400-vs-50 minus equal-400' in a['ladder_gap']
    assert 'learned_policy' not in deep['source_shares']
    assert json.loads((tmp_path / 'ladder.json').read_text())['schema'] == 'depth-ladder-v1'
    assert 'equal-400' in dl.render(report, a)
    with pytest.raises(ValueError):  # never overwrite a report
        dl.run(tmp_path / 'ladder.json', empty_policy, deals=4, saved=load(empty_policy))


def test_shards_cover_every_deal_once():
    conds = dl.conditions((50, 100))
    jobs = dl.shards(conds, ('tight-v1',), 10, 4)
    covered = sorted((j[0][0], d) for j in jobs for d in range(j[3], j[4]))
    assert covered == sorted((c[0], d) for c in conds for d in range(10))


def _result(t, table, seat, final, payouts=(), actions=None, hand='h'):
    actions = actions if actions is not None else [{'seat': s, 'action': 'fold'} for s in final]
    return dict(time=t, type='hand_result', seat=seat, result=dict(
        table_id=table, hand_id=hand, final_stacks={str(k): v for k, v in final.items()},
        payouts=[dict(seat=s, amount=a) for s, a in payouts], actions=actions))


def test_live_report_chains_stacks_and_handles_rebuys_and_gaps(tmp_path):
    events = [
        _result(0, 'T', 0, {0: 1000, 1: 1000, 2: 4000}, hand='a'),                        # start unknown
        dict(time=1, type='decision', hand_id='b', decision_source='learned_policy'),
        dict(time=2, type='decision', hand_id='b', decision_source='equity_fallback'),
        _result(30, 'T', 0, {0: 1200, 1: 800, 2: 4000}, [(0, 400)], hand='b'),            # +200 chips
        _result(60, 'T', 0, {0: 0, 1: 2000, 2: 4000}, [(1, 2400)], hand='c'),             # bust -1200
        _result(90, 'T', 0, {0: 980, 1: 2020, 2: 4000}, [(1, 40)], hand='d'),             # rebuy 1000, -20
        _result(2000, 'T', 0, {0: 5000, 1: 2020, 2: 0}, [], hand='e'),                    # unexplained chips
    ]
    path = tmp_path / 'session.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in events) + '\n{"partial')
    rows = lr.hands(lr.read_events(path), big_blind=20, buy_in=1000, max_gap=300)
    assert [r['net'] for r in rows] == [None, 200, -1200, -20, None]
    assert rows[0]['excluded'] == 'first hand at table' and rows[4]['excluded'].startswith('unexplained')
    assert rows[1]['effective'] == 1000 and rows[1]['typical_effective'] == 1000  # hero 1000 vs 1000/4000
    assert rows[3]['hero_start'] == 1000
    assert rows[4]['duration'] is None and rows[1]['duration'] == 30
    assert rows[1]['sources'] == {'learned_policy': 1, 'equity_fallback': 1}
    rep = lr.report([path], big_blind=20)
    assert rep['hands_with_net'] == 3
    assert rep['overall']['bb_per_100'] == pytest.approx((10 - 60 - 1) / 3 * 100)
    assert rep['overall']['hands_per_hour'] == pytest.approx(120)
    stack = rep['by_hero_stack']['40-75']
    assert stack['hands'] == 3 and stack['source_shares'] == {'learned_policy': .5, 'equity_fallback': .5}
    assert 'By hero starting stack' in lr.render(rep)
    assert lr.report([path], since=50)['hands_with_net'] == 2


def test_buckets_are_ordered_and_closed():
    assert lr.bucket(39.9, lr.HERO_EDGES) == '<40' and lr.bucket(40, lr.HERO_EDGES) == '40-75'
    assert lr.bucket(1500, lr.HERO_EDGES) == '>=1500'
    assert lr.bucket_order(lr.HERO_EDGES)[0] == '<40'
    assert lr.parse_time('1970-01-01T00:01:00Z') == 60 and lr.parse_time('5') == 5
