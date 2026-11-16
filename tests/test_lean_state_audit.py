from scripts.lean_state_audit import audit
from poker.fast_game import FastGame
from poker.lean_abstraction import observation_key


def test_history_collision_is_reachable_on_both_paths():
    result = audit(300)
    assert result['history_collisions']
    for collision in result['history_collisions']:
        histories = []
        for name in ('first', 'second'):
            game = FastGame(result['seed'], lean=True)
            for event in collision[name]['history']:
                game.act(event[3])
            o = game.observe()
            assert observation_key(o) == collision['key']
            assert list(o.legal) == list(collision['legal'])
            assert list(o.hole) == list(collision['hole'])
            assert list(o.board) == list(collision['board'])
            histories.append(tuple(o.history))
        assert histories[0] != histories[1]
