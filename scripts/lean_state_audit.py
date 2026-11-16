"""Reproduce strategic information loss in lean-v3 without training a policy."""
import argparse
import json
from random import Random
from poker.fast_game import FastGame
from poker.lean_abstraction import observation_key, card_features
from poker.fast_eval import CARD_ID, evaluate


def river_equity(hole, board):
    rest = [c for c in range(52) if c not in hole + board]
    score = evaluate(hole + board)
    wins = ties = total = 0
    for j, a in enumerate(rest):
        for b in rest[j+1:]:
            other = evaluate((a,b) + board)
            wins += score > other
            ties += score == other
            total += 1
    return (wins + ties / 2) / total


def audit(paths=3000):
    rng = Random(8251)
    seen = {}
    collisions = []
    for _ in range(paths):
        game = FastGame(17, lean=True)
        while game.live:
            o = game.observe()
            key = observation_key(o)
            # Same private/public cards, actor and menu: history alone still differs.
            signature = (key, o.hole, o.board, o.legal)
            history = tuple(o.history)
            old = seen.get(signature)
            if old is not None and old['history'] != history and len(collisions) < 3:
                raises = lambda h: sum(isinstance(x[3], int) for x in h)
                if raises(old['history']) != raises(history):
                    collisions.append(dict(key=key, hole=o.hole, board=o.board, legal=o.legal,
                        first=old, second=dict(history=history, pot=o.pot, call=o.call, stacks=o.stacks)))
            else:
                seen[signature] = dict(history=history, pot=o.pot, call=o.call, stacks=o.stacks)
            game.act(rng.choice(o.legal))
    # Board is a straight flush: every legal holding must split. Hero-card
    # overlapping impossible opponent hands are included in the current feature.
    board = tuple(CARD_ID[x] for x in ('As','Ks','Qs','Js','Ts'))
    hole = tuple(CARD_ID[x] for x in ('2c','3d'))
    return dict(paths=paths, seed=17, history_collisions=collisions,
                board_plays=dict(hole=['2c','3d'], board=['As','Ks','Qs','Js','Ts'],
                    feature=card_features(hole,board), exact_heads_up_equity=river_equity(hole,board)),
                note='Constructive information-loss examples, not measurements of payoff cost.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = audit()
    with open(a.out, 'x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
