"""Open-size translation for the learned preflop tree (Triplo v9, opt-in).

The lean action menu only ever opens to 2 or 2.5 BB (half-pot and pot raises), so the
compact history key of a player facing a bigger open (3 BB and up) lands in facing/pot
buckets the tree never visited, and the decision falls through to the tight preflop
rules, which fold almost everything. Live Triplo folded 71% to 3 BB opens and 92-97%
to 3.5 BB+ opens this way.

``translate`` maps such a spot onto the nearest trained one (standard action
translation): the same seats, folds and calls, but the open re-sized to ``target``
(2.5 BB, the largest trained open). The learned policy answers the translated spot and
``back`` maps its action onto the real menu: fold and call keep their meaning (the
call pays the real price), raises keep their place on the menu (half-pot, pot, all-in).

Only the first decision facing exactly one open is translated; anything with a limper
before the open, an all-in, a 3-bet, or a menu that does not line up is left alone.
"""
from dataclasses import replace

from .fast_game import FastGame

TARGET = 5          # 2.5 BB in normalized chips (1 = small blind)


def _committed(history):
    """Preflop chips each position has put in (blinds included), from a public history."""
    paid = {0: 1, 1: 2}
    for _, pos, street, action, _, call in history:
        if street:
            break
        if isinstance(action, int):
            paid[pos] = action
        elif action == 'call':
            paid[pos] = paid.get(pos, 0) + call
    return paid


def _replay(o, open_index, size):
    """Replay o's preflop history with the open re-sized; (game, history) or None."""
    paid = _committed(o.history)
    initial = [o.stacks[i] + paid.get(i, 0) for i in range(len(o.stacks))]
    if len(initial) != 6 or min(initial) <= 2:
        return None
    game = FastGame(0, 6, button=5, stacks=initial, lean=True)
    history = []
    for j, (player, pos, street, action, _, _) in enumerate(o.history):
        if street or not game.live or game.street or game.actors[0] != pos:
            return None
        pot, call, _ = game.menu()
        if j == open_index:
            action = size
        history.append((player, pos, street, action, pot, call))
        game.act(action, checked=False)
    if not game.live or game.street or game.actors[0] != o.position:
        return None
    return game, tuple(history)


def translate(o, target=TARGET, max_open=None):
    """The observation re-played with the open at ``target``, or None if this is not such a spot.

    ``max_open`` (normalized chips) leaves bigger opens to the rules.
    """
    if o.street or not o.call or 'fold' not in o.legal:
        return None
    raises = [j for j, h in enumerate(o.history) if isinstance(h[3], int)]
    if len(raises) != 1:
        return None
    k = raises[0]
    size = o.history[k][3]
    if size <= target or (max_open is not None and size > max_open):
        return None
    if any(h[3] != 'fold' for h in o.history[:k]):
        return None                                 # limpers: an isolation raise, not an open
    if any(h[1] == o.position for h in o.history):
        return None                                 # not hero's first decision
    if any(h[3] not in ('fold', 'call') for h in o.history[k + 1:]):
        return None
    if any(s <= 0 for s, a in zip(o.stacks, o.active) if a):
        return None                                 # somebody is all in
    real = _replay(o, None, None)
    if real is None:
        return None
    pot, call, legal = real[0].menu()
    if (pot, call) != (o.pot, o.call) or tuple(real[0].stacks) != tuple(o.stacks):
        return None                                 # history does not reproduce this spot
    replayed = _replay(o, k, target)
    if replayed is None:
        return None
    game, history = replayed
    pot, call, legal = game.menu()
    if 'fold' not in legal:
        return None
    return replace(o, stacks=tuple(game.stacks), pot=pot, call=call, legal=legal, history=history)


def back(action, translated, o):
    """Map an action chosen on the translated menu onto o's real menu, or None."""
    if not isinstance(action, int):
        return action if action in o.legal else None
    small = [a for a in translated.legal if isinstance(a, int)]
    real = [a for a in o.legal if isinstance(a, int)]
    if not real:
        return 'call'
    if len(small) == len(real):
        return real[small.index(action)]
    # Menus differ (an all-in on one side only): keep the raise's size relative to the pot.
    top_t = translated.call + _committed(translated.history).get(translated.position, 0)
    top_r = o.call + _committed(o.history).get(o.position, 0)
    frac = (action - top_t) / max(1, translated.pot)
    return min(real, key=lambda a: (abs((a - top_r) / max(1, o.pot) - frac), a))
