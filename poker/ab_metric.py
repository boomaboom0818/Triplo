"""Lower-variance A/B metrics for comparing Triplo builds (Triplo v9 harness).

One live hand varies by about 25 BB (season 17, v8), so an uncapped BB/100 needs
~20,000 hands to resolve +-35 BB/100. Capping each hand's net at +-25 BB (SD ~7.4)
or +-50 BB (SD ~12) needs ~1,700 or ~4,700 hands for the same resolution, and the share
of positive 10-minute windows adds a robust, outlier-free view.

The cap changes what is measured: Triplo's profit is concentrated in big pots (live v8:
+207 BB/100 uncapped, -8 capped at +-25, +47 capped at +-50). A capped difference is
therefore only a valid comparison for changes that leave big pots alone (preflop
sizing, small flop stabs). For changes that alter big pots (value sizing), also report
the uncapped metric and the big-pot breakdown (``by_pot``).

Every function takes per-hand nets in big blinds; results are BB/100 with normal 95%
intervals. ``paired`` is for simulations where both arms played the same deals.
"""
from math import ceil, floor, nan, sqrt
from statistics import mean, stdev

Z95 = 1.959963984540054
CAPS = (25, 50, None)                       # None = uncapped
POT_EDGES = (10, 25, 50, 100, 200)          # final pot, big blinds


def clip(x, cap):
    return x if cap is None else max(-cap, min(cap, x))


def cap_label(cap):
    return 'uncapped' if cap is None else f'cap{cap}'


def estimate(values, cap=None):
    """Mean BB/100 with a 95% interval; ``sd_bb`` is the per-hand SD after capping."""
    v = [clip(x, cap) for x in values]
    n = len(v)
    if not n:
        return dict(n=0, bb100=nan, ci95=[nan, nan], sd_bb=nan)
    m = mean(v)
    sd = stdev(v) if n > 1 else nan
    se = sd / sqrt(n) if n > 1 else nan
    return dict(n=n, bb100=round(100 * m, 2), ci95=[round(100 * (m - Z95 * se), 2), round(100 * (m + Z95 * se), 2)],
                sd_bb=round(sd, 3))


def paired(base, arm, cap=None):
    """arm - base on the same deals (each hand capped before differencing)."""
    if len(base) != len(arm):
        raise ValueError('paired arms need the same deals')
    return estimate([clip(b, cap) - clip(a, cap) for a, b in zip(base, arm)])


def welch(base, arm, cap=None):
    """arm - base for independent samples (live windows), 95% Welch-normal interval."""
    a, b = [clip(x, cap) for x in base], [clip(x, cap) for x in arm]
    if len(a) < 2 or len(b) < 2:
        return dict(bb100=nan, ci95=[nan, nan], n=[len(a), len(b)])
    d = mean(b) - mean(a)
    se = sqrt(stdev(a) ** 2 / len(a) + stdev(b) ** 2 / len(b))
    return dict(bb100=round(100 * d, 2), ci95=[round(100 * (d - Z95 * se), 2), round(100 * (d + Z95 * se), 2)],
                n=[len(a), len(b)])


def hands_for(sd_bb, half_width_bb100):
    """Hands for a 95% interval of +-half_width BB/100 at per-hand SD ``sd_bb``."""
    return ceil((Z95 * 100 * sd_bb / half_width_bb100) ** 2)


def hands_for_difference(sd_a, sd_b, half_width_bb100):
    """Hands PER ARM for a +-half_width interval on an unpaired difference."""
    return ceil((Z95 * 100 / half_width_bb100) ** 2 * (sd_a ** 2 + sd_b ** 2))


def window_share(rows, seconds=600., min_hands=1):
    """Share of fixed wall-clock windows with a positive net (live logs).

    ``rows`` are (time, net_bb) pairs. Windows are aligned to the first hand; windows
    with fewer than ``min_hands`` hands (lobby waits, reconnects) are skipped.
    """
    windows = {}
    rows = sorted(rows)
    if not rows:
        return dict(windows=0, positive_share=nan)
    t0 = rows[0][0]
    for t, net in rows:
        k = floor((t - t0) / seconds)
        n, s = windows.get(k, (0, 0.))
        windows[k] = (n + 1, s + net)
    kept = [s for n, s in windows.values() if n >= min_hands]
    if not kept:
        return dict(windows=0, positive_share=nan)
    pos = sum(s > 0 for s in kept)
    p = pos / len(kept)
    se = sqrt(p * (1 - p) / len(kept))
    return dict(windows=len(kept), positive_share=round(p, 4),
                ci95=[round(max(0., p - Z95 * se), 4), round(min(1., p + Z95 * se), 4)])


def block_share(values, block):
    """Share of consecutive ``block``-hand blocks with a positive net (simulation analogue of
    10-minute windows; live Triplo plays ~14 hands per 10 minutes)."""
    sums = [sum(values[i:i + block]) for i in range(0, len(values) - block + 1, block)]
    if not sums:
        return dict(blocks=0, positive_share=nan)
    return dict(blocks=len(sums), positive_share=round(sum(s > 0 for s in sums) / len(sums), 4))


def paired_block_share(base, arm, block):
    """Share of paired blocks where the arm out-earned the baseline (ties count half)."""
    sums = [sum(arm[i:i + block]) - sum(base[i:i + block]) for i in range(0, len(base) - block + 1, block)]
    if not sums:
        return dict(blocks=0, arm_ahead_share=nan)
    return dict(blocks=len(sums), arm_ahead_share=round((sum(s > 0 for s in sums) + .5 * sum(s == 0 for s in sums))
                                                       / len(sums), 4))


def pot_bucket(pot_bb, edges=POT_EDGES):
    lo = 0
    for hi in edges:
        if pot_bb < hi:
            return f'{lo}-{hi}'
        lo = hi
    return f'{edges[-1]}+'


def bucket_names(edges=POT_EDGES):
    return [pot_bucket(e - 1e-9, edges) for e in edges] + [f'{edges[-1]}+']


def by_pot(nets, pots, edges=POT_EDGES):
    """Per final-pot bucket: hands, share, mean net and contribution to the overall BB/100."""
    total = len(nets)
    groups = {}
    for net, pot in zip(nets, pots):
        groups.setdefault(pot_bucket(pot, edges), []).append(net)
    out = {}
    for name in bucket_names(edges):
        g = groups.get(name)
        if not g:
            continue
        out[name] = dict(hands=len(g), share=round(len(g) / total, 4), mean_net_bb=round(mean(g), 3),
                         contribution_bb100=round(100 * sum(g) / total, 2))
    return out


def paired_by_pot(base, arm, base_pots, arm_pots, edges=POT_EDGES):
    """Where a paired difference comes from: deals bucketed by the larger of the two final pots.

    ``contribution_bb100`` sums to the overall paired difference, with a 95% interval
    on each bucket's contribution (the per-deal difference times the bucket indicator)."""
    total = len(base)
    diffs = [b - a for a, b in zip(base, arm)]
    keys = [pot_bucket(max(p, q), edges) for p, q in zip(base_pots, arm_pots)]
    out = {}
    for name in bucket_names(edges):
        contrib = [d if k == name else 0. for d, k in zip(diffs, keys)]
        n = sum(k == name for k in keys)
        if not n:
            continue
        e = estimate(contrib)
        out[name] = dict(deals=n, share=round(n / total, 4), contribution_bb100=e['bb100'], ci95=e['ci95'])
    return out


def big_pot_summary(nets, pots, threshold=50.):
    """Hands whose final pot reached ``threshold`` BB: count, rate and net BB/100 they add."""
    big = [(n, p) for n, p in zip(nets, pots) if p >= threshold]
    total = len(nets)
    return dict(threshold_bb=threshold, hands=len(big), rate=round(len(big) / total, 4) if total else nan,
                contribution_bb100=round(100 * sum(n for n, _ in big) / total, 2) if total else nan,
                won_share=round(sum(n > 0 for n, _ in big) / len(big), 4) if big else nan)


def summary(nets, pots=None, block=14):
    """All single-arm metrics in one dict."""
    out = {cap_label(c): estimate(nets, c) for c in CAPS}
    out['positive_blocks'] = block_share(nets, block)
    if pots is not None:
        out['by_pot'] = by_pot(nets, pots)
        out['big_pots'] = big_pot_summary(nets, pots)
    return out


def compare(base, arm, base_pots=None, arm_pots=None, block=14):
    """All paired metrics for arm - base."""
    out = {cap_label(c): paired(base, arm, c) for c in CAPS}
    out['arm_ahead_blocks'] = paired_block_share(base, arm, block)
    if base_pots is not None and arm_pots is not None:
        out['by_pot'] = paired_by_pot(base, arm, base_pots, arm_pots)
    return out
