"""Versioned stochastic training profiles; no private simulator information."""
from .cfr import normalize


def profile_strategy(observation,style):
    call,fold,raise_mass={'passive-v1':(.75,.15,.10),'pressure-v1':(.35,.10,.55)}[style]
    raises=sum(isinstance(a,int) for a in observation.legal)
    return normalize([call if a=='call' else fold if a=='fold' else raise_mass/raises for a in observation.legal])


def population_fork_allowed(before,after):
    before=dict(before);after=dict(after)
    old=before.pop('opponent_population','historical-v1')
    new=after.pop('opponent_population','historical-v1')
    return old=='historical-v1' and new=='diverse-v1' and before==after
