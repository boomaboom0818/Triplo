"""Versioned stochastic training profiles; no private simulator information."""
from .cfr import normalize


def profile_strategy(observation,style):
    if style=='guarded-v2':
        raises=sum(isinstance(a,int) for a in observation.legal);has_fold='fold' in observation.legal
        if observation.call:call,fold,raise_mass=.13+(.85 if not has_fold else 0),.85,.02
        else:call,fold,raise_mass=.95,0.,.05
        if not raises:call+=raise_mass
        return normalize([call if a=='call' else fold if a=='fold' else raise_mass/raises for a in observation.legal])
    call,fold,raise_mass={'passive-v1':(.75,.15,.10),'pressure-v1':(.35,.10,.55)}[style]
    raises=sum(isinstance(a,int) for a in observation.legal)
    return normalize([call if a=='call' else fold if a=='fold' else raise_mass/raises for a in observation.legal])


def population_fork_allowed(before,after):
    before=dict(before);after=dict(after)
    old=before.pop('opponent_population','historical-v1')
    new=after.pop('opponent_population','historical-v1')
    return old=='historical-v1' and new in ('diverse-v1','guarded-tables-v2') and before==after
