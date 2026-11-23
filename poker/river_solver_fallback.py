"""Triplo v5 fallback with a calibrated heads-up river subgame."""
from .equity_pricing import GuardedEquityPolicyV4, VERSION_V5


class RiverSolverFallback(GuardedEquityPolicyV4):
    VERSION = VERSION_V5

    def __init__(self, *args, solve_river=True, **kwargs):
        super().__init__(*args, **kwargs)
        if solve_river:
            import numpy  # Check the optional dependency before play begins.
        self.solve_river = solve_river

    def act(self, o):
        baseline = super().act(o)
        if o.street != 3 or not self.solve_river:
            return baseline
        from .calibrated_river_solver import propose
        proposal = propose(o, baseline)
        if proposal is None:
            return baseline
        action, details = proposal
        self.last = dict(self.last or {}, river_solver=details)
        return action
