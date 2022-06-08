import numpy as np

from numpy.random import uniform, normal


class NQS:
    """Stores variational parameters and effective variables for lazy computation."""
    def __init__(self, config=None, state=None, vars=None):
        # variational parameters
        if vars is not None:
            self.vars = vars
        else:
            self.vars = np.array([uniform(0.0, 5.0)])

        # visible state
        if state is not None:
            self.state = state
        else:
            self.state = np.array([normal(-2, 2)])


def propose_update(nqs, config):
    new_nqs = NQS(state=nqs.state + np.random.normal(), vars=nqs.vars)
    return new_nqs