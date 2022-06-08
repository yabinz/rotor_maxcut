from copy import deepcopy

import numpy as np

from numpy.random import uniform, normal, random
from util import cartesian, g_func, angle, unroll, center

# import time
# np.random.seed(int(time.time())) # manually seeds according to time

class NQS:
    """Stores variational parameters and effective variables for lazy computation."""
    def __init__(self, config=None, state=None, vars=None):
        # variational parameters
        n, H = config['num_visible'], config['num_hidden']
        if vars is not None:
            self.vars = vars
        else:
            var_list = [uniform(-1, 1, (n, 2)), uniform(-1, 1, (H, 2)), normal(0.0, 1.0, (H, n))]
            self.vars = np.concatenate([np.reshape(var, -1) for var in var_list])
            # print(self.vars[0:5])
            print(random())

        self.cs = np.reshape(self.vars[:2 * n], (n, 2))
        self.bs = np.reshape(self.vars[2 * n:2 * (n + H)], (H, 2))
        self.weights = np.reshape(self.vars[2 * (n + H): 2 * (n + H) + n * H], (H, n))
        
        # visible state
        if state is not None:
            self.state = state
        else:
            self.state = unroll(np.random.uniform(-np.pi, np.pi, len(self.cs))) # len gives outersize =n
            # print(self.state[0:5])

        self.xs = cartesian(self.state) # stack [cosine(state), sine(state)]

        # import pdb; pdb.set_trace()
        # effective variable stuff
        self.xs_act = self.bs + np.tensordot(self.weights, self.xs, axes=[1, 0])
        self.r_norms = np.array([np.linalg.norm(q) for q in self.xs_act])
        self.g_r_norms = g_func(self.r_norms)
        self.g_over_r_norms = self.g_r_norms / self.r_norms

    def update_state(self, new_val, site):
        """Updates effective variables given Metropolis update information."""
        theta_new, theta_old = new_val, self.state[site]
        self.state[site] = theta_new
        x_new = np.array([np.cos(angle(theta_new)), np.sin(angle(theta_new))])
        self.xs[site] = x_new
        x_old = np.array([np.cos(angle(theta_old)), np.sin(angle(theta_old))])
        w = self.weights[:, site]
        self.xs_act += np.tensordot(w, x_new - x_old, axes=0)
        self.r_norms = np.array([np.linalg.norm(q) for q in self.xs_act])
        self.g_r_norms = g_func(self.r_norms)
        self.g_over_r_norms = self.g_r_norms / self.r_norms


def propose_update(nqs, config):
    site = np.random.randint(len(nqs.state))
    bump_size = config['bump_size']
    theta_euclid = nqs.state[site]
    bump = np.random.uniform(-bump_size, bump_size)
    theta_euclid_new = unroll(center(angle(theta_euclid) + bump))
    new_nqs = deepcopy(nqs)
    new_nqs.update_state(new_val=theta_euclid_new, site=site)  # lazy calculation
    return new_nqs