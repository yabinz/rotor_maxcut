import numpy as np


def local_energy(nqs, config):
    return nqs.vars + np.square(nqs.state) * (0.5 - 2 * np.square(nqs.vars))


def log_psi(nqs):
    return - nqs.vars * np.square(nqs.state)


def log_psi_vars(nqs):
    return -np.square(nqs.state)


def log_psi_state(nqs):
    return -2 * nqs.state * nqs.vars