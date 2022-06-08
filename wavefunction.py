import numpy as np

from scipy.special._ufuncs import iv
from util import perp


def local_energy(nqs, config):
    U = config['U']
    kinetic = -local_kinetic_energy(nqs)
    potential = 0
    for edge in config['edges']:
        j = edge['j']
        k = edge['k']
        G = edge['G']
        potential -= G * np.dot(nqs.xs[j], nqs.xs[k])
    return 2 * potential + (U / 2) * kinetic


def local_kinetic_energy(nqs):
    zs_bar = log_psi_bs(nqs)
    zs_act_bar = nqs.cs + np.matmul(nqs.weights.T, zs_bar)

    zs_bar_2 = zs_bar[:, None, :] * zs_bar[:, :, None]
    zs_act_bar_2 = zs_act_bar[:, None, :] * zs_act_bar[:, :, None]
    us = nqs.xs_act / nqs.r_norms[:, None]
    us_us = us[:, None, :] * us[:, :, None]
    scale = 1 - 2 * nqs.g_over_r_norms
    shift = np.tensordot(nqs.g_over_r_norms, np.identity(2), axes=0)
    zs_zs_bar = shift + us_us * scale[:, None, None]
    zs_cov = zs_zs_bar - zs_bar_2

    metric = zs_act_bar_2 + np.tensordot(np.square(nqs.weights), zs_cov, axes=[0, 0])
    xs_perp = perp(nqs.xs)
    xs_perp_2 = xs_perp[:, None, :] * xs_perp[:, :, None]
    return np.sum(xs_perp_2 * metric) - np.sum(nqs.xs * zs_act_bar)


def amplitude(nqs):
    """Returns wavefunction given classical state and variational parameters."""
    amp = np.exp(np.sum(nqs.cs * nqs.xs))
    for norm in nqs.r_norms:
        amp *= 2 * np.pi * iv(0, norm)

    return amp


def log_psi(nqs):
    """Returns logarithm of wavefunction given classical state and variational parameters."""
    log_amplitude = np.sum(nqs.cs * nqs.xs)

    for norm in nqs.r_norms:
        log_amplitude += np.log(iv(0, norm))

    return log_amplitude


def log_psi_vars(nqs):
    grad_cs = np.reshape(log_psi_cs(nqs), -1)
    grad_bs = np.reshape(log_psi_bs(nqs), -1)
    grad_weights = np.reshape(log_psi_weights(nqs), -1)
    return np.concatenate([grad_cs, grad_bs, grad_weights])


def log_psi_cs(nqs):
    """Returns gradient of logarithm of wavefunction with respect to visible biases."""
    return nqs.xs


def log_psi_bs(nqs):
    """Returns gradient of logarithm of wavefunction with respect to hidden biases."""
    return nqs.g_over_r_norms[:, None] * nqs.xs_act


def log_psi_weights(nqs):
    """Returns gradient of logarithm of wavefunction with respect to weights."""
    return nqs.g_over_r_norms[:, None] * np.matmul(nqs.xs_act, nqs.xs.T)



