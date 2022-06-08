import numpy as np
import yaml

from scipy.special import iv


def average(x):
    return sum(x) / len(x)

# new std function 
def standard_deviation(x):
    x_av = average(x)
    x_var = sum((t - x_av)**2 for t in x) / (len(x) - 1)
    return np.sqrt(x_var)
    
def g_func(x):
    return iv(1, x) / iv(0, x)


def cartesian(state, scalar=None):
    angles = angle(state)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    if scalar is not None:
        cosines = cosines * scalar
        sines = sines * scalar
    return np.stack([cosines, sines], axis=1)


def perp(state):
    state_perp = np.copy(state)
    state_perp[:, 0], state_perp[:, 1] = -state[:, 1], state[:, 0]
    return state_perp


def center(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi


# HMC helper
def unroll(angle):
    # return np.arctanh(angle / np.pi)
    return angle


# HMC helper
def angle(real):
    # return np.pi * np.tanh(real)
    return real


def sech(angle):
    return 1.0 / np.cosh(angle)


def read_config(config_path):
    with open(config_path, "r") as f:
        config_data = f.read()
        params = yaml.load(config_data, Loader=yaml.Loader)
        # params = yaml.load(config_data, Loader=yaml.CLoader)
        ## incuring error: AttributeError: module 'yaml' has no attribute 'CLoader'
    return params


