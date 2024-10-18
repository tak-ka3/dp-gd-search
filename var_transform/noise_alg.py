import numpy as np

def laplace_func(x, loc, sensitivity=1, eps=0.1):
    b = sensitivity / eps
    if type(loc) == np.ndarray or type(loc) == list:
        return [np.exp(-np.abs(x - val)/b) / (2*b) for val in loc]
    return np.exp(-np.abs(x - loc)/b) / (2*b)
