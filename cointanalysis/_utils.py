import numpy as np


def rms(array):
    """Return root-mean square."""
    return np.sqrt(array.mean() ** 2 + array.std() ** 2)
