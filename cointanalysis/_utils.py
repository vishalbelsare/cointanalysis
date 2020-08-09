import numpy as np


def rms(array):
    """Return root-mean square."""
    return np.sqrt(array.mean() ** 2 + array.std() ** 2)


def check_shape(X, n_features=None):
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X.shape[1] should be {n_features}")
    return X
