import pytest

import numpy as np
from fracdiff._stat import StationarityTester


list_seed = [42, 1, 2, 3]
list_n_samples = [100, 1000, 10000]


def make_stationary(seed, n_samples):
    np.random.seed(seed)
    return np.random.randn(n_samples)


def make_nonstationary(seed, n_samples):
    np.random.seed(seed)
    return np.random.randn(n_samples).cumsum()


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('seed', list_seed)
@pytest.mark.parametrize('n_samples', list_n_samples)
def test_stationary(seed, n_samples):
    X = make_stationary(seed, n_samples)

    assert StationarityTester().pvalue(X) < 0.1
    assert StationarityTester().is_stationary(X)


@pytest.mark.parametrize('seed', list_seed)
@pytest.mark.parametrize('n_samples', list_n_samples)
def test_nonstationary(seed, n_samples):
    X = make_nonstationary(seed, n_samples)

    assert StationarityTester().pvalue(X) > 0.1
    assert not StationarityTester().is_stationary(X)
