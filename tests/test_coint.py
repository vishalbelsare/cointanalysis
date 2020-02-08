import pytest

from itertools import product
import numpy as np

from cointanalysis import CointAnalysis


params_seed = [42]
params_method = ['AEG']
params_axis = ['0', '1', 'PCA']
params_trend = ['c']
params_adjust = [True]
params_gamma = [1.0, 2.0, 10.0, -1.0, -2.0, -10.0]
params_n_samples = [1000]


# def make_gauss(seed, n_samples, n_series):
#     np.random.seed(seed)
#     return np.random.randn(n_samples, n_series)


def make_stationary(seed, n_samples, n_series):
    np.random.seed(seed)
    return np.random.randn(n_samples, n_series)


def make_nonstationary(seed, n_samples, n_series):
    np.random.seed(seed)
    return np.random.randn(n_samples, n_series).cumsum(axis=0)


def make_cointegrated(seed, n_samples, gamma):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = gamma * x0_t + gauss[:, 1]
    for various gamma.

    cf: Hamilton [19.11.1, 19.11.2]
    """
    np.random.seed(seed)
    x0 = np.random.randn(n_samples).cumsum()
    x1 = gamma * x0 + np.random.randn(n_samples)
    return np.stack([x0, x1], axis=1)


def make_notcointegrated(seed, n_samples, gamma):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = x1_t-1 + gamma * x0_t + gauss[:, 1]
    for various gamma.
    """
    np.random.seed(seed)
    x0 = np.random.randn(n_samples).cumsum()
    x1 = np.random.randn(n_samples).cumsum()
    return np.stack([x0, x1], axis=1)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('method', params_method)
@pytest.mark.parametrize('axis', params_axis)
@pytest.mark.parametrize('trend', params_trend)
@pytest.mark.parametrize('gamma', params_gamma)
@pytest.mark.parametrize('n_samples', params_n_samples)
def test_cointegrated(seed, method, axis, trend, gamma, n_samples):
    coint = CointAnalysis(method=method, axis=axis, trend=trend)
    X = make_cointegrated(seed=seed, n_samples=n_samples, gamma=gamma)

    assert coint.test(X).pvalue_ < 0.2


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('method', params_method)
@pytest.mark.parametrize('axis', params_axis)
@pytest.mark.parametrize('trend', params_trend)
@pytest.mark.parametrize('gamma', params_gamma)
@pytest.mark.parametrize('n_samples', params_n_samples)
def test_notcointegrated(seed, method, axis, trend, gamma, n_samples):
    coint = CointAnalysis(method=method, axis=axis, trend=trend)
    X = make_notcointegrated(seed=seed, n_samples=n_samples, gamma=gamma)

    assert coint.test(X).pvalue_ > 0.1


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('method', params_method)
@pytest.mark.parametrize('axis', params_axis)
@pytest.mark.parametrize('trend', params_trend)
@pytest.mark.parametrize('n_samples', params_n_samples)
def test_stationary(seed, method, axis, trend, n_samples):
    coint = CointAnalysis(method=method, axis=axis, trend=trend)
    x0 = make_stationary(seed=seed, n_samples=n_samples, n_series=1)
    x1 = make_nonstationary(seed=seed, n_samples=n_samples, n_series=1)
    X = np.concatenate([x0, x1], axis=1)

    assert coint.test(X).pvalue_ is np.nan


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('method', params_method)
@pytest.mark.parametrize('axis', params_axis)
@pytest.mark.parametrize('trend', params_trend)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('gamma', params_gamma)
def test_fit(seed, method, axis, trend, n_samples, gamma):
    coint = CointAnalysis(method=method, axis=axis, trend=trend)
    X = make_cointegrated(seed=seed, n_samples=n_samples, gamma=gamma)
    coint.fit(X)

    assert np.isclose(-coint.coef_[0] / coint.coef_[1], gamma, rtol=1e-1)
    assert np.isclose(coint.mean_, 0, atol=abs(coint.coef_[1] * 1e-1))
    assert np.isclose(coint.std_, np.abs(coint.coef_[1]), rtol=1e-1)


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('method', params_method)
@pytest.mark.parametrize('axis', params_axis)
@pytest.mark.parametrize('trend', params_trend)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('gamma', params_gamma)
@pytest.mark.parametrize('adjust', params_adjust)
def test_transform(seed, method, axis, trend, n_samples, gamma, adjust):
    coint = CointAnalysis(method=method, axis=axis, trend=trend)
    X = make_cointegrated(seed=seed, n_samples=n_samples, gamma=gamma)

    spread = coint.fit_transform(X)

    if adjust:
        assert np.isclose(spread.mean(), 0.0, atol=1e-1)
        assert np.isclose(spread.std(), 1.0, atol=1e-1)
    else:
        assert np.isclose(spread.mean(), coint.mean_, atol=1e-1)
        assert np.isclose(spread.std(), coint.std_, atol=1e-1)
