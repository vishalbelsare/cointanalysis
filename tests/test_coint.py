import pytest

from itertools import product
import numpy as np

from cointanalysis import CointAnalysis


@pytest.fixture
def gauss():
    np.random.seed(42)
    return np.random.randn(1000, 2)

@pytest.fixture
def methods():
    return ('AEG', )

@pytest.fixture
def axes():
    return ('0', '1', 'PCA')

@pytest.fixture
def trends():
    return ('c', )

@pytest.fixture
def adjusts():
    return (True, False)


def test_positive(gauss, methods, axes, trends):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = gamma * x0_t + gauss[:, 1]
    for various gamma.

    cf: Hamilton [19.11.1, 19.11.2]
    """
    for (method, axis, trend) in product(methods, axes, trends):
        coint = CointAnalysis(
            method=method,
            axis=axis,
            trend=trend,
        )
        for gamma in (1.0, 2.0, 10.0, -1.0, -2.0, -10.0):
            x0 = gauss[:, 0].cumsum()
            x1 = gamma * x0 + gauss[:, 1]
            X = np.array([x0, x1]).T

            assert coint.pvalue(X) < 0.1

def test_negative(gauss, methods, axes, trends):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = x1_t-1 + gamma * x0_t + gauss[:, 1]
    for various gamma.
    """
    for (method, axis, trend) in product(methods, axes, trends):
        coint = CointAnalysis(
            method=method,
            axis=axis,
            trend=trend,
        )
        for gamma in (1.0, 2.0, 10.0, -1.0, -2.0, -10.0):
            x0 = gauss[:, 0].cumsum()
            x1 = (gamma * x0 + gauss[:, 1]).cumsum()
            X = np.array([x0, x1]).T

            assert coint.pvalue(X) > 0.1

def test_stationary(gauss, methods, axes, trends):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = gauss[:, 1]  (stationary as such)
    """
    for (method, axis, trend) in product(methods, axes, trends):
        coint = CointAnalysis(
            method=method,
            axis=axis,
            trend=trend,
        )
        x0 = gauss[:, 0].cumsum()
        x1 = gauss[:, 1]
        X = np.array([x0, x1]).T

        assert coint.pvalue(X) is np.nan

def test_fit(gauss, methods, axes, trends):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = gamma * x0_t + gauss[:, 1]
    for various gamma.
    """
    for (method, axis, trend) in product(methods, axes, trends):
        coint = CointAnalysis(
            method=method,
            axis=axis,
            trend=trend,
        )
        for gamma in (1.0, 2.0, 10.0, -1.0, -2.0, -10.0):
            x0 = gauss[:, 0].cumsum()
            x1 = gamma * x0 + gauss[:, 1]
            X = np.array([x0, x1]).T

            coint.fit(X)

            gamma_fit = - coint.coef_[0] / coint.coef_[1]
            gamma_exp = gamma
            mean_fit = coint.mean_
            mean_exp = 0.0
            std_fit = coint.std_
            std_exp = np.abs(coint.coef_[1])

            assert np.isclose(gamma_fit, gamma_exp, rtol=1e-1)
            assert np.isclose(mean_fit, mean_exp, atol=np.abs(coint.coef_[1]) * 1e-1)
            assert np.isclose(std_fit, std_exp, rtol=1e-1)

def test_transform(gauss, methods, axes, trends, adjusts):
    """
    cointegrated pair:
        - x0_t = x0_t-1 + gauss[:, 0]
        - x1_t = gamma * x0_t + gauss[:, 1]
    for various gamma.
    """
    for (method, axis, trend, adjust) in product(methods, axes, trends, adjusts):
        coint = CointAnalysis(
            method=method,
            axis=axis,
            trend=trend,
            adjust_mean=adjust,
            adjust_std=adjust,
        )
        for gamma in (1.0, 2.0, 10.0, -1.0, -2.0, -10.0):
            x0 = gauss[:, 0].cumsum()
            x1 = gamma * x0 + gauss[:, 1]
            X = np.array([x0, x1]).T

            spread = coint.fit_transform(X)

            if adjust:
                mean_exp = 0.0
                std_exp = 1.0
            else:
                mean_exp = coint.mean_
                std_exp = coint.std_

            assert np.isclose(spread.mean(), mean_exp, atol=1e-2)
            assert np.isclose(spread.std(), std_exp, rtol=1e-2)
