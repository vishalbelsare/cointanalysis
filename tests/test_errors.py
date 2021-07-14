import numpy as np
import pytest

from cointanalysis import CointAnalysis
from cointanalysis._stat import StationarityTester
from cointanalysis._utils import check_shape

# --------------------------------------------------------------------------------


@pytest.mark.parametrize("n_features", [0, 1, 3])
def test_check_shape(n_features):
    with pytest.raises(ValueError):
        X = np.random.randn(100, n_features)
        check_shape(X, n_features=2)


def test_stat_method():
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        StationarityTester(method="hoge").pvalue(X)


def test_stat_regression():
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        StationarityTester(regression="hoge").pvalue(X)


def test_coint_fit():
    X = np.random.randn(100, 2)

    with pytest.raises(ValueError):
        coint = CointAnalysis(method="hoge")
        coint.fit(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(axis=2)
        coint.fit(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(trend="ct")
        coint.fit(X)


def test_coint_test():
    X = np.random.randn(100, 2)

    with pytest.raises(ValueError):
        coint = CointAnalysis(method="hoge")
        coint.test(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(axis=2)
        coint.test(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(trend="ct")
        coint.test(X)


# def test_collinear():
#     coint = CointAnalysis(axis='PCA')
#     x = np.random.randn(1000).cumsum()
#     small_noise = 0.001 * np.random.randn(1000)
#     X = np.stack([x, x + small_noise], axis=1)

#     with pytest.raises(RuntimeWarning):
#         coint.test(X)
