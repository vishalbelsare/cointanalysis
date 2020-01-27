import pytest

import numpy as np
from cointanalysis._stat import StationarityTester
from cointanalysis import CointAnalysis


# --------------------------------------------------------------------------------


def test_stat_method():
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        StationarityTester(method='hoge').pvalue(X)


def test_stat_regression():
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        StationarityTester(regression='hoge').pvalue(X)


def test_coint_fit():
    X = np.random.randn(100, 2)

    with pytest.raises(ValueError):
        coint = CointAnalysis(method='hoge')
        coint.fit(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(axis=2)
        coint.fit(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(trend='ct')
        coint.fit(X)


def test_coint_test():
    X = np.random.randn(100, 2)

    with pytest.raises(ValueError):
        coint = CointAnalysis(method='hoge')
        coint.test(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(axis=2)
        coint.test(X)

    with pytest.raises(ValueError):
        coint = CointAnalysis(trend='ct')
        coint.test(X)
