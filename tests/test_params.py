import pytest

import numpy as np
from fracdiff._stat import StationarityTester


# --------------------------------------------------------------------------------


def test_stat_method():
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        StationarityTester(method='hoge').pvalue(X)
