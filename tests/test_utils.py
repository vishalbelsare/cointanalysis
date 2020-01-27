import pytest

import numpy as np
from cointanalysis._utils import rms


array_rms = [
    (np.full((100, ), value), np.abs(value))
    for value in [1.0, 2.0, 10.0, -1.0, -2.0, -10.0]
]
array_rms += [
    (np.arange(1.0, 5.0 + 1.0), np.sqrt(55.0 / 5))
]


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('array, rms_expected', array_rms)
def test_rms(array, rms_expected):
    assert np.isclose(rms(array), rms_expected)
