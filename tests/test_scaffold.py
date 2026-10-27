import numpy as np

import numerical_foundations


def test_package_imports():
    assert numerical_foundations is not None


def test_numpy_is_available():
    assert np.allclose(np.array([1.0, 2.0]).sum(), 3.0)