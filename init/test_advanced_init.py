import numpy as np

from init.advanced_init import _reshape_flattened_conv_filters, _flatten_conv_filters


def test_flatten_conv_filters():
    m = np.zeros((24, 3, 5, 5))
    m[0] = 1  # Make a filter all to ones

    exp = np.zeros((75, 24))
    exp[:, 0] = 1

    r = _flatten_conv_filters(m)

    np.testing.assert_almost_equal(r, exp)


def test_reshape_flattened_conv_filters():
    m = np.zeros((75, 24))
    m[:, 0] = 1

    r = _reshape_flattened_conv_filters(m, 5)

    exp = np.zeros((24, 3, 5, 5))
    exp[0] = 1  # Make a filter all to ones

    np.testing.assert_almost_equal(r, exp)
