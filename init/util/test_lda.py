import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from init.util.lda import transform, discriminants


def test_lda_transform():
    X = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4], [9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    W, B = transform(X, y)

    # From MATLAB
    exp_W = np.asarray([[0.9196, -0.5952], [0.3930, 0.8036]])
    exp_B = np.asarray(np.mean(X, axis=0))

    np.testing.assert_almost_equal(W, exp_W, decimal=4)
    np.testing.assert_almost_equal(B, exp_B, decimal=4)

    # From sklearn
    clf = LinearDiscriminantAnalysis(solver='eigen')
    clf.fit(X, y)

    # The '-' is necessary to match the Matlab ones and does not affect the results
    np.testing.assert_almost_equal(W, -clf.scalings_)
    np.testing.assert_almost_equal(B, np.mean(X, axis=0))


def test_lda_discriminant():
    X = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4], [9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    W, B = discriminants(X, y)

    exp_W = np.asarray([[2.0282, 5.5519], [1.2599, 2.7657]])
    exp_B = np.asarray([-6.0033, -34.5205])

    # exp_W and exp_B are transposed to match PyTorch requirement of weight matrix orientation
    np.testing.assert_almost_equal(W, exp_W.T, decimal=4)
    np.testing.assert_almost_equal(B, exp_B.T, decimal=4)

#
# import time
# def test_lda_speed():
#
#     X = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4], [9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
#     y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
#
#     start = time.time()
#
#     for i in range(10000):
#         W, B = transform(X, y)
#
#     end = time.time()
#     print("\nOurs")
#     print(end - start)
#
#     start = time.time()
#     clf = LinearDiscriminantAnalysis(solver='eigen')
#     for i in range(10000):
#
#         clf.fit(X, y)
#
#     end = time.time()
#     print("\nCLF")
#     print(end - start)
