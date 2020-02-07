import numpy as np
from sklearn.decomposition import PCA
from util.pca import transform

def test_pca_transform():
    X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

    P, C = transform(X)

    # From MATLAB
    exp_P = np.asarray([[-0.6779, -0.7352], [-0.7352, 0.6779]])
    exp_C = np.asarray(np.mean(X, axis=0))

    np.testing.assert_almost_equal(P, exp_P, decimal=4)
    np.testing.assert_almost_equal(C, exp_C, decimal=4)

    # From sklearn
    pca = PCA().fit(X)
    p = pca.components_.T  # Don't even think about touching this T!
    c = pca.mean_

    np.testing.assert_almost_equal(P, p)
    np.testing.assert_almost_equal(C, c)
