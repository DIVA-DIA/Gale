"""
Linear Discriminant Analysis Algorithm

It computes both the transformation matrix and the linear discriminants
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def transform(X, y, solver, **kwargs):
    """Computes the transformation matrix with sklearn

    Parameters
    ----------
    X: nd.array(N, m)
        Input data. On the rows are the samples and on the columns are the
        dimensions. Typically you want to have NUM_ROWS >> NUM_COLS otherwise
        you get a singular-matrix error
    y: nd.array(N,)
        Labels of the input data in the range [0, num_classes-1].
    solver : str
        Either 'eigen' or 'svd'

    Returns
    -------
    L: nd.array(N,m)
        LDA scaling (transformation) coefficient matrix L
    C: nd.array(N,)
        Vector with the mean of the input data
    """
    # Check for sizes
    assert len(X.shape) == 2
    assert len(y.shape) == 1

    clf = LinearDiscriminantAnalysis(solver=solver)
    clf.fit(X, y)

    # The negative is returned to match mathematical definition
    return -clf.scalings_, np.mean(X, axis=0)


def discriminants(X, y, solver, **kwargs):
    """Computes the discriminant coefficients with sklearn

    Parameters
    ----------
    X: nd.array(N, m)
        Input data. On the rows are the samples and on the columns are the
        dimensions. Typically you want to have NUM_ROWS >> NUM_COLS otherwise
        you get a singular-matrix error
    y: nd.array(N,)
        Labels of the input data in the range [0, num_classes-1].
    solver : str
        Either 'eigen' or 'svd'

    Returns
    -------
    L: nd.array(N,m)
        LDA classification coefficient matrix L
    C: nd.array(N,)
        Vector with the bias for the classification coefficients
    """
    # Check for sizes
    assert len(X.shape) == 2
    assert len(y.shape) == 1

    # Create the solver
    clf = LinearDiscriminantAnalysis(solver=solver)
    clf.fit(X=X, y=y)
    return clf.coef_, clf.intercept_
