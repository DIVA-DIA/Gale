"""
Principal Component Analysis Algorithm
"""

import logging
import numpy as np

def transform(X):

    # Check for sizes
    assert len(X.shape) == 2

    ###########################################################################
    # Step 1: Computing the mean vectors
    means = np.mean(X, axis=0)

    ###########################################################################
    # Step 2: Center data
    X = X - means

    ###########################################################################
    # Step 3: Computing the Covariance matrix
    C = np.cov(X.transpose())

    ###########################################################################
    # Step 4: Solving the generalized eigenvalue problem
    # beware of pinv() instead of inv()!!
    try:
        eig_vals, eig_vecs = np.linalg.eig(C)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        logging.error('np.linalg.LinAlgError raised, trying with pinv() instead')
        eig_vals, eig_vecs = np.linalg.eig(C)
        pass

    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    ###########################################################################
    # Stp 5: Sort the (eigenvalue, eigenvector) tuples from high to low
    L = eig_vecs[:, np.argsort(eig_vals)[::-1]]

    # Biases are needed to re-center the data
    B = means

    return L, B
